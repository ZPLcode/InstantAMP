"""Training observations for Motion Train's 43 kg Zenbot AMP locomotion actor."""

from typing import NamedTuple

import numpy as np

# Motion Train's 43 kg Zenbot robot_cfg.usd2urdf. Each value indexes an MJCF hinge.
LAB_ORDER = np.array(
    [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
)
_history = np.arange(5 * 78).reshape(5, 78)[-3:]
MOTION_TRAIN_INDEX = np.concatenate(
    [
        block.ravel()
        for block in (
            _history[:, :3],
            _history[:, 3:6],
            _history[:, 6:9],
            _history[:, 9 + LAB_ORDER],
            _history[:, 32 + LAB_ORDER],
            _history[:, 55 + LAB_ORDER],
        )
    ]
)
MOTION_TRAIN_WIDTH = len(MOTION_TRAIN_INDEX)
TERM_HISTORY_INDEX = MOTION_TRAIN_INDEX - 2 * 78
SENSOR_NOISE = np.repeat([0.28, 0.042, 0.0, 0.03, 1.5, 0.0], [3, 3, 3, 23, 23, 23]).astype(
    np.float32
)


# The sagittal reflection swaps the matching limbs. Pitch and knee flexion
# keep their signs. Roll, yaw and the unpaired waist hinge change signs.
_MIRROR_JOINTS = np.r_[6:12, 0:6, 12, 18:23, 13:18]
_MIRROR_SIGNS = np.array([1, -1, -1, 1, 1, -1] * 2 + [-1] + [1, -1, -1, 1, -1] * 2, np.float32)
_MIRROR_LAB_JOINTS = np.argsort(LAB_ORDER)[_MIRROR_JOINTS[LAB_ORDER]]


def mirror_action(action):
    """Reflect a 23-channel action in the physical MJCF joint order."""
    return action[..., _MIRROR_JOINTS] * _MIRROR_SIGNS


def mirror_observation(observation):
    """Reflect all three frames of the term-major 234-channel Motion Train observation."""
    import jax.numpy as jnp

    shape = observation.shape[:-1]
    vectors = observation[..., :27].reshape(*shape, 3, 3, 3)
    # Gyroscope is an axial vector, projected gravity is a polar vector,
    # and the command contains forward, lateral and yaw speeds.
    vector_signs = jnp.array([[-1, 1, -1], [1, -1, 1], [1, -1, -1]], observation.dtype)
    vectors = vectors * vector_signs[:, None, :]
    joints = observation[..., 27:].reshape(*shape, 3, 3, 23)
    joints = joints[..., _MIRROR_LAB_JOINTS] * _MIRROR_SIGNS[LAB_ORDER]
    return jnp.concatenate((vectors.reshape(*shape, 27), joints.reshape(*shape, 207)), axis=-1)


class MotionTrainSensors(NamedTuple):
    delay_s: object
    current_imu: object
    previous_imu: object
    frames: object  # (world, oldest -> newest time, 78 physical-order features)

    def observation(self):
        return self.frames.reshape((*self.frames.shape[:-2], -1))[..., TERM_HISTORY_INDEX]


def map_imu_delay(samples):
    """Motion Train's per-episode 90% nominal / 10% tail latency distribution, in seconds."""
    import jax.numpy as jnp

    return jnp.where(
        samples[..., 0] < 0.9, 0.002 + 0.008 * samples[..., 1], 0.010 + 0.025 * samples[..., 1]
    )


def interpolate_delayed_imu(current, previous, older, delay_s, step_dt=0.02):
    """Interpolate clean gyro and projected gravity, then normalize the delayed gravity vector."""
    import jax.numpy as jnp

    delay = delay_s[..., None]
    first = current + jnp.clip(delay / step_dt, 0.0, 1.0) * (previous - current)
    second = previous + jnp.clip((delay - step_dt) / step_dt, 0.0, 1.0) * (older - previous)
    delayed = jnp.where(delay <= step_dt, first, second)
    gravity = delayed[..., 3:6]
    gravity /= jnp.maximum(jnp.linalg.norm(gravity, axis=-1, keepdims=True), 1e-6)
    return jnp.concatenate((delayed[..., :3], gravity), axis=-1)


def advance_sensors(
    previous, raw_imu, command, joint_angle, joint_speed, last_action, fresh, key, corrupt=True
):
    """Advance one 50 Hz actor observation, refilling both histories for reset worlds."""
    import jax
    import jax.numpy as jnp

    from . import zenbot

    delay_key, noise_key = jax.random.split(key)
    delay = (
        map_imu_delay(jax.random.uniform(delay_key, (*fresh.shape, 2)))
        if corrupt
        else jnp.zeros(fresh.shape)
    )
    if previous is None:
        # Both samples start at the same value. The collector donates the state to XLA,
        # so current_imu and previous_imu must own separate buffers.
        last, older = raw_imu.copy(), raw_imu
    else:
        delay = jnp.where(fresh, delay, previous.delay_s)
        last = jnp.where(fresh[..., None], raw_imu, previous.current_imu)
        older = jnp.where(fresh[..., None], raw_imu, previous.previous_imu)
    imu = interpolate_delayed_imu(raw_imu, last, older, delay) if corrupt else raw_imu
    frame = jnp.concatenate(
        (imu, command, joint_angle - zenbot.DEFAULT_Q, joint_speed, last_action), axis=-1
    )
    if corrupt:
        frame += jax.random.uniform(noise_key, frame.shape, minval=-1.0, maxval=1.0) * SENSOR_NOISE
    filled = jnp.repeat(frame[..., None, :], 3, axis=-2)
    if previous is not None:
        shifted = jnp.concatenate((previous.frames[..., 1:, :], frame[..., None, :]), axis=-2)
        filled = jnp.where(fresh[..., None, None], filled, shifted)
    return MotionTrainSensors(delay, raw_imu, last, filled)


def teacher_yaw(command, measured):
    """Calibrate the 14300 demonstrator's turn rate from measured body gyro while collecting labels."""
    sign = (command > 0) * 1.0 - (command < 0) * 1.0
    return (command != 0.0) * (1.42 * sign + command - measured)


def motion_train_observation(frames):
    """Select the 234 clean policy inputs from five full-state Zenbot frames."""
    from .zenbot_command_actor import gait_teacher_observation

    if frames.shape[-2:] != (5, 99):
        raise ValueError("Expected five consecutive 99-channel Zenbot sensor frames")
    return gait_teacher_observation(frames)[..., MOTION_TRAIN_INDEX]


def motion_train_actor_mean(actor, norm, observation):
    """Run the dense Zenbot actor with the float32 arithmetic used by Motion Train."""
    import jax
    import jax.numpy as jnp

    from . import nets

    if "raw" not in norm:
        observation = jnp.clip(nets.rn_apply(norm, observation), -10.0, 10.0)
    with jax.default_matmul_precision("highest"):
        return nets.mlp_apply(
            actor, observation, jax.nn.elu, ln_eps=1e-6, compute_dtype=jnp.float32
        )
