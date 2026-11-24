"""Sensor history, joint order, and observation symmetry for Zenbot policies."""

from functools import lru_cache
from typing import NamedTuple

import numpy as np

from . import zenbot_models


class JointLayout(NamedTuple):
    default_q: np.ndarray
    lab_order: np.ndarray
    history_index: np.ndarray
    frame_order: np.ndarray
    noise: np.ndarray
    mirror: np.ndarray
    signs: np.ndarray
    lab_mirror: np.ndarray


@lru_cache(maxsize=2)
def joint_layout(joints):
    """Build Motion Train's three-frame observation layout in MJCF joint order."""
    model = zenbot_models.for_joint_count(joints)
    order = model.lab_order
    frame_width = 9 + 3 * joints
    frames = np.arange(3 * frame_width).reshape(3, frame_width)
    terms = (
        frames[:, :3], frames[:, 3:6], frames[:, 6:9], frames[:, 9 + order],
        frames[:, 9 + joints + order], frames[:, 9 + 2 * joints + order],
    )
    index = np.concatenate([term.ravel() for term in terms])
    names = model.dof
    mirror_names = [
        name.replace("left_", "right_") if name.startswith("left_")
        else name.replace("right_", "left_") for name in names
    ]
    mirror = np.array([names.index(name) for name in mirror_names])
    signs = np.array(
        [-1 if name.endswith(("_roll", "_yaw")) else 1 for name in names],
        dtype=np.float32,
    )
    noise = np.repeat(
        [.28, .042, 0., .03, 1.5, 0.], [3, 3, 3, joints, joints, joints]
    ).astype(np.float32)
    return JointLayout(
        model.default_q, order, index, np.argsort(index), noise, mirror, signs,
        np.argsort(order)[mirror[order]],
    )


def mirror_action(action):
    """Reflect an action in the physical MJCF joint order."""
    layout = joint_layout(action.shape[-1])
    return action[..., layout.mirror] * layout.signs


def mirror_observation(observation):
    """Reflect all three frames of the term-major Motion Train observation."""
    import jax.numpy as jnp

    shape = observation.shape[:-1]
    vectors = observation[..., :27].reshape(*shape, 3, 3, 3)
    # Gyroscope is an axial vector, projected gravity is a polar vector,
    # and the command contains forward, lateral and yaw speeds.
    vector_signs = jnp.array([[-1, 1, -1], [1, -1, 1], [1, -1, -1]], observation.dtype)
    vectors = vectors * vector_signs[:, None, :]
    count = (observation.shape[-1] - 27) // 9
    layout = joint_layout(count)
    joints = observation[..., 27:].reshape(*shape, 3, 3, count)
    joints = joints[..., layout.lab_mirror] * layout.signs[layout.lab_order]
    return jnp.concatenate((vectors.reshape(*shape, 27), joints.reshape(*shape, 9 * count)), axis=-1)


class MotionTrainSensors(NamedTuple):
    delay_s: object
    current_imu: object
    previous_imu: object
    frames: object  # (world, oldest to newest time, physical-order features)

    def observation(self):
        layout = joint_layout((self.frames.shape[-1] - 9) // 3)
        return self.frames.reshape((*self.frames.shape[:-2], -1))[..., layout.history_index]


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

    layout = joint_layout(joint_angle.shape[-1])

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
        (imu, command, joint_angle - layout.default_q, joint_speed, last_action), axis=-1
    )
    if corrupt:
        frame += jax.random.uniform(noise_key, frame.shape, minval=-1.0, maxval=1.0) * layout.noise
    filled = jnp.repeat(frame[..., None, :], 3, axis=-2)
    if previous is not None:
        shifted = jnp.concatenate((previous.frames[..., 1:, :], frame[..., None, :]), axis=-2)
        filled = jnp.where(fresh[..., None, None], filled, shifted)
    return MotionTrainSensors(delay, raw_imu, last, filled)


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
