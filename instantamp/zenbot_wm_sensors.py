"""Motion Train sensor replay and delayed Zenbot PD targets for world-model rollouts."""

import jax
import jax.numpy as jnp
import numpy as np

from . import env, zenbot
from .rot import mv3, tr3
from .zenbot_motion_train_obs import MotionTrainSensors, TERM_HISTORY_INDEX, advance_sensors

# observation() interleaves histories by term and reorders each set of joints.
FRAME_ORDER = np.argsort(TERM_HISTORY_INDEX)


def restore_sensors(observation, imu_delay, current_imu, previous_imu):
    frames = observation[..., FRAME_ORDER].reshape(*observation.shape[:-1], 3, 78)
    return MotionTrainSensors(imu_delay, current_imu, previous_imu, frames)


def pd_targets(action, previous, older, bias, context, *, elapsed=None):
    """The 4 × 23 angle targets delivered during the current 20 ms control interval."""
    delay = context[..., 2 + zenbot.GROUP].astype(jnp.int32)
    age = jnp.arange(env.DECIMATION)[:, None] - delay[..., None, :]
    if elapsed is not None:
        # Newton reads the oldest received command while the delay buffer fills after reset.
        age = jnp.maximum(age, -env.DECIMATION * jnp.asarray(elapsed)[..., None, None])
    applied = jnp.where(age >= 0, action[..., None, :],
                        jnp.where(age >= -env.DECIMATION, previous[..., None, :], older[..., None, :]))
    targets = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * applied - bias[..., None, :]
    return targets.reshape(*action.shape[:-1], -1)


def predicted_sensors(previous, state, command, applied_action, key, corrupt):
    """Sense the next predicted physical state with the same IMU clock and joint ordering as the simulator."""
    _, rotation, _, world_omega, angle, speed = state
    imu = jnp.concatenate((mv3(tr3(rotation), world_omega), -rotation[..., 2, :]), axis=-1)
    return advance_sensors(previous, imu, command, angle, speed, applied_action,
                           jnp.zeros(command.shape[:-1], bool), jax.random.fold_in(key, 234), corrupt)
