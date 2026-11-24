"""Motion Train sensor replay and delayed Zenbot PD targets for world-model rollouts."""

import jax
import jax.numpy as jnp

from . import env, zenbot
from .rot import mv3, tr3
from .zenbot_motion_train_obs import (
    MotionTrainSensors,
    advance_sensors,
    joint_layout,
)


def restore_sensors(observation, imu_delay, current_imu, previous_imu):
    width = observation.shape[-1] // 3
    order = joint_layout((width - 9) // 3).frame_order
    frames = observation[..., order].reshape(*observation.shape[:-1], 3, width)
    return MotionTrainSensors(imu_delay, current_imu, previous_imu, frames)


def pd_targets(action, previous, older, bias, context, *, elapsed=None, profile=zenbot):
    """Per-motor angle targets delivered at each physics substep of the 20 ms control interval."""
    delay = context[..., 2 + profile.GROUP].astype(jnp.int32)
    age = jnp.arange(env.DECIMATION)[:, None] - delay[..., None, :]
    if elapsed is not None:
        # Newton reads the oldest received command while the delay buffer fills after reset.
        age = jnp.maximum(age, -env.DECIMATION * jnp.asarray(elapsed)[..., None, None])
    applied = jnp.where(age >= 0, action[..., None, :],
                        jnp.where(age >= -env.DECIMATION, previous[..., None, :], older[..., None, :]))
    targets = profile.DEFAULT_Q + profile.ACTION_SCALE * applied - bias[..., None, :]
    return targets.reshape(*action.shape[:-1], -1)


def predicted_sensors(previous, state, command, applied_action, key, corrupt):
    """Sense the next predicted physical state with the same IMU clock and joint ordering as the simulator."""
    _, rotation, _, world_omega, angle, speed = state
    imu = jnp.concatenate((mv3(tr3(rotation), world_omega), -rotation[..., 2, :]), axis=-1)
    return advance_sensors(previous, imu, command, angle, speed, applied_action,
                           jnp.zeros(command.shape[:-1], bool), jax.random.fold_in(key, 234), corrupt)
