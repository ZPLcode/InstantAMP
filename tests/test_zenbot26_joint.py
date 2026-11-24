import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import zenbot26, zenbot_source_gait
from instantamp.train import parse
from instantamp.zenbot_motion_train_obs import (
    MotionTrainSensors,
    joint_layout,
    mirror_action,
    mirror_observation,
)
from instantamp.zenbot_wm_sensors import restore_sensors


def test_native_and_imagined_sensor_history_preserve_all_26_joint_channels():
    frames = jnp.arange(3 * 87, dtype=jnp.float32).reshape(1, 3, 87)
    sensors = MotionTrainSensors(None, None, None, frames)
    observed = np.asarray(sensors.observation())
    layout = joint_layout(26)
    expected = np.concatenate([np.asarray(frames[0, :, i:i+3]).ravel() for i in (0, 3, 6)] +
                              [np.asarray(frames[0])[:, i + layout.lab_order].ravel() for i in (9, 35, 61)])
    np.testing.assert_array_equal(observed[0], expected)
    np.testing.assert_array_equal(restore_sensors(observed, None, None, None).frames, frames)
    np.testing.assert_array_equal(mirror_observation(mirror_observation(observed)), observed)
    action = np.arange(1, 27)
    reflection = mirror_action(action)
    np.testing.assert_array_equal(reflection[12:16], action[12:16] * [-1, -1, 1, -1])


def test_joint_task_trains_both_motion_priors_and_samples_running_speeds():
    cfg = parse(["--robot", zenbot26.MIXED_TASK])
    assert cfg.amp_getup and cfg.wm_global_gait and cfg.amp_dir == cfg.walk_dir
    assert cfg.amp_getup_dir == cfg.floor_dir == "data/zenbot26_walk_run_recovery/recovery"
    assert cfg.walk_dir == "data/zenbot26_walk_run_recovery/walk_run"
    pose, velocity = zenbot_source_gait.stand_reset(jax.random.key(4), 8, zenbot26.DEFAULT_Q)
    assert pose.shape == (8, 33) and velocity.shape == (8, 32)
    cmd = zenbot_source_gait.sample_command(jax.random.key(5), 2048, stage=2, side_share=cfg.wm_side_share)
    assert float(cmd.vel[:, 0].max()) > 2.9 and float(cmd.vel[:, 0].min()) < -.9
