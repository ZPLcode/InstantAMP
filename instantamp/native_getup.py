"""Native MuJoCo sensing and actuator stepping for the LeggedLab G1 get-up task."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import env, g1_getup as robot


def observation_terms(data, links, previous):
    """Native MuJoCo equivalent of Sim.obs_frame for the named task."""
    q, qd = data.qpos[7:], data.qvel[6:]
    rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_matrix()
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    local = Rotation.from_rotvec([0., 0., -yaw]).as_matrix() @ rotation
    points = ((data.xpos[links] - data.qpos[:3]) @ rotation).flatten()
    return data.qvel[3:6], np.r_[local[:, 0], local[:, 2]], q, qd, previous, points


def advance(model, data, action):
    """Use the get-up position drives, effort saturation and speed cutoffs at 200 Hz."""
    target = robot.DEFAULT_Q + robot.ACTION_SCALE * action
    for _ in range(env.DECIMATION):
        q, qd = data.qpos[7:], data.qvel[6:]
        tau = np.clip(robot.KP * (target - q) - robot.KD * qd, -robot.EFFORT, robot.EFFORT)
        tau[(np.abs(qd) >= robot.SPEED) & (tau * qd > 0)] = 0.
        data.ctrl[:] = q + (tau + robot.KD * qd) / robot.KP
        mujoco.mj_step(model, data)
    mujoco.mj_kinematics(model, data)
