"""Estimate G1 double-support capture feasibility from the five proprioceptive frames."""

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .native_getup import advance


def support_features(observation):
    """Kinematic estimates in the pelvis yaw frame. Lower feet get more odometry weight."""
    raw = np.asarray(observation)
    gyro = raw[..., 12:15]
    rotation = raw[..., 39:45]
    matrix = np.stack((rotation[..., :3], np.cross(rotation[..., 3:], rotation[..., :3]),
                       rotation[..., 3:]), axis=-1)
    landmarks = raw[..., 480:].reshape(*raw.shape[:-1], 5, 6, 3)
    feet = landmarks[..., :2, :]
    current = feet[..., -1, :, :]
    center = np.einsum('...ij,...fj->...fi', matrix, current)
    depth = -center[..., 2]
    midpoint = center.mean(-2)
    slope = np.einsum('t,...tfj->...fj', np.arange(-2, 3) / .20, feet)
    foot_speeds = -np.einsum('...ij,...fj->...fi', matrix, slope + np.cross(gyro[..., None, :], current))
    weights = np.exp(np.clip((depth - depth.max(-1, keepdims=True)) / .022, -30, 0))
    weights /= weights.sum(-1, keepdims=True)
    velocity = (weights[..., None] * foot_speeds).sum(-2)
    capture = -midpoint[..., :2] + np.sqrt(np.maximum(.24, depth.max(-1, keepdims=True) + .04) / 9.81) * velocity[..., :2]
    spread = center[..., 0, :] - center[..., 1, :]
    angles, speeds = raw[..., 161:190], raw[..., 306:335]
    return np.concatenate((depth.mean(-1, keepdims=True), np.abs(depth[..., :1] - depth[..., 1:2]),
                           rotation[..., 5:6], np.linalg.norm(gyro[..., :2], axis=-1, keepdims=True),
                           np.linalg.norm(gyro, axis=-1, keepdims=True), velocity, np.abs(velocity),
                           capture, np.abs(capture), midpoint[..., :2], np.abs(midpoint[..., :2]),
                           spread, np.abs(spread), np.linalg.norm(foot_speeds, axis=-1),
                           np.linalg.norm(foot_speeds[..., 0, :] - foot_speeds[..., 1, :], axis=-1, keepdims=True),
                           gyro, rotation[..., [2, 3, 4]], angles[..., [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]],
                           speeds[..., [0, 3, 4, 6, 9, 10]], angles[..., [15, 22]]), axis=-1)


def solver_observation(data):
    """Synthetic IMU/encoder sample used only inside an internal hypothetical rollout."""
    r = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()
    yaw = np.arctan2(r[1, 0], r[0, 0])
    local = Rotation.from_rotvec([0, 0, -yaw]).as_matrix() @ r
    raw = np.zeros(570, np.float32)
    raw[39:45] = np.r_[local[:, 0], local[:, 2]]
    raw[12:15] = data.qvel[3:6]
    raw[161:190] = data.qpos[7:]
    raw[306:335] = data.qvel[6:]
    return raw


class CaptureEligibility:
    """Fit on physical intervention outcomes. Forecast only promising new support states."""

    def __init__(self, archive, model):
        prefix = 'momentum_capture/general/'
        values = {name[len(prefix):]: archive[name] for name in archive if name.startswith(prefix)}
        self.threshold, self.velocity_radius = values['settings']
        self.splits = values['splits']
        self.thresholds = values['thresholds']
        self.left = values['left']
        self.right = values['right']
        self.success = values['success']
        self.tree_index = np.arange(len(self.splits))
        self.known = cKDTree(values['known'])
        self.network = values
        self.parameters = values['parameters']
        self.data = mj.MjData(model)

    def probabilities(self, features):
        node = np.zeros(len(self.splits), int)
        for _ in range(12):
            index = (self.tree_index, node)
            node = np.where(features[self.splits[index]] < self.thresholds[index], self.left[index], self.right[index])
        return self.success[self.tree_index, node].reshape(4, -1).mean(-1)

    def estimate(self, raw, features):
        x = np.concatenate((features, raw[12:15], raw[39:45], raw[161:176], raw[306:321], raw[451:466], raw[552:]))
        net = self.network
        y = np.clip((x - net['mean']) / net['std'], -6., 6.)
        for i in range(3):
            y = y @ net['w' + str(i)] + net['b' + str(i)]
            if i < 2:
                y = np.where(y > 0, y, np.expm1(np.minimum(y, 0.)))
        return y * net['ys'] + net['ym']

    def choose(self, raw, policy_features, solver):
        if raw[44] < .57 or self.known.query(policy_features, distance_upper_bound=.025)[0] < .025:
            return -1, 0
        features = support_features(raw)
        prob = self.probabilities(features)
        mode = int(np.argmax(prob))
        if prob[mode] < self.threshold:
            return -1, 0
        estimate = self.estimate(raw, features)
        solver._estimate_state(raw)
        qpos = solver.data.qpos.copy()
        velocity = solver.data.qvel.copy()
        qpos[2] = .9 * qpos[2] + .1 * estimate[3]
        velocity[:3] = .1 * velocity[:3] + .9 * estimate[:3]
        direction = velocity[:2] / max(.1, np.linalg.norm(velocity[:2]))
        for delta in (0., self.velocity_radius, -self.velocity_radius):
            trial = velocity.copy()
            trial[:2] += delta * direction
            if not self.can_stand(qpos, trial, raw[451:480], solver, self.parameters[mode]):
                return -1, 3
        return mode, 0

    def can_stand(self, position, velocity, previous, solver, params):
        model, data = solver.model, self.data
        mj.mj_resetData(model, data)
        data.qpos[:] = position
        data.qvel[:] = velocity
        mj.mj_forward(model, data)
        data.qacc_warmstart[:] = 0.
        height, arms, settled = 0., position[22:].copy(), 0
        for age in range(82):
            raw = solver_observation(data)
            target, height = solver.command(raw, params, age, height, arms)
            fraction = min(1., (age + 1) / params[1])
            advance(model, data, previous + fraction * (target - previous))
            up = 1. - 2. * np.sum(data.qpos[4:6]**2)
            if age > 12 and (data.qpos[2] < .375 or up < .30):
                return False
            standing = .65 < data.qpos[2] < .85 and up > .866 and np.linalg.norm(data.qvel[:2]) < .21
            settled = settled + 1 if standing else 0
        return settled >= 30
