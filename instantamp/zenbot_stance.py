"""Zenbot double-support capture from the ordinary 480-value encoder and IMU history."""

from threading import Lock

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import zenbot as robot
from .sim import build_scene


def stance_key(frame):
    """Yaw-invariant onboard state used to recognize physical capture examples."""
    return np.concatenate((
        .10 * frame[..., :3], frame[..., 3:9], .75 * frame[..., 9:32],
        .038 * frame[..., 32:55], 1.20 * frame[..., 78:],
        .030 * frame[..., 55 + np.array([0, 1, 3, 4, 6, 7, 9, 10])],
    ), axis=-1)


class _ZenbotDynamics:
    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        self.feet = [model.body("robot_" + name).id for name in robot.FEET]
        self.torso = model.body("robot_" + robot.TORSO).id
        self.identity = np.eye(model.nv)
        self.cone = np.array([[.7, 0, 1], [-.7, 0, 1], [0, .7, 1], [0, -.7, 1]])
        self.sole = np.array([.045, 0., -.033])
        self.corners = self.sole + np.array([[x, y, 0.] for x in (-.088, .088) for y in (-.031, .031)])
        self.edges = self.sole + np.array([[x, y, 0.] for x in (-.110, .110) for y in (-.040, .040)])
        self.regularizer = .025 * np.eye(32)
        self.load_prior = np.full(32, .025 * model.body_mass.sum() * 9.81 / 32)

    def estimate(self, frame):
        """Estimate height and translation velocity from the low sole corners and the IMU."""
        m, d = self.model, self.data
        x, z = frame[3:6].astype(float), frame[6:9].astype(float)
        x /= max(1e-6, np.linalg.norm(x))
        z -= x * (x @ z)
        z /= max(1e-6, np.linalg.norm(z))
        d.qpos[:3] = 0.
        d.qpos[3:7] = Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z))).as_quat(scalar_first=True)
        d.qpos[7:] = frame[9:32]
        d.qvel[:3] = 0.
        d.qvel[3:6] = frame[:3]
        d.qvel[6:] = frame[32:55]
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        heights, rates = [], []
        jac = np.empty((3, m.nv))
        for foot in self.feet:
            rotation = d.xmat[foot].reshape(3, 3)
            corners = d.xpos[foot] + self.edges @ rotation.T
            heights.append(corners[:, 2].min())
            weights = np.exp(-np.minimum(35., (corners[:, 2] - heights[-1]) / .012))
            contact_point = np.average(corners, axis=0, weights=weights)
            mj.mj_jac(m, d, jac, None, contact_point, foot)
            rates.append(jac @ d.qvel)
        low = min(heights)
        supports = np.exp(-np.minimum(35., (np.asarray(heights) - low) / .008))
        d.qvel[:3] = -np.average(rates, axis=0, weights=supports)
        d.qpos[2] = -low - .0005
        mj.mj_forward(m, d)
        return np.ptp(heights)

    def target(self, lookahead, gain, anchors=None):
        """Regulate soles, COM and torso using bounded joint torques and a compressive contact wrench."""
        m, d = self.model, self.data
        jp, jr, jpd, jrd = [np.empty((3, m.nv)) for _ in range(4)]
        positions, contacts, tasks, targets = [], [], [], []

        def task(jac, error, rate, weight, bias=0.):
            tasks.append(weight * jac)
            targets.append(weight * (np.clip(rate**2 * error - 2 * rate * jac @ d.qvel, -80., 80.) - bias))

        for foot in self.feet:
            rotation = Rotation.from_matrix(d.xmat[foot].reshape(3, 3))
            sole = d.xpos[foot] + rotation.apply(self.sole)
            positions.append(sole)
            flat = Rotation.from_euler("z", np.arctan2(d.xmat[foot, 3], d.xmat[foot, 0]))
            mj.mj_jac(m, d, jp, jr, sole, foot)
            mj.mj_jacDot(m, d, jpd, jrd, sole, foot)
            error=np.array([0., 0., .003 - sole[2]])
            if anchors is not None:
                error[:2]=np.clip(anchors[len(positions)-1]-sole[:2],-.09,.09)
            task(jp, error, 7.5 * gain, 5., jpd @ d.qvel)
            task(jr, (flat * rotation.inv()).as_rotvec(), 6.8 * gain, 3.5, jrd @ d.qvel)
            for corner in d.xpos[foot] + rotation.apply(self.corners):
                mj.mj_jac(m, d, jp, None, corner, foot)
                contacts.append(self.cone @ jp)

        center = np.mean(positions, axis=0)
        if anchors is not None:
            center[:2]=np.mean(anchors,axis=0)
        center[2] = .715
        mj.mj_jacSubtreeCom(m, d, jp, 1)
        task(jp, center - d.subtree_com[1], 3.9 * gain, 5.)
        torso = Rotation.from_matrix(d.xmat[self.torso].reshape(3, 3))
        yaw = np.arctan2(d.xmat[self.torso, 3], d.xmat[self.torso, 0])
        desired = Rotation.from_euler("zyx", [yaw, .025, 0])
        mj.mj_jac(m, d, None, jr, d.xpos[self.torso], self.torso)
        task(jr, (desired * torso.inv()).as_rotvec(), 6. * gain, 2.4)
        task(self.identity[6:], robot.DEFAULT_Q - d.qpos[7:], 3.7, .30)
        a = np.vstack(tasks)
        accel = solve(a.T @ a + .001 * self.identity, a.T @ np.concatenate(targets), assume_a="pos")
        mass = np.empty((m.nv, m.nv))
        mj.mj_fullM(m, d, mass)
        effort = mass @ accel + d.qfrc_bias - d.qfrc_passive
        contact = np.vstack(contacts).T
        wrench, required = contact[:6].copy(), effort[:6].copy()
        wrench[3:] *= 6.
        required[3:] *= 6.
        force = nnls(np.vstack((wrench, self.regularizer)), np.r_[required, self.load_prior], maxiter=400)[0]
        torque = np.clip((effort - contact @ force)[6:], -robot.EFFORT, robot.EFFORT)
        target = d.qpos[7:] + lookahead * d.qvel[6:] + (torque + robot.KD * (d.qvel[6:] + lookahead * accel[6:])) / robot.KP
        return (target - robot.DEFAULT_Q) / robot.ACTION_SCALE


@jax.tree_util.register_static
class ZenbotStance:
    """Memory is per rollout. The callback shares only scratch MuJoCo storage protected by a lock."""

    def __init__(self, keys, parameters, settings, corrections=None):
        self.tree = cKDTree(keys)
        self.parameters = parameters
        self.corrections = corrections
        self.settings = settings
        self.dynamics = _ZenbotDynamics(build_scene(robot_name="zenbot23_getup"))
        self.lock = Lock()

    @classmethod
    def from_archive(cls, archive):
        names = ("keys", "parameters", "settings")
        if not any("zenbot_stance/" + name in archive for name in names):
            return None
        keys, params, settings = (np.asarray(archive["zenbot_stance/" + field], np.float32) for field in names)
        if (keys.ndim != 2 or not len(keys) or keys.shape[1] != 81 or params.shape != (len(keys), 2)
                or settings.shape != (3,) or not all(np.isfinite(x).all() for x in (keys, params, settings))
                or not (0 < settings[0] < 3 and 0 <= settings[1] <= 3 and 0 <= settings[2] <= 2)
                or np.any((params[:, 0] < 0) | (params[:, 0] > .05))
                or np.any((params[:, 1] < .3) | (params[:, 1] > 2))):
            raise ValueError("Invalid Zenbot sensor stance dimensions or parameters")
        corrections = archive.get("zenbot_stance/motion_correction")
        if corrections is not None:
            corrections = np.asarray(corrections, np.float32)
            if corrections.shape != (len(keys), 4) or not np.isfinite(corrections).all():
                raise ValueError("Invalid Zenbot sensor stance motion correction")
        return cls(keys, params, settings, corrections)

    def _host(self, observations, base_actions, memory):
        obs = np.asarray(observations).reshape(-1, 480)
        actions = np.array(base_actions, copy=True).reshape(-1, 23)
        memories = np.array(memory, copy=True).reshape(-1, 3)
        radius, max_speed, max_gyro = self.settings
        with self.lock:
            for frame, action, mem in zip(obs[:, -96:], actions, memories):
                upright = frame[8]
                if upright < .58:
                    mem[:] = 0.
                    continue
                near, idx = self.tree.query(stance_key(frame))
                if not mem[0] and near > radius and (max_speed == 0 or upright < .91):
                    continue
                gap = self.dynamics.estimate(frame)
                d = self.dynamics.data
                if self.corrections is not None and near < .65:
                    # Correct the ground-foot assumption locally in slipping or unloaded configurations.
                    # The training target is measured heading-relative velocity and root clearance.
                    weight = np.exp(-(near / .30)**2)
                    delta = weight * self.corrections[idx]
                    d.qpos[2] += delta[0]
                    d.qvel[:3] += delta[1:]
                    mj.mj_forward(self.dynamics.model, d)
                height, speed = d.qpos[2], np.linalg.norm(d.qvel[:2])
                if mem[0] and (height < .40 or np.linalg.norm(frame[:3]) > 6.0 or gap > .29):
                    mem[:] = 0.
                if not mem[0]:
                    sensed_support = .54 < height < .77 and upright > .93 and gap < .045 and speed < max_speed and np.linalg.norm(frame[:3]) < max_gyro and abs(d.qvel[2]) < .42
                    if not ((near <= radius and height > .42) or sensed_support):
                        continue
                    mem[:] = [1., *self.parameters[idx]]
                action[:] = self.dynamics.target(mem[1], mem[2])
        return actions.reshape(base_actions.shape), memories.reshape(memory.shape)

    def step(self, obs, base_actions, memory):
        specs = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (base_actions, memory))
        return jax.pure_callback(self._host, specs, obs, base_actions, memory, vmap_method="broadcast_all")
