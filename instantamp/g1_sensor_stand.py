"""Double-support standing feedback from the G1 actor's encoder and IMU observations.

At 50 Hz, the controller estimates pelvis velocity assuming that the lower foot is planted,
solves for accelerations to level both feet and regulate the trunk and center of mass, and
maps the resulting inverse-dynamics effort into the task's bounded position drives. A host
callback is used for the native MuJoCo Jacobians and the nonnegative contact-force solve.
"""

from functools import lru_cache
from threading import Lock

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from . import g1_getup as g1
from .sim import build_scene


def pose_features(anchor):
    """Initial action context of the posture policy, fixed while double support persists."""
    joints = [3, 9, 12, 13, 14, 15, 16, 18, 19, 22, 23, 25, 26]
    return np.r_[np.clip(anchor[joints] / 3., -2., 2.), 1.]


class _StandingDynamics:
    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        self.feet = [model.body("robot_" + name).id for name in g1.KEY_BODIES[:2]]
        self.trunk = model.body("robot_torso_link").id
        self.colliders = [np.flatnonzero((model.geom_bodyid == foot) & (model.geom_contype > 0))
                          for foot in self.feet]
        self.identity = np.eye(model.nv)
        self.foot_offset = np.array([.030, 0., -.030])
        self.corners = np.array([[x, y, 0.] for x in (-.055, .075) for y in (-.021, .021)])
        self.cone = np.array([[.6, 0., 1.], [-.6, 0., 1.], [0., .6, 1.], [0., -.6, 1.]])
        self.contact_regularizer = .015 * np.eye(32)
        self.neutral_force = np.full(32, model.body_subtreemass[1] * 9.81 * .015 / 32)
        self.posture = g1.DEFAULT_Q.copy()
        for name, target in {
            "left_knee": .26, "right_knee": .26,
            "left_shoulder_pitch": .04, "right_shoulder_pitch": .04,
            "left_shoulder_roll": .20, "right_shoulder_roll": -.20,
            "left_elbow": .86, "right_elbow": .86,
            "left_wrist_roll": .35, "right_wrist_roll": -.22,
        }.items():
            self.posture[g1.DOF.index(name)] = target

    def _estimate_state(self, obs):
        m, d = self.model, self.data
        x, z = obs[39:42].astype(float), obs[42:45].astype(float)
        x /= max(np.linalg.norm(x), 1e-9)
        z -= x * np.dot(x, z)
        z /= max(np.linalg.norm(z), 1e-9)
        rotation = Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z)))
        d.qpos[:3] = 0.
        d.qpos[3:7] = rotation.as_quat(scalar_first=True)
        d.qpos[7:] = obs[161:190]
        d.qvel[:3] = 0.
        d.qvel[3:6] = obs[12:15]
        d.qvel[6:] = obs[306:335]
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)

        bottoms, velocities = [], []
        jac = np.empty((3, m.nv))
        for foot, geoms in zip(self.feet, self.colliders):
            extent = m.geom_size[geoms, 0].copy()
            boxes = m.geom_type[geoms] == mj.mjtGeom.mjGEOM_BOX
            extent[boxes] = (np.abs(d.geom_xmat[geoms[boxes], 6:9]) * m.geom_size[geoms[boxes]]).sum(-1)
            bottoms.append(np.min(d.geom_xpos[geoms, 2] - extent))
            sole = d.xpos[foot] + d.xmat[foot].reshape(3, 3) @ self.foot_offset
            mj.mj_jac(m, d, jac, None, sole, foot)
            velocities.append(jac @ d.qvel)
        low = min(bottoms)
        weights = np.exp(-np.minimum(30., (np.array(bottoms) - low) / .011))
        d.qvel[:3] = -np.average(velocities, weights=weights, axis=0)
        d.qpos[2] = -low - .001
        mj.mj_forward(m, d)

    def __call__(self, obs, pose=None):
        self._estimate_state(obs)
        m, d = self.model, self.data
        trunk_rotation = Rotation.from_matrix(d.xmat[self.trunk].reshape(3, 3))
        yaw = trunk_rotation.as_euler("ZYX")[0]
        posture, pitch = self.posture, .10
        if pose is not None:
            # The high-level action adjusts bilateral arms and sagittal posture within a fixed range.
            posture = self.posture.copy()
            posture[[15, 22]] += .40 * pose[0] + np.array([1., -1.]) * .23 * pose[1]
            posture[[16, 23]] += np.array([1., -1.]) * .18 * pose[2]
            posture[[18, 25]] += .45 * pose[3] + np.array([1., -1.]) * .25 * pose[4]
            posture[[19, 26]] += np.array([1., -1.]) * .32 * pose[5]
            pitch += .075 * pose[6]
        target_trunk = Rotation.from_euler("ZYX", [yaw, pitch, 0.])
        foot_positions, forces, matrices, targets = [], [], [], []

        def task(jac, error, rate, weight, bias=0.):
            matrices.append(weight * jac)
            acceleration = rate**2 * error - 2 * rate * (jac @ d.qvel)
            targets.append(weight * (np.clip(acceleration, -90., 90.) - bias))

        jac, rot_jac, jac_dot, rot_dot = (np.empty((3, m.nv)) for _ in range(4))
        for foot in self.feet:
            rotation = Rotation.from_matrix(d.xmat[foot].reshape(3, 3))
            sole = d.xpos[foot] + rotation.apply(self.foot_offset)
            foot_positions.append(sole)
            flat = Rotation.from_euler("z", rotation.as_euler("ZYX")[0])
            mj.mj_jac(m, d, jac, rot_jac, sole, foot)
            mj.mj_jacDot(m, d, jac_dot, rot_dot, sole, foot)
            task(jac, np.array([0., 0., .004 - sole[2]]), 8., 6., jac_dot @ d.qvel)
            task(rot_jac, (flat * rotation.inv()).as_rotvec(), 8., 3.6, rot_dot @ d.qvel)
            for corner in sole + rotation.apply(self.corners):
                mj.mj_jac(m, d, jac, None, corner, foot)
                forces.append(self.cone @ jac)

        center = np.mean(foot_positions, axis=0) + Rotation.from_euler("z", yaw).apply([.018, 0., 0.])
        center[2] = .748
        mj.mj_jacSubtreeCom(m, d, jac, 1)
        task(jac, center - d.subtree_com[1], 4., 5.)
        mj.mj_jac(m, d, None, rot_jac, d.xpos[self.trunk], self.trunk)
        mj.mj_jacDot(m, d, None, rot_dot, d.xpos[self.trunk], self.trunk)
        task(rot_jac, (target_trunk * trunk_rotation.inv()).as_rotvec(), 7., 2., rot_dot @ d.qvel)
        task(self.identity[6:], posture - d.qpos[7:], 4., .22)
        a = np.vstack(matrices)
        accel = solve(a.T @ a + .0005 * self.identity, a.T @ np.concatenate(targets), assume_a="pos")
        mass = np.empty((m.nv, m.nv))
        mj.mj_fullM(m, d, mass)
        effort = mass @ accel + d.qfrc_bias - d.qfrc_passive

        # Satisfy the floating-base wrench with compressive contacts inside friction pyramids.
        # The small force prior shares otherwise indeterminate loads between the support points.
        contact = np.vstack(forces).T
        wrench, required = contact[:6].copy(), effort[:6].copy()
        wrench[3:] *= 6.
        required[3:] *= 6.
        load = nnls(np.vstack((wrench, self.contact_regularizer)), np.r_[required, self.neutral_force],
                    maxiter=200)[0]
        torque = np.clip((effort - contact @ load)[6:], -g1.EFFORT, g1.EFFORT)
        target = d.qpos[7:] + (torque + g1.KD * d.qvel[6:]) / g1.KP
        return (target - g1.DEFAULT_Q) / g1.ACTION_SCALE


@jax.tree_util.register_static
class SensorStance:
    """Evaluate only active environments. Inactive rows return zero actions for the caller to mask."""

    def __init__(self, pose_weight=None):
        self.pose_weight = pose_weight
        self.model = build_scene(robot_name="g1_getup")
        # JAX can invoke callbacks on short-lived external threads. Keep native MuJoCo
        # storage owned by the controller, including when that callback thread exits.
        self.dynamics = _StandingDynamics(self.model)
        self.lock = Lock()

    @classmethod
    def read(cls, archive):
        """Load an optional AMP-trained standing pose policy from the enclosing checkpoint."""
        if "quiet_pose/weight" not in archive:
            return sensor_stance()
        weight = np.asarray(archive["quiet_pose/weight"], np.float32)
        if weight.shape != (14, 7) or not np.isfinite(weight).all():
            raise ValueError("Expected finite 14-by-7 G1 quiet pose weights")
        return cls(weight.copy())

    def _actions(self, observations, active, anchor):
        observations, active = np.asarray(observations), np.asarray(active)
        result = np.zeros((*observations.shape[:-1], len(g1.DOF)), observations.dtype)
        with self.lock:
            rows = observations.reshape(-1, observations.shape[-1])
            actions = result.reshape(-1, result.shape[-1])
            anchors = np.asarray(anchor).reshape(-1, len(g1.DOF))
            for i in np.flatnonzero(active):
                pose = None
                if self.pose_weight is not None:
                    # Frozen at the instant of acquisition. A new capture chooses a new posture.
                    pose = np.tanh(pose_features(anchors[i]) @ self.pose_weight)
                actions[i] = self.dynamics(rows[i], pose)
        return result

    def __call__(self, observations, active, anchor=None):
        if anchor is None:
            if self.pose_weight is not None:
                raise ValueError("The learned quiet pose requires its captured 29-joint action")
            anchor = observations[..., 451:480]
        spec = jax.ShapeDtypeStruct((*observations.shape[:-1], len(g1.DOF)), observations.dtype)
        return jax.lax.cond(jnp.any(active),
                            lambda: jax.pure_callback(self._actions, spec, observations, active, anchor,
                                                      vmap_method="broadcast_all"),
                            lambda: jnp.zeros(spec.shape, spec.dtype))


@lru_cache(maxsize=1)
def sensor_stance():
    return SensorStance()
