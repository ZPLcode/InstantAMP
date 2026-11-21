"""Capture the G1 in a loaded crouch and regulate its momentum before finishing the rise."""

from threading import Lock

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from . import g1_getup as robot
from .g1_capture_eligibility import CaptureEligibility
from .g1_sensor_stand import _StandingDynamics
from .sim import build_scene


def features(obs):
    return jnp.concatenate((.28 * obs[..., 12:15], 1.7 * obs[..., 39:45], obs[..., 161:190],
                            .055 * obs[..., 306:335], .09 * obs[..., 451:480], 1.65 * obs[..., -18:]), axis=-1)


class MomentumDynamics(_StandingDynamics):
    def __init__(self, model):
        super().__init__(model)
        self.corners = np.array([[x, y, 0.] for x in (-.065, .10) for y in (-.027, .027)])
        self.cone = np.array([[.8, 0., 1.], [-.8, 0., 1.], [0., .8, 1.], [0., -.8, 1.]])
        self.force_prior = np.full(32, .01 * model.body_subtreemass[1] * 9.81 / 32)

    def command(self, obs, parameters, age, height, arms):
        self._estimate_state(obs)
        m, d = self.model, self.data
        cop_limit, _, initial_height_offset, gain, damping, posture_weight = parameters
        rotation = Rotation.from_matrix(d.xmat[self.trunk].reshape(3, 3))
        yaw = rotation.as_euler("ZYX")[0]
        heading = Rotation.from_euler("z", yaw)
        foot_rotations = [Rotation.from_matrix(d.xmat[f].reshape(3, 3)) for f in self.feet]
        feet = np.array([d.xpos[f] + rot.apply(self.foot_offset) for f, rot in zip(self.feet, foot_rotations)])
        soles = np.array([foot + rot.apply(self.corners) for foot, rot in zip(feet, foot_rotations)])
        bottom = soles[:, :, 2].min(axis=1)
        center = feet.mean(axis=0)
        center[:2] += heading.apply([.010, 0., 0.])[:2]
        com = d.subtree_com[1]
        com_jac = np.empty((3, m.nv))
        mj.mj_jacSubtreeCom(m, d, com_jac, 1)
        velocity = com_jac @ d.qvel
        pendulum_height = max(.25, com[2] - min(bottom))
        capture = com[:2] + velocity[:2] * np.sqrt(pendulum_height / 9.81)
        offset = heading.inv().apply(np.r_[capture - center[:2], 0.])
        if age == 0:
            height = np.clip(com[2] + initial_height_offset, .43, .745)
        if (abs(offset[0]) < .13 and abs(offset[1]) < .14 and np.linalg.norm(velocity[:2]) < .5) or age > 55:
            height = min(.748, height + .005)
        center[2] = height

        matrices, objectives, contacts = [], [], []

        def task(jacobian, acceleration, weight, bias=0.):
            matrices.append(weight * jacobian)
            objectives.append(weight * (np.clip(acceleration, -95., 95.) - bias))

        jac, angular, derivative, angular_derivative = (np.empty((3, m.nv)) for _ in range(4))
        for foot, rot, pos, corners in zip(self.feet, foot_rotations, feet, soles):
            mj.mj_jac(m, d, jac, angular, pos, foot)
            mj.mj_jacDot(m, d, derivative, angular_derivative, pos, foot)
            flat = Rotation.from_euler("z", rot.as_euler("ZYX")[0])
            task(jac, 81. * np.array([0., 0., .003 - pos[2]]) - 18. * (jac @ d.qvel), 5., derivative @ d.qvel)
            task(angular, 81. * (flat * rot.inv()).as_rotvec() - 18. * (angular @ d.qvel), 3.,
                 angular_derivative @ d.qvel)
            for corner in corners:
                mj.mj_jac(m, d, jac, None, corner, foot)
                contacts.append(self.cone @ jac)

        desired = gain**2 * (center - com) - damping * gain * velocity
        if cop_limit:
            # Clip the implied center of pressure to the yaw-aligned extent of the near-floor soles.
            corners = np.concatenate(soles[bottom < min(bottom) + .045])
            local = heading.inv().apply(corners - center)
            requested = com[:2] - pendulum_height / 9.81 * desired[:2]
            requested = heading.inv().apply(np.r_[requested - center[:2], 0.])[:2]
            pressure = np.clip(requested, local[:, :2].min(0) + .008, local[:, :2].max(0) - .008)
            pressure = center[:2] + heading.apply(np.r_[pressure, 0.])[:2]
            desired[:2] = 9.81 / pendulum_height * (com[:2] - pressure)
        task(com_jac, desired, 7.)
        mj.mj_jac(m, d, None, angular, d.xpos[self.trunk], self.trunk)
        mj.mj_jacDot(m, d, None, angular_derivative, d.xpos[self.trunk], self.trunk)
        trunk = Rotation.from_euler("ZYX", [yaw, .07, 0.])
        task(angular, 49. * (trunk * rotation.inv()).as_rotvec() - 14. * (angular @ d.qvel), 2.4,
             angular_derivative @ d.qvel)

        pose = self.posture.copy()
        knee = np.interp(height, [.48, .60, .75], [1.75, 1.2, .26])
        pose[[3, 9]], pose[[0, 6]], pose[[4, 10]] = knee, -.55 * knee, -.45 * knee
        pose[15:] = arms + min(age / 50., 1.) * (pose[15:] - arms)
        task(self.identity[6:], 16. * (pose - d.qpos[7:]) - 8. * d.qvel[6:], posture_weight)

        # Eliminate the acceleration from the constrained least-squares problem.
        # The nonnegative corner forces must supply the unactuated base wrench.
        matrix = np.vstack(matrices)
        normal = cho_factor(matrix.T @ matrix + .0006 * self.identity, check_finite=False)
        free = cho_solve(normal, matrix.T @ np.concatenate(objectives), check_finite=False)
        mass = np.empty((m.nv, m.nv))
        mj.mj_fullM(m, d, mass)
        coupling = cho_solve(normal, mass[:6].T, check_finite=False)
        chol = np.linalg.cholesky(mass[:6] @ coupling)
        bias = d.qfrc_bias - d.qfrc_passive
        required = mass[:6] @ free + bias[:6]
        contact = np.vstack(contacts).T
        wrench = solve_triangular(chol, contact[:6], lower=True, check_finite=False)
        rhs = solve_triangular(chol, required, lower=True, check_finite=False)
        load = nnls(np.vstack((wrench, .01 * np.eye(32))), np.r_[rhs, self.force_prior], maxiter=300)[0]
        acceleration = free + coupling @ cho_solve((chol, True), contact[:6] @ load - required, check_finite=False)
        torque = np.clip((mass @ acceleration + bias - contact @ load)[6:], -robot.EFFORT, robot.EFFORT)
        target = d.qpos[7:] + (torque + robot.KD * d.qvel[6:]) / robot.KP
        return (target - robot.DEFAULT_Q) / robot.ACTION_SCALE, height


@jax.tree_util.register_static
class MomentumCapture:
    # Table row plus one, elapsed control samples, requested COM height, initial arm pose and pre-capture action.
    memory_size = 3 + 14 + 29

    def __init__(self, keys, radii, parameters, archive):
        self.keys, self.radii = jnp.asarray(keys), jnp.asarray(radii)
        self.parameters = parameters
        self.solver = MomentumDynamics(build_scene(robot_name="g1_getup"))
        self.eligibility = None
        if "momentum_capture/general/settings" in archive:
            self.eligibility = CaptureEligibility(archive, self.solver.model)
            self.parameters = np.concatenate((self.parameters, self.eligibility.parameters))
        self.lock = Lock()

    @classmethod
    def read(cls, archive):
        prefix = "momentum_capture/"
        if prefix + "keys" not in archive:
            return None
        keys, radii, parameters = (np.asarray(archive[prefix + k], np.float32) for k in ("keys", "radii", "parameters"))
        if (keys.ndim != 2 or keys.shape[1] != 114 or len(keys) == 0 or radii.shape != (len(keys),)
                or parameters.shape != (len(keys), 6) or np.any(radii <= 0.) or np.any(parameters[:, 1] < 1.)
                or not all(np.isfinite(x).all() for x in (keys, radii, parameters))):
            raise ValueError("Invalid G1 momentum capture table")
        return cls(keys, radii, parameters, archive)

    def _step(self, obs, base, memory, input_features, quiet_fraction):
        obs, base = np.asarray(obs), np.asarray(base)
        result, state = base.copy(), np.asarray(memory).copy()
        with self.lock:
            for index in np.ndindex(obs.shape[:-1]):
                row = int(state[index][0]) - 1
                if row < 0:
                    if self.eligibility is None or quiet_fraction[index] >= .9:
                        continue
                    if state[index][1] > 0:
                        state[index][1] -= 1
                        continue
                    mode, cooldown = self.eligibility.choose(obs[index], input_features[index], self.solver)
                    if mode < 0:
                        state[index][1] = cooldown
                        continue
                    row = len(self.keys) + mode
                    state[index][:3] = row + 1, 0., 0.
                    state[index][3:17] = obs[index][176:190]
                    state[index][17:] = obs[index][451:480]
                age = int(state[index][1])
                params = self.parameters[row]
                target, height = self.solver.command(obs[index], params, age, state[index][2], state[index][3:17])
                share = min(1., (age + 1) / params[1])
                anchor = state[index][17:]
                result[index] = anchor + share * (target - anchor)
                state[index][1:3] = age + 1, height
        return result, state

    def step(self, obs, base, memory, quiet_fraction):
        input_features = features(obs)
        distance = jnp.linalg.norm(input_features[..., None, :] - self.keys, axis=-1)
        index = jnp.argmin(distance, axis=-1)
        acquire = (memory[..., 0] == 0.) & (jnp.min(distance, axis=-1) < self.radii[index]) & (quiet_fraction < .9)
        initial = jnp.concatenate(((index + 1)[..., None], jnp.zeros_like(memory[..., 1:3]),
                                   obs[..., 176:190], obs[..., 451:480]), axis=-1)
        state = jnp.where(acquire[..., None], initial, memory)
        spec = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (base, state))
        active = jnp.any(state[..., 0] > 0.)
        if self.eligibility is not None:
            active |= jnp.any((quiet_fraction < .9) & (obs[..., 44] > .55))
        return jax.lax.cond(active,
                            lambda: jax.pure_callback(self._step, spec, obs, base, state, input_features,
                                                      quiet_fraction, vmap_method="broadcast_all"),
                            lambda: (base, state))
