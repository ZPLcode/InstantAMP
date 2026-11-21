"""Track physically executed G1 recoveries from encoder/IMU history and capture their support footprint."""

from threading import Lock

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import g1_getup as robot
from .g1_sensor_stand import _StandingDynamics
from .native_getup import advance
from .sim import build_scene


def residual_features(obs):
    return jnp.concatenate(
        (
            0.12 * obs[..., 12:15],
            1.4 * obs[..., 39:45],
            obs[..., 161:190],
            0.05 * obs[..., 306:335],
            0.08 * obs[..., 451:480],
        ),
        axis=-1,
    )


def _row(obs):
    return np.r_[obs[12:15], obs[39:45], obs[161:190], obs[306:335], obs[451:480]]


_SCALE = np.r_[
    np.full(3, 0.065),
    np.full(6, 1.5),
    np.ones(29),
    np.full(29, 0.043),
    np.full(29, 0.050),
]


def _orientation(six):
    x, z = six[:3].copy(), six[3:].copy()
    x /= max(np.linalg.norm(x), 1e-6)
    z -= x * (x @ z)
    z /= max(np.linalg.norm(z), 1e-6)
    return np.column_stack((x, np.cross(z, x), z))


class _AnchoredStance(_StandingDynamics):
    def action(self, feet_xy, pelvis_xy, gain, pelvis_strength, desired_height=0.748, support_shift=None):
        m, d = self.model, self.data
        orientation = Rotation.from_matrix(d.xmat[self.trunk].reshape(3, 3))
        yaw = orientation.as_euler("ZYX")[0]
        target_trunk = Rotation.from_euler("ZYX", [yaw, 0.10, 0.0])
        contact, task_a, task_b = [], [], []

        def task(jac, error, rate, weight, bias=0.0):
            task_a.append(weight * jac)
            task_b.append(
                weight
                * (
                    np.clip(rate**2 * error - 2 * rate * jac @ d.qvel, -90.0, 90.0)
                    - bias
                )
            )

        j, jr, jd, jrd = [np.empty((3, m.nv)) for _ in range(4)]
        for i, foot in enumerate(self.feet):
            rot = Rotation.from_matrix(d.xmat[foot].reshape(3, 3))
            sole = d.xpos[foot] + rot.apply(self.foot_offset)
            flat = Rotation.from_euler("z", rot.as_euler("ZYX")[0])
            mj.mj_jac(m, d, j, jr, sole, foot)
            mj.mj_jacDot(m, d, jd, jrd, sole, foot)
            error = np.r_[np.clip(feet_xy[i] - sole[:2], -0.10, 0.10), 0.004 - sole[2]]
            task(j, error, 8.0 * gain, 6.0, jd @ d.qvel)
            task(jr, (flat * rot.inv()).as_rotvec(), 8.0 * gain, 3.6, jrd @ d.qvel)
            for corner in sole + rot.apply(self.corners):
                mj.mj_jac(m, d, j, None, corner, foot)
                contact.append(self.cone @ j)
        center = np.r_[feet_xy.mean(0), desired_height] + Rotation.from_euler(
            "z", yaw
        ).apply([0.018, 0, 0])
        if support_shift is not None:
            center[:2] -= support_shift
            pelvis_xy = pelvis_xy - support_shift
        mj.mj_jacSubtreeCom(m, d, j, 1)
        task(j, center - d.subtree_com[1], 4.0 * gain, 5.0)
        mj.mj_jac(m, d, None, jr, d.xpos[self.trunk], self.trunk)
        mj.mj_jacDot(m, d, None, jrd, d.xpos[self.trunk], self.trunk)
        task(
            jr,
            (target_trunk * orientation.inv()).as_rotvec(),
            7.0 * gain,
            2.0,
            jrd @ d.qvel,
        )
        if pelvis_strength:
            mj.mj_jac(m, d, j, None, d.qpos[:3], 1)
            task(j[:2], pelvis_xy - d.qpos[:2], 7.0 * gain, pelvis_strength)
        task(self.identity[6:], self.posture - d.qpos[7:], 4.0, 0.22)
        a = np.vstack(task_a)
        acceleration = solve(
            a.T @ a + 0.0005 * self.identity,
            a.T @ np.concatenate(task_b),
            assume_a="pos",
        )
        mass = np.empty((m.nv, m.nv))
        mj.mj_fullM(m, d, mass)
        effort = mass @ acceleration + d.qfrc_bias - d.qfrc_passive
        contact = np.vstack(contact).T
        wrench, required = contact[:6].copy(), effort[:6].copy()
        wrench[3:] *= 6.0
        required[3:] *= 6.0
        load = nnls(
            np.vstack((wrench, self.contact_regularizer)),
            np.r_[required, self.neutral_force],
            maxiter=300,
        )[0]
        torque = np.clip((effort - contact @ load)[6:], -robot.EFFORT, robot.EFFORT)
        target = d.qpos[7:] + (torque + robot.KD * d.qvel[6:]) / robot.KP
        return (target - robot.DEFAULT_Q) / robot.ACTION_SCALE


@jax.tree_util.register_static
class TrackingCapture:
    """A bounded local observation search and finite-horizon feedback with independent memory per robot.

    Memory carries acquisition mode, the last matched example, integrated yaw error, relative sole
    anchors and the captured pelvis position. No simulator state or reference index is accepted from callers.
    """

    def __init__(self, arrays):
        self.row = arrays["observations"]
        self.keys = self.row * _SCALE
        self.q = arrays["qpos"]
        self.v = arrays["qvel"]
        self.u = arrays["action"]
        self.gain = arrays["gain"]
        self.support = arrays["support_jacobian"]
        self.params = arrays["stance_parameters"]
        self.end = arrays["end"].astype(int)
        self.start = arrays["start"].astype(int)
        self.starts = np.flatnonzero(self.start == np.arange(len(self.u)))
        self.tree = cKDTree(self.keys)
        self.initial = cKDTree(self.keys[self.starts])
        self.settings = arrays["settings"]
        self.residual_w = jnp.asarray(arrays["residual_w"])
        self.residual_b = jnp.asarray(arrays["residual_b"])
        self.residual_scale = jnp.asarray(arrays["residual_scale"])
        self.dynamics = _AnchoredStance(build_scene(robot_name="g1_getup"))
        self.lock = Lock()
        self.observer = arrays.get("observer")
        self.capture = arrays.get("capture")
        # Correct the simulated state from real body/encoder innovations through the landing impact.
        # A foot can be near the floor and still be moving quickly during the transition.
        self.contact_observer = arrays.get("contact_observer")
        self.model = self.dynamics.model
        self.predicted = mj.MjData(self.model)
        self.probe = mj.MjData(self.model)
        self.search = arrays.get("capture_search")
        self.branch_probe = arrays.get("branch_probe")
        self.branch_priority = arrays.get("branch_priority")
        self.ramps = arrays.get("capture_ramps")
        if self.search is not None:
            self.search_data = mj.MjData(self.model)
            self.search_stance = _AnchoredStance(self.model)
            self.search_stance.data = self.search_data
            self.cpu_residual = (
                np.asarray(self.residual_w),
                np.asarray(self.residual_b),
                float(self.residual_scale),
            )
        self.memory_size = 119 if self.observer is not None else 12

    @classmethod
    def from_archive(cls, archive):
        prefix = "tracking_capture/"
        if not any(k.startswith(prefix) for k in archive):
            return None
        fields = (
            "observations",
            "qpos",
            "qvel",
            "action",
            "gain",
            "support_jacobian",
            "stance_parameters",
            "start",
            "end",
            "settings",
            "residual_w",
            "residual_b",
            "residual_scale",
        )
        try:
            a = {k: np.asarray(archive[prefix + k]) for k in fields}
        except KeyError as exc:
            raise ValueError("Incomplete G1 tracking capture") from exc
        n = len(a["action"])
        shapes = (
            (n, 96),
            (n, 36),
            (n, 35),
            (n, 29),
            (n, 29, 70),
            (n, 3, 32),
            (n, 2),
            (n,),
            (n,),
            (5,),
            (96, 29),
            (29,),
            (),
        )
        if n < 2 or any(
            a[k].shape != shape or not np.isfinite(a[k]).all()
            for k, shape in zip(fields, shapes)
        ):
            raise ValueError("Invalid G1 tracking capture arrays")
        index = np.arange(n)
        if (
            not np.issubdtype(a["start"].dtype, np.integer)
            or not np.issubdtype(a["end"].dtype, np.integer)
            or np.any(a["start"] < 0)
            or np.any(a["start"] > index)
            or np.any(a["end"] < index)
            or np.any(a["end"] >= n)
            or not np.array_equal(a["start"][a["end"]], a["start"])
            or not np.array_equal(a["end"][a["start"]], a["end"])
            or np.any(a["settings"] < 0)
        ):
            raise ValueError("Invalid G1 tracking capture segment layout")
        if prefix + "observer" in archive:
            a["observer"] = np.asarray(archive[prefix + "observer"], float)
            a["capture"] = np.asarray(archive.get(prefix + "capture", ()), int)
            if (
                a["observer"].shape != (4,)
                or not np.isfinite(a["observer"]).all()
                or a["capture"].shape != (n,)
                or np.any(a["capture"] < a["start"])
                or np.any(a["capture"] > a["end"])
            ):
                raise ValueError("Invalid G1 floating-base observer")
        if prefix + "contact_observer" in archive:
            a["contact_observer"] = np.asarray(
                archive[prefix + "contact_observer"], float
            )
            if (
                a.get("observer") is None
                or a["contact_observer"].shape != (3,)
                or not np.isfinite(a["contact_observer"]).all()
                or np.any(a["contact_observer"] < 0)
                or np.any(a["contact_observer"] > 1)
            ):
                raise ValueError("Invalid foot-contact observer settings")
        if prefix + "capture_search" in archive:
            a["capture_search"] = np.asarray(archive[prefix + "capture_search"], int)
            if (
                a.get("observer") is None
                or a["capture_search"].shape != (3,)
                or np.any(a["capture_search"] <= 0)
                or np.max(a["capture_search"]) > 180
            ):
                raise ValueError("Invalid G1 sensor-based support search")
        if prefix + "capture_ramps" in archive:
            a["capture_ramps"] = np.asarray(archive[prefix + "capture_ramps"], int)
            if (
                a.get("capture_search") is None
                or a["capture_ramps"].ndim != 1
                or not 1 <= len(a["capture_ramps"]) <= 4
                or np.any(a["capture_ramps"] < 0)
                or np.max(a["capture_ramps"]) > 80
            ):
                raise ValueError("Invalid G1 capture extension times")
        if prefix + "branch_probe" in archive:
            a["branch_probe"] = np.asarray(archive[prefix + "branch_probe"], int)
            a["branch_priority"] = np.asarray(
                archive.get(prefix + "branch_priority", ()), float
            )
            if (
                a.get("capture_search") is None
                or a["branch_probe"].shape != (2,)
                or min(a["branch_probe"]) <= 0
                or a["branch_priority"].shape != (n,)
                or not np.isfinite(a["branch_priority"]).all()
            ):
                raise ValueError("Invalid onboard trajectory selection parameters")
        return cls(a)

    def memory(self, obs):
        return jnp.zeros((*obs.shape[:-1], self.memory_size), dtype=obs.dtype)

    def _quiet(self, obs, mem, k, first):
        controller = self.dynamics
        controller._estimate_state(obs)
        d = controller.data
        if d.qpos[2] < 0.46 or obs[44] < 0.65 or np.linalg.norm(obs[12:15]) > 5.0:
            mem[:] = 0.0
            return None
        soles = np.asarray(
            [
                (d.xpos[f] + d.xmat[f].reshape(3, 3) @ controller.foot_offset)[:2]
                for f in controller.feet
            ]
        )
        if first:
            mem[3:7] = soles.ravel()
            mem[7:9] = d.qpos[:2]
        # Recover translation in the stance-foot coordinate frame; heading can drift under a stationary contact.
        anchors = mem[3:7].reshape(2, 2)
        d.qpos[:2] += np.mean(anchors - soles, axis=0)
        mj.mj_forward(controller.model, d)
        mem[0] = 2.0
        return controller.action(anchors, mem[7:9], *self.params[k])

    @staticmethod
    def _measurement(q, v):
        rotation = Rotation.from_quat(q[3:7], scalar_first=True).as_matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        local = Rotation.from_euler("z", -yaw).as_matrix() @ rotation
        return np.r_[0.14 * v[3:6], local[:, 0], local[:, 2], q[7:], 0.038 * v[6:]]

    def _advance_estimate(self, d, previous, applied, delta=None):
        m = self.model
        mj.mj_resetData(m, d)
        d.qpos[:] = previous[:36]
        d.qvel[:] = previous[36:71]
        if delta is not None:
            d.qvel[:3] += delta[:3]
            d.qpos[2] += delta[3]
        mj.mj_forward(m, d)
        d.qacc_warmstart[:] = previous[71:]
        advance(m, d, applied)
        return self._measurement(d.qpos, d.qvel)

    def _observe(self, r, mem, k, distance):
        d = self.predicted
        q, v = self.q[k], self.v[k]
        orientation = _orientation(r[3:9])
        if mem[9] == 0 or distance < 0.00012:
            mj.mj_resetData(self.model, d)
            d.qpos[:] = q
            d.qvel[:] = v
            mj.mj_forward(self.model, d)
        else:
            previous = mem[13:119].astype(float)
            # Use the actual previous float32 command, including learned residuals and any blending.
            applied = r[67:].astype(np.float32)
            predicted = self._advance_estimate(d, previous, applied)
            actual = np.r_[0.14 * r[:3], r[3:9], r[9:38], 0.038 * r[38:67]]
            error = actual - predicted
            if (
                self.observer[0]
                and error @ error > 1e-7
                and (mem[0] != 2 or self.contact_observer is not None)
            ):
                # Ground contact makes floating velocity observable through the encoder/IMU response to a known command.
                # Difference the physical prediction around its current velocity and height, and accept a correction
                # only if it explains the measured response better after a second forward simulation.
                steps = np.array([0.045, 0.045, 0.035, 0.0015])
                columns = []
                for j in range(4):
                    delta = np.zeros(4)
                    delta[j] = steps[j]
                    response = self._advance_estimate(
                        self.probe, previous, applied, delta
                    )
                    columns.append(response - predicted)
                J = np.column_stack(columns)
                coeff = np.linalg.solve(
                    J.T @ J + max(0.00002, self.observer[1]) * np.eye(4), J.T @ error
                )
                delta = steps * np.clip(coeff, -2.0, 2.0)
                candidate = self._advance_estimate(self.probe, previous, applied, delta)
                if np.sum((actual - candidate) ** 2) < 0.995 * (error @ error):
                    mj.mj_copyData(d, self.model, self.probe)
        rotation = Rotation.from_quat(d.qpos[3:7], scalar_first=True).as_matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        d.qpos[3:7] = (
            Rotation.from_euler("z", yaw) * Rotation.from_matrix(orientation)
        ).as_quat(scalar_first=True)
        d.qpos[7:] = r[9:38]
        d.qvel[3:6] = r[:3]
        d.qvel[6:] = r[38:67]
        angle = (
            Rotation.from_quat(q[3:7], scalar_first=True).inv()
            * Rotation.from_quat(d.qpos[3:7], scalar_first=True)
        ).as_rotvec()
        geom = -self.support[k] @ np.r_[angle, d.qpos[7:] - q[7:]]
        if mem[0] != 2:
            # Until support is fixed, measured shape relative to the expected contact provides a weak height anchor.
            d.qpos[2] += self.observer[2] * np.clip(
                q[2] + geom[2] - d.qpos[2], -0.040, 0.040
            )
        mj.mj_forward(self.model, d)
        return d

    def _remember(self, d, mem):
        mem[9] = max(mem[9], 1.0)
        mem[13:119] = np.r_[d.qpos, d.qvel, d.qacc_warmstart]

    @staticmethod
    def _ramp_height(initial, age, duration):
        if duration <= 0:
            return 0.748
        w = np.clip(age / duration, 0.0, 1.0)
        return initial + (0.748 - initial) * (w * w * (3.0 - 2.0 * w))

    def _forecast_capture(self, estimate, k, acquire, previous, multiple, ramp=0):
        """Predict the footprint from the onboard state estimate under a proposed support time."""
        model, d, controller = self.model, self.search_data, self.search_stance
        mj.mj_copyData(d, model, estimate)
        motor = previous.copy()
        w, b, scale = self.cpu_residual
        center = feet = None
        max_root = max_feet = 0.0
        penalties = 0.0
        upright_frames = longest = 0
        fell = False
        root_at_rise = feet_at_rise = None
        last = int(self.end[k])
        for t in range(acquire - k + int(self.search[2])):
            row = min(last, k + t)
            rotation = Rotation.from_quat(d.qpos[3:7], scalar_first=True).as_matrix()
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            local = Rotation.from_euler("z", -yaw).as_matrix() @ rotation
            observable = np.r_[
                0.12 * d.qvel[3:6],
                1.4 * local[:, 0],
                1.4 * local[:, 2],
                d.qpos[7:],
                0.05 * d.qvel[6:],
                0.08 * motor,
            ].astype("f")
            if row < acquire:
                dx = np.empty(self.model.nv)
                mj.mj_differentiatePos(model, dx, 1.0, self.q[row], d.qpos)
                u = self.u[row] + np.clip(
                    multiple
                    * self.settings[3]
                    * (self.gain[row] @ np.r_[dx, d.qvel - self.v[row]])
                    / 0.25,
                    -5.0,
                    5.0,
                )
            else:
                if feet is None:
                    feet = np.array(
                        [
                            (
                                d.xpos[f]
                                + d.xmat[f].reshape(3, 3) @ controller.foot_offset
                            )[:2]
                            for f in controller.feet
                        ]
                    )
                    initial_height = np.clip(d.subtree_com[1, 2], 0.5, 0.748)
                if d.qpos[2] < 0.32 or d.xmat[1, 8] < 0.40:
                    return (
                        100.0
                        + max_root
                        + 2.0 * (1.0 - t / (acquire - k + self.search[2]))
                    )
                mj.mj_forward(model, d)
                if center is None and d.qpos[2] > 0.635 and d.xmat[1, 8] > 0.84:
                    center = d.qpos[:2].copy()
                u = controller.action(
                    feet,
                    d.qpos[:2] if center is None else center,
                    float(self.params[row, 0]),
                    0.0 if center is None else float(self.params[row, 1]),
                    self._ramp_height(initial_height, k + t - acquire, ramp),
                )
            motor = np.asarray(u, np.float32) + scale * np.tanh(observable @ w + b)
            advance(model, d, motor)
            upr = 0.65 < d.qpos[2] < 0.85 and d.xmat[1, 8] > 0.866
            if upr and upright_frames == 0 and root_at_rise is None:
                root_at_rise = d.qpos[:2].copy()
                feet_at_rise = d.xpos[controller.feet, :2].copy()
            upright_frames = int(upr) * (upright_frames + 1)
            longest = max(longest, upright_frames)
            fell |= longest >= 10 and (d.qpos[2] < 0.48 or d.xmat[1, 8] < 0.50)
            if root_at_rise is not None:
                max_root = max(max_root, np.linalg.norm(d.qpos[:2] - root_at_rise))
                max_feet = max(
                    max_feet,
                    np.linalg.norm(
                        d.xpos[controller.feet, :2] - feet_at_rise, axis=-1
                    ).max(),
                )
            if row >= acquire:
                penalties += (
                    max(0.0, np.linalg.norm(d.qvel[:2]) - 0.12) / self.search[2]
                )
        bad_stand = max(0.0, 0.67 - d.qpos[2]) + max(0.0, 0.90 - d.xmat[1, 8])
        if self.branch_probe is not None and (
            longest < 10 or upright_frames < 10 or fell
        ):
            return 14.0 + 10.0 * int(fell) + 8.0 * bad_stand
        return (
            12.0 * bad_stand
            + 3.0 * np.linalg.norm(d.qvel[:2])
            + max_root
            + 0.55 * max_feet
            + 0.45 * penalties
            + 0.30 * (upright_frames < 15)
        )

    def _support_time(self, d, mem, k, previous):
        nominal = int(self.capture[k])
        if self.search is None:
            return nominal, 1.0
        if mem[11] <= 0 and k >= nominal - self.search[0]:
            step = int(self.search[1])
            best = (np.inf, nominal, 1.0, 0)
            options = (
                (1.0, 0),
                (1.0, -step),
                (1.0, step),
                (1.0, -2 * step),
                (1.0, 2 * step),
                (0.72, 0),
                (1.28, 0),
                (0.72, step),
                (1.28, -step),
            )
            for ramp in [0] if self.ramps is None else self.ramps:
                for factor, offset in options:
                    target = max(k, min(nominal + offset, int(self.end[k])))
                    cost = self._forecast_capture(d, k, target, previous, factor, ramp)
                    cost += 0.0003 * abs(target - nominal) + 0.001 * abs(factor - 1.0)
                    if cost < best[0]:
                        best = (cost, target, factor, ramp)
                    if cost < 0.13:
                        break
                if best[0] < 0.13:
                    break
            mem[2] = best[3]
            mem[11] = best[1] + 1
            mem[12] = best[2]
        return (int(mem[11]) - 1, float(mem[12])) if mem[11] > 0 else (nominal, 1.0)

    def _choose_branch(self, d, r, mem, k):
        age = k - self.start[k]
        if mem[11] != 0 or age < min(
            self.branch_probe[0], max(1, (self.capture[k] - self.start[k]) // 2)
        ):
            return k
        root = self.start[k]
        near = self.starts[
            np.max(abs(self.q[self.starts, 7:] - self.q[root, 7:]), axis=1) < 0.02
        ]
        options = []
        for start in near:
            target = min(start + age, self.capture[start] - 1)
            delta = np.linalg.norm(self.keys[target] - r * _SCALE)
            if delta < max(2.1, self.settings[1]):
                options.append((self.branch_priority[start], delta, int(target)))
        best = (np.inf, k)
        for prior, dist, row in sorted(options)[: int(self.branch_probe[1])]:
            loss = self._forecast_capture(d, row, int(self.capture[row]), r[67:], 1.0)
            value = loss + 0.02 * dist + prior
            if value < best[0]:
                best = value, row
            if value < 0.13:
                break
        mem[11] = -1.0
        return best[1]

    def _observed_action(self, r, mem, k, distance, fallback):
        d = self._observe(r, mem, k, distance)
        if self.branch_probe is not None and mem[0] != 2:
            k = self._choose_branch(d, r, mem, k)
            mem[1] = k
            distance = np.linalg.norm(self.keys[k] - r * _SCALE)
        acquire, factor = self._support_time(d, mem, k, r[67:])
        if mem[0] != 2 and k < acquire:
            dx = np.empty(35)
            mj.mj_differentiatePos(self.model, dx, 1.0, self.q[k], d.qpos)
            tangent = np.r_[dx, d.qvel - self.v[k]]
            self._remember(d, mem)
            return (
                self.u[k]
                if distance < 0.00012
                else self.u[k]
                + np.clip(
                    factor * self.settings[3] * (self.gain[k] @ tangent) / 0.25,
                    -5.0,
                    5.0,
                )
            )
        if d.qpos[2] < 0.34 or r[8] < 0.49 or np.linalg.norm(r[:3]) > 7.0:
            mem[:] = 0.0
            return fallback
        controller = self.dynamics
        mj.mj_copyData(controller.data, self.model, d)
        cd = controller.data
        soles = np.asarray(
            [
                cd.xpos[f] + cd.xmat[f].reshape(3, 3) @ controller.foot_offset
                for f in controller.feet
            ]
        )
        if mem[0] != 2:
            mem[3:7] = soles[:, :2].ravel()
            mem[10] = 0.0
            if self.ramps is not None:
                mem[12] = np.clip(cd.subtree_com[1, 2], 0.5, 0.748)
                mem[9] = 1.0
        mem[0] = 2.0
        anchors = mem[3:7].reshape(2, 2)
        # A fixed footprint determines local translation once the feet are loaded.
        high = soles[:, 2].min()
        velocity_gain, height_gain, speed_limit = (
            (0.45, 0.24, np.inf)
            if self.contact_observer is None
            else self.contact_observer
        )
        if (
            np.ptp(soles[:, 2]) < 0.065
            and high < 0.065
            and velocity_gain + height_gain + self.observer[3] > 0
        ):
            weights = np.exp(-np.minimum(30.0, (soles[:, 2] - high) / 0.012))
            velocities = []
            jac = np.empty((3, self.model.nv))
            for foot, sole in zip(controller.feet, soles):
                mj.mj_jac(self.model, cd, jac, None, sole, foot)
                velocities.append(jac @ cd.qvel)
            if np.max(np.linalg.norm(velocities, axis=-1)) < speed_limit:
                d.qvel[:3] -= velocity_gain * np.clip(
                    np.average(velocities, weights=weights, axis=0), -0.55, 0.55
                )
                d.qpos[:2] += self.observer[3] * np.mean(anchors - soles[:, :2], axis=0)
                d.qpos[2] += height_gain * np.clip(0.002 - high, -0.025, 0.025)
            mj.mj_forward(self.model, d)
            mj.mj_copyData(cd, self.model, d)
        if mem[10] == 0 and d.qpos[2] > 0.635 and r[8] > 0.84:
            mem[7:9] = d.qpos[:2]
            mem[10] = 1.0
        self._remember(d, mem)
        height = 0.748
        if self.ramps is not None:
            height = self._ramp_height(mem[12], mem[9] - 1.0, mem[2])
            mem[9] += 1.0
        strength = self.params[k, 1] if mem[10] else 0.0
        # Near a sensor-identical physical example, retain its precisely realized float32 motor command.
        if distance < 0.00012 and k < self.end[k]:
            return self.u[k]
        return controller.action(anchors, mem[7:9], self.params[k, 0], strength, height)

    def _controller(self, obs, fallback, mem):
        r = _row(obs).astype(np.float64)
        key = r * _SCALE
        if mem[0] == 2 and self.observer is None:
            action = self._quiet(obs, mem, int(mem[1]), False)
            if action is not None:
                return action
        acquire, release, phase_bias, feedback, quiet_margin = self.settings
        if mem[0] == 0:
            distance, index = self.initial.query(key)
            k = self.starts[index]
            if distance > acquire:
                distance, k = self.tree.query(key)
            if distance > acquire:
                return fallback
            mem[:] = 0.0
            mem[0] = 1.0
        else:
            old = int(mem[1])
            options = np.arange(
                max(self.start[old], old - 1), min(self.end[old], old + 4) + 1
            )
            scores = np.square(self.keys[options] - key).sum(
                -1
            ) + phase_bias * np.square(options - np.minimum(old + 1, self.end[old]))
            k = options[np.argmin(scores)]
            distance = np.linalg.norm(self.keys[k] - key)
            if distance > release and mem[0] != 2:
                # Release rather than execute a distant example after a push or a new fall.
                mem[:] = 0.0
                return fallback
        mem[1] = k
        if self.observer is not None:
            return self._observed_action(r, mem, k, distance, fallback)
        expected = self.row[k].astype(float)
        if k >= self.end[k] - quiet_margin and obs[44] > 0.85:
            action = self._quiet(obs, mem, k, True)
            if action is not None:
                return action
            mem[0] = 1.0
        nominal = self.q[k]
        commanded = self.u[k]
        if distance < 0.00012:
            return commanded
        refR = Rotation.from_quat(nominal[3:7], scalar_first=True)
        orient = _orientation(r[3:9])
        ref_matrix = refR.as_matrix()
        yaw = np.arctan2(ref_matrix[1, 0], ref_matrix[0, 0])
        # The IMU supplies roll, pitch and body angular velocity. Integrate only the yaw error relative to the local path.
        omega_delta = r[:3] - expected[:3]
        mem[2] = np.clip(mem[2] + 0.02 * (orient[2] @ omega_delta), -0.36, 0.36)
        actual = Rotation.from_euler("z", yaw + mem[2]) * Rotation.from_matrix(orient)
        angle = (refR.inv() * actual).as_rotvec()
        body = np.r_[angle, r[9:38] - expected[9:38]]
        rates = np.r_[omega_delta, r[38:67] - expected[38:67]]
        jac = self.support[k]
        translation = np.clip(-jac @ body, -0.18, 0.18)
        velocity = np.clip(-jac @ rates, -1.5, 1.5)
        dx = np.r_[translation, body, velocity, rates]
        correction = np.asarray(self.gain[k], float) @ dx
        return commanded + np.clip(feedback * correction / 0.25, -5.0, 5.0)

    def _host(self, obs, fallback, memory):
        observations = np.asarray(obs).reshape(-1, 570)
        output = np.asarray(fallback).copy().reshape(-1, 29)
        states = np.asarray(memory).copy().reshape(-1, self.memory_size)
        with self.lock:
            for x, u, state in zip(observations, output, states):
                u[:] = self._controller(x, u, state)
        return output.reshape(fallback.shape), states.reshape(memory.shape)

    def base_step(self, obs, fallback, memory):
        specs = tuple(
            jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (fallback, memory)
        )
        return jax.pure_callback(
            self._host, specs, obs, fallback, memory, vmap_method="broadcast_all"
        )

    def step(self, obs, fallback, memory):
        action, state = self.base_step(obs, fallback, memory)
        residual = self.residual_scale * jnp.tanh(
            residual_features(obs) @ self.residual_w + self.residual_b
        )
        return action + jnp.where(state[..., :1] != 0, residual, 0.0), state
