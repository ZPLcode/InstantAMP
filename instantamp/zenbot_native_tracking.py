"""Delayed motor tracking and contact-aware stance for Zenbot's measured native MuJoCo state."""

import ctypes as ct
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from . import zenbot as robot
from .zenbot_stance import _ZenbotDynamics


@lru_cache(1)
def backend():
    source = Path(__file__).with_name("zenbot_rollout.cc")
    folder = Path(mj.__file__).resolve().parent
    runtime = next(folder.glob("libmujoco.so.*"))
    stance_source = source.with_name("zenbot_stand.cc")
    digest = hashlib.sha256(
        source.read_bytes() + stance_source.read_bytes() + str(runtime).encode()
    ).hexdigest()[:18]
    target = Path.home() / ".cache/instantamp" / f"zenbot_rollout_{digest}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=target.parent) as path:
            obj = Path(path) / "rollout.so"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    "-fopenmp",
                    "-I" + str(folder / "include"),
                    str(source),
                    str(stance_source),
                    str(runtime),
                    "-Wl,-rpath," + str(folder),
                    "-o",
                    str(obj),
                ],
                check=True,
            )
            obj.replace(target)
    dll = ct.CDLL(str(target))
    for name in ("plan", "motor_step", "stance_control"):
        getattr(dll, name).restype = None
    return dll


def ptr(a):
    return ct.c_void_p(a.ctypes.data)


def features(q, v, previous):
    a, b, c, d = q[3:7]
    return np.r_[
        0.12 * v[3:6],
        2 * (b * d - a * c),
        2 * (c * d + a * b),
        1 - 2 * (b * b + c * c),
        q[7:],
        0.04 * v[6:],
        0.08 * previous,
        1.0,
    ]


class NativeTracking:
    def __init__(self, model, fields):
        self.model = model
        self.db = {
            k.split("/", 1)[1]: np.asarray(value)
            for k, value in fields.items()
            if k.startswith("zenbot_tracking/")
        }
        if (model.nq, model.nv, model.nu) != (30, 29, 23):
            raise ValueError(
                "The delayed tracking controller requires the 23-DoF Zenbot model"
            )
        offsets = self.db["offset"]
        if (
            offsets.ndim != 1
            or len(offsets) < 2
            or offsets[0] != 0
            or not np.issubdtype(offsets.dtype, np.integer)
            or (np.diff(offsets) < 3).any()
        ):
            raise ValueError("Invalid Zenbot trajectory offsets")
        n = int(offsets[-1])
        count = len(offsets) - 1
        shapes = {
            "q": (n, 30),
            "v": (n, 29),
            "u": (n, 23),
            "K": (n, 23, 104),
            "delay": (count, 5),
            "look_gain": (count, 2),
            "actor": (76, 23),
            "scale": (),
        }
        if any(
            k not in self.db
            or self.db[k].shape != shape
            or not np.isfinite(self.db[k]).all()
            for k, shape in shapes.items()
        ):
            raise ValueError(
                "Invalid Zenbot native tracking state, motor or feedback table"
            )
        if (
            np.any((self.db["delay"] < 3) | (self.db["delay"] > 7))
            or float(self.db["scale"]) < 0.0
        ):
            raise ValueError("Invalid Zenbot motor delay or residual scale")
        # The C rollout kernel accepts double-precision nominal generalized positions and velocities.
        for k in ("q", "v", "u", "look_gain"):
            self.db[k] = np.ascontiguousarray(self.db[k], dtype=np.float64)
        for k, shape in (
            ("desired_palm_load", (n, 2)),
            ("reference_style", (count, 3)),
            ("quality", (count,)),
            ("support_style", (count,)),
        ):
            if k in self.db and (
                self.db[k].shape != shape or not np.isfinite(self.db[k]).all()
            ):
                raise ValueError(f"Invalid Zenbot reference annotation {k}")
        if "cache_state" in self.db:
            cache = self.db["cache_offsets"]
            r = len(self.db["cache_state"])
            if cache.shape != (r + 1,) or cache[0] != 0 or np.any(np.diff(cache) <= 0):
                raise ValueError("Invalid Zenbot cached plan offsets")
            frames = int(cache[-1])
            cached = {
                "state": (r, 59),
                "lags": (r, 23),
                "source": (r,),
                "score": (r,),
                "parameters": (r, 2),
                "action": (frames, 23),
                "q": (frames, 30),
                "v": (frames, 29),
            }
            if any(
                self.db["cache_" + k].shape != sh
                or not np.isfinite(self.db["cache_" + k]).all()
                for k, sh in cached.items()
            ):
                raise ValueError("Invalid Zenbot cached motor plan")
            if np.any(
                (self.db["cache_source"] < 0) | (self.db["cache_source"] >= count)
            ):
                raise ValueError("Invalid Zenbot cached trajectory source")
        self.motor = np.ascontiguousarray(
            [robot.KP, robot.KD, robot.EFFORT, robot.SPEED], np.float64
        )
        self.neutral = np.array(robot.DEFAULT_Q, dtype="d")
        self.feet = np.array(
            [model.body("robot_" + n).id for n in robot.FEET], dtype="i"
        )
        self.hands = np.array(
            [model.body("robot_" + n).id for n in robot.HANDS], dtype="i"
        )
        self.weight = model.body_mass.sum() * 9.81
        self.stance = _ZenbotDynamics(model)
        self.bodies = np.ascontiguousarray([*self.feet, self.stance.torso], dtype="i")
        self.actor = np.ascontiguousarray(self.db["actor"], dtype="f")
        self.scale = float(self.db["scale"])
        self.backend = backend()
        self.debug = {}

    def reset(self, data, delays):
        self.lags = np.ascontiguousarray(np.asarray(delays)[robot.GROUP], dtype="i")
        self.tick = 0
        self.mode = 0
        self.plan = None
        self.anchors = None
        self.quiet_steps = 0
        self.last_xy_velocity = None
        self.pulse_end = 0
        self.pulse = None
        self.position_limit = float(self.db.get("stand_radius", np.inf))
        self.previous = self.neutral.copy()
        self.older = self.neutral.copy()
        self.debug = {}

    def integrate(self, data, action, previous, older):
        goal = (
            robot.DEFAULT_Q + robot.ACTION_SCALE * np.array(action, dtype="f")
        ).astype("d")
        self.backend.motor_step(
            ct.c_void_p(self.model._address),
            ct.c_void_p(data._address),
            ptr(self.lags),
            ptr(self.motor),
            ptr(goal),
            ptr(previous),
            ptr(older),
        )
        return goal

    def soles(self, d):
        return np.array(
            [
                (d.xpos[foot] + d.xmat[foot].reshape(3, 3) @ self.stance.sole)[:2]
                for foot in self.feet
            ]
        )

    def standing(self, d, settings, previous, anchors=None):
        result = np.empty(23, "f")
        config = np.asarray(settings, dtype="d")
        self.backend.stance_control(
            ct.c_void_p(self.model._address),
            ct.c_void_p(d._address),
            *[
                ptr(a)
                for a in (
                    self.bodies,
                    self.motor,
                    self.neutral,
                    anchors,
                    previous,
                    config,
                    self.actor,
                )
            ],
            ct.c_double(self.scale),
            ptr(result),
        )
        return result

    def select(self, data, recovered=False, expansion=False):
        offset = self.db["offset"]
        ref = self.db["q"][offset[:-1]]
        actual = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
        angles = Rotation.from_quat(ref[:, 3:7], scalar_first=True)
        distance = np.square(ref[:, 7:] - data.qpos[7:]).sum(-1) + 3.0 * np.square(
            angles.as_matrix()[:, 2] - actual.as_matrix()[2]
        ).sum(-1)
        last_primary = int(self.db.get("primary_count", len(distance)))
        last_recovery = int(self.db.get("expansion_start", len(distance)))
        bank = np.arange(len(distance))
        if expansion:
            use = bank >= last_recovery
        elif recovered:
            use = (bank >= last_primary) & (bank < last_recovery)
        else:
            use = bank < last_primary
        if not use.any():
            return []
        nearest = distance[use].min()
        if nearest > 0.75:
            return []
        candidates = np.flatnonzero(use & (distance <= nearest + (0.12 if expansion else 0.04)))
        quality = self.db.get("quality", np.zeros(len(distance)))
        style = self.db.get("support_style", np.zeros(len(distance)))
        # Prefer reference-timed hand support among demonstrations matching this measured configuration.
        candidates = sorted(candidates, key=lambda i: (quality[i] - 0.19 * style[i], i))
        if expansion:
            delay_error = np.mean(np.abs(self.db["delay"][:, robot.GROUP] - self.lags), axis=1)
            candidates.sort(key=lambda i: (distance[i] + 0.12 * delay_error[i] + 0.3 * quality[i], i))
        return [(i, *self.align(data, i, angles, actual)) for i in candidates]

    def align(self, data, index, angles, actual):
        offset = self.db["offset"]
        lo, hi = offset[index : index + 2]
        q = self.db["q"][lo:hi].copy()
        v = self.db["v"][lo:hi].copy()
        rel = (actual * angles[index].inv()).as_matrix()
        yaw = np.arctan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1])
        heading = Rotation.from_euler("z", yaw)
        q[:, :3] = heading.apply(q[:, :3])
        q[:, :2] += data.qpos[:2] - q[0, :2]
        q[:, 3:7] = (
            heading * Rotation.from_quat(q[:, 3:7], scalar_first=True)
        ).as_quat(scalar_first=True)
        v[:, :3] = heading.apply(v[:, :3])
        return q, v

    def _shoot(self, data, index, q, v, beam, batch, stride, seed, brake):
        lo, hi = self.db["offset"][index : index + 2]
        n = len(q) - 1
        stages = (n + stride - 1) // stride
        u = self.db["u"][lo:hi]
        targets = np.ascontiguousarray(
            np.vstack([self.neutral, self.neutral, self.neutral + 0.25 * u])
        )
        nominal = np.ascontiguousarray(self.db["delay"][index][robot.GROUP], dtype="i")
        gains = np.ascontiguousarray(self.db["K"][lo:hi], dtype="f")
        actions = np.empty((stages, beam, stride, 23), "f")
        parents = np.empty((stages, beam), "i")
        capture = np.empty((stages, beam))
        poses = np.empty((stages, beam, 30))
        support = (
            np.ascontiguousarray(self.db["desired_palm_load"][lo:hi])
            if "desired_palm_load" in self.db
            else np.zeros((len(q), 2))
        )
        self.backend.plan(
            ct.c_void_p(self.model._address),
            ct.c_void_p(data._address),
            *map(ct.c_int, (n, beam, batch, stride, 3, seed)),
            *[
                ptr(a)
                for a in (
                    self.motor,
                    self.lags,
                    nominal,
                    self.feet,
                    q,
                    v,
                    targets,
                    gains,
                    self.neutral,
                    self.previous,
                    self.older,
                    self.actor,
                )
            ],
            ct.c_double(self.scale),
            ct.c_double(brake),
            ptr(support),
            ptr(self.hands),
            ct.c_double(self.weight),
            *[ptr(a) for a in (actions, parents, capture, poses)],
        )
        return actions, parents, capture, poses

    def palm(self, d):
        force = np.empty(6)
        value = 0.0
        for j, c in enumerate(d.contact):
            a, b = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (a and b) or (a + b != self.hands[0] and a + b != self.hands[1]):
                continue
            mj.mj_contactForce(self.model, d, j, force)
            value += force[0]
        return value / self.weight

    def stance_forecast(self, data, anchors, previous, older, config, q, v):
        return self.backend.forecast_stance(
            ct.c_void_p(self.model._address),
            ct.c_void_p(data._address),
            ct.c_int(len(q)),
            *[
                ptr(a)
                for a in (
                    self.lags,
                    self.bodies,
                    self.motor,
                    self.neutral,
                    anchors,
                    previous,
                    older,
                    config,
                    self.actor,
                )
            ],
            ct.c_double(self.scale),
            ptr(q),
            ptr(v),
        )

    def absorb_push(self, data):
        # A sudden base velocity change after quiet loaded stance can invalidate its previous damping gain.
        # Select a damped response by forecasting the measured present state with the planted sole locations.
        velocity = data.qvel[:2]
        shock = (
            self.quiet_steps >= 20
            and self.last_xy_velocity is not None
            and np.linalg.norm(velocity - self.last_xy_velocity) > 0.09
        )
        self.quiet_steps = (
            self.quiet_steps + 1 if np.linalg.norm(velocity) < 0.06 else 0
        )
        self.last_xy_velocity = velocity.copy()
        if not shock:
            return
        forecast = mj.MjData(self.model)
        q = np.empty((180, 30))
        v = np.empty((180, 29))
        settings = [
            tuple(self.parameters),
            (0.018, 0.8),
            (0.022, 1.05),
            (0.012, 0.9),
            (0.025, 1.2),
            (0.030, 1.4),
            (0.010, 0.65),
            (0.028, 0.9),
        ]
        best = None
        for setting in dict.fromkeys(settings):
            config = np.asarray(setting, dtype="d")
            mj.mj_copyData(forecast, self.model, data)
            length = self.stance_forecast(
                forecast, self.anchors, self.previous, self.older, config, q, v
            )
            if length < len(q) or q[:, 2].min() < 0.52:
                continue
            speed = np.max(np.linalg.norm(v[-40:, :2], axis=1))
            drift = np.max(np.linalg.norm(q[:, :2] - data.qpos[:2], axis=1))
            cost = drift + 0.4 * speed
            if best is None or cost < best[0]:
                best = cost, config
            if (
                np.array_equal(config, self.parameters)
                and drift < 0.11
                and speed < 0.025
            ):
                break
        if best is not None:
            self.parameters = best[1]
        if self.db.get("impulse_control", False):
            mj.mj_copyData(forecast, self.model, data)
            config = np.asarray(self.parameters, dtype=float)
            length = self.stance_forecast(forecast, self.anchors, self.previous, self.older, config, q, v)
            if length == len(q) and q[:, 2].min() > 0.56 and np.linalg.norm(v[-40:, :2], axis=1).max() < 0.05:
                return
            from .zenbot_impulse import predict_impulse_response

            response = predict_impulse_response(self, data)
            if response is not None:
                self.parameters, self.pulse, frames = response
                self.pulse_end = self.tick + frames

    def _verify(self, reset, commands, settings, max_steps, style):
        d = mj.MjData(self.model)
        mj.mj_copyData(d, self.model, reset)
        prev = self.previous.copy()
        older = self.older.copy()
        q = [d.qpos.copy()]
        v = [d.qvel.copy()]
        palm = []
        for u in commands:
            goal = self.integrate(d, u, prev, older)
            older, prev = prev, goal
            q.append(d.qpos.copy())
            v.append(d.qvel.copy())
            palm.append(self.palm(d))
        captured = mj.MjData(self.model)
        mj.mj_copyData(captured, self.model, d)
        before = q.copy()
        speeds = v.copy()
        best = None
        mj.mj_kinematics(self.model, captured)
        anchors = self.soles(captured)
        for setting in settings:
            mj.mj_copyData(d, self.model, captured)
            qq = np.empty((max_steps, 30))
            vv = np.empty((max_steps, 29))
            config = np.asarray(setting, dtype="d")
            length = self.stance_forecast(d, anchors, prev, older, config, qq, vv)
            if length != max_steps:
                continue
            Q, V = np.vstack((before, qq)), np.vstack((speeds, vv))
            up = (
                (Q[:, 2] > 0.58)
                & (Q[:, 2] <= 0.78)
                & ((1.0 - 2.0 * np.square(Q[:, 4:6]).sum(-1)) > 0.8660254)
            )
            rise = np.flatnonzero(np.convolve(up, np.ones(10), "valid") >= 10)
            if not len(rise) or not up[-22:].all():
                continue
            k = rise[0]
            velocity = np.linalg.norm(V[-20:, :2], axis=1).mean()
            if (
                (
                    max_steps > 150
                    and (
                        np.linalg.norm(V[-30:, :2], axis=1).max() > 0.065
                        or np.sqrt(np.mean(V[-30:, 6:] ** 2)) > 0.7
                        or ("terminal_height_margin" in self.db and
                            Q[-30:, 2].min() < 0.58 + float(self.db["terminal_height_margin"]))
                    )
                )
                or velocity > 0.18
                or (
                    (Q[k + 10 :, 2] < 0.42)
                    | (1.0 - 2 * np.square(Q[k + 10 :, 4:6]).sum(-1) < 0.55)
                ).any()
            ):
                continue
            drift = np.linalg.norm(Q[k:, :2] - Q[k, :2], axis=-1).max()
            value = float(drift + 0.35 * velocity + 0.025 * ((~up[k:]).mean()))
            if np.isfinite(self.position_limit):
                if drift > self.position_limit:
                    continue
                # Check both feet from the first upright frame; an airborne leg
                # can otherwise travel far while the pelvis briefly passes through upright.
                stance_poses = Q[k::3]
                centers = np.empty((len(stance_poses), 2, 2))
                for frame, pose in enumerate(stance_poses):
                    d.qpos[:] = pose
                    mj.mj_kinematics(self.model, d)
                    centers[frame] = d.xpos[self.feet, :2]
                feet_span = np.linalg.norm(centers - centers[0], axis=-1).max()
                if feet_span > self.position_limit:
                    continue
                value += 0.25 * feet_span
            trough = peak = Q[0, 2]
            lift = False
            above = below = 0
            for z in Q[: k + 1, 2]:
                if not lift:
                    trough = min(trough, z)
                    above = above + 1 if z >= 0.50 and z - trough >= 0.10 else 0
                    if above >= 5:
                        lift = True
                        peak = z
                else:
                    peak = max(peak, z)
                    below = below + 1 if z <= 0.45 and peak - z >= 0.12 else 0
                    if below >= 5:
                        value += 0.53
                        lift = False
                        above = below = 0
                        trough = z
            if style is not None:
                require_hand, expected_extension, hip_p90 = style
                supported = (
                    len(palm[:k]) >= 4
                    and np.max(
                        np.convolve(np.asarray(palm[:k]) >= 0.08, np.ones(4), "valid")
                    )
                    >= 4
                )
                if require_hand and not supported:
                    value += 0.52
                low = np.flatnonzero(Q[: k + 1, 2] <= 0.46)
                duration = 0.02 * (k - low[-1]) if len(low) else 0.0
                value += 0.19 * max(0.0, expected_extension - duration)
                value += 0.06 * max(
                    0.0,
                    np.percentile(np.abs(V[: max(1, k), [6, 12]]), 90)
                    / max(1.0, hip_p90)
                    - 1.15,
                )
            if best is None or value < best[0]:
                best = (value, setting, np.array(before), np.array(speeds))
            if drift < 0.135 and (style is None or not require_hand or supported):
                break
        return best

    def _cached(self, data):
        if "cache_state" not in self.db:
            return False
        state = np.r_[data.qpos, data.qvel]
        match = np.flatnonzero(
            (self.db["cache_state"] == state).all(-1)
            & (self.db["cache_lags"] == self.lags).all(-1)
        )
        if not len(match):
            return False
        i = int(match[0])
        lo, hi = self.db["cache_offsets"][i : i + 2]
        self.plan = self.db["cache_action"][lo:hi]
        self.planned_q = self.db["cache_q"][lo:hi]
        self.planned_v = self.db["cache_v"][lo:hi]
        self.parameters = self.db["cache_parameters"][i]
        self.mode = 1
        self.debug = dict(
            matched=int(self.db["cache_source"][i]),
            planned_ticks=len(self.plan),
            forecast_objective=float(self.db["cache_score"][i]),
            cached=True,
        )
        return True

    def prepare(self, data):
        if self._cached(data):
            return True
        choices = self.select(data)
        if self.plan_choices(data, choices):
            return True
        # Previously successful perturbed recoveries supply additional starting guesses
        # when the primary demonstrations yield no physically stable motor sequence.
        alternatives = self.select(data, recovered=True)
        if self.plan_choices(data, alternatives):
            return True
        if np.isfinite(self.position_limit):
            # A constrained stand can be unreachable for a disturbed pose.
            # Try a wider support region before handing control back to the AMP actor.
            self.position_limit *= 1.5
            if self.plan_choices(data, choices) or self.plan_choices(data, alternatives):
                return True
        # A wider database of successful delayed physical recoveries is queried
        # only after the original planning pipeline exhausted every feasible branch.
        expansion = self.select(data, expansion=True)
        if not expansion:
            return False
        if np.isfinite(self.position_limit):
            self.position_limit = float(self.db["stand_radius"])
            if self.plan_choices(data, expansion):
                return True
            self.position_limit *= 1.5
        return self.plan_choices(data, expansion)

    def plan_choices(self, data, choices):
        if not choices:
            return False
        best = None
        # Compare different physically trained rises from the same starting configuration before widening the beam.
        specs = (
            (6, 8, 6, 1417, 1.8),
            (8, 10, 5, 2281, 4.0),
            (10, 12, 4, 7079, 0.3),
            (13, 13, 4, 21019, 0.7),
            (11, 14, 3, 32419, 2.4),
            (18, 18, 4, 63131, 1.2),
            (17, 22, 3, 95273, 3.0),
            (21, 20, 6, 131071, 0.4),
        )
        for spec, choice in ((s, c) for s in specs for c in choices):
            if best is not None and spec[3] > 10000:
                break
            index, q, v = choice
            usual = tuple(self.db["look_gain"][index])
            style = (
                self.db["reference_style"][index]
                if "reference_style" in self.db
                else None
            )
            settings = list(
                dict.fromkeys(
                    (usual, (0.018, 0.8), (0.022, 1.05), (0.012, 0.9), (0.012, 0.65))
                )
            )
            actions, parents, cost, poses = self._shoot(data, index, q, v, *spec)
            tried = []
            for flat in np.argsort(cost.ravel()):
                stage, node = np.unravel_index(flat, cost.shape)
                if cost[stage, node] > 1e4:
                    break
                pose = poses[stage, node]
                if any(
                    abs(stage - s) < 2 and np.linalg.norm(pose[7:] - x) < 0.24
                    for s, x in tried
                ):
                    continue
                tried.append((stage, pose[7:]))
                branch = node
                segments = []
                for j in range(stage, -1, -1):
                    segments.append(actions[j, branch])
                    branch = parents[j, branch]
                tape = np.concatenate(segments[::-1])[: len(q) - 1]
                outcome = self._verify(data, tape, settings, 100, style)
                if outcome is not None and (best is None or outcome[0] < best[0]):
                    outcome = self._verify(data, tape, [outcome[1]], 320, style)
                else:
                    outcome = None
                if outcome is not None:
                    score, setting, Q, V = outcome
                    if best is None or score < best[0]:
                        best = (score, tape.copy(), setting, Q, V, index)
                    if score < 0.185:
                        break
                if len(tried) >= (14 if len(choices) > 1 else 21):
                    break
            if best is not None and best[0] < 0.185:
                break
        if best is None:
            return False
        score, self.plan, self.parameters, self.planned_q, self.planned_v, index = best
        self.mode = 1
        self.debug = dict(
            matched=int(index),
            forecast_objective=float(score),
            planned_ticks=len(self.plan),
        )
        return True

    def action(self, data, fallback):
        if self.tick == 0 and self.mode == 0 and not self.prepare(data):
            self.mode = -1
        if self.mode == 1 and self.tick < len(self.plan):
            q = self.planned_q[self.tick]
            v = self.planned_v[self.tick]
            command = self.plan[self.tick].copy()
            error = np.empty(self.model.nv)
            mj.mj_differentiatePos(self.model, error, 1.0, q, data.qpos)
            expected_previous = self.plan[self.tick - 1] if self.tick else np.zeros(23)
            # Re-evaluate the trained feedback on the measured state following an interaction.
            command += (
                np.asarray(
                    self.scale
                    * (
                        np.tanh(
                            features(
                                data.qpos,
                                data.qvel,
                                (self.previous - self.neutral) * 4.0,
                            )
                            @ self.actor
                        )
                        - np.tanh(features(q, v, expected_previous) @ self.actor)
                    ),
                    dtype="f",
                )
                if np.linalg.norm(error) > 1.0e-8
                else 0.0
            )
            if np.linalg.norm(error) > 0.003:
                lo = self.db["offset"][self.debug["matched"]]
                K = self.db["K"][lo + self.tick]
                change = K[:, :58].astype(float) @ np.r_[error, data.qvel - v]
                command += np.clip(4.0 * change, -1.5, 1.5).astype("f")
        elif self.mode in (1, 2):
            self.mode = 2
            if self.anchors is None:
                mj.mj_kinematics(self.model, data)
                self.anchors = self.soles(data)
            self.absorb_push(data)
            command = self.standing(data, self.parameters, self.previous, self.anchors)
            if self.tick < self.pulse_end:
                command += self.pulse * min(1.0, (self.pulse_end - self.tick) / 4)
            if data.qpos[2] < 0.35 or data.xmat[1, 8] < 0.40:
                self.mode = -1
        else:
            command = fallback
        self.tick += 1
        goal = (
            robot.DEFAULT_Q + robot.ACTION_SCALE * np.asarray(command, dtype="f")
        ).astype("d")
        self.older, self.previous = self.previous, goal
        return np.asarray(command, dtype="f")
