"""Native G1 feedback using the measured MuJoCo floating base, joints and foot locations."""

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .g1_tracking_capture import _AnchoredStance, residual_features
from .native_getup import advance, observation_terms
from .g1_getup import KEY_BODIES, KP


def supported_after_rise(samples):
    """Keep an upright posture and carry weight on the feet throughout each second at 50 Hz."""
    if len(samples) < 50:
        return False
    upright, loaded = np.asarray(samples, dtype=bool).T
    return bool(upright.all() and np.convolve(loaded, np.ones(50, int), "valid").min() >= 45)


class FloatingBaseTracking:
    """Select a local motion by its initial body configuration and hold the acquired footprint."""

    def __init__(
        self, model, policy, capture, planning=0, reference=None, support_timing=None
    ):
        self.model = model
        self.policy = policy
        self.solver = _AnchoredStance(model)
        self.capture = np.asarray(capture, dtype=int)
        self.starts = policy.starts
        self.zero = np.zeros(model.nv)
        if (
            self.capture.shape != self.starts.shape
            or np.any(self.capture < 1)
            or np.any(self.capture + self.starts > policy.end[self.starts])
        ):
            raise ValueError(
                "Invalid support acquisition offsets for floating-base tracking"
            )
        self.planning = int(planning)
        if self.planning not in range(14):
            raise ValueError("Unknown G1 acquisition planning method")
        self.reference = reference
        if self.planning >= 3 and (
            reference is None
            or reference.shape != (len(policy.q), 59)
            or not np.isfinite(reference).all()
        ):
            raise ValueError(
                "Native style planning requires the matched reference positions and velocities"
            )
        self.support_timing = support_timing
        if self.planning >= 4:
            if support_timing is None or support_timing.shape != (
                len(policy.q),
                7 if self.planning >= 6 else 6,
            ):
                raise ValueError(
                    "Paced G1 recovery requires timing and palm phase targets"
                )
            self.floor = model.geom("floor").id
            self.hand_side = np.full(model.ngeom, -1)
            self.hand_bodies = [
                model.body("robot_" + side + "_wrist_yaw_link").id
                for side in ("left", "right")
            ]
            for i, b in enumerate(self.hand_bodies):
                for gid in np.flatnonzero(model.geom_bodyid == b):
                    if model.geom_contype[gid]:
                        self.hand_side[gid] = i
            self.body_weight = model.body_mass.sum() * 9.81
            self.hand_wrench = np.empty(6)
            self.arm_jac = np.empty((3, model.nv))
        self.links = [model.body("robot_" + name).id for name in KEY_BODIES]
        if self.planning >= 9:
            self.foot_geoms = np.isin(model.geom_bodyid, self.links[:2])
        self.loaded_balance = None
        if self.planning >= 12:
            from .g1_loaded_balance import LoadedBalance
            self.loaded_balance = LoadedBalance(model, self.solver.feet, self.solver.foot_offset)
        if self.planning:
            self.probe = mj.MjData(model)
            self.probe_history = np.zeros((1, 570), np.float32)
            self.predict_residual = jax.jit(
                lambda h: (
                    policy.residual_scale
                    * jnp.tanh(
                        residual_features(h) @ policy.residual_w + policy.residual_b
                    )
                )
            )
        self.reset()

    def reset(self):
        if self.loaded_balance is not None:
            self.loaded_balance.reset()
        self.row = -1
        self.tick = 0
        self.footprint = None
        self.pelvis = None
        self.engaged = False
        self.feedback_scale = 1.0
        self.drive_filter = 0.0
        self.reference_weight = 0.0
        self.probe_style = float("inf")
        self.last_drive = None
        self.cap_tick = 0
        self.plan_attempts = 0
        self.plan_score = None
        self.pace = 1.0
        self.crouch_steps = 0
        self.press = 0.0
        self.crouch_height = None
        self.phase_cursor = 0.0
        self.capture_time = None
        self.support_frames = 0
        self.hand_consecutive = self.hand_steps = 0
        self.acquired_hand = False
        self.paced_score = None
        self.confirmed_profile = False
        self.quiet_frames = 0
        self.impulse_armed = False
        self.balance_pulse = None
        self.pulse_frame = 0
        self.balance_actor_frame = None
        self.balance_actor_share = 0.

    def _acquire(self, data):
        q, v = data.qpos, data.qvel
        ref = self.policy.q[self.starts]
        attitude = Rotation.from_quat(ref[:, 3:7], scalar_first=True)
        actual = Rotation.from_quat(q[3:7], scalar_first=True)
        gravity = attitude.as_matrix()[:, 2] - actual.as_matrix()[2]
        distance = (
            np.square(ref[:, 7:] - q[7:]).sum(-1)
            + 2.0 * np.square(gravity).sum(-1)
            + 0.0018 * np.square(self.policy.v[self.starts, 6:] - v[6:]).sum(-1)
        )
        row = int(np.argmin(distance))
        if distance[row] > 0.65**2 or abs(ref[row, 2] - q[2]) > 0.12:
            return False
        rel = (actual * attitude[row].inv()).as_matrix()
        yaw = np.arctan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1])
        self.heading = Rotation.from_euler("z", yaw)
        self.origin = q[:2] - self.heading.apply(ref[row, :3])[:2]
        self.row = row
        if self.planning >= 4:
            similar = np.abs(ref[:, 7:] - ref[row, 7:]).max(-1) < 0.025
            self.near_rows = np.flatnonzero(
                similar & (np.abs(ref[:, 2] - ref[row, 2]) < 0.025)
            )
        self.cap_tick = self.capture[row]
        return True

    def action(self, data, fallback, previous_action=None):
        self.engaged = False
        self.balance_actor_share = 0.
        if self.row == -2:
            return fallback
        if self.row < 0 and not self._acquire(data):
            # A new recovery starts in reset(). Keep the general learned policy on unfamiliar starts.
            self.row = -2
            return fallback
        if self.planning and not self.tick:
            self._select(data)
        if self.planning >= 10 and self.pelvis is not None and previous_action is not None:
            speed = np.linalg.norm(data.qvel[:2])
            lean = False
            if self.loaded_balance is not None:
                footprint, lean = self.loaded_balance.update(data)
                if footprint is not None:
                    self.footprint = footprint
                    self.pelvis = data.qpos[:2].copy()
            impulse = self.impulse_armed and speed > .12
            if (impulse or lean) and self.balance_actor_frame is None and self.balance_pulse is None:
                from .g1_stance_impulse import fit_stance_impulse
                if impulse and self.planning >= 13:
                    self.loaded_balance.after_impulse()
                self.balance_pulse = fit_stance_impulse(self, data, previous_action)
                if self.planning >= 11 and self.balance_pulse is None:
                    self.balance_actor_frame = 0
                self.pulse_frame = self.quiet_frames = 0
                self.impulse_armed = False
            elif self.balance_pulse is None and self.balance_actor_frame is None:
                quiet = speed < .04 and .68 < data.qpos[2] < .83 and data.xmat[1, 8] > .91
                self.quiet_frames = self.quiet_frames + 1 if quiet else 0
                self.impulse_armed |= self.quiet_frames >= 40
            if self.balance_actor_frame is not None:
                # A large disturbance can place the body outside the local ankle/hip
                # correction. Use the trained whole-body actor briefly before holding.
                t = self.balance_actor_frame
                self.balance_actor_share = min(1., (t + 1) / 3., (26 - t) / 5.)
                self.balance_actor_frame = t + 1 if t < 25 else None
                if t == 25 and self.loaded_balance is not None:
                    self.loaded_balance.replant = True
        return self._control(data, fallback)

    def balance_action(self, tracked, actor):
        if not self.balance_actor_share:
            return tracked
        return tracked + self.balance_actor_share * (actor - tracked)

    def _control(self, data, fallback):
        phase = self.tick * self.pace if self.planning >= 4 else self.tick
        if self.planning >= 5:
            phase = self.phase_cursor
        k = self.starts[self.row] + min(int(phase), self.cap_tick)
        if self.planning >= 4:
            hand_load = self._hand_load(data)
            self.last_palm_load = hand_load.sum()
            if data.qpos[2] < 0.64:
                self.hand_consecutive = (
                    self.hand_consecutive + 1 if hand_load.sum() >= 0.055 else 0
                )
                if self.planning >= 5:
                    self.support_frames = (
                        self.support_frames + 1 if hand_load.sum() >= 0.07 else 0
                    )
                    self.acquired_hand |= self.support_frames >= (
                        6 if self.planning >= 8 else 4
                    )
                else:
                    self.acquired_hand |= self.hand_consecutive >= 3
                self.hand_steps += int(hand_load.sum() >= 0.055)
        if self.footprint is not None and (
            data.qpos[2] < 0.34 or data.xmat[1, 8] < 0.44
        ):
            self.row = -2
            self.engaged = False
            return fallback
        self.engaged = True
        if phase < self.cap_tick:
            q = self.policy.q[k].copy()
            v = self.policy.v[k].copy()
            motor = self.policy.u[k]
            if self.planning >= 4 and self.pace != 1.0:
                frac = phase - int(phase)
                qnext = self.policy.q[k + 1]
                q = q + frac * (qnext - q)
                q[3:7] /= np.linalg.norm(q[3:7])
                v = self.pace * (v + frac * (self.policy.v[k + 1] - v))
                motor = motor + frac * (self.policy.u[k + 1] - motor)
            q[:3] = self.heading.apply(q[:3])
            q[:2] += self.origin
            q[3:7] = (
                self.heading * Rotation.from_quat(q[3:7], scalar_first=True)
            ).as_quat(scalar_first=True)
            v[:3] = self.heading.apply(v[:3])
            mj.mj_differentiatePos(self.model, self.zero, 1.0, q, data.qpos)
            dx = self.zero.copy()
            dv = data.qvel - v
            dx[:3] = self.heading.inv().apply(dx[:3])
            dv[:3] = self.heading.inv().apply(dv[:3])
            correction = np.asarray(self.policy.gain[k], np.float64) @ np.r_[dx, dv]
            u = motor + np.clip(self.feedback_scale * correction / 0.25, -5.0, 5.0)
        else:
            solver = self.solver
            if self.footprint is None:
                self.capture_time = self.tick
                self.footprint = np.array(
                    [
                        (
                            data.xpos[f]
                            + data.xmat[f].reshape(3, 3) @ solver.foot_offset
                        )[:2]
                        for f in solver.feet
                    ]
                )
            gain, strength = self.policy.params[k]
            if self.pelvis is None and data.qpos[2] > 0.635 and data.xmat[1, 8] > 0.84:
                self.pelvis = data.qpos[:2].copy()
            solver.data = data
            mj.mj_forward(self.model, data)
            center = data.qpos[:2] if self.pelvis is None else self.pelvis
            height = 0.748
            if self.planning >= 4 and self.crouch_steps:
                if self.crouch_height is None:
                    self.crouch_height = float(
                        np.clip(data.subtree_com[1, 2], 0.48, 0.748)
                    )
                elapsed = self.tick - (
                    self.capture_time
                    if self.planning >= 5
                    else self.cap_tick / self.pace
                )
                progress = np.clip(elapsed / self.crouch_steps, 0.0, 1.0)
                ease = progress * progress * (3.0 - 2.0 * progress)
                height = self.crouch_height + ease * (0.748 - self.crouch_height)
            u = solver.action(
                self.footprint,
                center,
                float(gain),
                0.0 if self.pelvis is None else float(strength),
                desired_height=height,
                support_shift=self.loaded_balance.shift if self.loaded_balance is not None else None,
            )
            if self.planning >= 10 and data is not self.probe and self.balance_pulse is not None:
                from .g1_stance_impulse import LEG_JOINTS, pulse_weight
                offset, duration = self.balance_pulse
                u[LEG_JOINTS] += pulse_weight(self.pulse_frame, duration) * offset
                self.pulse_frame += 1
                if self.pulse_frame >= duration:
                    self.balance_pulse = None
        if self.reference_weight and phase < self.cap_tick:
            ref = self.reference[k]
            error = (ref[:29] - data.qpos[7:]) + 0.035 * (ref[29:58] - data.qvel[6:])
            target = self.reference_weight * np.clip(error, -0.50, 0.50) / 0.25
            target[:12] *= 0.60
            # Acquire and release the pose bias over four control frames.
            u += (
                min(1.0, (self.tick + 1) / 4.0, (self.cap_tick - self.tick) / 4.0)
                * target
            )
        if (
            self.planning >= 4
            and self.press
            and phase < self.cap_tick
            and data.qpos[2] < 0.65
        ):
            self._hand_pressure(data, u, self.support_timing[k, :2], hand_load)
        if self.drive_filter and phase < self.cap_tick and self.last_drive is not None:
            u = (1.0 - self.drive_filter) * u + self.drive_filter * self.last_drive
        if self.drive_filter:
            self.last_drive = u
        if self.planning >= 5:
            gate = self.support_timing[k, 5] * self.cap_tick
            needs_load = self.support_timing[k, 2] and not self.acquired_hand
            self.phase_cursor = (
                min(phase + self.pace, gate)
                if needs_load and phase <= gate
                else phase + self.pace
            )
            if needs_load and phase >= gate - 1 and data.qpos[2] < 0.66:
                # Keep the support arm working against the physical floor until its load is confirmed.
                nearest = int(np.argmin(data.xpos[self.hand_bodies, 2]))
                self._hand_pressure(data, u, np.eye(2, dtype=bool)[nearest], hand_load)
        self.tick += 1
        return np.asarray(u, np.float32)

    def _hand_load(self, data):
        load = np.zeros(2)
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = c.geom
            g = b if a == self.floor else a if b == self.floor else -1
            if g < 0 or self.hand_side[g] < 0 or c.efc_address < 0:
                continue
            mj.mj_contactForce(self.model, data, i, self.hand_wrench)
            vertical = self.hand_wrench[:3] @ c.frame.reshape(3, 3)[:, 2]
            load[self.hand_side[g]] += abs(vertical) / self.body_weight
        return load

    def _hand_pressure(self, data, u, targets, load):
        for i in np.flatnonzero(targets):
            b = self.hand_bodies[i]
            if data.xpos[b, 2] > 0.32 or load[i] > 0.24:
                continue
            mj.mj_jac(self.model, data, self.arm_jac, None, data.xpos[b], b)
            downward = self.press * np.clip(
                (0.17 - load[i]) * self.body_weight + 85.0 * (data.xpos[b, 2] - 0.05),
                0.0,
                0.25 * self.body_weight,
            )
            joint = slice(15 + 7 * i, 22 + 7 * i)
            z_jac = self.arm_jac[2, 6:][joint]
            if self.planning >= 6 and load[i] < 0.05:
                # Lower an unloaded reference support palm to the contact surface through
                # the arm joints, including while the pelvis is still close to the ground.
                dz = np.clip(0.05 - data.xpos[b, 2], -0.14, 0.0)
                reach = z_jac * dz / (z_jac @ z_jac + 0.012)
                u[joint] += self.press * np.clip(reach / 0.25, -0.8, 0.8)
            torque = -z_jac * downward
            u[joint] += np.clip(torque / (0.25 * KP[joint]), -0.85, 0.85)

    def _feet_carry_weight(self, data):
        normal_force = 0.
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = contact.geom
            geom = b if a == self.floor else a if b == self.floor else -1
            if geom >= 0 and self.foot_geoms[geom] and contact.efc_address >= 0:
                mj.mj_contactForce(self.model, data, i, self.hand_wrench)
                normal_force += self.hand_wrench[0]
        return normal_force >= 0.5 * self.body_weight

    def _predicted_action(self, data, previous):
        gyro, axes, q, v, last, _ = observation_terms(data, self.links, previous)
        h = self.probe_history
        h[0, 12:15], h[0, 39:45], h[0, 161:190] = gyro, axes, q
        h[0, 306:335], h[0, 451:480] = v, last
        # Match the actuator rounding and trained motor correction of the live player.
        return np.asarray(self.predict_residual(h)[0])

    def _probe(self, data, gain, offset):
        self.last_drive = None
        state = self.probe
        mj.mj_copyData(state, self.model, data)
        self.tick = 0
        self.footprint = self.pelvis = None
        self.feedback_scale = gain
        self.cap_tick = int(
            np.clip(
                self.capture[self.row] + offset,
                1,
                self.policy.end[self.starts[self.row]] - self.starts[self.row],
            )
        )
        self.crouch_height = None
        self.phase_cursor = 0.0
        self.capture_time = None
        self.support_frames = 0
        self.hand_consecutive = self.hand_steps = 0
        self.acquired_hand = False
        self.paced_score = None
        self.confirmed_profile = False
        horizon = int(np.ceil(self.cap_tick / self.pace)) + max(
            230 if self.planning >= 8 else 94, self.crouch_steps + 40
        )
        previous = np.zeros(29, np.float32)
        peak, slide, repeats, quiet, upright = 0.0, 0.0, 0, 0, 0
        anchored = False
        first_up = None
        knee_history = []
        palm_history = []
        supported_stance = []
        first_extension = None
        hand_at_extension = False
        extension = 0
        was_up = False
        raised = False
        high = -np.inf
        loss = 0.0
        motion_loss = np.zeros(3)
        samples = 0
        self.probe_style = float("inf")
        for t in range(horizon):
            residual = self._predicted_action(state, previous)
            base = self._control(state, previous)
            if not self.engaged:
                return 25.0 + (horizon - t) * 0.045
            if self.planning >= 6:
                knee_history.append(state.qpos[[10, 16]].mean())
                palm_history.append(self.last_palm_load)
            command = base + residual
            if self.planning >= 3:
                before = state.qvel[6:].copy()
            advance(self.model, state, command)
            if self.planning >= 3:
                k = min(
                    self.policy.end[self.starts[self.row]],
                    self.starts[self.row] + int((t + 1) * self.pace),
                )
                ref = self.reference[k]
                if ref[-1]:
                    motion_loss += (
                        np.mean(np.square(state.qpos[7:] - ref[:29])),
                        np.mean(np.square(state.qvel[6:] - ref[29:58])),
                        np.mean(np.square((state.qvel[6:] - before) / 0.02)),
                    )
                    samples += 1
            previous = command
            p = state.qpos[:3]
            angle = state.xmat[1, 8]
            high = max(high, p[2])
            raised |= p[2] >= 0.51
            if not was_up and raised and high - p[2] > 0.12 and p[2] <= 0.44:
                repeats += 1
                high, raised = p[2], False
            standing = 0.65 < p[2] <= 0.85 and angle > 0.8660254
            if self.planning >= 9:
                supported_stance.append((standing, standing and self._feet_carry_weight(state)))
            if standing and not upright and not anchored:
                anchor = p[:2].copy()
                ankles = state.xpos[self.links[:2], :2].copy()
                peak = slide = 0.0
            upright = upright + 1 if standing else 0
            anchored |= upright >= 10
            if self.planning >= 4:
                extension = extension + 1 if p[2] > 0.595 and angle > 0.73 else 0
                if extension >= 5 and first_extension is None:
                    first_extension = t - 3
                    hand_at_extension = self.acquired_hand
                if anchored and first_up is None:
                    first_up = t - 8
            was_up |= anchored
            if anchored or upright:
                peak = max(peak, np.linalg.norm(p[:2] - anchor))
                slide = max(
                    slide,
                    np.linalg.norm(
                        state.xpos[self.links[:2], :2] - ankles, axis=-1
                    ).max(),
                )
            if anchored and (angle < 0.50 or p[2] < 0.50):
                return 24.0 + (horizon - t) * 0.025
            if t >= horizon - 25:
                quiet += int(
                    standing and angle > 0.945 and np.linalg.norm(state.qvel[:2]) < 0.14
                )
            if t < self.cap_tick:
                k = self.starts[self.row] + t
                loss += min(2.0, np.mean((state.qpos[7:] - self.policy.q[k, 7:]) ** 2))
        if (
            self.planning >= 3
            and anchored
            and quiet >= 23
            and peak <= 0.125
            and slide <= 0.12
        ):
            pose, speed, accel = np.sqrt(motion_loss / max(1, samples))
            self.probe_style = (
                2.2 * pose
                + 0.17 * speed
                + 0.006 * accel
                + 0.38 * (peak + slide)
                + 2.1 * max(0.0, peak - 0.09, slide - 0.085)
            )
        if (
            self.planning >= 4
            and anchored
            and quiet >= 23
            and peak < 0.18
            and slide < 0.18
        ):
            settings = self.support_timing[self.starts[self.row]]
            required, desired = bool(settings[2]), settings[3]
            hand_ok = not required or hand_at_extension
            coverage = 1.0
            if self.planning >= 6 and required:
                knees = np.asarray(knee_history[: first_up + 1])
                palms = np.asarray(palm_history[: first_up + 1])
                high = int(np.argmax(knees))
                extended = np.convolve(
                    (knees[high:] + 0.35 < knees[high]).astype(int),
                    np.ones(4, int),
                    mode="valid",
                )
                ids = np.flatnonzero(extended == 4)
                extension_at = high + ids[0] if len(ids) else len(knees) - 1
                required_frames = 6 if self.planning >= 8 else 4
                carries = np.convolve(
                    (palms[: extension_at + 1] > 0.07).astype(int),
                    np.ones(required_frames, int),
                    mode="valid",
                )
                hand_ok = bool((carries == required_frames).any())
                loc = np.minimum(
                    self.policy.end[self.starts[self.row]],
                    self.starts[self.row]
                    + np.rint(np.linspace(0.0, settings[6], len(palms))).astype(int),
                )
                expected = self.support_timing[loc, :2].any(-1)
                coverage = (palms[expected] > 0.07).mean() if expected.any() else 1.0
            self.confirmed_profile = (
                hand_ok
                and coverage >= 0.55
                and 0.89 * desired <= first_up <= 1.16 * desired
                and peak <= 0.10
                and slide <= 0.105
            )
            early = max(0.0, desired - first_up) / 50.0
            late = max(0.0, first_up - 1.20 * desired) / 50.0
            pose, velocity, accel = np.sqrt(motion_loss / max(1, samples))
            self.paced_score = (
                8.0 * (not hand_ok)
                + 1.25 * (1.0 - coverage)
                + 2.2 * early
                + 0.3 * late
                + 4.0 * max(0.0, peak - 0.09, slide - 0.09)
                + 0.42 * (peak + slide)
                + 1.5 * pose
                + 0.11 * velocity
                + 0.0038 * accel
            )
            if self.planning >= 9:
                # Rank the transfer onto the feet from the first sustained upright
                # posture, including the initial second of balance after knee extension.
                if not supported_after_rise(supported_stance[first_up - 1:]):
                    self.paced_score += 6.
                    self.confirmed_profile = False
        return (
            12.0 * (not anchored)
            + 4.0 * (1.0 - quiet / 25.0)
            + 3.0 * repeats
            + peak
            + 0.7 * slide
            + 0.12 * np.linalg.norm(state.qvel[:2])
            + 0.04 * loss / max(self.cap_tick, 1)
        )

    def _select(self, data):
        # Evaluate candidate local feedback gains and support timing on independent
        # copies of the sensed state. The live robot only receives its next motor command.
        start = self.row
        if self.planning >= 4:
            return self._select_paced(data)
        options = [
            (1.0, 0),
            (0.84, 4),
            (1.13, -3),
            (0.93, 10),
            (1.25, 8),
            (0.72, -4),
            (1.0, -8),
            (1.06, 5),
            (0.87, -2),
            (1.36, -2),
            (0.68, 6),
            (0.96, -5),
            (1.18, 2),
            (0.78, -10),
            (1.04, 14),
            (0.9, -12),
        ]
        best = (float("inf"), 1.0, 0)
        for gain, offset in options:
            self.row = start
            self.engaged = True
            value = self._probe(data, gain, offset)
            self.plan_attempts += 1
            if value < best[0]:
                best = (value, gain, offset)
            if value < 0.145:
                break
        # For a predicted failure, broaden the search to damped rising commands and
        # earlier/later support. The easy trajectories keep the first reliable profile.
        chosen_filter = 0.0
        if self.planning >= 2 and best[0] > 5.0:
            for filtering in (0.08, 0.19, 0.0, 0.32):
                for factor, offset in (
                    (1.0, 0),
                    (0.83, 4),
                    (1.19, -6),
                    (0.64, 10),
                    (1.38, -11),
                    (0.97, -16),
                    (0.75, 20),
                    (1.55, 3),
                    (0.56, -7),
                    (1.06, 22),
                    (1.64, -8),
                    (0.52, 15),
                    (1.23, -16),
                    (1.44, 13),
                    (0.89, -23),
                    (0.68, 28),
                ):
                    self.row = start
                    self.engaged = True
                    self.drive_filter = filtering
                    value = self._probe(data, factor, offset)
                    self.plan_attempts += 1
                    if value < best[0]:
                        best = (value, factor, offset)
                        chosen_filter = filtering
                    if best[0] < 0.15:
                        break
                if best[0] < 0.15:
                    break
        if self.planning >= 3:
            self.row = start
            self.engaged = True
            self.drive_filter = chosen_filter
            self._probe(data, best[1], best[2])
            score = self.probe_style
            selected_weight = 0.0
            for weight, filtering, gain, offset in (
                (0.28, 0.12, 1.0, 0),
                (0.48, 0.20, 0.94, 1),
                (0.72, 0.28, 1.07, -2),
                (0.4, 0.33, 1.1, 4),
                (0.17, 0.19, 0.89, -4),
                (0.60, 0.07, 1.17, 5),
                (0.82, 0.40, 1.0, 3),
                (0.0, 0.35, 0.94, 0),
            ):
                self.row = start
                self.engaged = True
                self.drive_filter = filtering
                self.reference_weight = weight
                value = self._probe(data, gain, offset)
                self.plan_attempts += 1
                if self.probe_style < score:
                    score = self.probe_style
                    best = (value, gain, offset)
                    chosen_filter, selected_weight = filtering, weight
            self.reference_weight = selected_weight
        self.drive_filter = chosen_filter
        self.last_drive = None
        self.row = start
        self.tick = 0
        self.footprint = self.pelvis = None
        self.feedback_scale = best[1]
        self.cap_tick = int(
            np.clip(
                self.capture[start] + best[2],
                1,
                self.policy.end[self.starts[start]] - self.starts[start],
            )
        )
        self.plan_score = best[0]

    def _select_paced(self, data):
        # Each branch has a causal, continuously advancing phase. Simulate from
        # measured state with real palm-floor forces before deciding how slowly to extend.
        best = None
        fallback = None
        rows = list(self.near_rows)
        arms = [
            (1.0, 0, 0.75, 32, 0.25),
            (0.90, 3, 0.81, 44, 0.55),
            (1.12, -4, 0.70, 52, 0.30),
            (1.0, 0, 0.88, 37, 0.35),
            (1.0, 0, 0.94, 48, 0.35),
            (1.0, 0, 1.0, 32, 0.0),
            (0.84, 4, 1.0, 28, 0.20),
            (1.13, -3, 1.0, 35, 0.35),
            (0.93, 10, 0.85, 28, 0.20),
        ]
        if self.planning >= 6:
            arms += [
                (1.0, -13, 0.96, 75, 0.55),
                (0.92, -22, 1.0, 100, 0.28),
                (1.18, -8, 0.90, 78, 0.65),
                (0.85, 0, 0.71, 66, 0.72),
            ]
        if self.planning >= 7:
            # First validate the learned trajectory at its own trained pace. An exact
            # fit can be accepted immediately. The fallback search covers other starts.
            for row in rows:
                if not self.support_timing[self.starts[row], 4]:
                    continue
                self.row = row
                self.engaged = True
                self.pace = 1.0
                self.crouch_steps = 0
                self.press = 0.0
                value = self._probe(data, 1.0, 0)
                self.plan_attempts += 1
                choice = (row, 1.0, 0, 1.0, 0, 0.0)
                if fallback is None or value < fallback[0]:
                    fallback = (value, choice)
                if self.paced_score is not None and (
                    best is None or self.paced_score < best[0]
                ):
                    best = (self.paced_score, choice)
                if self.confirmed_profile:
                    return self._use_profile((self.paced_score, choice))
        for gain, offset, pace, extension, pressure in arms:
            for row in rows:
                self.row = row
                self.engaged = True
                self.pace = pace
                self.crouch_steps = extension
                self.press = pressure
                self.reference_weight = 0.0
                self.drive_filter = 0.0
                # A deliberately retimed, contact-qualified example already embeds its pace.
                if self.support_timing[self.starts[row], 4]:
                    self.pace = min(1.05, pace / 0.75)
                value = self._probe(data, gain, offset)
                self.plan_attempts += 1
                choice = (row, gain, offset, self.pace, extension, pressure)
                if fallback is None or value < fallback[0]:
                    fallback = (value, choice)
                if self.paced_score is not None and (
                    best is None or self.paced_score < best[0]
                ):
                    best = (self.paced_score, choice)
            if best is not None and best[0] < 1.10:
                break
        if best is None or best[0] >= 4.0:
            # Search more contact timings before allowing a poor first rise.
            for gain, offset, pace, extension, pressure in (
                (0.74, -5, 0.82, 35, 0.55),
                (1.29, 5, 0.93, 27, 0.18),
                (1.0, -8, 1.0, 0, 0.0),
                (1.36, -2, 1.0, 12, 0.0),
                (0.68, 6, 1.0, 0, 0.0),
                (0.96, -5, 1.0, 0, 0.0),
            ):
                for row in rows:
                    self.row = row
                    self.engaged = True
                    self.pace = pace
                    self.crouch_steps = extension
                    self.press = pressure
                    value = self._probe(data, gain, offset)
                    self.plan_attempts += 1
                    choice = (row, gain, offset, self.pace, extension, pressure)
                    if value < fallback[0]:
                        fallback = (value, choice)
                    if self.paced_score is not None and (
                        best is None or self.paced_score < best[0]
                    ):
                        best = (self.paced_score, choice)
                if best is not None and best[0] < 1.5:
                    break
        return self._use_profile(best or fallback)

    def _use_profile(self, outcome):
        self.plan_score, choice = outcome
        row, gain, offset, self.pace, self.crouch_steps, self.press = choice
        self.row = row
        self.feedback_scale = gain
        self.cap_tick = int(
            np.clip(
                self.capture[row] + offset,
                1,
                self.policy.end[self.starts[row]] - self.starts[row],
            )
        )
        self.tick = 0
        self.phase_cursor = 0.0
        self.capture_time = None
        self.footprint = self.pelvis = self.crouch_height = self.last_drive = None
        self.support_frames = 0
        self.hand_consecutive = self.hand_steps = 0
        self.acquired_hand = False


def load_native_tracker(model, policy, checkpoint):
    with np.load(checkpoint) as z:
        if "floating_base/sensor" not in z:
            return None
        if (
            str(z["floating_base/sensor"]) != "mujoco_root_and_foot_poses"
            or policy.tracking_capture is None
        ):
            raise ValueError("Unknown native G1 floating-base tracking configuration")
        return FloatingBaseTracking(
            model,
            policy.tracking_capture,
            z["floating_base/capture_offsets"],
            int(z.get("floating_base/planning", 0)),
            z.get("floating_base/reference"),
            z.get("floating_base/support_timing"),
        )
