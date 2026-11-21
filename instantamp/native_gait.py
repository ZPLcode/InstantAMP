"""Run the exported G1 velocity policy on the same physical articulation used for recovery."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from . import env, g1_getup as robot


@dataclass(frozen=True)
class RunGains:
    hip: float
    knee: float
    ankle: float
    knee_offset: float
    hip_offset: float
    push: float
    swing: float
    waist_pitch: float

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values = values.get("gains", values) if isinstance(values, dict) else values
        gains = cls(**values) if isinstance(values, dict) else cls(*values)
        if not np.isfinite(list(vars(gains).values())).all():
            raise ValueError("Expected eight finite running adaptation gains")
        return gains


def ground_contacts(model, data, floor, feet):
    """A bit per supporting foot and a separate flag for any other active floor contact."""
    mask = 0
    other_contact = False
    for contact in data.contact:
        geom1, geom2 = contact.geom
        if geom1 == floor:
            other = geom2
        elif geom2 == floor:
            other = geom1
        else:
            continue
        if contact.efc_address < 0 or contact.dist > .001:
            continue
        link = model.geom_bodyid[other]
        if link == feet[0]:
            mask |= 1
        elif link == feet[1]:
            mask |= 2
        else:
            other_contact = True
    return mask, other_contact


class JogSpeedCalibration:
    """Map desired speed to the ONNX command and hip swing. Grounded feet estimate forward speed."""

    def __init__(self, speed, neural, swing, feedback):
        self.speed = speed
        self.neural = neural
        self.swing = swing
        self.feedback = feedback
        self.reset()

    @classmethod
    def read(cls, profile, speed):
        config = json.loads(Path(profile).read_text())
        table = config.get("jog_speed_calibration") if isinstance(config, dict) else None
        if table is None:
            return None
        values = np.asarray(table["knots"], dtype=float)
        if (values.ndim != 2 or values.shape[1] != 4 or len(values) < 2 or not np.isfinite(values).all()
                or np.any(np.diff(values[:, 0]) <= 0) or np.any(values[:, :3] <= 0) or np.any(values[:, 3] < 0)):
            raise ValueError("Expected increasing, finite [speed, neural command, hip scale, feedback] calibration knots")
        if not values[0, 0] <= speed <= values[-1, 0]:
            return None
        neural, swing, feedback = [float(np.interp(speed, values[:, 0], column)) for column in values[:, 1:].T]
        if neural == speed and swing == 1. and feedback == 0.:
            return None
        return cls(speed, neural, swing, feedback)

    def reset(self):
        self.last_soles = None
        self.forward_speed = 0.

    def command(self, gait, model, data, requested):
        # Root translation cancels in these pelvis-relative sole positions. A planted sole's
        # motion relative to the moving pelvis gives forward speed; retain the estimate during flight.
        if self.feedback:
            root = data.xmat[1].reshape(3, 3)
            rotations = data.xmat[gait.feet].reshape(2, 3, 3)
            soles = (data.xpos[gait.feet] + rotations @ np.asarray([.03, 0., -.03]) - data.qpos[:3]) @ root
            if self.last_soles is not None:
                mask, _ = ground_contacts(model, data, gait.floor, gait.feet)
                supported = [i for i in range(2) if mask & (1 << i)]
                if supported:
                    relative = (soles - self.last_soles) * 50. + np.cross(data.qvel[3:6], soles)
                    self.forward_speed += .09 * (-relative[supported, 0].mean() - self.forward_speed)
            self.last_soles = soles.copy()
        output = np.asarray(requested, dtype=float).copy()
        share = np.clip((requested[0] / self.speed - .78) / .22, 0., 1.)
        correction = np.clip((self.speed - self.forward_speed) * self.feedback, -.11, .11)
        output[0] = requested[0] * (self.neural / self.speed) + share * correction
        return output


class G1VelocityPolicy:
    """Four 96-feature observations, the ONNX velocity actor and its published torque-speed model."""

    def __init__(self, export, model):
        import onnxruntime as ort

        export = Path(export).expanduser()
        with export.with_suffix(".json").open() as stream:
            metadata = json.load(stream)
        expected = [name + "_joint" for name in robot.DOF]
        if (metadata["joint_names"] != expected or metadata["decimation"] != env.DECIMATION
                or not np.isclose(metadata["physics_options"]["timestep"], env.SIM_DT)
                or not metadata.get("normalization_in_onnx")):
            raise ValueError("Expected a normalized 29-joint G1 ONNX export running at 200/50 Hz")
        self.default = np.asarray(metadata["default_q"], dtype=float)
        self.scale = np.asarray(metadata["action_scale"], dtype=float)
        self.motors = {name: np.asarray([motor[name] for motor in metadata["motors"]], dtype=float)
                       for name in ("stiffness", "damping", "effort_limit", "X1", "X2", "Y1", "Y2", "Fs", "Fd", "Va")}
        vectors = [self.default, self.scale, *self.motors.values()]
        if (any(v.shape != (len(expected),) or not np.isfinite(v).all() for v in vectors)
                or np.any(self.motors["X2"] <= self.motors["X1"]) or np.any(self.motors["Va"] <= 0)):
            raise ValueError("Malformed G1 velocity actor motor calibration")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(export), sess_options=options, providers=["CPUExecutionProvider"])
        ip, = self.session.get_inputs()
        op, = self.session.get_outputs()
        if (ip.name, ip.shape, op.name, op.shape) != ("observation", [1, 384], "action", [1, 29]):
            raise ValueError("Expected a 384-input, 29-output G1 velocity ONNX graph")
        self.feet = [model.body("robot_" + name).id for name in robot.KEY_BODIES[:2]]
        self.floor = model.geom("floor").id
        self.model = model
        self.jog = None
        self.hips = np.asarray([robot.DOF.index(side + "_hip_pitch") for side in ("left", "right")])
        self.knees = np.asarray([robot.DOF.index(side + "_knee") for side in ("left", "right")])
        self.ankles = np.asarray([robot.DOF.index(side + "_ankle_pitch") for side in ("left", "right")])
        self.waist = robot.DOF.index("waist_pitch")
        self.reset()

    def reset(self):
        self.history = None
        self.previous = np.zeros(len(self.default))
        if self.jog is not None:
            self.jog.reset()

    def target(self, data, command, gains=None):
        if self.jog is not None and gains is not None:
            command = self.jog.command(self, self.model, data, command)
            gains = replace(gains, hip=gains.hip * self.jog.swing)
        frame = np.r_[data.qvel[3:6], -data.xmat[1].reshape(3, 3)[2], command,
                      data.qpos[7:] - self.default, data.qvel[6:], self.previous].astype(np.float32)
        self.history = np.repeat(frame[None], 4, axis=0) if self.history is None else np.vstack((self.history[1:], frame))
        self.previous = self.session.run(["action"], {"observation": self.history.reshape(1, 384)})[0][0]
        raw = self.default + self.scale * self.previous
        if gains is None or command[0] <= .65:
            return raw
        share = np.clip((command[0] - .65) / .69, 0., 1.)
        feet = (data.xpos[self.feet] - data.qpos[:3]) @ data.xmat[1].reshape(3, 3)
        trailing = np.clip((-feet[:, 0] - .01) / .13, 0., 1.)
        swing = np.clip((feet[:, 0] - .01) / .12, 0., 1.)
        adapted = raw.copy()
        adapted[self.hips] = self.default[self.hips] + (raw[self.hips] - self.default[self.hips]) * gains.hip + gains.hip_offset
        adapted[self.knees] = (self.default[self.knees] + (raw[self.knees] - self.default[self.knees]) * gains.knee
                               + gains.knee_offset - gains.push * .75 * trailing + gains.swing * swing)
        adapted[self.ankles] = (self.default[self.ankles] + (raw[self.ankles] - self.default[self.ankles]) * gains.ankle
                                - gains.push * trailing - gains.knee_offset * .5)
        adapted[self.waist] += gains.waist_pitch
        return raw + share * (adapted - raw)

    def advance(self, model, data, target):
        motors = self.motors
        flight_steps = 0
        for _ in range(env.DECIMATION):
            q, v = data.qpos[7:], data.qvel[6:]
            effort = motors["stiffness"] * (target - q) - motors["damping"] * v
            direction_limit = np.where(effort * v > 0., motors["Y1"], np.minimum(motors["Y2"], motors["effort_limit"]))
            limit = np.minimum(motors["effort_limit"], direction_limit)
            limit *= np.clip((motors["X2"] - abs(v)) / (motors["X2"] - motors["X1"]), 0., 1.)
            torque = np.clip(effort, -limit, limit) - motors["Fs"] * np.tanh(v / motors["Va"]) - motors["Fd"] * v
            torque = np.clip(torque, -robot.EFFORT, robot.EFFORT)
            torque[(abs(v) >= robot.SPEED) & (torque * v > 0)] = 0.
            data.ctrl[:] = q + (torque + robot.KD * v) / robot.KP
            mujoco.mj_step(model, data)
            mask, contact = ground_contacts(model, data, self.floor, self.feet)
            flight_steps += int(mask == 0 and not contact)
        mujoco.mj_kinematics(model, data)
        return flight_steps
