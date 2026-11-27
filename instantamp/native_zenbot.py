"""Run a combined Zenbot gait and recovery policy in native MuJoCo at 50 Hz."""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from . import env, nets, zenbot_models
from .inference import load
from .rot import quat_to_mat
from .sim import build_scene, robot_profile
from .zenbot_motion_train_obs import joint_layout, motion_train_actor_mean
from .zenbot_wm_sensors import pd_targets


class ZenbotNative:
    """One joint policy with measured IMU, encoders and delayed torque motors."""

    def __init__(self, directory, checkpoint, seed=42, sensor_noise=0.):
        actor, norm, ode = load(checkpoint)
        if not isinstance(actor, list) or not actor:
            raise ValueError("Expected a dense Zenbot gait and recovery policy")
        joints = actor[-1]["b"].shape[0]
        model = zenbot_models.for_joint_count(joints)
        layout = joint_layout(joints)
        width = len(layout.history_index)
        if actor[0]["w"].shape[0] != width:
            raise ValueError(f"Expected {width} inputs for {joints} Zenbot motors")
        self.profile = robot_profile(model.mixed_task)
        self.model = build_scene(robot_name=model.mixed_task)
        self.data = mujoco.MjData(self.model)
        self.motor_delay_groups = int(model.group.max()) + 1
        self.directory = (Path(directory) if directory is not None else
                          Path(__file__).resolve().parents[1] / model.mixed_recovery_dir)
        with np.load(checkpoint, allow_pickle=False) as archive:
            self._first_command = bool(
                archive.get("zenbot/delay_uses_first_command", False)
            )

        if not np.isfinite(sensor_noise) or sensor_noise < 0:
            raise ValueError("Sensor noise must be finite and nonnegative")
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 713]))
        self._noise_scale = sensor_noise * layout.noise
        self._zero_noise = jnp.zeros_like(layout.noise)
        self._actuator_ctrl = jax.jit(self.profile.actuator_ctrl)

        names = (*self.profile.FEET, *self.profile.HANDS, "left_elbow_Link",
                 "right_elbow_Link", *self.profile.KNEES)
        bodies = {
            self.model.body("robot_" + name).id: i for i, name in enumerate(names)
        }
        self._support_geoms = {
            i: bodies[body] for i, body in enumerate(self.model.geom_bodyid)
            if body in bodies
        }
        self._floor = self.model.geom("floor").id
        self._weight = self.model.body_mass.sum() * abs(self.model.opt.gravity[2])
        self._force = np.empty(6)
        self.support_load = np.zeros(len(names))
        self.flight = np.zeros(env.DECIMATION, dtype=bool)
        self.motor_torque = np.zeros((env.DECIMATION, joints))
        self.motor_velocity = np.zeros_like(self.motor_torque)

        @jax.jit
        def control(qpos, qvel, previous, history, command, delays, noise, tick):
            frame = self._frame(qpos, qvel, previous, command) + noise
            history = jnp.concatenate((history[1:], frame[None]))
            observation = history.ravel()[layout.history_index][None]
            action = motion_train_actor_mean(actor, norm, observation)
            if ode is not None:
                action = nets.ode_step(action, previous[None], *ode)
            context = jnp.r_[1., 0., delays, 0.]
            targets = pd_targets(
                action[0], history[-1, -joints:], history[-2, -joints:],
                jnp.zeros(joints), context, profile=self.profile,
                elapsed=tick if self._first_command else None,
            )
            return action[0], history, targets.reshape(env.DECIMATION, joints)

        self._control = control

    def _noise(self):
        if not self._noise_scale.any():
            return self._zero_noise
        sample = self._rng.uniform(-1., 1., size=self._noise_scale.shape)
        return jnp.asarray(sample * self._noise_scale, dtype=jnp.float32)

    def _frame(self, qpos, qvel, previous, command):
        gravity = -quat_to_mat(qpos[3:7])[2]
        return jnp.concatenate((
            qvel[3:6], gravity, command.reshape(3),
            qpos[7:] - self.profile.DEFAULT_Q, qvel[6:], previous,
        ))

    def reset(self, pose, velocity, delays, command=(0., 0., 0.)):
        """Reset the physical state and fill the policy history from its first frame."""
        if np.shape(delays) != (self.motor_delay_groups,):
            raise ValueError(f"Expected {self.motor_delay_groups} motor-group delays")
        self.delays = jnp.asarray(delays)
        self.command = jnp.asarray(command, dtype=jnp.float32).reshape(1, 3)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = pose
        self.data.qvel[:] = velocity
        mujoco.mj_forward(self.model, self.data)
        self.previous = jnp.zeros(self.model.nu)
        frame = self._frame(
            self.data.qpos, self.data.qvel, self.previous, self.command
        )
        self.history = jnp.repeat((frame + self._noise())[None], 3, axis=0)
        self.support_load.fill(0.)
        self.flight.fill(False)
        self.motor_torque.fill(0.)
        self.motor_velocity.fill(0.)

    def step(self):
        """Apply one policy action and return average foot load in body weights."""
        data = self.data
        self.previous, self.history, targets = self._control(
            data.qpos, data.qvel, self.previous, self.history, self.command,
            self.delays, self._noise(), round(data.time / env.DT),
        )
        self.support_load.fill(0.)
        self.flight.fill(True)
        for substep, target in enumerate(np.asarray(targets)):
            data.ctrl[:] = self._actuator_ctrl(data.qpos[7:], data.qvel[6:], target)
            self.motor_torque[substep] = data.ctrl
            self.motor_velocity[substep] = data.qvel[6:]
            mujoco.mj_step(self.model, data)
            for k, contact in enumerate(data.contact):
                if self._floor not in contact.geom:
                    continue
                geom = contact.geom2 if contact.geom1 == self._floor else contact.geom1
                mujoco.mj_contactForce(self.model, data, k, self._force)
                if self._force[0] > 1.:
                    self.flight[substep] = False
                body = self._support_geoms.get(geom)
                if body is not None:
                    self.support_load[body] += self._force[0]
        self.support_load /= self._weight * env.DECIMATION
        return self.support_load[:2].sum()
