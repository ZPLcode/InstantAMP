"""Single-robot Zenbot policy control and native MuJoCo dynamics at 50 Hz."""

from functools import partial
from pathlib import Path
from zipfile import BadZipFile
from zlib import error as DecompressionError

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

from . import env, nets, zenbot
from .inference import ZenbotWalkRecovery, load, policy_memory, policy_step
from .sim import Sim, unflatten
from .zenbot_command_actor import COMMAND_HISTORY_WIDTH
from .zenbot_motion_train_obs import MOTION_TRAIN_WIDTH, motion_train_actor_mean, motion_train_observation
from .zenbot_ground_policy import ZenbotGroundPolicy
from .zenbot_rise import ZenbotRisePolicy


class ZenbotNative:
    def __init__(self, directory, checkpoint, seed=42, sensor_noise=0., *, walk_dir=None):
        self.actor, norm, ode = load(checkpoint)
        layers = self.actor.layers if isinstance(self.actor, ZenbotRisePolicy) else self.actor
        if isinstance(self.actor, ZenbotWalkRecovery):
            layers = self.actor.rise
        if isinstance(layers, ZenbotGroundPolicy):
            layers = layers.rise
        self.actor_width = layers[0]["w"].shape[0]
        self.command_gait = isinstance(self.actor, list) and self.actor_width in (COMMAND_HISTORY_WIDTH, MOTION_TRAIN_WIDTH)
        profile = zenbot.MIXED_TASK if walk_dir is not None and not self.command_gait else "zenbot23_getup"
        if self.command_gait and walk_dir is None:
            # Sim constructs an upright reset bank. Native episodes reset directly from the supplied pose.
            walk_dir = Path(__file__).resolve().parents[1] / "data/zenbot23_walk_run_recovery/standing"
        sim = Sim(1, robot=profile, walk_dir=str(walk_dir or directory), floor_dir=str(directory),
                  physics=False, obs_noise=False, assist_newtons=0., randomize_world=False,
                  zenbot_getup_stand=self.command_gait, zenbot_standing_amp=self.command_gait,
                  zenbot_support_weight=8. if self.command_gait else 0., zenbot_command_gait=self.command_gait)
        self.sim, self.model = sim, sim.m
        self.data = mj.MjData(self.model)
        if (not isinstance(layers, list) or self.actor_width not in (sim.obs_dim, MOTION_TRAIN_WIDTH)
                or layers[-1]["b"].shape != (self.model.nu,)):
            raise ValueError(f"Expected a {sim.obs_dim}-input, 23-output {profile} policy")
        self.native_tracking = None
        self.delay_uses_first_command = False
        if checkpoint is not None:
            with np.load(checkpoint) as archive:
                self.delay_uses_first_command = bool(archive.get("zenbot/delay_uses_first_command", False))
                if "zenbot_tracking/offset" in archive:
                    from .zenbot_native_tracking import NativeTracking
                    fields={k:archive[k] for k in archive.files if k.startswith("zenbot_tracking/")}
                    self.native_tracking=NativeTracking(self.model,fields)
        self.sensor_noise = sensor_noise
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, 713]))
        self.zeros = jnp.zeros((1, sim.frame_dim), dtype=jnp.float32)
        self.floor, self.feet = self.model.geom("floor").id, set(sim.foot_geoms)
        self.weight = self.model.body_mass.sum() * 9.81
        self.force = np.zeros(6)
        names=(*zenbot.FEET,*zenbot.HANDS,"left_elbow_Link","right_elbow_Link")
        if sim.mixed or self.command_gait:
            names = (*names, *zenbot.KNEES)
        bodies={self.model.body("robot_"+name).id:i for i,name in enumerate(names)}
        self.support_geoms={i:bodies[b] for i,b in enumerate(self.model.geom_bodyid) if b in bodies}
        self.support_load=np.zeros(len(names))
        self.flight = np.zeros(env.DECIMATION, dtype=bool)
        self.foot_pair_force = np.zeros(env.DECIMATION)

        def control(actor, norm, ode, qpos, qvel, old, history, memory, delays, noise, command, elapsed=None):
            state = unflatten(qpos[None], qvel[None])
            frame = sim.obs_frame(state, old[None], command, None)
            if sensor_noise:
                frame += noise
            history = sim.push_actor(history, frame)
            if self.command_gait and actor[0]["w"].shape[0] == MOTION_TRAIN_WIDTH:
                actor_obs = motion_train_observation(history.reshape(1, 5, 99))
                out = motion_train_actor_mean(actor, norm, actor_obs)
            else:
                out, memory = policy_step(actor, norm, history, memory)
            action = nets.ode_step(out, old[None], *ode) if ode is not None else out
            context = jnp.r_[1., 0., delays, 0.][None]
            targets = sim.control_targets(action, jnp.zeros_like(action), history, context, elapsed=elapsed)
            return action[0], history, memory, targets.reshape(env.DECIMATION, self.model.nu)

        # Pass dense weights as JIT arguments so replacing a checkpoint with the
        # same architecture reuses the executable and applies the new parameters.
        self._dense_control = jax.jit(control)
        # Fixed-checkpoint evaluators use the original constant parameter compilation.
        self.control = jax.jit(partial(control, self.actor, norm, ode))

        @jax.jit
        def sense(qpos,qvel,old,history,noise):
            state=unflatten(qpos[None],qvel[None])
            frame=sim.obs_frame(state,old[None],None,None)+noise
            return sim.push_actor(history,frame)
        self.sense=sense

    def replace_policy(self, checkpoint):
        """Load another dense get-up actor in place, preserving the live motor and simulation state."""
        if self.sim.mixed or not isinstance(self.actor, list) or self.native_tracking is not None:
            raise ValueError("Live replacement requires a standalone dense Zenbot get-up actor")
        try:
            with open(checkpoint, "rb") as stream:
                with np.load(stream, allow_pickle=False) as archive:
                    if "zenbot_tracking/offset" in archive:
                        raise ValueError("Live replacement requires a dense policy checkpoint")
                    first_command = bool(archive.get("zenbot/delay_uses_first_command", False))
                stream.seek(0)
                actor, norm, ode = load(stream)
            if not isinstance(actor, list):
                raise ValueError("Live replacement requires a dense policy checkpoint")
            width = actor[0]["w"].shape[0]
            allowed_widths = (COMMAND_HISTORY_WIDTH, MOTION_TRAIN_WIDTH) if self.command_gait else (self.sim.obs_dim,)
            if width not in allowed_widths:
                raise ValueError("Incompatible Zenbot observation layout")
            actor_width = width
            for layer in actor:
                w, b = layer["w"], layer["b"]
                if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],) or not w.shape[1]:
                    raise ValueError("Incompatible Zenbot policy layer")
                if "ln_g" in layer or "ln_b" in layer:
                    if layer["ln_g"].shape != b.shape or layer["ln_b"].shape != b.shape:
                        raise ValueError("Incompatible Zenbot layer normalization")
                width = w.shape[1]
            if (width != self.model.nu or any(norm[k].shape != (actor_width,) for k in ("mean", "var"))
                    or np.any(np.asarray(norm["var"]) < 0) or (ode and any(v is not None and v <= 0 for v in ode))
                    or not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves((actor, norm, ode)))):
                raise ValueError("Invalid Zenbot policy outputs, parameters or observation normalization")
        except (KeyError, IndexError, TypeError, EOFError, BadZipFile, DecompressionError) as error:
            raise ValueError(f"Incomplete policy checkpoint {checkpoint}") from error
        candidate = partial(self._dense_control, actor, norm, ode)
        if hasattr(self, "history"):
            # Validate the compiled control path on the current observation before
            # publishing. The trial call consumes neither a noise draw nor a step.
            action, _, _, _ = candidate(jnp.asarray(self.data.qpos), jnp.asarray(self.data.qvel), self.previous,
                                        self.history, self.memory, self.delays, self.zeros, self.command)
            if not np.isfinite(np.asarray(action)).all():
                raise ValueError("Replacement policy produced a non-finite motor command")
        self.actor, self.control, self.actor_width = actor, candidate, actor_width
        self.delay_uses_first_command = first_command

    def sample_noise(self):
        if self.sensor_noise == 0.:
            return self.zeros
        noise = self.sensor_rng.uniform(-1., 1., self.zeros.shape) * self.sim.obs_noise * self.sensor_noise
        return jnp.asarray(noise, dtype=jnp.float32)

    def reset(self, pose, velocity, delays, command=(0., 0., 0.)):
        d = self.data
        mj.mj_resetData(self.model, d)
        d.qpos[:] = pose
        d.qvel[:] = velocity
        mj.mj_forward(self.model, d)
        self.delays = delays
        if self.native_tracking is not None:
            self.native_tracking.reset(d,delays)
        self.previous = jnp.zeros(self.model.nu)
        initial = unflatten(jnp.asarray(d.qpos)[None], jnp.asarray(d.qvel)[None])
        self.command = jnp.asarray(command, dtype=jnp.float32).reshape(1, 3)
        first = self.sim.obs_frame(initial, self.previous[None], self.command, None)
        if self.sensor_noise:
            first += self.sample_noise()
        self.history = self.sim.push_actor(None, first)
        self.memory = policy_memory(self.actor, self.history)

    def step(self):
        d = self.data
        tracker=self.native_tracking
        if tracker is not None and tracker.tick==0 and tracker.mode==0:
            if not tracker.prepare(d):
                tracker.mode=-1
        if tracker is not None and tracker.mode in (1,2):
            self.history=self.sense(jnp.asarray(d.qpos),jnp.asarray(d.qvel),self.previous,self.history,self.sample_noise())
            previous,older=tracker.previous,tracker.older
            action=tracker.action(d,np.asarray(self.previous))
            self.previous=jnp.asarray(action)
            age=np.arange(env.DECIMATION)[:,None]-tracker.lags
            targets=np.where(age>=0,tracker.previous,np.where(age>=-env.DECIMATION,previous,older))
        else:
            elapsed = round(d.time / env.DT) if self.delay_uses_first_command else None
            self.previous, self.history, self.memory, targets = self.control(
                jnp.asarray(d.qpos), jnp.asarray(d.qvel), self.previous, self.history,
                self.memory, self.delays, self.sample_noise(), self.command, elapsed)
        load_n = 0.
        self.support_load.fill(0.)
        self.foot_pair_force.fill(0.)
        for substep, target in enumerate(np.asarray(targets)):
            q, v = d.qpos[7:], d.qvel[6:]
            torque = np.clip(zenbot.KP * (target - q) - zenbot.KD * v, -zenbot.EFFORT, zenbot.EFFORT)
            d.ctrl[:] = np.where((abs(v) >= zenbot.SPEED) & (torque * v > 0.), 0., torque)
            mj.mj_step(self.model, d)
            self.flight[substep] = True
            for k, contact in enumerate(d.contact):
                if contact.geom1 in self.feet and contact.geom2 in self.feet:
                    mj.mj_contactForce(self.model, d, k, self.force)
                    self.foot_pair_force[substep] = max(self.foot_pair_force[substep], self.force[0])
                    continue
                other = contact.geom2 if contact.geom1==self.floor else contact.geom1 if contact.geom2==self.floor else -1
                body = self.support_geoms.get(other)
                if other >= 0 and (self.sim.mixed or self.command_gait):
                    mj.mj_contactForce(self.model, d, k, self.force)
                    if self.force[0] > 1.:
                        self.flight[substep] = False
                if body is not None:
                    if not (self.sim.mixed or self.command_gait):
                        mj.mj_contactForce(self.model, d, k, self.force)
                    load=self.force[0]/env.DECIMATION
                    self.support_load[body]+=load/self.weight
                    if body<2:
                        load_n+=load
        return load_n / self.weight
