"""Single-robot Zenbot policy control and native MuJoCo dynamics at 50 Hz."""

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

from . import env, nets, zenbot
from .inference import load, policy_memory, policy_step
from .sim import Sim, unflatten
from .zenbot_rise import ZenbotRisePolicy


class ZenbotNative:
    def __init__(self, directory, checkpoint, seed=42, sensor_noise=0.):
        sim = Sim(1, robot="zenbot23_getup", walk_dir=str(directory), floor_dir=str(directory),
                  physics=False, obs_noise=False, assist_newtons=0., randomize_world=False)
        self.sim, self.model = sim, sim.m
        self.data = mj.MjData(self.model)
        self.actor, norm, ode = load(checkpoint)
        layers = self.actor.layers if isinstance(self.actor, ZenbotRisePolicy) else self.actor
        if (not isinstance(layers, list) or layers[0]["w"].shape[0] != sim.obs_dim
                or layers[-1]["b"].shape != (self.model.nu,)):
            raise ValueError("Expected a 480-input, 23-output Zenbot policy")
        self.native_tracking = None
        if checkpoint is not None:
            with np.load(checkpoint) as archive:
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
        bodies={self.model.body("robot_"+name).id:i for i,name in enumerate(names)}
        self.support_geoms={i:bodies[b] for i,b in enumerate(self.model.geom_bodyid) if b in bodies}
        self.support_load=np.zeros(len(names))

        @jax.jit
        def control(qpos, qvel, old, history, memory, delays, noise):
            state = unflatten(qpos[None], qvel[None])
            frame = sim.obs_frame(state, old[None], None, None)
            if sensor_noise:
                frame += noise
            history = sim.push_actor(history, frame)
            out, memory = policy_step(self.actor, norm, history, memory)
            action = nets.ode_step(out, old[None], *ode) if ode is not None else out
            context = jnp.r_[1., 0., delays, 0.][None]
            targets = sim.control_targets(action, jnp.zeros_like(action), history, context)
            return action[0], history, memory, targets.reshape(env.DECIMATION, self.model.nu)

        self.control = control

        @jax.jit
        def sense(qpos,qvel,old,history,noise):
            state=unflatten(qpos[None],qvel[None])
            frame=sim.obs_frame(state,old[None],None,None)+noise
            return sim.push_actor(history,frame)
        self.sense=sense

    def sample_noise(self):
        if self.sensor_noise == 0.:
            return self.zeros
        noise = self.sensor_rng.uniform(-1., 1., self.zeros.shape) * self.sim.obs_noise * self.sensor_noise
        return jnp.asarray(noise, dtype=jnp.float32)

    def reset(self, pose, velocity, delays):
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
        first = self.sim.obs_frame(initial, self.previous[None], None, None)
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
            self.previous, self.history, self.memory, targets = self.control(
                jnp.asarray(d.qpos), jnp.asarray(d.qvel), self.previous, self.history,
                self.memory, self.delays, self.sample_noise())
        load_n = 0.
        self.support_load.fill(0.)
        for target in np.asarray(targets):
            q, v = d.qpos[7:], d.qvel[6:]
            torque = np.clip(zenbot.KP * (target - q) - zenbot.KD * v, -zenbot.EFFORT, zenbot.EFFORT)
            d.ctrl[:] = np.where((abs(v) >= zenbot.SPEED) & (torque * v > 0.), 0., torque)
            mj.mj_step(self.model, d)
            for k, contact in enumerate(d.contact):
                other = contact.geom2 if contact.geom1==self.floor else contact.geom1 if contact.geom2==self.floor else -1
                body = self.support_geoms.get(other)
                if body is not None:
                    mj.mj_contactForce(self.model, d, k, self.force)
                    load=self.force[0]/env.DECIMATION
                    self.support_load[body]+=load/self.weight
                    if body<2:
                        load_n+=load
        return load_n / self.weight
