from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from instantamp import env, zenbot
from instantamp.sim import Sim, unflatten

pytestmark = pytest.mark.usefixtures("zenbot23_motion_banks")


class CpuState(NamedTuple):
    qpos: object
    qvel: object
    qacc_warmstart: object

    def replace(self, **updates):
        return self._replace(**updates)

    @property
    def ctrl(self):
        return jnp.zeros_like(self.qpos[..., 7:])

    @property
    def _impl(self):
        return SimpleNamespace(nacon=jnp.zeros(1), contact__dist=jnp.zeros(1))


def simulator(wm=True, n=16, **options):
    data=Path('data/zenbot23_walk_run_recovery')
    sim=Sim(n,physics=False,robot=zenbot.MIXED_TASK,walk_dir=str(data/'walk_run'),floor_dir=str(data/'recovery'),
            zenbot_motion_train_wm=wm,reset_floor=.5,obs_noise=False,push=False,randomize_world=False,
            random_start=False,**options)
    sim.dx=CpuState(jnp.zeros(30),jnp.zeros(29),jnp.zeros(29))
    return sim


def native_substeps(sim, world, state, targets, assist, fresh):
    # Only the Warp transport is replaced. Integrate the real scene for all four motor requests.
    pos,vel=[],[]
    for i in range(sim.N):
        d=mujoco.MjData(sim.m)
        d.qpos[:],d.qvel[:]=state.qpos[i],state.qvel[i]
        d.qacc_warmstart[:]=state.qacc_warmstart[i]
        mujoco.mj_forward(sim.m,d)
        for target in np.asarray(targets[i]).reshape(4,23):
            d.ctrl[:]=zenbot.actuator_ctrl(d.qpos[7:],d.qvel[6:],target)
            mujoco.mj_step(sim.m,d)
        pos.append(d.qpos.copy())
        vel.append(d.qvel.copy())
    # Contacts do not enter these six terms; termination and timeout use the real post-step pelvis state.
    zeros=jnp.zeros(sim.N)
    return state.replace(qpos=jnp.asarray(pos),qvel=jnp.asarray(vel)),jnp.zeros((sim.N,2)),zeros,zeros.astype(bool),None,None


def test_fixed_commands_match_real_initial_and_resampled_sensor_inputs():
    sim=simulator()
    s=sim.init(jax.random.key(352))
    assert set(sim.weights)=={'track_lin','track_ang','rise','action_rate','termination','recovery_timeout_cost'}
    assert sim.weights['action_rate']==-.015 and sim.recovery_amp_steps==200
    assert s.obs.shape==(sim.N,390) and s.motion_train_sensors.observation().shape==(sim.N,234)
    assert np.any(s.down) and np.any(~s.down)
    assert set(np.unique(s.delay_steps))=={0,3,4,5,6,7}
    assert np.all(np.all(s.delay_steps==0,axis=-1)|np.all(s.delay_steps>=3,axis=-1))
    np.testing.assert_array_equal(s.cmd.time_left,10.)
    np.testing.assert_array_equal(s.c_obs,jnp.where(s.was_up[:,None],s.cmd.vel,0.))
    np.testing.assert_array_equal(s.motion_train_sensors.frames[:,-1,6:9],s.c_obs)
    draw=jax.jit(lambda k:sim.sample_wm_command(k,12000))(jax.random.key(24))
    counts=np.all(np.asarray(draw.vel)[:,None]==zenbot.DIRECTIONAL_COMMANDS,axis=-1).sum(axis=0)
    np.testing.assert_allclose(counts/12000,1/8,atol=.015)
    assert not draw.is_heading.any()
    np.testing.assert_array_equal(draw.is_standing,np.all(draw.vel==0.,axis=1))
    np.testing.assert_array_equal(draw.time_left,10.)

    # Resample the first half, release walking when the timer goes from one to zero,
    # and keep the remaining floor/standing command clocks identical to their real predecessors.
    countdown=jnp.where(s.was_up,jnp.where(jnp.arange(sim.N)%2==0,1,2),s.recovery_left)
    s=s._replace(recovery_left=countdown,cmd=s.cmd._replace(time_left=jnp.where(jnp.arange(sim.N)<sim.N//2,.01,3.)))
    s=s._replace(c_obs=sim.wm_velocity_command(s.cmd,unflatten(s.data.qpos,s.data.qvel)[1],s.was_up,countdown))
    sim.physics=lambda *args:native_substeps(sim,*args)
    act=jnp.ones((sim.N,23))*.15
    nxt,tr=sim.step(s,act,auto_reset=False)
    key=jax.random.split(s.key,6)[2]
    expected=env.tick(s.cmd,key,sample=lambda k:sim.sample_wm_command(k,sim.N))
    for actual,required in zip(nxt.cmd,expected,strict=True):
        np.testing.assert_array_equal(actual,required)
    np.testing.assert_array_equal(nxt.c_obs,jnp.where((nxt.was_up&(nxt.recovery_left==0))[:,None],expected.vel,0.))
    np.testing.assert_array_equal(nxt.motion_train_sensors.frames[:,-1,6:9],nxt.c_obs)
    np.testing.assert_array_equal(nxt.motion_train_sensors.frames[:,-1,55:],act)
    np.testing.assert_array_equal(nxt.obs.reshape(sim.N,5,78)[:,-1,55:],act)
    assert np.any(nxt.recovery_left==0) and np.any(nxt.recovery_left==1)
    assert np.any(np.asarray(nxt.data.qpos)!=np.asarray(s.data.qpos))
    np.testing.assert_allclose(tr.parts[:,sim.terms.index('action_rate')],env.DT*(-.015)*23*.15**2,rtol=2e-6)
    np.testing.assert_array_equal(tr.reward,tr.parts.sum(-1))
    expired=jnp.arange(sim.N)%2==0
    nxt=nxt._replace(t=jnp.where(expired,sim.episode_steps-1,nxt.t))
    final,timed=sim.step(nxt,act)
    np.testing.assert_array_equal(timed.truncated,expired)
    np.testing.assert_array_equal(final.cmd.time_left[expired],10.)
    np.testing.assert_array_equal(final.cmd.time_left[~expired],nxt.cmd.time_left[~expired]-.02)
    np.testing.assert_array_equal(final.motion_train_sensors.frames[expired,0],final.motion_train_sensors.frames[expired,-1])
    np.testing.assert_array_equal(final.motion_train_sensors.frames[expired,-1,55:],0.)


def test_world_model_and_real_rewards_use_body_heading_and_recovery_clock():
    sim=simulator(n=4)
    q=jnp.tile(jnp.r_[jnp.array([0,0,.69,np.sqrt(.5),0,0,np.sqrt(.5)]),jnp.asarray(zenbot.DEFAULT_Q)],(4,1))
    q=q.at[3,2].set(.25)
    v=jnp.zeros((4,29)).at[:,1].set(1.).at[:,5].set(.6)
    state = unflatten(q, v)
    view = sim.view_state(state)
    cmd=jnp.tile(jnp.array([1.,0.,.6]),(4,1)).at[1,:].set(jnp.array([.5,0.,.1]))
    action=jnp.ones((4,23))*.5
    down=jnp.array([False,False,False,True])
    terms=sim.reward_terms(state,state,view.kin,action,0.*action,cmd,down,down.astype(float),jnp.zeros((4,2)),
                           jnp.zeros(4),sim.q_soft,recovery_active=jnp.array([False,False,True,True]),
                           recovery_deadline=jnp.array([False,False,False,True]))
    np.testing.assert_allclose(terms['track_lin'],[1.,np.exp(-1),0,0],rtol=1e-5)
    np.testing.assert_allclose(terms['track_ang'],[1.,np.exp(-1),0,0],rtol=1e-5)
    assert terms['rise'][3]>0 and terms['recovery_timeout_cost'][3]>0
    assert not np.any(terms['rise'][:3])
    np.testing.assert_array_equal(terms['termination'],[0,0,0,1])
    np.testing.assert_array_equal(terms['action_rate'],np.full(4,23*.5**2))


def test_wm_mixes_native_cold_openings_with_moving_recovery_frames():
    from scipy.spatial.transform import Rotation

    from instantamp.zenbot_trials import fallen_starts

    sim = simulator(n=12288, prob_shove=.2)
    sim.reset_floor = .35
    world = SimpleNamespace(walk=sim.walk, floor=sim.floor)
    key = jax.random.key(7149)
    sample = jax.jit(lambda k: sim._fresh(k, world, return_source=True))(key)
    (pos, vel, up), (clip, frame, yaw) = jax.tree.map(np.asarray, sample)
    floor = clip >= 0
    cold, phase = floor & (frame == 0), floor & (frame > 0)
    assert abs(cold.mean() - .175) < .012 and abs(phase.mean() - .175) < .012
    assert abs(cold.sum() / floor.sum() - .5) < .035
    assert sim.assist_newtons == 0. and not up[cold].any()
    names, starts, _ = fallen_starts("data/zenbot23_walk_run_recovery/recovery")
    native = np.asarray(starts)[[i for i, (_, t) in enumerate(names) if t == 0.]]
    assert len(native) == 11
    np.testing.assert_allclose(np.bincount(clip[cold], minlength=11) / cold.sum(), 1 / 11, atol=.025)
    base = native[clip[cold]]
    np.testing.assert_array_equal(pos[cold, :3], base[:, :3])
    assert np.abs(pos[cold, 7:] - base[:, 7:]).max() <= np.deg2rad(2.) + 1e-6
    perturbation = Rotation.from_quat(base[:, 3:7], scalar_first=True).inv() * Rotation.from_quat(pos[cold, 3:7], scalar_first=True)
    euler = perturbation.as_euler("xyz", degrees=True)
    assert np.abs(euler[:, :2]).max() < 2.0001 and np.abs(euler[:, 2]).max() < .0001
    assert .06 < np.abs(vel[cold, :2]).max() <= .07
    np.testing.assert_array_equal(vel[cold, 2:], 0.)
    np.testing.assert_array_equal(yaw[cold], 0.)
    assert np.abs(vel[phase, 6:]).max() > 2. and np.max(frame[phase]) > 100
    assert np.max(np.abs(yaw[phase])) > 3.

    # Gait frames and any upright shove are identical to the ordinary four-second mixed task.
    old = simulator(wm=False, n=sim.N, prob_shove=.2, recovery_amp_seconds=4.)
    old.reset_floor = sim.reset_floor
    old_pos, old_vel, old_up = old._fresh(key, SimpleNamespace(walk=old.walk, floor=old.floor))
    for actual, expected in zip((pos, vel, up), (old_pos, old_vel, old_up)):
        np.testing.assert_allclose(actual[~floor], np.asarray(expected)[~floor], rtol=1e-6, atol=2e-6)


def test_default_and_new_motor_writes_preserve_real_unsaturated_angle_and_old_delay_history():
    a=simulator(wm=False,n=2)
    b=simulator(n=2)
    assert a.zero_motor_delay_probability==0. and a.recovery_amp_steps==0
    assert a.weights==env.WEIGHTS and a.obs_dim==b.obs_dim==390
    old=a.init(jax.random.key(21))
    assert old.motion_train_sensors is None and np.all(np.asarray(old.delay_steps)>=3)
    with pytest.raises(ValueError, match='mixed Zenbot'):
        Sim(1,physics=False,robot='g1',zenbot_motion_train_wm=True)
    history=jnp.zeros((2,5,78)).at[:,-1,55:].set(-20.).at[:,-2,55:].set(30.)
    delays=np.array([[0]*5,[3,4,5,6,7]],dtype=np.int32)
    ctx=jnp.array(np.column_stack((np.ones(2),np.zeros(2),delays,np.zeros(2))))
    action=jnp.full((2,23),50.)
    expected=np.where(np.arange(4)[:,None]>=delays[:,None,zenbot.GROUP],50.,
                      np.where(np.arange(4)[:,None]+4>=delays[:,None,zenbot.GROUP],-20.,30.))
    for sim in (a,b):
        raw=sim.control_targets(action,jnp.zeros((2,23)),history.reshape(2,390),ctx)
        np.testing.assert_array_equal(raw.reshape(2,4,23),zenbot.DEFAULT_Q+.25*expected.astype(np.float32))
        assert np.any(np.asarray(raw).reshape(2,4,23)>sim.joint_range[:,1])
