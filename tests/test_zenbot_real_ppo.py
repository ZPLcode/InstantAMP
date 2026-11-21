"""Physical PPO minibatch and bootstrap regressions on CPU."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, critic, nets, optim, support_discriminator, zenbot
from instantamp.sim import Sim
from instantamp.train import InstantAMP, parse
from tools.prepare_zenbot import convert


def test_real_ppo_uses_newest_physical_steps_and_updates_actor_and_value(tmp_path):
    base = ["--robot", zenbot.MIXED_TASK, "--recovery_amp_loss", "legged_support", "--legged_real_ppo", "1"]
    with pytest.raises(ValueError, match="env_steps"):
        parse(base)
    cfg = parse([*base, "--env_steps", "4", "--ppo_len", "4", "--pi_updates", "2", "--det_share", "0",
                 "--ppo_frac", "1", "--recovery_amp_stride", "1", "--amp_frames", "3"])
    q = np.tile(zenbot.DEFAULT_Q, (12, 1))
    convert(tmp_path / "floor.npz", np.tile([0., 0., .3], (12, 1)), np.tile([1., 0., 0., 0.], (12, 1)),
            q, np.arange(23), 50)
    n, t = 4, 8
    xs = InstantAMP.__new__(InstantAMP)
    xs.cfg = cfg
    xs.sim = Sim(n, robot=zenbot.MIXED_TASK, physics=False, walk_dir=tmp_path, floor_dir=tmp_path,
                 recovery_amp_seconds=4., legged_support=True)
    xs.amp = xs.getup = xs.ppo = True
    xs.recovery_clock = xs.ode = False
    xs.ground_base = None
    xs.ppo_envs = np.arange(n)
    xs.sim.context = lambda current: jnp.zeros((n, 8))
    key = jax.random.PRNGKey(42)
    pose = jnp.tile(jnp.r_[0., 0., .3, 1., 0., 0., 0., jnp.asarray(q[0])], (t, n, 1))
    vel = jnp.zeros((t, n, 29))
    obs = jax.random.normal(key, (t, n, 8))
    pi = nets.mlp_init(key, [8, 16, 23])
    mean = nets.pi_apply(pi, obs)
    xs.pi_mask = jax.tree.map(lambda _: False, pi)
    zeros = jnp.zeros((t, n))
    current = namedtuple("Current", "data a_last c_obs was_up t recovery_left")
    data = namedtuple("Data", "qpos qvel")
    r = jnp.broadcast_to(jnp.arange(t)[:, None], (t, n))
    buf = dict(s_qpos=pose, s_qvel=vel, s1_qpos=pose, s1_qvel=vel, was_up=zeros, t=r.astype(jnp.int32),
               c_obs=jnp.zeros((t, n, 3)), ok=jnp.ones((t, n)), done=zeros.at[-2, 0].set(1), term=zeros,
               a_last=mean, a_ode=mean*0., mu=mean, a=mean + cfg.action_noise * jax.random.normal(key, mean.shape),
               obs=obs, xi=jnp.zeros((t, n, 8)), r=r, rlin=zeros, recovery_left=200-r.astype(jnp.int32),
               ptr=jnp.array(0), size=jnp.array(t))
    env = current(data(pose[-1], vel[-1]), mean[-1], buf["c_obs"][-1], zeros[-1], r[-1], buf["recovery_left"][-1])
    heads = {"native": amp.init(key, 3, hidden=(8,), frame_dim=73),
             "support": support_discriminator.init(key, 3, amp.ORIENTED_SUPPORT_DIM)}
    carry = dict(env=env, buf=buf, key=key, it=jnp.array(100), nonfinite=jnp.array(0), pi=pi, avg=pi,
                 opt_pi=optim.init(pi, xs.pi_mask), obs_norm=nets.rn_init(8), critic=critic.init(key, 95, (8,)),
                 amp=amp.init(key, 3, hidden=(8,), frame_dim=74), amp_up=heads)
    batch, _ = xs.ppo_batch(carry)
    np.testing.assert_array_equal(batch["obs"], obs[-4:].reshape(-1, 8))
    assert jnp.isfinite(batch["value_target"]).all()
    result, stats = jax.jit(xs.real_ppo_iter)(carry)
    assert result["it"] == 101 and result["opt_pi"]["t"] == 8 and result["critic"]["opt"]["t"] == 8
    assert result["nonfinite"] == 0 and all(np.isfinite(value).all() for value in stats.values())
    assert not np.array_equal(result["pi"][-1]["w"], pi[-1]["w"])
    assert not np.array_equal(result["critic"]["net"][-1]["w"], carry["critic"]["net"][-1]["w"])
