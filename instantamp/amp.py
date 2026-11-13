"""Adversarial motion prior (AMP, Peng et al. 2021) for the G1, in JAX and on the world model's imagination.

A discriminator tells windows of `frames` consecutive 50 Hz frames of the robot's motion from windows of the human walking and running
clips (data/amp/walk_run, tools/download_data.sh). Its score gives a style reward for the imagined motion, which the policy follows
through the world model like every other term of the reward: the style reward is differentiable in the imagined states.

A frame holds the joint positions and velocities, the pelvis's angular velocity in the pelvis frame, the direction of gravity in the
pelvis frame, and the positions of the feet and hands relative to the pelvis in its heading frame (the ankle-roll and wrist-yaw links):
29 + 29 + 3 + 3 + 12 = 76 values (64 with use_wrists(False)).

The critic is a WGAN-GP critic (Gulrajani et al. 2017): the loss mean D(robot) - mean D(demo) + 1e-3 mean D(demo)^2 plus 10 times the squared
deviation of the gradient norm from 1 at random points between a demo window and a robot window. The style reward is
(1 + tanh((d - mu_robot) / scale)) / 2, centered at the robot's running mean score with a fixed scale (see `style`). Why WGAN-GP and not
LSGAN: the policy follows the reward's gradient through the world model, so the critic's slope matters.
"""

from __future__ import annotations

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import env, nets, optim
from .rot import mv3, quat_to_mat, tr3, yaw_mat

WRIST = (19, 20, 21, 26, 27, 28)  # the G1's wrist roll, pitch and yaw in action order
KEPT = np.arange(29)
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
FRAME_DIM = 2 * len(KEPT) + 3 + 3 + 3 * len(KEY_BODIES)


def use_wrists(on: bool):
    """Keep the wrists in the frame features (on, the default) or leave them out (off). Left out, nothing in the reward held the
    wrists, and a policy turned them to about 1.5 rad, each side another way. Call before Demos and init: the critic's input width
    follows."""
    global KEPT, FRAME_DIM
    KEPT = np.arange(29) if on else np.asarray([j for j in range(29) if j not in WRIST])
    FRAME_DIM = 2 * len(KEPT) + 3 + 3 + 3 * len(KEY_BODIES)


EMA = 0.99  # running mean scores of the critic
GP_WEIGHT = 10.0
DRIFT = 1e-3
SCORE_FLOOR = 0.5  # smallest half-gap in the style reward


def frame(S, key_pos):
    """Frame features (.., FRAME_DIM) of states S = (p, R, v, w, q, qd) (world-frame w) and the key bodies' world positions (.., 4, 3)."""
    p, R, _, w, q, qd = S
    return jnp.concatenate((q[..., KEPT], qd[..., KEPT], mv3(tr3(R), w), -R[..., 2, :], mv3(tr3(yaw_mat(R))[..., None, :, :], key_pos - p[..., None, :]).reshape(*p.shape[:-1], -1)), -1)


class Demos:
    """The frame features of all demo clips, and the start of every window of `frames` frames that stays inside a clip."""

    def __init__(self, directory: str, sim, frames: int):
        files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert files, f"no clips in {directory} (bash tools/download_data.sh)"
        feats, starts, n = [], [], 0
        for f in files:
            d = np.load(f)
            assert int(d["fps"]) == 50, f
            x, y, z, w = d["root_quat"].T  # (x, y, z, w) -> the rotation matrix
            R = quat_to_mat(jnp.stack([w, x, y, z], -1))
            assert d["dof_pos"].shape[1] == env.J, f
            q, qd = jnp.asarray(d["dof_pos"]), jnp.asarray(d["dof_vel"])
            S = (jnp.asarray(d["root_pos"]), R, None, jnp.asarray(d["root_ang_vel"]), q, qd)
            feats.append(frame(S, sim.key_positions(S[0], R, q)))
            starts.append(np.arange(n, n + len(x) - frames + 1))
            n += len(x)
        self.frames = jnp.concatenate(feats, 0)
        self.starts = jnp.asarray(np.concatenate(starts), jnp.int32)
        self.window = frames

    def sample(self, key, B):
        """B windows (B, frames, FRAME_DIM) uniform over the windows of all clips."""
        i = self.starts[jax.random.randint(key, (B,), 0, len(self.starts))]
        return self.frames[i[:, None] + jnp.arange(self.window)]


# ---------------------------------------------------------------- the critic
def init(key, frames: int, hidden=(1024, 512)):
    """The critic's parameters (an MLP with ELU on the flattened window, one output), the running normalizer of the frames, the running
    mean scores of the demos and the robot, and the optimizer state."""
    layers = nets.mlp_init(key, [frames * FRAME_DIM, *hidden, 1])
    mask = jax.tree.map(lambda _: False, layers)
    return {"net": layers, "norm": nets.rn_init(FRAME_DIM), "mu": jnp.zeros((2,)), "opt": optim.init(layers, mask), "ramp": jnp.zeros(())}


NORM_CLIP = 10.0  # the normalized frames are clipped, so that an extreme imagined state cannot overflow the critic or its gradient


def normalize(norm, window):
    return jnp.clip(nets.rn_apply(norm, window, eps=1e-2), -NORM_CLIP, NORM_CLIP)


def score(net, norm, window):
    """The critic's score of windows (.., frames, FRAME_DIM) of raw frame features."""
    x = normalize(norm, window).reshape(*window.shape[:-2], -1)
    return nets.mlp_apply(net, x, jax.nn.elu)[..., 0]


def style(st, d, scale: float = 0.0):
    """The style reward in [0, 1] of scores d, from the running mean scores of the demos and the robot.

    scale = 0: centered at the midpoint of the two means, with their half-gap as the scale. The robot's mean then always gets about
    0.12, and the slope is 1 / (2 half-gap): with a gap of about 140 the reward is nearly flat and its gradient through the world
    model nearly 0.
    scale > 0: centered at the robot's mean score, with this fixed scale: the robot's mean gets 0.5, and the slope where the robot
    is stays 1 / (2 scale) whatever the gap. (Not centered at the midpoint with a capped scale: the robot would sit deep in the
    saturated tail.)"""
    mu_d, mu_r = st["mu"][0], st["mu"][1]
    if scale > 0:
        return 0.5 * (1 + jnp.tanh((d - mu_r) / scale))
    m, s = (mu_d + mu_r) / 2, jnp.maximum((mu_d - mu_r) / 2, SCORE_FLOOR)
    return 0.5 * (1 + jnp.tanh((d - m) / s))


def loss(net, norm, demo, robot, wt, key):
    """WGAN-GP loss of the critic on windows of the demos and of the robot (B, frames, FRAME_DIM) and the statistics; wt (B,) weighs the
    robot windows (0: not usable) and, by pairs, the interpolates."""
    xd, xr = normalize(norm, demo), normalize(norm, robot)
    f = lambda x: nets.mlp_apply(net, x.reshape(x.shape[0], -1), jax.nn.elu)[:, 0]  # noqa: E731
    d_demo, d_robot = f(xd), f(xr)
    eps = jax.random.uniform(key, (xd.shape[0], 1, 1))
    xh = (eps * xd + (1 - eps) * xr).reshape(xd.shape[0], -1)
    g = jax.vmap(jax.grad(lambda x: nets.mlp_apply(net, x[None], jax.nn.elu)[0, 0]))(xh)
    n = jnp.maximum(wt.sum(), 1.0)
    gp = GP_WEIGHT * jnp.sum(wt * (jnp.linalg.norm(g, axis=-1) - 1.0) ** 2) / n
    d_robot = jnp.sum(wt * d_robot) / n
    total = d_robot - d_demo.mean() + DRIFT * jnp.mean(d_demo**2) + gp
    return total, {"d_demo": d_demo.mean(), "d_robot": d_robot, "gp": gp}


def train_step(st, demo, robot, wt, key, lr: float):
    """One critic step on a batch of demo and robot windows; the robot windows with weight 0 take no part."""
    (_, aux), grads = jax.value_and_grad(loss, has_aux=True)(st["net"], st["norm"], demo, robot, wt, key)
    grads, _ = optim.clip_by_global_norm(grads, 1.0)
    net, opt = optim.step(st["net"], grads, st["opt"], jax.tree.map(lambda _: False, st["net"]), lr, betas=(0.5, 0.9))
    norm = nets.rn_update(st["norm"], robot.reshape(-1, robot.shape[-1]), weights=jnp.repeat(wt, robot.shape[1]))
    first = (st["mu"][0] == 0) & (st["mu"][1] == 0)
    now = jnp.stack([aux["d_demo"], aux["d_robot"]])
    live = wt.sum() > 0  # no usable robot window: the scores of this step say nothing about the robot
    mu = jnp.where(live, jnp.where(first, now, EMA * st["mu"] + (1 - EMA) * now), st["mu"])
    return {**st, "net": net, "opt": opt, "norm": norm, "mu": mu}, aux
