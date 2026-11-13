"""A value function for imagination (`--critic 1`, on by default).

XSoma's published recipe has no critic: the policy maximizes the discounted reward of 48 imagined steps (about 1 s). Getting up takes 3 to
5 s, longer than the horizon, so a critic can bootstrap the return at the horizon, as in SHAC (Xu et al. 2022) and as SuperMimic's
repository does:

  actor    maximizes  sum_k gamma^k alive_k r_k  +  gamma^H alive_H V(s_H)
           with the critic's parameters held fixed; V(s_H) is differentiable in the imagined state, so the bootstrap's gradient reaches
           the policy through the world model like the rewards' do.
  critic   fits TD(lambda) targets of the same imagined rollouts (detached): G_k = r_k + gamma (1 - d_k) ((1 - lambda) V(s_{k+1}) + lambda G_{k+1}),
           G_H = V(s_H), for `epochs` passes over the rollouts' steps in `minibatches` minibatches (SuperMimic's repository: 5 and 4).

The critic sees the privileged state (train.InstantAMP.critic_in): the heading-invariant robot state, the command, the last action, the
dynamics context xi, and the recovery state (whether the robot has been up, and the share of its time to rise that is left).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import nets, optim


def init(key, in_dim: int, hidden=(512, 512, 256)):
    net = nets.mlp_init(key, [in_dim, *hidden, 1], layer_norm=True)
    net[-1]["w"] = jnp.zeros_like(net[-1]["w"])  # the untrained critic predicts 0
    net[-1]["b"] = jnp.zeros_like(net[-1]["b"])
    mask = jax.tree.map(lambda _: False, net)
    return {"net": net, "norm": nets.rn_init(in_dim), "opt": optim.init(net, mask)}


def value(st, x):
    """V of critic inputs x (.., in_dim)."""
    return nets.mlp_apply(st["net"], nets.rn_apply(st["norm"], x), jax.nn.silu)[..., 0]


def lambda_targets(r, d, V, gamma: float, lam: float):
    """TD(lambda) targets (H, B) of rewards r (H, B), terminations d (H, B; 1 ends the rollout after the step) and values V (H + 1, B)."""

    def back(G, x):
        r_k, d_k, v1 = x
        G = r_k + gamma * (1 - d_k) * ((1 - lam) * v1 + lam * G)
        return G, G

    _, G = jax.lax.scan(back, V[-1], (r, d, V[1:]), reverse=True)
    return G


def fit(st, x, r, d, key, gamma: float, lam: float, lr: float, epochs: int, minibatches: int):
    """Fit the critic to the TD(lambda) targets of imagined rollouts: x (H + 1, B, in_dim) the inputs of the states s_0 .. s_H, r and d (H, B).
    Steps after a termination have no target. Returns the new critic state and the loss of the last minibatch."""
    alive_before = jnp.concatenate((jnp.ones_like(d[:1]), jnp.cumprod(1 - d, 0)[:-1]), 0)
    st = {**st, "norm": nets.rn_update(st["norm"], x.reshape(-1, x.shape[-1]))}
    G = jax.lax.stop_gradient(lambda_targets(r, d, value(st, x), gamma, lam))
    X, Y, W = x[:-1].reshape(-1, x.shape[-1]), G.reshape(-1), alive_before.reshape(-1)
    n = X.shape[0]
    m = n // minibatches
    mask = jax.tree.map(lambda _: False, st["net"])

    def loss(net, xb, yb, wb):
        v = nets.mlp_apply(net, nets.rn_apply(st["norm"], xb), jax.nn.silu)[..., 0]
        return jnp.sum(wb * (v - yb) ** 2) / jnp.maximum(wb.sum(), 1.0)

    def epoch(carry, k):
        net, opt = carry
        perm = jax.random.permutation(k, n)[: m * minibatches].reshape(minibatches, m)

        def mb(c2, idx):
            net, opt = c2
            l, g = jax.value_and_grad(loss)(net, X[idx], Y[idx], W[idx])
            g, _ = optim.clip_by_global_norm(g, 1.0)
            net, opt = optim.step(net, g, opt, mask, lr)
            return (net, opt), l

        (net, opt), ls = jax.lax.scan(mb, (net, opt), perm)
        return (net, opt), ls[-1]

    (net, opt), ls = jax.lax.scan(epoch, (st["net"], st["opt"]), jax.random.split(key, epochs))
    return {**st, "net": net, "opt": opt}, ls[-1]
