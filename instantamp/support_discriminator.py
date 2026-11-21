"""Adversarial prior on unpaired windows of ground-support motion."""

import jax
import jax.numpy as jnp
import numpy as np

from . import nets, optim


def load_legged_prior(path):
    """Read the converted 23-DoF LeggedLab discriminator in native joint order."""
    widths = (730, 1024, 512, 1)
    with np.load(path, allow_pickle=False) as data:
        def field(name, shape):
            if name not in data:
                raise ValueError(f"Missing LeggedLab discriminator field {name}")
            value = data[name]
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"Invalid LeggedLab discriminator field {name}; expected finite floats of shape {shape}")
            return jnp.asarray(value, jnp.float32)

        norm = {k: field("norm/" + k, () if k == "count" else (73,)) for k in ("mean", "var", "count")}
        if np.any(data["norm/var"] < 0) or data["norm/count"].item() <= 0:
            raise ValueError("Invalid LeggedLab normalization statistics")
        layers = [{"w": field(f"net/{i}/w", (n, m)), "b": field(f"net/{i}/b", (m,))}
                  for i, (n, m) in enumerate(zip(widths, widths[1:]))]
    return {"net": layers, "norm": norm}


def legged_logit(st, window):
    """Apply the source policy-window preprocessing and the original float32 network."""
    if window.shape[-2:] != (10, 73):
        raise ValueError("The LeggedLab Zenbot discriminator expects ten 73-dimensional frames")
    norm = st["norm"]
    x = jnp.clip((window - norm["mean"]) / (jnp.sqrt(norm["var"]) + .01), -10., 10.)
    x = x.reshape(*x.shape[:-2], 730)
    for i, layer in enumerate(st["net"]):
        x = jnp.matmul(x, layer["w"], precision=jax.lax.Precision.HIGH) + layer["b"]
        if i < len(st["net"]) - 1:
            x = jax.nn.leaky_relu(x)
    return x[..., 0]


def legged_generator_reward(logit, scale):
    """Keep LeggedLab's LS reward on accepted motion, extending its lower tail for policy gradients."""
    native = jnp.where(logit > 3., 0., 1. - .25 * jnp.square(logit - 1.))
    # At -1 the two branches agree in both value (0) and slope (1). The
    # logarithmic tail reduces sensitivity to extreme simulated velocities.
    tail = 2. * jnp.arcsinh((logit + 1.) / 2.)
    return scale * jnp.where(logit < -1., tail, native)


def init(key, frames, feature_dim):
    net = nets.mlp_init(key, (frames * feature_dim, 128, 64, 1))
    adam = jax.tree.map(lambda _: False, net)
    return {"net": net, "opt": optim.init(net, adam)}


def _logit(net, window):
    flat = window.reshape(*window.shape[:-2], -1)
    return nets.mlp_apply(net, flat, jax.nn.leaky_relu)[..., 0]


def reward(st, windows, scale):
    # Zero at equal odds. A confidently rejected policy window still has a
    # generator gradient, including when its discriminator logit is below -1.
    return scale * (jnp.log(2.) + jax.nn.log_sigmoid(_logit(st["net"], windows)))


def train_step(st, demo, robot, mask, key, lr):
    """Fit the two window populations. The mask excludes interrupted policy samples."""
    kd, kr, km = jax.random.split(key, 3)
    demo = demo + .015 * jax.random.normal(kd, demo.shape)
    robot = robot + .015 * jax.random.normal(kr, robot.shape)
    fraction = jax.random.uniform(km, (len(robot), 1, 1))
    mixed = fraction * robot + (1. - fraction) * demo
    count = jnp.maximum(mask.sum(), 1.)

    def objective(net):
        positive = _logit(net, demo)
        negative = _logit(net, robot)
        # Symmetric soft labels limit the incentive to increase the score gap
        # on easy examples from a small motion bank.
        positive_loss = jax.nn.softplus(-positive) + .08 * positive
        negative_loss = jax.nn.softplus(negative) - .08 * negative
        grad_input = jax.vmap(jax.grad(lambda sample: _logit(net, sample)))(mixed)
        gp = .5 * (mask * jnp.square(grad_input).sum(axis=(-2, -1))).sum() / count
        loss = .5 * (positive_loss.mean() + (mask * negative_loss).sum() / count) + gp
        return loss, {"d_demo": positive.mean(), "d_robot": (mask * negative).sum() / count, "gp": gp}

    (_, stats), grads = jax.value_and_grad(objective, has_aux=True)(st["net"])
    grads, norm = optim.clip_by_global_norm(grads, 1.)
    adam = jax.tree.map(lambda _: False, st["net"])
    net, opt = optim.step(st["net"], grads, st["opt"], adam, lr, betas=(.5, .99),
                          ok=(mask.sum() > 0) & jnp.isfinite(norm))
    return {**st, "net": net, "opt": opt}, stats
