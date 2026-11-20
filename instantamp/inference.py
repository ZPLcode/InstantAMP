"""Load an exported InstantAMP policy and evaluate its raw action before ODE integration."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import env, nets


class G1LimbBlend(NamedTuple):
    base: list
    style: list
    weights: object
    gated: object


def _actor(archive, which, prefix=""):
    root = prefix + which + "/"
    layers = []
    while root + f"{len(layers)}/w" in archive:
        layer = root + f"{len(layers)}/"
        layers.append({key[len(layer):]: jnp.asarray(archive[key]) for key in archive.files if key.startswith(layer)})
    if not layers:
        raise ValueError(f"Missing {root} policy weights")
    norm = {key: jnp.asarray(archive[prefix + "obs_norm/" + key]) for key in ("mean", "var")}
    ode = None
    if prefix + "ode/dmax" in archive:
        ode = tuple(float(archive[prefix + "ode/" + key]) or None for key in ("dmax", "amax"))
    return layers, norm, ode


def load(path, which="avg"):
    """Return (policy, observation normalizer, ODE limits) for an NPZ export."""
    with np.load(path, allow_pickle=False) as archive:
        if "blend" not in archive:
            return _actor(archive, which)
        base, base_norm, ode = _actor(archive, which, "base/")
        style, style_norm, style_ode = _actor(archive, which, "style/")
        weights = archive["blend"]
        if ode != style_ode or weights.shape != (env.J,) or any(
            actor[0]["w"].shape[0] != env.OBS_DIM or actor[-1]["b"].shape != weights.shape
            for actor in (base, style)
        ):
            raise ValueError(f"Incompatible G1 actors or joint weights in {path}")
        return G1LimbBlend(base, style, jnp.asarray(weights), jnp.asarray(archive["gate"])), (base_norm, style_norm), ode


def g1_recovery_gate(obs):
    """Weight from projected gravity and bilateral knee flexion in the latest 96-value G1 frame."""
    frame = obs[..., -env.FRAME_DIM:]
    knees = np.array([env.DOF.index("left_knee"), env.DOF.index("right_knee")])
    tilt = 1 + frame[..., 5]
    knee_angles = frame[..., 9 + knees] + env.DEFAULT_Q[knees]
    tilt_gate = jnp.clip((tilt - .08) / .35, 0., 1.)
    kneel_gate = jnp.clip((knee_angles.min(-1) - 1.) / .65, 0., 1.)
    return jnp.maximum(tilt_gate, kneel_gate)


def policy_output(policy, norm, obs, obs_clip=10.):
    """Return the raw output from unnormalized observation history, batched or unbatched."""
    def apply(layers, stats):
        return nets.pi_apply(layers, jnp.clip(nets.rn_apply(stats, obs), -obs_clip, obs_clip))

    if isinstance(policy, G1LimbBlend):
        base = apply(policy.base, norm[0])
        style = apply(policy.style, norm[1])
        gate = jnp.where(policy.gated, g1_recovery_gate(obs), 1.)
        return base + (gate[..., None] * policy.weights) * (style - base)
    return apply(policy, norm)
