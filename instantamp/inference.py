"""Load dense policy exports and evaluate the G1 actor before ODE integration."""

import jax.numpy as jnp
import numpy as np

from . import nets
from .zenbot_motion_train_obs import motion_train_actor_mean


def load(path, which="avg"):
    """Read the dense actor, observation normalizer and optional ODE limits."""
    with np.load(path, allow_pickle=False) as archive:
        if any(
            key in archive
            for key in (
                "blend",
                "stand_thresholds",
                "ground_max_share",
                "zenbot_library/keys",
                "gait/pi/0/w",
                "zenbot_tracking/offset",
            )
        ):
            raise ValueError("Expected a single dense policy export")
        layers = []
        while f"{which}/{len(layers)}/w" in archive:
            prefix = f"{which}/{len(layers)}/"
            layers.append(
                {
                    key[len(prefix) :]: jnp.asarray(archive[key])
                    for key in archive.files
                    if key.startswith(prefix)
                }
            )
        if not layers:
            raise ValueError(f"Missing {which} policy weights")
        norm = {
            key: jnp.asarray(archive[f"obs_norm/{key}"])
            for key in ("mean", "var")
        }
        if "obs_norm/raw" in archive:
            norm["raw"] = jnp.asarray(True)
        ode = None
        if "ode/dmax" in archive:
            ode = tuple(
                float(archive[f"ode/{key}"]) or None for key in ("dmax", "amax")
            )
    return layers, norm, ode


def policy_output(policy, norm, obs, obs_clip=10.0):
    """Evaluate the G1 policy on a batch of physical observation histories."""
    if "raw" not in norm:
        obs = jnp.clip(nets.rn_apply(norm, obs), -obs_clip, obs_clip)
    return nets.pi_apply(policy, obs)


def policy_action(sim, state, policy, norm, ode, obs_clip=10.0):
    """Produce a physical action using the observation layout used in training."""
    if sim.zenbot_motion_train_wm:
        obs = state.motion_train_sensors.observation()
        output = motion_train_actor_mean(policy, norm, obs)
    else:
        output = policy_output(policy, norm, state.obs, obs_clip)
    return (
        nets.ode_step(output, state.a_ode, *ode) if ode is not None else output
    )
