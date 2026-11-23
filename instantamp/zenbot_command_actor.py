"""Warm-start a single command-conditioned Zenbot actor from its get-up sensor history."""

import jax
import jax.numpy as jnp
import numpy as np

from . import zenbot


SENSORS_PER_FRAME = 9 + 3 * len(zenbot.DOF) + 3 * len(zenbot.KEY_BODIES)
COMMAND_HISTORY_WIDTH = zenbot.HIST * (SENSORS_PER_FRAME + 3)


def gait_teacher_observation(frames):
    """Recover Motion Train's five IMU, command, joint and action frames from the command actor's sensors."""
    first, third = frames[..., 3:6], frames[..., 6:9]
    # The vertical projection of the second rotation column comes from third × first.
    gravity = -jnp.stack((first[..., 2], jnp.cross(third, first)[..., 2], third[..., 2]), axis=-1)
    joints = len(zenbot.DOF)
    frame = jnp.concatenate((frames[..., :3], gravity, frames[..., -3:], frames[..., 9:9+joints] - zenbot.DEFAULT_Q,
                             frames[..., 9+joints:9+3*joints]), axis=-1)
    return frame.reshape((*frame.shape[:-2], -1))


def initialize_command_actor(template, path):
    """Preserve the source actor's actions for every command until its new weights learn."""
    from . import optim
    from .train import initialize_actor

    if template["pi"][0]["w"].shape[0] != COMMAND_HISTORY_WIDTH:
        raise ValueError("Expected a command-conditioned Zenbot training template")
    sensor_width = zenbot.HIST * SENSORS_PER_FRAME
    with np.load(path, allow_pickle=False) as archive:
        source_width = archive["obs_norm/mean"].shape
    if source_width == (COMMAND_HISTORY_WIDTH,):
        return initialize_actor(template, path, expected_ode=None)
    if source_width != (sensor_width,):
        raise ValueError(f"Expected a {sensor_width}- or {COMMAND_HISTORY_WIDTH}-input standalone Zenbot actor")

    old_first = {**template["pi"][0], "w": template["pi"][0]["w"][:sensor_width]}
    old_norm = {**template["obs_norm"], **{k: template["obs_norm"][k][:sensor_width] for k in ("mean", "var")}}
    source = initialize_actor(
        {**template, "pi": [old_first, *template["pi"][1:]], "obs_norm": old_norm}, path, expected_ode=None
    )

    def append_channels(array, fill):
        frames = array.reshape((zenbot.HIST, SENSORS_PER_FRAME, *array.shape[1:]))
        padding = jnp.full((zenbot.HIST, 3, *array.shape[1:]), fill, dtype=array.dtype)
        return jnp.concatenate((frames, padding), axis=1).reshape((COMMAND_HISTORY_WIDTH, *array.shape[1:]))

    first = {**source["pi"][0], "w": append_channels(source["pi"][0]["w"], 0.0)}
    pi = [first, *source["pi"][1:]]
    norm = source["obs_norm"]
    source.update(
        pi=pi,
        avg=jax.tree.map(jnp.copy, pi),
        obs_norm={**norm, "mean": append_channels(norm["mean"], 0.0), "var": append_channels(norm["var"], 1.0)},
    )
    if "opt_pi" in source:
        source["opt_pi"] = optim.init(pi, jax.tree.map(lambda _: False, pi))
    return source
