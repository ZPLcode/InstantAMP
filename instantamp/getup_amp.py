"""Expert motion windows for the G1 get-up discriminator.

Untrimmed fallAndGetUp takes use the original all-down window filter. A segmented
bank contains one forward recovery per file. All its windows, including the
transition to upright and the standing tail, are valid. These clips are sampled
with equal probability so long recoveries do not dominate the prior.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .rot import quat_to_mat


def getup_frame(sim, state):
    """G1 AMP features augmented with pelvis height (m) and world vertical speed (m/s)."""
    position, _, linear_velocity, *_ = state
    return jnp.concatenate((sim.amp_frame(state), position[..., 2:3], linear_velocity[..., 2:3]), axis=-1)


class GetupDemos:
    def __init__(self, directory: str, sim, frames: int, *, segmented=False):
        directory = Path(directory)
        files = sorted(directory.glob("*.npz"))
        if not files or frames < 1:
            raise ValueError(f"Expected get-up clips and a positive window length in {directory}")
        if segmented and (directory / "manifest.json").exists():
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest.get("kind") == "g1_rise_clips":
                names = sorted(clip["file"] for clip in manifest["clips"])
                if names != [path.name for path in files]:
                    raise ValueError(f"Get-up clip files differ from the manifest in {directory}")

        fields = ["root_pos", "root_quat", "root_ang_vel", "dof_pos", "dof_vel"]
        if segmented:
            fields.append("root_lin_vel")
        parts = {name: [] for name in fields}
        lengths = []
        for path in files:
            with np.load(path) as clip:
                if float(clip["fps"]) != 50 or clip["dof_pos"].shape[1] != sim.J:
                    raise ValueError(f"{path}: expected 50 Hz motion with {sim.J} joints")
                length = len(clip["root_pos"])
                if segmented and length < frames:
                    raise ValueError(f"{path}: {length} frames cannot fill a {frames}-frame window")
                lengths.append(length)
                for name in fields:
                    value = clip[name]
                    if len(value) != length:
                        raise ValueError(f"{path}: {name} and root_pos have different frame counts")
                    parts[name].append(value)

        # Run FK on a single batch. The segmented bank can contain many different
        # clip lengths; compiling JAX kernels for each length slows every launch.
        motion = {name: jnp.asarray(np.concatenate(values)) for name, values in parts.items()}
        rot = quat_to_mat(motion["root_quat"][:, (3, 0, 1, 2)])
        state = (motion["root_pos"], rot, motion.get("root_lin_vel"), motion["root_ang_vel"],
                 motion["dof_pos"], motion["dof_vel"])
        offsets = np.cumsum([0, *lengths[:-1]], dtype=np.int32)
        counts = np.maximum(np.asarray(lengths) - frames + 1, 0)
        self.clip_offsets = None
        if segmented:
            self.frames = getup_frame(sim, state)
            self.clip_offsets = jnp.asarray(offsets)
            self.clip_counts = jnp.asarray(counts, dtype=jnp.int32)
            starts = [offset + np.arange(count) for offset, count in zip(offsets, counts)]
        else:
            self.frames = sim.amp_frame(state)
            down = ~np.asarray(sim.is_up(state[0][:, 2], rot))
            starts = []
            for offset, length in zip(offsets, lengths):
                if length >= frames:
                    clip_down = down[offset:offset + length]
                    all_down = np.lib.stride_tricks.sliding_window_view(clip_down, frames).all(-1)
                    starts.append(offset + np.flatnonzero(all_down))
        if not starts or not any(len(indices) for indices in starts):
            raise ValueError(f"No valid {frames}-frame get-up windows in {directory}")
        self.starts = jnp.asarray(np.concatenate(starts), dtype=jnp.int32)
        self.window = frames
        self.n_windows = len(self.starts)

    def sample(self, key, B):
        if self.clip_offsets is None:
            i = self.starts[jax.random.randint(key, (B,), 0, self.n_windows)]
        else:
            kc, kt = jax.random.split(key)
            clip = jax.random.randint(kc, (B,), 0, len(self.clip_offsets))
            i = self.clip_offsets[clip] + jax.random.randint(kt, (B,), 0, self.clip_counts[clip])
        return self.frames[i[:, None] + jnp.arange(self.window)]
