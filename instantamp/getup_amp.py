"""Demo windows of the get-up motion prior: the second discriminator.

The clips are the LAFAN1 fallAndGetUp takes (data/recovery/lafan1, tools/download_data.sh). A demo window is `frames` consecutive
frames of a take in which the robot is not up in any frame (env.is_up: pelvis at or below 0.6 m, or 1 - R_zz at or above 0.3): lying,
sitting, getting up, and falling down. The frame features are those of the locomotion prior (amp.frame), so the same critic code,
normalizer and style reward serve both discriminators.
"""

from __future__ import annotations

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env
from .rot import quat_to_mat


class GetupDemos:
    def __init__(self, directory: str, sim, frames: int):
        files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert files, f"no get-up clips in {directory}"
        feats, starts, n = [], [], 0
        for f in files:
            d = np.load(f)
            assert int(d["fps"]) == 50, f
            x, y, z, w = d["root_quat"].T
            R = quat_to_mat(jnp.stack([w, x, y, z], -1))
            p, q, qd = jnp.asarray(d["root_pos"]), jnp.asarray(d["dof_pos"]), jnp.asarray(d["dof_vel"])
            S = (p, R, None, jnp.asarray(d["root_ang_vel"]), q, qd)
            feats.append(amp.frame(S, sim.key_positions(p, R, q)))
            down = ~np.asarray(env.is_up(d["root_pos"][:, 2], np.asarray(R)))
            ok = np.lib.stride_tricks.sliding_window_view(down, frames).all(-1)  # windows of not-up frames only
            starts.append(n + np.nonzero(ok)[0])
            n += len(x)
        self.frames = jnp.concatenate(feats, 0)
        self.starts = jnp.asarray(np.concatenate(starts), jnp.int32)
        assert len(self.starts) > 0, "no window of not-up frames in the get-up clips"
        self.window = frames
        self.n_windows = int(len(self.starts))

    def sample(self, key, B):
        i = self.starts[jax.random.randint(key, (B,), 0, len(self.starts))]
        return self.frames[i[:, None] + jnp.arange(self.window)]
