"""Sample G1 recovery examples whose entire discriminator window is down."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .rot import quat_to_mat


class GetupDemos:
    """Collect valid windows within each source motion for the recovery prior."""

    def __init__(self, directory: str, sim, frames: int):
        files = sorted(Path(directory).glob("*.npz"))
        if not files or frames < 1:
            raise ValueError(
                f"Expected get-up clips and a positive window length in {directory}"
            )

        fields = ("root_pos", "root_quat", "root_ang_vel", "dof_pos", "dof_vel")
        parts = {name: [] for name in fields}
        lengths = []
        for path in files:
            with np.load(path, allow_pickle=False) as clip:
                if (
                    float(clip["fps"]) != 50
                    or clip["dof_pos"].shape[1] != sim.J
                ):
                    raise ValueError(
                        f"{path}: expected 50 Hz motion with {sim.J} joints"
                    )
                length = len(clip["root_pos"])
                lengths.append(length)
                for name in fields:
                    value = clip[name]
                    if len(value) != length:
                        raise ValueError(
                            f"{path}: {name} and root_pos have different frame counts"
                        )
                    parts[name].append(value)

        motion = {
            name: jnp.asarray(np.concatenate(values))
            for name, values in parts.items()
        }
        rot = quat_to_mat(motion["root_quat"][:, (3, 0, 1, 2)])
        state = (
            motion["root_pos"],
            rot,
            None,
            motion["root_ang_vel"],
            motion["dof_pos"],
            motion["dof_vel"],
        )
        self.frames = sim.amp_frame(state)
        down = ~np.asarray(sim.is_up(state[0][:, 2], rot))
        starts = []
        offset = 0
        for length in lengths:
            if length >= frames:
                windows = np.lib.stride_tricks.sliding_window_view(
                    down[offset : offset + length], frames
                )
                starts.append(offset + np.flatnonzero(windows.all(axis=-1)))
            offset += length
        if not any(len(indices) for indices in starts):
            raise ValueError(
                f"No valid {frames}-frame get-up windows in {directory}"
            )
        self.starts = jnp.asarray(np.concatenate(starts), dtype=jnp.int32)
        self.window = frames
        self.n_windows = len(self.starts)

    def sample(self, key, batch_size):
        indices = self.starts[
            jax.random.randint(key, (batch_size,), 0, self.n_windows)
        ]
        return self.frames[indices[:, None] + jnp.arange(self.window)]
