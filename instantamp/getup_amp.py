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

from . import amp, nets
from .kin import Kinematics
from .rot import mm3, mv3, quat_to_mat, tr3, yaw_mat


# legged_lab G1 get-up key links, in the exact order of its discriminator.
LEGGED_BODIES = (*amp.KEY_BODIES, "left_shoulder_roll_link", "right_shoulder_roll_link")


def legged_frame(state, points):
    """G1 get-up AMP: heading-free tangent/normal, body gyro, q, qdot, body-relative six link origins."""
    p, r, _, w, q, qd = state
    local = mm3(tr3(yaw_mat(r)), r)
    offsets = mv3(tr3(r)[..., None, :, :], points - p[..., None, :])
    return jnp.concatenate((local[..., :, 0], local[..., :, 2], mv3(tr3(r), w),
                            q, qd, offsets.reshape(*p.shape[:-1], 18)), -1)


def getup_frame(sim, state):
    """G1 AMP features augmented with pelvis height (m) and world vertical speed (m/s)."""
    position, _, linear_velocity, *_ = state
    return jnp.concatenate((sim.amp_frame(state), position[..., 2:3], linear_velocity[..., 2:3]), axis=-1)


class GetupDemos:
    def __init__(self, directory: str, sim, frames: int, *, segmented=False, legged=False):
        directory = Path(directory)
        files = sorted(directory.glob("*.npz"))
        if not files or frames < 1:
            raise ValueError(f"Expected get-up clips and a positive window length in {directory}")
        self.files = files
        if legged:
            if not segmented or sim.J != 29 or frames != 10:
                raise ValueError("The G1 legged recovery prior needs 29 joints and segmented 10-frame windows")
            self.kin = Kinematics(sim.m, [f"robot_{name}" for name in LEGGED_BODIES])
            self.walk_joints = sim.amp_joints
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
            if legged:
                points = self.kin.forward(state[0], state[1], state[4])[0]
                self.frames = legged_frame(state, points)
            else:
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

    def policy_frames(self, state):
        """Compute the unmodified walking frame and the 85-dimensional recovery frame with a shared FK."""
        points = self.kin.forward(state[0], state[1], state[4])[0]
        walking = amp.frame(state, points[..., :4, :], self.walk_joints)
        return jnp.concatenate((walking, legged_frame(state, points)), axis=-1)


class LeggedRecoveryPrior:
    """One LSGAN with the same source-time distribution for both classes and both A/B recipes."""

    def __init__(self, sim, demos, timed):
        ref = sim.reference
        if (not sim.g1_reference or demos.files != ref.files or
                not np.array_equal(demos.clip_counts + demos.window - 1, ref.lengths)):
            raise ValueError("The G1 legged recovery prior requires identical reset and demonstration clips")
        self.demos, self.timed = demos, timed
        self.lengths, self.offsets = ref.lengths, ref.offsets

    def _indices(self, clip, start, elapsed):
        clip, start, elapsed = jnp.broadcast_arrays(clip, start, elapsed)
        i = jnp.clip(clip, 0, len(self.lengths) - 1)
        last = self.lengths[i] - 1
        end = start + elapsed
        valid = (i == clip) & (start >= 0) & (elapsed >= self.demos.window - 1) & (end <= last)
        ticks = jnp.clip(end[..., None] - (self.demos.window - 1) + jnp.arange(self.demos.window), 0, last[..., None])
        return i, ticks, valid

    def _pack(self, raw, i, ticks):
        if not self.timed:
            return raw
        clock = ticks / jnp.maximum(self.lengths[i][..., None] - 1, 1)
        return jnp.concatenate((raw, clock[..., None]), -1)

    def pair(self, windows, clip, start, elapsed):
        """Window ends at source frame start+elapsed. Frame zero is the physical reset pose."""
        i, ticks, valid = self._indices(clip, start, elapsed)
        expert = self.demos.frames[self.offsets[i][..., None] + ticks]
        return self._pack(expert, i, ticks), self._pack(windows, i, ticks), valid

    def style(self, disc, windows, clip, start, elapsed, scale):
        i, ticks, valid = self._indices(clip, start, elapsed)
        score = amp.recovery_style(disc, self._pack(windows, i, ticks), scale)
        return jnp.where(valid, score, 0.)

    def init(self, key, hidden=(1024, 512)):
        demo = self.demos.frames
        if self.timed:
            clock = np.concatenate([np.linspace(0., 1., int(n), dtype=np.float32) for n in self.lengths])
            demo = jnp.concatenate((demo, jnp.asarray(clock)[:, None]), -1)
        disc = amp.init(key, self.demos.window, hidden, demo.shape[-1])
        if self.timed:
            # Same initial function as A: preserve all motion weights and let
            # the discriminator learn the extra per-frame clock connections.
            raw_dim = self.demos.frames.shape[-1]
            plain = nets.mlp_init(key, [self.demos.window * raw_dim, *hidden, 1])[0]
            weights = plain["w"].reshape(self.demos.window, raw_dim, -1)
            padded = jnp.pad(weights, ((0, 0), (0, 1), (0, 0)))
            disc["net"][0] = {"w": padded.reshape(disc["net"][0]["w"].shape), "b": plain["b"]}
        disc["norm"] = nets.rn_update(disc["norm"], demo)
        return disc
