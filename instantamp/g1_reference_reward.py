"""Training-only, phase-aligned recovery targets for the autonomous G1 actor."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import env
from .rot import quat_to_mat


class G1RecoveryReference:
    """Native reference poses and world-frame ankle/wrist positions, indexed by reset clip and frame."""

    def __init__(self, directory, key_positions):
        self.files = sorted(Path(directory).glob("*.npz"))
        positions, quats, joints, lengths = [], [], [], []
        for path in self.files:
            with np.load(path) as clip:
                q = clip["dof_pos"]
                p = clip["root_pos"]
                quat = clip["root_quat"]
                if (float(clip["fps"]) != 50 or q.shape != (len(p), env.J) or p.shape != (len(p), 3)
                        or quat.shape != (len(p), 4) or not len(p)):
                    raise ValueError(f"Expected a segmented native 50 Hz G1 get-up in {path}")
                joints.append(q)
                positions.append(p)
                quats.append(quat[:, [3, 0, 1, 2]])
                lengths.append(len(p))
        if not lengths:
            raise ValueError(f"No G1 recovery references in {directory}")
        self.position = jnp.asarray(np.concatenate(positions), jnp.float32)
        self.rotation = quat_to_mat(jnp.asarray(np.concatenate(quats), jnp.float32))
        self.joints = jnp.asarray(np.concatenate(joints), jnp.float32)
        self.landmarks = key_positions(self.position, self.rotation, self.joints)
        self.offsets = jnp.asarray(np.cumsum([0, *lengths[:-1]]), jnp.int32)
        self.lengths = jnp.asarray(lengths, jnp.int32)

    def active(self, clip, start, elapsed):
        return (clip >= 0) & (start >= 0) & (elapsed >= 0) & (start + elapsed < self.lengths[jnp.maximum(clip, 0)])

    def score(self, state, points, clip, start, yaw, elapsed):
        p, r, _, _, q, _ = state
        valid_clip = jnp.maximum(clip, 0)
        base = self.offsets[valid_clip]
        last = self.lengths[valid_clip] - 1
        idx = base + jnp.clip(start + elapsed, 0, last)
        # Reference XY is translated to put its reset frame at (0, 0) and rotated by the
        # same random world yaw that augmented the floor reset. Gravity is unchanged.
        origin = self.position[base + jnp.clip(start, 0, last)].at[..., 2].set(0.)
        zeros, ones = jnp.zeros_like(yaw), jnp.ones_like(yaw)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        rz = jnp.stack((c, -s, zeros, s, c, zeros, zeros, zeros, ones), -1).reshape(*yaw.shape, 3, 3)
        target_pos = (rz @ (self.position[idx] - origin)[..., None])[..., 0]
        target_rot = rz @ self.rotation[idx]
        target_pts = ((rz[..., None, :, :] @ (self.landmarks[idx] - origin[..., None, :])[..., None])[..., 0])
        # Independent bounded terms retain a gradient when another body part is far away.
        matched = lambda error, scale: 1. / (1. + error / scale**2)
        translation = p - target_pos
        xyz = .5 * (matched(translation[..., 2] ** 2, .12)
                    + matched((translation[..., :2] ** 2).sum(-1), .20))
        rotation = matched(.5 * ((r - target_rot) ** 2).sum(axis=(-2, -1)), .55)
        joint = matched(((q - self.joints[idx]) ** 2).mean(-1), .40)
        landmark = matched(((points - target_pts) ** 2).sum(-1), .17).mean(-1)
        score = .25 * (xyz + rotation + joint + landmark)
        return jnp.where(self.active(clip, start, elapsed), score, 0.)
