"""Phase-matched pose targets from native MuJoCo Zenbot motion, measured at 50 Hz."""

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import env, zenbot
from .rot import quat_to_mat
from tools.prepare_zenbot_support_frames import first_support_frame


PHYSICAL_FLOOR = Path(__file__).resolve().parents[1] / "data/zenbot23_physical_recovery"
SETTLE_TICKS = round(1. / env.DT)


class PhysicalRecoveryReference(zenbot.RecoveryReference):
    def __init__(self, physical_dir, source_dir, key_positions):
        self.joints = len(zenbot.DOF)
        self.coupled_pose = True
        sources = sorted(Path(source_dir).glob("*.npz"))
        self.clip_names = tuple(p.name for p in sources)
        by_source = {}
        for path in sorted(Path(physical_dir).glob("*.npz")):
            with np.load(path, allow_pickle=False) as trajectory:
                name = str(trajectory["source_clip"])
                if name in by_source:
                    raise ValueError(f"Duplicate physical recovery reference for {name}")
                q, v = trajectory["qpos"], trajectory["qvel"]
                if (q.shape != (len(q), 30) or v.shape != (len(q), 29) or not np.isfinite(q).all()
                        or not np.isfinite(v).all() or float(trajectory["dt"]) != env.DT
                        or int(trajectory["source_frame"]) != 0 or float(trajectory["external_assist_newtons"]) != 0.):
                    raise ValueError(f"Expected a finite unassisted 50 Hz physical recovery from frame zero: {path}")
                physical = trajectory["physical"]
                if len(physical) != len(q) - 1:
                    raise ValueError(f"Expected post-action contact records aligned to state t+1: {path}")
                support = first_support_frame(physical)
                by_source[name] = (path, q, v, support, str(trajectory["source_sha256"]))
        if not sources or set(self.clip_names) != set(by_source):
            raise ValueError(f"Source clip names in {source_dir} must match the physical references in {physical_dir}")
        self.files, poses, velocities, support_frames = [], [], [], []
        for source in sources:
            path, q, v, support, source_hash = by_source[source.name]
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                raise ValueError(f"Physical recovery plan {path} came from a different source clip: {source}")
            self.files.append(path)
            poses.append(q)
            velocities.append(v)
            support_frames.append(support)
        lengths = [len(q) for q in poses]
        self.lengths = jnp.asarray(lengths, jnp.int32)
        self.offsets = jnp.asarray(np.cumsum([0, *lengths[:-1]]), jnp.int32)
        self.first_support = jnp.asarray(support_frames, jnp.int32)
        self.qpos = jnp.asarray(np.concatenate(poses), jnp.float32)
        self.qvel = jnp.asarray(np.concatenate(velocities), jnp.float32)
        p, rot, joints = self.qpos[:, :3], quat_to_mat(self.qpos[:, 3:7]), self.qpos[:, 7:]
        self.poses = self.pose(p, rot, joints, key_positions(p, rot, joints))

    def tracked_score(self, pose, clip, start_frame, elapsed, recovering, fell):
        """The post-step pose score within the first matched rise and up to one second of settling."""
        paired = (clip >= 0) & (start_frame >= 0)
        index = jnp.maximum(clip, 0)
        frame = start_frame + elapsed
        active = paired & recovering & ~jnp.asarray(fell, bool)
        active &= (frame >= 0) & (frame <= self.first_support[index] + SETTLE_TICKS)
        return jnp.where(active, self.score(pose, index, frame), 0.)

    def failed_rise(self, score, clip, start_frame, elapsed, active, fell, minimum):
        """End a training attempt that loses its matched rise, or falls after standing."""
        frame = start_frame + elapsed
        tracked = (clip >= 0) & active & (elapsed >= 5) & (frame <= self.first_support[jnp.maximum(clip, 0)] + SETTLE_TICKS)
        return fell | (tracked & (score < minimum))
