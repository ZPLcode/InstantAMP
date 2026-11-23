"""World-model training windows from measured native MuJoCo get-up and standing transitions."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import env, zenbot
from .zenbot_wm_reference import PHYSICAL_FLOOR
from .zenbot_wm_sensors import pd_targets


class PhysicalMotionReplay:
    """Sample equally from each verified clip's rise and final standing segment."""

    def __init__(self, physical_dir=PHYSICAL_FLOOR):
        directory = Path(physical_dir)
        manifest = json.loads((directory / "manifest.json").read_text())
        entries = sorted(manifest["trajectories"].values(), key=lambda entry: entry["filename"])
        if not entries:
            raise ValueError("Expected verified physical recovery trajectories")
        positions, velocities, targets, contexts, stance_start = [], [], [], [], []
        for entry in entries:
            path = directory / entry["filename"]
            if not entry["quality"]["accepted"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Physical motion failed its manifest check: {path}")
            with np.load(path, allow_pickle=False) as motion:
                q, v = motion["qpos"], motion["qvel"]
                raw, history, delays = motion["raw_action"], motion["raw_action_history"], motion["delays"]
                steps = len(raw)
                joints = len(zenbot.DOF)
                if (steps != entry["ticks"] or not 0 < entry["stance_start"] < steps
                        or q.shape != (steps + 1, 7 + joints) or v.shape != (steps + 1, 6 + joints)
                        or raw.shape != (steps, joints) or history.shape != (2, joints)
                        or delays.shape != (5,) or np.any((delays < 3) | (delays > 7))
                        or float(motion["dt"]) != env.DT or float(motion["external_assist_newtons"]) != 0.):
                    raise ValueError(f"Expected consecutive unassisted 50 Hz states and delayed actions: {path}")
            # Context matches Sim.context for the nominal 43 kg model. The stored history is oldest first.
            context = np.asarray([.9, 0., *delays, 0.], dtype=np.float32)
            actions = jnp.asarray(np.concatenate((history, raw)), jnp.float32)
            target = pd_targets(actions[2:], actions[1:-1], actions[:-2], jnp.zeros_like(actions[0]), jnp.asarray(context))
            positions.append(q)
            velocities.append(v)
            targets.append(target)
            contexts.append(context)
            stance_start.append(entry["stance_start"])

        self.files = tuple(directory / entry["filename"] for entry in entries)
        self.qpos = jnp.asarray(np.stack(positions), jnp.float32)
        self.qvel = jnp.asarray(np.stack(velocities), jnp.float32)
        self.targets = jnp.stack(targets)
        self.contexts = jnp.asarray(np.stack(contexts))
        self.stance_start = jnp.asarray(stance_start, jnp.int32)
        self.max_window = self.targets.shape[1] - max(stance_start)

    def sample_windows(self, key, batch_size, window):
        """Return the seven consecutive-transition fields consumed by InstantAMP.wm_loss."""
        if batch_size < 1 or not 1 <= window <= self.max_window:
            raise ValueError(f"Expected a positive batch and a window of 1 to {self.max_window} steps")
        group_key, time_key = jax.random.split(key)
        count = 2 * len(self.files)
        groups = jax.random.permutation(group_key, count)[jnp.arange(batch_size) % count]
        clip, standing = jnp.divmod(groups, 2)
        boundary = self.stance_start[clip]
        # A window starting at the end of the rise may continue into the same clip's stance.
        low = jnp.where(standing, boundary, 0)
        high = jnp.where(standing, self.targets.shape[1] - window + 1, boundary)
        start = jax.random.randint(time_key, (batch_size,), low, high)
        ticks = start[:, None] + jnp.arange(window)
        rows = clip[:, None]
        return {
            "s_qpos": self.qpos[rows, ticks],
            "s_qvel": self.qvel[rows, ticks],
            "s1_qpos": self.qpos[rows, ticks + 1],
            "s1_qvel": self.qvel[rows, ticks + 1],
            "target": self.targets[rows, ticks],
            "xi": jnp.broadcast_to(self.contexts[rows], (batch_size, window, self.contexts.shape[-1])),
            "mask": jnp.ones((batch_size, window), jnp.float32),
        }
