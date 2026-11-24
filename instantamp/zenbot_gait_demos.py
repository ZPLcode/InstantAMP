"""Zenbot walking and running windows from the weighted Motion Train locomotion bank."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import amp, zenbot, zenbot_source_gait as source_gait
from .rot import mat_to_quat, quat_to_mat


# The four transitions contain steady forward walking between the start and stop.
# Earlier motion_train banks use the two CMU walk takes instead.
STRAIGHT_CLIPS = {
    "walk": ("Male1_B1_StandToWalk", "Male1_B2_WalkToStand", "Male2_B1_StandToWalk",
             "Male2_B2_WalkToStand", "143_02_stageii", "143_03_stageii"),
    "run": ("run__C3_-_run_stageii",),
}

DIRECTION_CLIPS = {
    "male_walk__Walk_B4_-_Stand_to_Walk_Back_stageii": (0, -1),
    "droidupe1_side_left": (1, 1), "droidupe1_side_right": (1, -1),
    "droidupe1_turn_left": (2, 1), "droidupe1_turn_right": (2, -1),
}


def _body_velocities(clip):
    x, y, z, w = clip["root_quat"].T
    yaw = np.unwrap(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    vx, vy = clip["root_lin_vel"][:, :2].T
    return np.cos(yaw) * vx + np.sin(yaw) * vy, -np.sin(yaw) * vx + np.cos(yaw) * vy, yaw


def straight_window_starts(clip, frames, gait):
    """Return within-clip starts where the pelvis moves forward throughout the window."""
    target, tolerance, floor = {"walk": (1., .35, .45), "run": (2.5, .5, 1.7)}[gait]
    if frames < 1 or float(clip["fps"]) != 50.:
        raise ValueError("Straight gait windows require a positive length and 50 Hz motion")
    if len(clip["root_quat"]) < frames:
        return np.empty(0, dtype=np.int32)
    forward, lateral, headings = (sliding_window_view(v, frames) for v in _body_velocities(clip))
    steady = (forward.min(axis=1) > floor) & (np.abs(forward.mean(axis=1) - target) <= tolerance)
    straight = (np.abs(lateral.mean(axis=1)) <= .35) & (np.ptp(headings, axis=1) <= .40)
    return np.flatnonzero(steady & straight).astype(np.int32)


def source_command_speeds(directory):
    """Add the typical sustained speed of each original reverse, side-step and pivot take."""
    recorded = zenbot.DIRECTIONAL_COMMANDS.copy()
    for name, (axis, sign) in DIRECTION_CLIPS.items():
        with np.load(Path(directory) / (name + ".npz"), allow_pickle=False) as clip:
            fps = float(clip["fps"])
            forward, lateral, heading = _body_velocities(clip)
        length = min(round(2 * fps), len(heading))
        if fps != 50. or length < 2:
            raise ValueError(f"Expected at least two 50 Hz source frames in {name}")
        windows = sliding_window_view((forward, lateral, heading)[axis], length)
        speed = ((windows[:, -1] - windows[:, 0]) * fps / (length - 1)
                 if axis == 2 else windows.mean(axis=1))
        active = speed[sign * speed > (.25 if axis == 2 else .10)]
        if not len(active):
            raise ValueError(f"No sustained motion in the recorded direction of {name}")
        recorded[gait_command_index(name), axis] = np.median(active)
    return np.concatenate((zenbot.DIRECTIONAL_COMMANDS, recorded))


def directional_window_starts(clip, frames, name, *, match_speed=False, target=None):
    """Keep active segments of each named demonstration in its actual body-frame direction."""
    if frames < 2 or float(clip["fps"]) != 50.:
        raise ValueError("Directional gait windows require at least two frames at 50 Hz")
    if len(clip["root_quat"]) < frames:
        return np.empty(0, dtype=np.int32)
    axis, sign = DIRECTION_CLIPS[name]
    forward, lateral, heading = _body_velocities(clip)
    segment = sliding_window_view((forward, lateral, heading)[axis], frames)
    if axis == 2:
        rate = (segment[:, -1] - segment[:, 0]) * 50. / (frames - 1)
        moving = sign * rate > .25
    else:
        moving = sign * segment.mean(axis=-1) > .10
    if match_speed:
        # A fast pivot and a slow pivot have the same sign. Only teach this
        # command with recordings that also have compatible translation and yaw.
        headings = sliding_window_view(heading, frames)
        mean = np.column_stack((sliding_window_view(forward, frames).mean(axis=1),
                                sliding_window_view(lateral, frames).mean(axis=1),
                                (headings[:, -1] - headings[:, 0]) * 50. / (frames - 1)))
        if target is None:
            target = zenbot.DIRECTIONAL_COMMANDS[gait_command_index(name)]
        moving &= (np.abs(mean - target) <= np.array([.20, .20, .25])).all(axis=1)
    return np.flatnonzero(moving).astype(np.int32)


def gait_command_index(name):
    """Map an AMP clip to the command it demonstrates. Return None for other motions."""
    if name == "stand_4s_120f":
        return 0
    for index, gait in enumerate(("walk", "run"), start=1):
        if name in STRAIGHT_CLIPS[gait]:
            return index
    if name in DIRECTION_CLIPS:
        axis, sign = DIRECTION_CLIPS[name]
        return np.flatnonzero(np.sign(zenbot.DIRECTIONAL_COMMANDS[3:, axis]) == sign).item() + 3
    return None


def command_motion_windows(clip, name, frames, *, match_speed=False, targets=None):
    """Yield command indices and contiguous native reference windows with that motion."""
    group = gait_command_index(name)
    if group == 0:
        yield group, np.arange(max(0, len(clip["dof_pos"]) - frames + 1), dtype=np.int32)
    elif name in DIRECTION_CLIPS:
        target = None if targets is None else targets[group]
        yield group, directional_window_starts(clip, frames, name, match_speed=match_speed, target=target)
    else:
        # Start/stop, walk/run transitions and curved clips can contain steady straight sections.
        for index, gait in enumerate(("walk", "run"), 1):
            yield index, straight_window_starts(clip, frames, gait)


def reference_command_index(commands, labels):
    """Match the dominant translation to a demo, using a turn demo for pure yaw."""
    labels = jnp.asarray(labels)
    xy = commands[..., :2]
    axis = jnp.where(jnp.any(xy != 0., axis=-1), jnp.argmax(jnp.abs(xy), axis=-1), 2)
    target = jnp.take_along_axis(commands, axis[..., None], axis=-1)
    candidates = labels.T[axis]
    distance = jnp.where(target * candidates > 0., jnp.abs(candidates - target), jnp.inf)
    return jnp.where(target[..., 0] == 0., 0, jnp.argmin(distance, axis=-1))


class GlobalGaitDemos(amp.Demos):
    """Draw consecutive 50 Hz frames at fractional start times using Motion Train's clip weights."""

    def __init__(self, directory, sim, frames=source_gait.AMP_STEPS):
        def world_frame(state):
            pos, rotation, _, angular, joints, rates = state
            offsets = sim.key_positions(pos, rotation, joints) - pos[..., None, :]
            return jnp.concatenate((mat_to_quat(rotation), angular, joints, rates, offsets.reshape(len(pos), -1)), -1)

        super().__init__(directory, sim, frames, per_clip=True, frame_fn=world_frame)
        if self.clip_probabilities is None:
            raise ValueError(f"Missing Motion Train gait weights in {directory}/sampling.json")

    def sample(self, key, count):
        choose, time = jax.random.split(key)
        clip = jax.random.choice(choose, len(self.clip_lengths), (count,), p=self.clip_probabilities)
        # Reserve one control tick beyond the last sampled frame, as in Motion Train.
        start = jax.random.uniform(time, (count,)) * jnp.maximum(self.clip_lengths[clip] - 2, 0)
        return self.at_time(clip, start)

    def at_time(self, clip, start):
        first = self.clip_offsets[clip]
        index = first[:, None] + start.astype(jnp.int32)[:, None] + jnp.arange(self.window)
        last = first + self.clip_lengths[clip] + self.window - 2
        a, b = self.frames[index], self.frames[jnp.minimum(index + 1, last[:, None])]
        t = (start % 1.)[:, None, None]
        qa, qb = a[..., :4], b[..., :4]
        dot = jnp.sum(qa * qb, axis=-1, keepdims=True)
        qb = jnp.where(dot < 0., -qb, qb)
        angle = jnp.arccos(jnp.clip(jnp.abs(dot), 0., 1.))
        # sinc form of slerp is well defined for identical orientations.
        quat = ((1. - t) * jnp.sinc((1. - t) * angle / jnp.pi) * qa
                + t * jnp.sinc(t * angle / jnp.pi) * qb)
        angular, joints, rates, points = jnp.split(a[..., 4:] + t * (b[..., 4:] - a[..., 4:]),
                                                  [3, 3 + len(zenbot.DOF), 3 + 2 * len(zenbot.DOF)], axis=-1)
        return source_gait.amp_frame(quat_to_mat(quat), angular, joints, rates,
                                     points.reshape(*points.shape[:-1], len(zenbot.KEY_BODIES), 3))


class CommandGaitDemos(amp.Demos):
    """Sample independent AMP windows from the gait requested of each physical robot."""

    def __init__(self, directory, sim, frames):
        super().__init__(directory, sim, frames, per_clip=True)
        def reduced_motion(state):
            p, rotation, velocity, angular, q, qd = state
            neutral = q.mean(axis=0)
            return sim.amp_frame((p, rotation, velocity, angular, neutral + .25 * (q - neutral), .25 * qd), None)

        # Keep the original root trajectory and recompute the limb landmarks of the smaller steps.
        self.reduced_frames = amp.Demos(directory, sim, frames, per_clip=True, frame_fn=reduced_motion).frames
        labels = np.asarray(getattr(sim, "wm_command_speeds", zenbot.DIRECTIONAL_COMMANDS))
        self.labels = jnp.asarray(labels)
        kinds = len(zenbot.DIRECTIONAL_COMMANDS)
        command_tables = labels.reshape(-1, kinds, 3)
        groups = [[] for _ in labels]
        offset = 0
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path, allow_pickle=False) as clip:
                for variant, targets in enumerate(command_tables):
                    for group, starts in command_motion_windows(clip, path.stem, frames, match_speed=True, targets=targets):
                        groups[variant * kinds + group].extend(offset + starts)
                offset += len(clip["dof_pos"])
        counts = np.array([len(group) for group in groups], dtype=np.int32)
        if np.any(counts == 0) or offset != len(self.frames):
            raise ValueError(f"Missing AMP motion windows for commands {labels[counts == 0]}")
        self.command_counts = jnp.asarray(counts)
        self.command_offsets = jnp.asarray(counts.cumsum() - counts, jnp.int32)
        self.command_starts = jnp.asarray(np.concatenate(groups), jnp.int32)

    def sample_commands(self, key, commands, *, mismatch=False, reduced=False):
        group = reference_command_index(commands, self.labels)
        if mismatch:
            key, other = jax.random.split(key)
            kinds = len(zenbot.DIRECTIONAL_COMMANDS)
            # A different speed of the same action is still the same motion class.
            group = group // kinds * kinds + (group + jax.random.randint(other, group.shape, 1, kinds)) % kinds
        row = self.command_offsets[group] + jax.random.randint(key, group.shape, 0, self.command_counts[group])
        start = self.command_starts[row]
        frames = self.reduced_frames if reduced else self.frames
        return frames[start[:, None] + jnp.arange(self.window)]


def command_reset_bank(directory, bank, upright, directional, *, command_speeds=None):
    """Group upright motion states by the command used for the first gait segment."""
    targets = zenbot.DIRECTIONAL_COMMANDS if directional else zenbot.DIRECTIONAL_COMMANDS[:3]
    if command_speeds is not None:
        targets = command_speeds
    groups = [[] for _ in targets]
    offset = 0
    upright = np.asarray(upright)
    # load_bank concatenates all clips in sorted filename order.
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path) as clip:
            count = len(clip["root_pos"])
            name = path.stem
            kind = gait_command_index(name)
            if kind is None or kind >= len(groups):
                offset += count
                continue
            for group in range(kind, len(groups), len(zenbot.DIRECTIONAL_COMMANDS)):
                if kind == 0:
                    rows = np.arange(count)
                elif kind <= 2:
                    rows = straight_window_starts(clip, 10, "walk" if kind == 1 else "run")
                else:
                    rows = directional_window_starts(clip, 10, name, match_speed=command_speeds is not None,
                                                     target=targets[group])
                rows = rows + offset
                groups[group].extend(rows[upright[rows]])
            offset += count
    counts = np.array(list(map(len, groups)), np.int32)
    if np.any(counts == 0):
        raise ValueError(f"No upright motion resets for commands {targets[counts == 0].tolist()} in {directory}")
    indices = np.concatenate(groups)
    return {**{name: value[indices] for name, value in bank.items()},
            "counts": jnp.asarray(counts), "offsets": jnp.asarray(np.cumsum(counts) - counts, jnp.int32)}


def guided_gait_reset_bank(walk, stand_dir):
    """Start stand commands in a measured stance; include a quarter of cold starts for moving commands."""
    from .sim import load_bank

    stand = load_bank(stand_dir, walk["q"].shape[1], sample_clip_times=True)
    lengths, starts = np.asarray(stand["lengths"]), np.asarray(stand["offsets"])
    standing_rows = np.arange(len(stand["z"]), dtype=np.int32)
    group_rows = [standing_rows]
    for first, frames in zip(np.asarray(walk["offsets"])[1:], np.asarray(walk["counts"])[1:]):
        # Keep each original motion frame once and include at least one stance per recording.
        motion = len(standing_rows) + first + np.arange(frames, dtype=np.int32)
        n_cold = max(len(lengths), (int(frames) + 1) // 3)
        per_clip, extra = divmod(n_cold, len(lengths))
        cold = []
        for i, (start, length) in enumerate(zip(starts, lengths)):
            count = per_clip + int(i < extra)
            # Center each selection in a distinct temporal interval of this recording.
            cold.append(start + ((2 * np.arange(count) + 1) * length // (2 * count)))
        group_rows.append(np.concatenate([*cold, motion]))

    rows = np.concatenate(group_rows).astype(np.int32)
    counts = np.array(list(map(len, group_rows)), np.int32)
    fields = {name: jnp.concatenate((stand[name], value))[rows]
              for name, value in walk.items() if name not in ("counts", "offsets")}
    return {**fields, "counts": jnp.asarray(counts), "offsets": jnp.asarray(np.cumsum(counts) - counts, jnp.int32)}


class StraightGaitDemos:
    """Sample weighted clips, then uniformly sample their valid contiguous 50 Hz windows."""

    def __init__(self, directory, sim, frames, gait, directional=False):
        if gait not in STRAIGHT_CLIPS:
            raise ValueError(f"Unknown straight gait {gait!r}")
        directory = Path(directory)
        weights = json.loads((directory / "sampling.json").read_text())["weights"]
        features, starts, probabilities = [], [], []
        self.clip_names = []
        self.forward_starts = []
        offset = 0
        names = STRAIGHT_CLIPS[gait] + (tuple(DIRECTION_CLIPS) if directional and gait == "walk" else ())
        for name in names:
            if name not in weights:
                continue
            weight = weights[name]
            if not np.isfinite(weight) or weight < 0.:
                raise ValueError(f"Invalid gait sampling weight for {name}")
            if weight == 0.:
                continue
            with np.load(directory / (name + ".npz"), allow_pickle=False) as clip:
                if name in DIRECTION_CLIPS:
                    indices = directional_window_starts(clip, frames, name)
                else:
                    indices = straight_window_starts(clip, frames, gait)
                if not indices.size:
                    continue
                if clip["dof_pos"].shape != (len(clip["root_quat"]), sim.J):
                    raise ValueError(f"Incorrect number of joints or frames in {name}")
                rot = quat_to_mat(jnp.asarray(clip["root_quat"][:, [3, 0, 1, 2]]))
                state = (clip["root_pos"], rot, clip["root_lin_vel"], clip["root_ang_vel"],
                         clip["dof_pos"], clip["dof_vel"])
                features.append(sim.amp_frame(tuple(map(jnp.asarray, state))))
            starts.append(indices + offset)
            if name not in DIRECTION_CLIPS:
                self.forward_starts.append(starts[-1])
            offset += len(features[-1])
            self.clip_names.append(name)
            probabilities.append(weight)
        if not starts:
            raise ValueError(f"No valid straight {gait} windows of {frames} frames in {directory}")
        counts = np.array([len(indices) for indices in starts], dtype=np.int32)
        self.clip_counts = jnp.asarray(counts)
        self.offsets = jnp.asarray(np.cumsum(counts) - counts)
        self.starts = jnp.asarray(np.concatenate(starts))
        self.frames = jnp.concatenate(features)
        self.clip_probabilities = jnp.asarray(np.asarray(probabilities) / np.sum(probabilities))
        self.window = frames

    def sample(self, key, count):
        choose, time = jax.random.split(key)
        clip = jax.random.choice(choose, len(self.clip_names), (count,), p=self.clip_probabilities)
        candidate = self.offsets[clip] + jax.random.randint(time, (count,), 0, self.clip_counts[clip])
        start = self.starts[candidate]
        return self.frames[start[:, None] + jnp.arange(self.window)]

    def joint_angle_range(self, joints):
        """Central 90% of forward gait angles, counting each real frame only once."""
        if not self.forward_starts:
            raise ValueError("Forward gait labels require at least one forward motion window")
        indices = np.unique(np.concatenate(self.forward_starts)[:, None] + np.arange(self.window))
        angles = np.asarray(self.frames)[indices, 9:9+joints]
        return jnp.asarray(np.percentile(angles, [5., 95.], axis=0), dtype=self.frames.dtype)
