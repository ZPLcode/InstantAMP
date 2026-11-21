"""Fit a local rise policy to physically executed sensor/action trajectories and their feedback gains."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from instantamp.g1_rise import features, LocalRise


def sensor_history(q, v, previous, points):
    """Reconstruct the five term-major actor observations from native MuJoCo trajectory samples."""
    root = Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
    heading = np.arctan2(root[:, 1, 0], root[:, 0, 0])
    rotation = Rotation.from_euler("z", -heading[:, None]).as_matrix() @ root
    local_points = ((points - q[:, None, :3]) @ root).reshape(len(q), 18)
    terms = (v[:, 3:6], np.c_[rotation[:, :, 0], rotation[:, :, 2]], q[:, 7:], v[:, 6:], previous, local_points)
    indices = np.maximum(np.arange(len(q))[:, None] + np.arange(-4, 1), 0)
    return np.concatenate([term[indices].reshape(len(q), -1) for term in terms], axis=1).astype(np.float32)


def fit(paths, action_weight=0.65, radius=2.0, gain_scale=2.0):
    if not paths:
        raise ValueError("At least one physical get-up demonstration is required")
    keys, poses, actions, gains = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            action = z["action"]
            q = z["qpos"][:-1]
            if q.shape != (len(action), 36) or action.shape[1:] != (29,):
                raise ValueError(f"Invalid native get-up action and state dimensions in {path}")
            previous = np.vstack((np.zeros(29), action[:-1]))
            obs = (
                z["actor_obs"]
                if "actor_obs" in z
                else sensor_history(q, z["qvel"][:-1], previous, z["sim_points"][:-1])
            )
            limit = min(len(action), 270)
            if "source_qpos" in z:
                lost = np.flatnonzero((q[:, 2] < 0.50) & (z["source_qpos"][:-1, 2] > 0.68))
                if len(lost):
                    limit = min(limit, int(lost[0]) + 1)
            key, pose = features(obs[:limit], action_weight)
            keys.append(np.asarray(key))
            poses.append(np.asarray(pose))
            actions.append(action[:limit].astype(np.float32))
        # Project the trajectory's fitted task-space gains onto the components observable by the actor.
        factors = path.with_name("smooth_feedback.npz")
        with np.load(factors, allow_pickle=False) as f:
            k = f["feedback"][:limit]
            gains.append(np.concatenate((k[:, :, 3:35], k[:, :, 38:70]), axis=-1).astype(np.float16))
    library = dict(
        keys=np.concatenate(keys),
        poses=np.concatenate(poses),
        actions=np.concatenate(actions),
        gain=np.concatenate(gains),
        settings=np.asarray([action_weight, radius, gain_scale], np.float32),
    )
    LocalRise.read(library)
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollouts", type=Path, nargs="+", help="Native rollout.npz files with neighbouring smooth_feedback.npz"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--action-weight", type=float, default=0.65)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--gain-scale", type=float, default=2.0)
    parser.add_argument(
        "--handoff-samples",
        type=Path,
        help="Optional NPZ of observed 570-value histories 'obs' around a successful balance hand-off",
    )
    parser.add_argument(
        "--handoff-override",
        type=float,
        nargs=4,
        metavar=("RADIUS", "DEPTH", "UPRIGHT", "INCREMENT"),
        help="Local hand-off calibration for those neighborhoods",
    )
    parser.add_argument(
        "--aligned-handoff",
        type=float,
        nargs=4,
        metavar=("UPRIGHT_COSINE", "ANKLE_SPREAD", "DEPTH", "INCREMENT"),
        help="Optional balance hand-off calibration when the torso is upright and both ankles are at similar heights",
    )
    args = parser.parse_args()
    if (args.handoff_samples is None) != (args.handoff_override is None):
        parser.error("--handoff-samples and --handoff-override must be supplied together")
    library = fit(sorted(args.rollouts), args.action_weight, args.radius, args.gain_scale)
    if args.handoff_samples is not None:
        with np.load(args.handoff_samples, allow_pickle=False) as samples:
            library["handoff_keys"] = np.asarray(features(samples["obs"], args.action_weight)[0])
        library["handoff_settings"] = np.asarray(args.handoff_override, dtype=np.float32)
        LocalRise.read(library)
    if args.aligned_handoff is not None:
        library["aligned_handoff"] = np.asarray(args.aligned_handoff, dtype=np.float32)
        LocalRise.read(library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        np.savez_compressed(f, **library)
    print(f"Exported {len(library['keys'])} local sensor/action examples from {len(args.rollouts)} trajectories")


if __name__ == "__main__":
    main()
