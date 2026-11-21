"""Score eleven MuJoCo floor-start rollouts with the original, frozen LeggedLab discriminator."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import env, support_discriminator, zenbot
from instantamp.sim import Sim, unflatten


def score(run, traces, prior, seconds):
    config = json.loads((run / "config.json").read_text())
    if config["robot"] != zenbot.MIXED_TASK or not np.isfinite(seconds) or seconds < 10 * env.DT:
        raise ValueError("Expected the mixed Zenbot task and at least ten frames")
    floor = Path(config["floor_dir"])
    sim = Sim(1, robot=zenbot.MIXED_TASK, walk_dir=config["walk_dir"], floor_dir=floor,
              random_start=False, reset_floor=0., physics=False)
    fixed = support_discriminator.load_legged_prior(prior)
    features = jax.jit(lambda q, v: sim.amp_support_frame(unflatten(q, v), with_native=True)[..., :73])
    logits = jax.jit(lambda x: support_discriminator.legged_logit(fixed, x))

    def summarize(d):
        if not len(d):
            return {"windows": 0}
        reward = np.maximum(1. - .25 * np.square(d - 1.), 0.)
        return {"windows": len(d), "median_logit": float(np.median(d)), "zero_reward_fraction": float((reward == 0.).mean()),
                "mean_source_style": float(reward.mean())}

    def population(clips):
        poses = [q for _, q, _ in clips]
        feat = np.asarray(features(np.concatenate(poses), np.concatenate([v for _, _, v in clips])))
        windows, low, counts = [], [], []
        for f, q in zip(np.split(feat, np.cumsum([len(q) for q in poses])[:-1]), poses):
            if len(f) < 10:
                raise ValueError("A floor-start rollout has fewer than ten complete 50 Hz samples")
            windows.append(f[np.arange(len(f)-9)[:, None] + np.arange(10)])
            low.append(q[9:, 2] < zenbot.UP_HEIGHT)
            counts.append(len(windows[-1]))
        d = np.asarray(logits(np.concatenate(windows)))
        by_clip = {}
        for (name, _, _), values, below in zip(clips, np.split(d, np.cumsum(counts)[:-1]), low):
            by_clip[name] = {"all": summarize(values), "low_pelvis": summarize(values[below])}
        return {"all": summarize(d), "low_pelvis": summarize(d[np.concatenate(low)]), "clips": by_clip}

    demos, policy = [], []
    n = round(seconds / env.DT)
    for motion in sorted(floor.glob("*.npz")):
        trace = traces / ("recover_to_walk_" + motion.stem + "_0.npz")
        with np.load(motion) as z, np.load(trace) as t:
            if float(z["fps"]) != 50 or float(t["fps"]) != 50:
                raise ValueError("The original discriminator uses contiguous 50 Hz windows")
            quat = z["root_quat"]
            q = np.column_stack((z["root_pos"], quat[:, [3, 0, 1, 2]], z["dof_pos"]))
            v = np.column_stack((z["root_lin_vel"], Rotation.from_quat(quat).inv().apply(z["root_ang_vel"]), z["dof_vel"]))
            demos.append((motion.stem, q, v))
            policy.append((motion.stem, t["qpos"][:n], t["qvel"][:n]))
    if not demos:
        raise ValueError("No recovery demonstrations found in the run's floor bank")
    return {"prior": str(prior), "traces": str(traces), "policy_duration_s": seconds,
            "low_pelvis_threshold_m": zenbot.UP_HEIGHT, "demonstrations": population(demos), "policy": population(policy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--seconds", type=float, default=4.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.run, args.traces, args.prior, args.seconds)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {g: v for g, v in result[k].items() if g != "clips"}
                      for k in ("demonstrations", "policy")}, indent=2))
