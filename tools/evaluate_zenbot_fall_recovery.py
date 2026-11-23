"""Push a standing Zenbot in native MuJoCo, then score the same actor through impact and recovery."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from instantamp import env, zenbot
from instantamp.evaluate import fallen_starts
from instantamp.native_zenbot import ZenbotNative
from tools.evaluate_zenbot_source_getup import measure


def standing_pose(clip):
    with np.load(clip, allow_pickle=False) as data:
        return np.r_[data["root_pos"][0], data["root_quat"][0, [3, 0, 1, 2]], data["dof_pos"][0]]


def attempt(sim, pose, delay, impulse, *, from_floor=False):
    sim.reset(pose, np.zeros(29), delay)
    stance = []
    for _ in range(400 if from_floor else 100):
        load = sim.step()
        q = sim.data.qpos
        upright = .58 < q[2] < .78 and 1. - 2. * np.square(q[4:6]).sum() > .8660254
        stance.append(upright and load >= .5 and (not from_floor or np.linalg.norm(sim.data.qvel[:2]) < .15))
    if np.mean(stance[-50:]) < .90:
        return dict(started_upright=False, fell=False, recovered=False, held=False, kept_balance=False)
    sim.data.qvel[:2] += impulse
    q, v, loads = [], [], []
    for _ in range(700):
        loads.append(sim.step())
        q.append(sim.data.qpos.copy())
        v.append(sim.data.qvel.copy())
    q, v = np.asarray(q), np.asarray(v)
    tilt = 1. - 2. * np.square(q[:, 4:6]).sum(axis=-1)
    down = np.flatnonzero((q[:, 2] < .42) & (tilt < .65))
    if not len(down):
        final = measure(q, v, np.asarray(loads))
        return dict(started_upright=True, fell=False, recovered=False, held=False, kept_balance=final["held"])
    first = int(down[0])
    if len(q) - first < 100:
        return dict(started_upright=True, fell=True, kept_balance=False, recovered=False, held=False, impact_s=(first + 1) * env.DT)
    stats = measure(q[first:], v[first:], np.asarray(loads[first:]))
    return dict(started_upright=True, fell=True, kept_balance=False, impact_s=round((first + 1) * env.DT, 2),
                recovered=stats["rise"] is not None, rise_after_impact_s=stats["rise"],
                held=stats["held"], first_attempt=stats["first"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[932277, 1654679, 2718869, 3950477, 5969183])
    parser.add_argument("--from-floor", action="store_true", help="Have the actor recover from an AMP opening before the push")
    args = parser.parse_args()
    motions = Path(__file__).resolve().parents[1] / "data/zenbot23_walk_run_recovery"
    if args.from_floor:
        all_names, all_poses, _ = fallen_starts(motions / "recovery", zenbot.RESET_LIFT)
        openings = [(name, np.asarray(pose)) for (name, offset), pose in zip(all_names, all_poses) if offset == 0.]
    else:
        openings = [("stand_4s_120f.npz", standing_pose(motions / "walk_run/stand_4s_120f.npz"))]
    sim = ZenbotNative(motions / "recovery", args.checkpoint)
    directions = np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    cases = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for direction in directions:
            delay = rng.integers(3, 8, 5)
            impulse = direction * 2.5
            name, pose = openings[len(cases) % len(openings)]
            cases.append(dict(seed=seed, opening=name, impulse_mps=impulse.tolist(), delays=delay.tolist(),
                              **attempt(sim, pose, delay, impulse, from_floor=args.from_floor)))
    summary = {name: sum(row[name] for row in cases) for name in ("started_upright", "fell", "recovered", "held", "kept_balance")}
    summary["trials"] = len(cases)
    output = dict(start="floor" if args.from_floor else "standing_pose", checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), summary=summary, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
