"""Follow a timed G1 get-up in native physical simulation, using GPU lookahead.

Use --clip PATH for any segmented native 50 Hz clip. Use --fixed for the eleven
published comparison starts. World poses are initialized once at the reference
start with a 2 cm lift. Only the joint motors are commanded after initialization.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from instantamp import env, nets
from instantamp.inference import load, policy_output
from instantamp.reference_mpc import PlanningConfig, build_planner, servo_target
from instantamp.sim import Sim, build_scene, unflatten
from instantamp.train import set_jax
from tools.evaluate_g1_style import STARTS_30HZ, TAKE, preview
from tools.score_g1_tracking import load_native_reference, score_tracking

# End frames of the same source segments used in legged_lab's 11-clip recovery benchmark.
ENDS_30HZ = (565, 785, 936, 1140, 1320, 1461, 1684, 1916, 2210, 2391, 4962)


def read_cases(clip, fixed, bank):
    if fixed:
        p = Path("data/recovery/lafan1") / (TAKE + ".npz")
        q, v = load_native_reference(p)
        paths = []
        for start, end in zip(STARTS_30HZ, ENDS_30HZ):
            interval = slice(round(start / (30 * env.DT)), round(end / (30 * env.DT)))
            paths.append((f"{TAKE}_{start}_{end}", q[interval], v[interval]))
    else:
        files = [clip] if clip else sorted(bank.glob("*.npz"))
        if not files:
            raise ValueError("No 50 Hz get-up clips found")
        paths = [(f.stem, *load_native_reference(f)) for f in files]
    result = []
    for name, q, v in paths:
        q = q.copy()
        if len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError(f"Invalid reference: {name}")
        q[:, :2] -= q[0, :2].copy()
        up = (q[:, 2] > env.UP_HEIGHT) & (2 * (q[:, 4:6] ** 2).sum(1) < env.UP_TILT)
        if up[0] or not up[-1]:
            raise ValueError(f"Expected a fallen start and standing end in {name}")
        result.append((name, q, v))
    return result


def run(cases, checkpoint, cfg, seed, seconds, support_friction):
    count = len(cases)
    steps = max(round(seconds / env.DT), max(len(q) - 1 for _, q, _ in cases) + round(1 / env.DT))
    ends = np.array([len(q) - 1 for _, q, _ in cases])
    length = steps + cfg.horizon + cfg.stride + 1
    reference = np.stack([q[np.minimum(np.arange(length), len(q) - 1)] for _, q, _ in cases])
    model = build_scene(g1_support_friction=support_friction)
    # Differentiate the commanded positions using MuJoCo's free-joint quaternion convention.
    vel = np.zeros((count, length, model.nv))
    for i in range(count):
        for j in range(ends[i]):
            mujoco.mj_differentiatePos(model, vel[i, j], env.DT, reference[i, j], reference[i, j + 1])
    predictions = Sim(count * cfg.population, reset_floor=0, push=False, obs_noise=False,
                      random_start=False, g1_support_friction=support_friction)
    solve, center = build_planner(predictions, reference, vel, cfg)
    robots = [mujoco.MjData(model) for _ in cases]
    for i, d in enumerate(robots):
        d.qpos[:] = reference[i, 0]
        d.qpos[2] += .02
        d.qvel[:] = cases[i][2][0]
        mujoco.mj_forward(model, d)

    actor, norm, ode = load(checkpoint)
    if ode is None:
        raise ValueError("Expected a G1 baseline checkpoint with an ODE actor")
    stand = jax.jit(lambda h, a: nets.ode_step(policy_output(actor, norm, h), a, *ode))
    frame = jax.jit(lambda q, v, a: env.obs_frame(unflatten(q, v), a, jnp.zeros((count, 3))))
    physical_ctrl = jax.jit(env.actuator_ctrl)
    def read(field):
        return np.stack([getattr(d, field) for d in robots])
    q, v = read("qpos"), read("qvel")
    action = np.zeros((count, env.J), np.float32)
    history = np.tile(np.asarray(frame(q, v, action)), (1, env.HIST))
    positions, speeds, targets, motors = [q], [v], [], []
    handoff = np.full(count, -1)
    key = jax.random.PRNGKey(seed)
    for tick in range(0, steps, cfg.stride):
        if np.any(handoff < 0):
            key, draw = jax.random.split(key)
            offsets, center = solve(q, v, read("qacc_warmstart"), np.int32(tick), center, draw)
            offsets = np.asarray(offsets)
        else:
            offsets = np.zeros((cfg.stride, count, env.J))
        for j in range(min(cfg.stride, steps - tick)):
            t = tick + j
            up = (q[:, 2] > env.UP_HEIGHT) & (2 * (q[:, 4:6] ** 2).sum(1) < env.UP_TILT)
            handoff[(handoff < 0) & (t >= ends) & up] = t
            target = servo_target(q[:, 7:], v[:, 6:], reference[:, t + 1, 7:], vel[:, t + 1, 6:], offsets[j], cfg)
            if (handoff >= 0).any():
                stable = np.asarray(stand(history, action))
                target = np.where((handoff >= 0)[:, None], env.DEFAULT_Q + env.ACTION_SCALE * stable, target)
            action = ((target - env.DEFAULT_Q) / env.ACTION_SCALE).astype(np.float32)
            sub_controls = []
            for _ in range(env.DECIMATION):
                ctrl = np.asarray(physical_ctrl(q[:, 7:], v[:, 6:], target))
                for d, u in zip(robots, ctrl):
                    d.ctrl[:] = u
                    mujoco.mj_step(model, d)
                sub_controls.append(ctrl)
                q, v = read("qpos"), read("qvel")
            history = np.column_stack((history[:, env.FRAME_DIM:], np.asarray(frame(q, v, action))))
            positions.append(q)
            speeds.append(v)
            targets.append(target)
            motors.append(sub_controls)
    return dict(qpos=np.stack(positions, 1), qvel=np.stack(speeds, 1), target=np.stack(targets, 1),
                ctrl=np.asarray(motors).transpose(2, 0, 1, 3), reference=reference[:, :steps + 1],
                reference_end=ends, balance_start=handoff, names=np.array([x[0] for x in cases]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", type=Path)
    source.add_argument("--fixed", action="store_true")
    source.add_argument("--bank", type=Path)
    p.add_argument("--balance", type=Path, default=Path("runs/g1_4096_20261120_01/ckpt/0011.npz"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=768)
    p.add_argument("--seconds", type=float, default=6.)
    p.add_argument("--batch", type=int, default=11, help="Clips per planning batch")
    p.add_argument("--baseline-contacts", action="store_true", help="Use the original frictionless non-foot contacts")
    p.add_argument("--gif", action="store_true")
    for f in dataclasses.fields(PlanningConfig):
        if isinstance(f.default, bool):
            p.add_argument("--" + f.name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = p.parse_args()
    if args.batch <= 0 or not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("batch size and duration must be positive")
    set_jax(".venv/cache/jax")
    cfg = PlanningConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PlanningConfig)})
    cases = read_cases(args.clip, args.fixed, args.bank)
    args.out.mkdir(parents=True, exist_ok=False)
    tic = time.monotonic()
    scores = []
    model = build_scene(g1_support_friction=not args.baseline_contacts)
    for lower in range(0, len(cases), args.batch):
        chunk = cases[lower:lower + args.batch]
        result = run(chunk, args.balance, cfg, args.seed + lower, args.seconds, not args.baseline_contacts)
        np.savez_compressed(args.out / f"trajectories_{lower:03d}.npz", **result)
        for i, (name, raw, _) in enumerate(chunk):
            score = score_tracking(model, raw, result["qpos"][i])
            score["name"] = name
            score["balance_start_s"] = float(result["balance_start"][i] * env.DT) if result["balance_start"][i] >= 0 else None
            scores.append(score)
        print(f"{len(scores)}/{len(cases)} physical trials complete", flush=True)
        if lower == 0 and args.gif:
            preview(args.out / "comparison.gif", [("G1 reference", chunk[0][1]), ("Physical tracking", result["qpos"][0])],
                    min(len(chunk[0][1]) - 1, round((scores[0]["reference_rise_s"] + .55) / env.DT)))
    median = {k: float(np.median([r["rise_and_settle"][k] for r in scores])) for k in scores[0]["rise_and_settle"]}
    errors = [abs(r["rise_time_error_s"]) for r in scores if r["rise_time_error_s"] is not None]
    summary = dict(cases=len(scores), rose=sum(r["rose_by_deadline"] for r in scores),
                   held=sum(r["final_half_second_upright"] for r in scores), median_errors=median,
                   median_abs_rise_time_error_s=float(np.median(errors)) if errors else None,
                   max_abs_rise_time_error_s=max(errors, default=None))
    output = {**summary, "config": dataclasses.asdict(cfg), "balance_policy": str(args.balance),
              "support_friction": not args.baseline_contacts, "seed": args.seed, "dt_s": env.DT,
              "elapsed_s": time.monotonic() - tic, "trials": scores}
    (args.out / "scores.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
