"""Compare G1 policies to the native AMP reference from 11 fixed fallen starts.

Example: python tools/evaluate_g1_style.py OLD.npz NEW.npz --out comparison.json --gif example.gif
The predefined times are the 30 Hz source frames of the curated subject1 get-ups.
All poses and reference trajectories used here come from the 29-DoF G1 retarget.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import env, nets
from instantamp.inference import load, policy_output
from instantamp.rot import mv3, quat_to_mat, tr3, yaw_mat
from instantamp.sim import Sim, unflatten
from instantamp.train import set_jax

TAKE = "fallAndGetUp1_subject1"
STARTS_30HZ = (458, 670, 845, 1058, 1262, 1395, 1584, 1808, 2088, 2301, 4873)
STEPS = env.GETUP_STEPS + round(1 / env.DT)
PROBE_SPEEDS = (0., 0.8)


def upright(qpos):
    rot = quat_to_mat(qpos[..., 3:7])
    return np.asarray(env.is_up(qpos[..., 2], rot))


def read_motion(path):
    with np.load(path) as data:
        if not np.isclose(float(data["fps"]) * env.DT, 1) or data["dof_pos"].shape[1] != env.J:
            raise ValueError(f"Expected a 50 Hz G1 reference in {path}")
        quat = data["root_quat"][:, [3, 0, 1, 2]]
        qpos = np.column_stack((data["root_pos"], quat, data["dof_pos"]))
        ang = mv3(tr3(quat_to_mat(quat)), data["root_ang_vel"])
        qvel = np.column_stack((data["root_lin_vel"], ang, data["dof_vel"]))
    return qpos, qvel


def segmented_starts(directory, count):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["kind"] != "g1_rise_clips" or not np.isclose(manifest["fps"] * env.DT, 1):
        raise ValueError(f"Expected a G1 segmented get-up manifest in {directory}")
    if count < 1 or not manifest["clips"]:
        raise ValueError("At least one diagnostic clip is required")
    candidates = []
    for clip in sorted(manifest["clips"], key=lambda c: c["file"]):
        with np.load(directory / clip["file"]) as data:
            quat = data["root_quat"][0, [3, 0, 1, 2]]
        vertical = np.asarray(quat_to_mat(quat))[2]
        if vertical[2] > 0.7:
            pose = "low_upright"
        elif abs(vertical[0]) >= abs(vertical[1]):
            pose = "supine" if vertical[0] > 0 else "prone"
        else:
            pose = "side"
        candidates.append({**clip, "start_pose": pose})
    # Balance takes first, then pose categories. Filename order breaks ties deterministically.
    selected, take_counts, pose_counts = [], {}, {}
    while candidates and len(selected) < count:
        clip = min(candidates, key=lambda c: (take_counts.get(c["source_take"], 0), pose_counts.get(c["start_pose"], 0)))
        candidates.remove(clip)
        take_counts[clip["source_take"]] = take_counts.get(clip["source_take"], 0) + 1
        pose_counts[clip["start_pose"]] = pose_counts.get(clip["start_pose"], 0) + 1
        selected.append(clip)
    paths, initial_velocities = [], []
    for clip in selected:
        qpos, qvel = read_motion(directory / clip["file"])
        qpos[:, :2] -= qpos[0, :2].copy()
        # The stored clip ends after reaching stance. All pose metrics end at first rise.
        paths.append(qpos[np.minimum(np.arange(env.GETUP_STEPS + 1), len(qpos) - 1)])
        initial_velocities.append(qvel[0])
    return selected, np.stack(paths), np.stack(initial_velocities)


def reference_starts(take, clips_dir=None, count=12):
    if clips_dir:
        cases, paths, initial_velocities = segmented_starts(clips_dir, count)
    else:
        qpos, qvel = read_motion(take)
        starts = np.rint(np.array(STARTS_30HZ) / (30 * env.DT)).astype(int)
        paths = qpos[starts[:, None] + np.arange(env.GETUP_STEPS + 1)[None]]
        paths[:, :, :2] -= qpos[starts, None, :2]
        cases = [{"source_take": take.stem, "source_frame_30hz": frame} for frame in STARTS_30HZ]
        initial_velocities = qvel[starts]
    q0 = paths[:, 0].copy()
    q0[:, 2] += 0.02
    if upright(q0).any() or not upright(paths[:, :env.GETUP_STEPS + 1]).any(axis=1).all():
        raise ValueError("The fixed starts must be down and reach upright in the supplied G1 reference")
    probes = np.tile(np.r_[0., 0., env.DEFAULT_HEIGHT, 1., 0., 0., 0., env.DEFAULT_Q], (len(PROBE_SPEEDS), 1))
    velocities = np.zeros((len(probes), initial_velocities.shape[1]))
    return cases, paths, jnp.asarray(np.vstack((q0, probes)), dtype=jnp.float32), jnp.asarray(np.vstack((initial_velocities, velocities)), dtype=jnp.float32)


def fixed_state(sim, qpos, qvel, standing, commands, key=None):
    """Initialize reproducible physical rollouts with nominal dynamics and specified world-frame poses."""
    state = sim.init(jax.random.PRNGKey(42) if key is None else key)
    world = state.world._replace(body_ipos=jnp.broadcast_to(sim.m.body_ipos, state.world.body_ipos.shape),
                                 geom_friction=jnp.broadcast_to(sim.m.geom_friction, state.world.geom_friction.shape),
                                 bias=jnp.zeros_like(state.world.bias))
    cmd = state.cmd._replace(vel=commands, is_heading=jnp.zeros_like(standing), is_standing=jnp.all(commands == 0, axis=-1),
                             time_left=jnp.full_like(state.cmd.time_left, 20.))
    frame = sim.obs_frame(unflatten(qpos, qvel), state.a_last, commands, None)
    return state._replace(data=state.data.replace(qpos=qpos, qvel=qvel), world=world,
                          obs=jnp.tile(frame, (1, sim.history)), cmd=cmd, c_obs=commands,
                          was_up=standing, down=~standing, t=jnp.zeros_like(state.t))


def rollout_function(sim, qpos, qvel, predict=None):
    n_fallen = len(qpos) - len(PROBE_SPEEDS)
    commands = jnp.zeros((len(qpos), 3)).at[n_fallen:, 0].set(jnp.array(PROBE_SPEEDS))
    standing = jnp.arange(len(qpos)) >= n_fallen

    def rollout(pi, norm, ode, obs_clip):
        state = fixed_state(sim, qpos, qvel, standing, commands)

        def step(state, _):
            if predict is None:
                output = policy_output(pi, norm, state.obs, obs_clip)
            else:
                output = predict(pi, norm, state.obs, obs_clip)
            action = nets.ode_step(output, state.a_ode, *ode) if ode is not None else output
            state, transition = sim.step(state, action)
            return state, (transition.qpos1, transition.qvel1[..., :2], transition.terminated | transition.truncated)

        _, (poses, velocity_xy, ended) = jax.lax.scan(step, state, None, length=STEPS)
        return (jnp.concatenate((qpos[None], poses)).swapaxes(0, 1), velocity_xy.swapaxes(0, 1), ended.T)

    return jax.jit(rollout, static_argnums=(2,))


def first_episode(path, ended):
    # A reset or divergence ends scoring. Freeze the last valid pose for fixed-time comparisons.
    stop = np.flatnonzero(ended)
    return path[:1 + int(stop[0])] if len(stop) else path


def pose_errors(qpos, points, demo, demo_points):
    delta = points - demo_points
    return {"wrist_rmse_cm": float(100 * np.sqrt((delta[:, 2:] ** 2).sum(-1).mean())),
            "ankle_rmse_cm": float(100 * np.sqrt((delta[:, :2] ** 2).sum(-1).mean())),
            "joint_rmse_rad": float(np.sqrt(((qpos[:, 7:] - demo[:, 7:]) ** 2).mean()))}


def compare_trajectories(sim, cases, refs, paths, velocities, ended):
    def landmarks(qpos):
        rot = quat_to_mat(qpos[..., 3:7])
        pos = qpos[..., :3]
        world = sim.key_positions(pos, rot, qpos[..., 7:])
        return np.asarray(mv3(tr3(yaw_mat(rot))[..., None, :, :], world - pos[..., None, :]))

    points, ref_points = landmarks(paths[:len(refs)]), landmarks(refs)
    records = []
    for i, demo in enumerate(refs):
        path = first_episode(paths[i], ended[i])
        up = upright(path)
        threshold = np.flatnonzero(up[:env.GETUP_STEPS + 1])
        demo_step = int(np.flatnonzero(upright(demo))[0])
        valid_steps = np.minimum(np.arange(demo_step + 1), len(path) - 1)
        row = {**cases[i], "demo_rise_s": demo_step * env.DT,
               "rose_in_5s": bool(len(threshold)), "upright_last_second": bool(len(up) == STEPS + 1 and up[-round(1 / env.DT):].all()),
               "fixed_time": pose_errors(path[valid_steps], points[i, valid_steps], demo[:demo_step + 1], ref_points[i, :demo_step + 1])}
        if len(threshold):
            rise = int(threshold[0])
            phase = np.linspace(0, 1, 51)
            policy_idx, demo_idx = np.rint(phase * rise).astype(int), np.rint(phase * demo_step).astype(int)
            row["policy_rise_s"] = rise * env.DT
            row["rise_time_abs_error_s"] = abs(rise - demo_step) * env.DT
            row["phase"] = pose_errors(path[policy_idx], points[i, policy_idx], demo[demo_idx], ref_points[i, demo_idx])
        records.append(row)

    probes = []
    for i, speed in enumerate(PROBE_SPEEDS, len(refs)):
        path = first_episode(paths[i], ended[i])
        # Ignore the first second of acceleration for tracking. Count early falls explicitly.
        steps = len(path) - 1
        sample = slice(round(1 / env.DT), steps)
        rot = yaw_mat(quat_to_mat(path[1:, 3:7]))[sample, :2, :2]
        local_xy = np.asarray(mv3(tr3(rot), velocities[i, sample]))
        error = np.linalg.norm(local_xy - [speed, 0.], axis=1)
        probes.append({"command_vx_m_s": speed, "survived_6s": steps == STEPS, "scored_steps": len(error),
                       "mean_tracking_error_m_s": float(error.mean()) if len(error) else None})
    return records, probes


def summarize(records):
    matched = [r for r in records if r["rose_in_5s"]]

    def med_errors(rows, key):
        if not rows:
            return None
        return {name: float(np.median([r[key][name] for r in rows])) for name in rows[0][key]}

    return {"starts": len(records), "rose_in_5s": len(matched), "upright_last_second": sum(r["upright_last_second"] for r in records),
            "median_fixed_time": med_errors(records, "fixed_time"), "median_phase": med_errors(matched, "phase"),
            **{key: float(np.median([r[key] for r in matched])) if matched else None
               for key in ("demo_rise_s", "policy_rise_s", "rise_time_abs_error_s")}}


def preview(output, sequences, finish):
    import mujoco
    from PIL import Image, ImageDraw
    from tools.render import font, visual_model

    model = visual_model()
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type, cam.azimuth, cam.elevation, cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, 125, -14, 2.45
    cam.lookat[:] = [0.12, 0.06, 0.55]
    width, height = 336, 240
    frames = []
    label = font(15)
    with mujoco.Renderer(model, height, width) as renderer:
        for t in range(0, finish + 1, 2):
            image = Image.new("RGB", ((width + 2) * len(sequences) - 2, height + 52), "#15222a")
            draw = ImageDraw.Draw(image)
            for i, (name, path) in enumerate(sequences):
                x = i * (width + 2)
                pose = path[min(t, len(path) - 1)]
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                image.paste(Image.fromarray(renderer.render()), (x, 26))
                draw.text((x + 8, 2), name, fill="white", font=label)
                caption = "episode ended" if t >= len(path) else f"{t * env.DT:.2f} s    pelvis {pose[2]:.2f} m"
                draw.text((x + 8, height + 28), caption, fill="white", font=label)
            frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--reference", type=Path, default=Path("data/recovery/lafan1") / f"{TAKE}.npz")
    parser.add_argument("--clips-dir", type=Path, help="Evaluate a fixed sample of a segmented bank instead of the 11 subject1 starts")
    parser.add_argument("--count", type=int, default=12, help="Maximum number of diagnostic clips, balanced across source takes and starting poses")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gif", type=Path, help="Show the same supine recovery for each policy, in real time")
    args = parser.parse_args()
    set_jax(".venv/cache/jax")
    cases, refs, qpos, qvel = reference_starts(args.reference, args.clips_dir, args.count)
    sim = Sim(len(qpos), reset_floor=0., push=False, prob_shove=0., random_start=False, obs_noise=False)
    sim.obs_noise = np.zeros_like(sim.obs_noise)
    rollout = rollout_function(sim, qpos, qvel)
    results = {}
    panels = [("G1 AMP reference", refs[0])]
    finish = int(np.flatnonzero(upright(refs[0]))[0])
    for label, checkpoint in (("baseline", args.baseline), ("candidate", args.candidate)):
        if checkpoint is None:
            continue
        cfg = json.loads((checkpoint.parent.parent / "config.json").read_text())
        if cfg.get("robot", "g1") != "g1":
            raise ValueError(f"{checkpoint}: expected a G1 policy")
        paths, velocities, ended = map(np.asarray, rollout(*load(checkpoint), cfg.get("obs_clip", 10.)))
        records, probes = compare_trajectories(sim, cases, refs, paths, velocities, ended)
        meta = json.loads(checkpoint.with_suffix(".json").read_text())
        results[label] = {"checkpoint": str(checkpoint), "iteration": meta["it"], "summary": summarize(records),
                          "locomotion_probes": probes, "trials": records}
        panels.append((f"{label.title()}    iteration {meta['it']}", first_episode(paths[0], ended[0])))
        if records[0]["rose_in_5s"]:
            finish = max(finish, round(records[0]["policy_rise_s"] / env.DT))
        else:
            finish = env.GETUP_STEPS
        print(label, json.dumps(results[label]["summary"]), flush=True)
    report = {"reference": str(args.clips_dir or args.reference), "initial_lift_m": 0.02, "dt_s": env.DT,
              "protocol": "Fixed same native G1 fallen states and velocities, nominal physics, standing command, no pushes or noise. "
                          "Success is the first up threshold before reset within 5 s. The final hold requires upright throughout seconds 5–6. "
                          "Fixed-time errors span demo onset to its first up threshold, freezing the last valid pre-reset pose on failure. "
                          "Phase errors use 51 equally spaced fractions of each trajectory's own onset-to-rise. Wrists and ankles are in the pelvis heading frame. "
                          "Two separate nominal upright starts probe standing and walking for 6 s.",
              **results}
    if args.candidate:
        shared = [i for i, (a, b) in enumerate(zip(results["baseline"]["trials"], results["candidate"]["trials"]))
                  if a["rose_in_5s"] and b["rose_in_5s"]]
        report["common_rises"] = {label: summarize([result["trials"][i] for i in shared]) for label, result in results.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.gif:
        preview(args.gif, panels, min(len(refs[0]) - 1, finish + 25))


if __name__ == "__main__":
    main()
