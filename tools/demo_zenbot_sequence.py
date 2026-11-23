"""Run one native MuJoCo episode through recovery, standing, locomotion and stopping."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax.numpy as jnp
import numpy as np

from instantamp import zenbot
from tools.evaluate_zenbot_directions import COMMANDS, heading_kinematics
from tools.evaluate_zenbot_getup import DT, evaluate
from tools.play_zenbot_getup import ROOT, ZenbotPlayer


def record_video(path, poses, commands, names, stage_index, model_xml=None):
    import mujoco
    from PIL import Image, ImageDraw

    from tools.render import FPS, H, W, font, visual_model

    model = (visual_model("zenbot23_getup", ROOT / "data/zenbot23_getup/meshes") if model_xml is None
             else mujoco.MjModel.from_xml_path(str(model_xml)))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 120.0, -14.0
    look = poses[0, :2].copy()
    title, subtitle = font(20), font(15)
    header = 62
    with mujoco.Renderer(model, H, W) as renderer, TemporaryDirectory(dir=path.parent) as tmp:
        movie = Path(tmp) / "sequence.mp4"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H + header}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            for i in range(0, len(commands), round(1 / FPS / DT)):
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                look = 0.8 * look + 0.2 * poses[i, :2]
                camera.lookat[:] = [*look, 0.5]
                renderer.update_scene(data, camera=camera)
                frame = Image.new("RGB", (W, H + header), (28, 35, 43))
                frame.paste(Image.fromarray(renderer.render()), (0, header))
                draw = ImageDraw.Draw(frame)
                vx, vy, wz = commands[i]
                draw.text(
                    (12, 4),
                    f"{i * DT:05.2f} s    {names[stage_index[i]].replace('_', ' ').title()}",
                    fill="white",
                    font=title,
                )
                draw.text(
                    (12, 34),
                    f"Command   forward {vx:+.1f} m/s    left {vy:+.1f} m/s    turn {wz:+.1f} rad/s",
                    fill=(231, 235, 239),
                    font=subtitle,
                )
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed to encode the recorded Zenbot sequence")
        movie.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path, help="Directory for summary.json and requested recording files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trial", type=int, default=0, help="Recovery opening index")
    p.add_argument("--cold-start", action="store_true", help="Start the fallen robot at rest before adding the test perturbation")
    p.add_argument(
        "--stage-seconds", type=float, default=6.0, help="Duration per locomotion/stop command; minimum 2"
    )
    p.add_argument(
        "--save-trace", action="store_true", help="Save physical state, contact loads and velocity commands at 50 Hz"
    )
    p.add_argument("--video", action="store_true", help="Render an original-speed MP4; requires meshes and ffmpeg")
    args = p.parse_args()
    if not np.isfinite(args.stage_seconds) or args.stage_seconds < 2.0 or args.seed < 0:
        p.error("Use a nonnegative seed and finite --stage-seconds >= 2")
    player = ZenbotPlayer(args.checkpoint, trial=args.trial, seed=args.seed, visuals=False)
    native = player.native
    if not native.command_gait or native.native_tracking is not None:
        p.error("Expected a single command-conditioned actor")
    move_steps = round(args.stage_seconds / DT)
    # The first 12 seconds have the same zero command as the independent floor evaluation.
    stages = [
        ("get_up", COMMANDS["still"], 400),
        ("stand", COMMANDS["still"], 200),
        ("walk", COMMANDS["forward"], move_steps),
        ("run", (2.5, 0.0, 0.0), move_steps),
        ("stop", COMMANDS["still"], move_steps),
        ("back", COMMANDS["back"], move_steps),
        ("left", COMMANDS["left"], move_steps),
        ("right", COMMANDS["right"], move_steps),
        ("turn_left", COMMANDS["turn_left"], move_steps),
        ("turn_right", COMMANDS["turn_right"], move_steps),
        ("final_stop", COMMANDS["still"], move_steps),
    ]
    floor_steps = stages[0][2] + stages[1][2]
    floor, initial = evaluate(player, args.trial, args.seed, floor_steps, -1, cold_start=args.cold_start)
    times = [n for _, _, n in stages]
    names = [name for name, _, _ in stages]
    stage_index = np.repeat(np.arange(len(stages)), times)
    commands = np.repeat([cmd for _, cmd, _ in stages], times, axis=0)
    total = len(commands)
    poses = np.empty((total + 1, native.model.nq))
    velocities = np.empty((total + 1, native.model.nv))
    loads = np.empty((total, len(native.support_load)))
    poses[: floor_steps + 1], velocities[: floor_steps + 1], loads[:floor_steps] = (
        initial["qpos"],
        initial["qvel"],
        initial["support_load_bw"],
    )
    for i in range(floor_steps, total):
        if stage_index[i] != stage_index[i - 1]:
            native.command = jnp.asarray(commands[i], dtype=jnp.float32)[None]
        native.step()
        poses[i + 1], velocities[i + 1], loads[i] = native.data.qpos, native.data.qvel, native.support_load

    rows = []
    start = 0
    for name, target, count in stages:
        end = start + count
        q = poses[start : end + 1]
        motion, yaw = heading_kinematics(q)
        up = 1.0 - 2.0 * np.square(q[:, 4:6]).sum(1)
        upright = (q[:, 2] > zenbot.UP_HEIGHT) & (up > 0.8)
        tail = slice(-round(2 / DT), None)
        rows.append(
            {
                "stage": name,
                "start_s": start * DT,
                "end_s": end * DT,
                "command": target,
                "mean_last_2s": motion[tail].mean(0).tolist(),
                "upright_fraction_last_2s": float(upright[tail].mean()),
                "upright_all_last_2s": bool(upright[tail].all()),
                "total_yaw_rad": float(yaw[-1] - yaw[0]),
                "pelvis_max_departure_m": float(np.linalg.norm(q[:, :2] - q[0, :2], axis=1).max()),
                "min_height_m": float(q[:, 2].min()),
                "fell_after_initial_rise": bool(((q[:, 2] < 0.42) | (up < 0.55)).any())
                if start >= floor_steps
                else None,
            }
        )
        start = end
    report = {
        "checkpoint": str(args.checkpoint),
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "trial": args.trial,
        "fps": 1 / DT,
        "duration_s": total * DT,
        "continuous": True,
        "cold_start": args.cold_start,
        "coordinate_order": ["forward_mps", "left_mps", "counterclockwise_radps"],
        "ground": floor,
        "stages": rows,
    }
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.save_trace:
        np.savez_compressed(
            args.out / "sequence.npz",
            fps=1 / DT,
            qpos=poses,
            qvel=velocities,
            support_load_bw=loads,
            command=commands,
            stage_index=stage_index,
            stage_names=names,
            sha256=report["sha256"],
        )
    if args.video:
        record_video(args.out / "sequence.mp4", poses, commands, names, stage_index)
    print(f"Recorded {total * DT:.2f} simulated seconds from {floor['clip']}: {args.out}")


if __name__ == "__main__":
    main()
