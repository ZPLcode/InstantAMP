"""Render all floor-start trials beside the source at their original speed."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from instantamp import zenbot
from instantamp.sim import build_scene
from tools.evaluate_zenbot_walk_recovery import limb_clearances
from tools.render import font, visual_model


def trial(directory, clip):
    with np.load(directory / f"recover_to_walk_{clip}_0.npz") as data:
        if not np.isclose(float(data["fps"]), 50.):
            raise ValueError(f"{directory} must contain 50 Hz physics traces")
        pose = np.vstack([data["initial_qpos"], data["qpos"]])
        loads = data["support_load_fraction"]
        if loads.shape != (len(pose)-1, 8):
            raise ValueError(f"Missing measured hand loads for {clip} in {directory}")
        return pose, loads[:, 2:4], data["delays"], data["command_mps"]


def render(reference, previous, selected, iteration, out, encoder, seconds):
    clips = sorted(reference.glob("*.npz"))
    if len(clips) != 11:
        raise ValueError(f"Expected all eleven native recovery clips in {reference}")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("The duration per clip must be positive")
    width, height, top, bottom, fps = 512, 366, 66, 57, 25
    canvas_size = (3 * width, top + height + bottom + 1)
    phys = build_scene(robot_name=zenbot.MIXED_TASK)
    model = visual_model("zenbot23_getup")
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 2.55, -15.
    big, mid, small = font(23), font(18), font(16)
    columns = ("LeggedLab demonstration", "Previous AMP   15,250", f"New AMP   {iteration:,}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp, mujoco.Renderer(model, height, width) as renderer:
        stage = Path(temp) / "comparison.mp4"
        args = [encoder, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas_size[0]}x{canvas_size[1]}",
                "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(stage)]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as writer:
            for seq, path in enumerate(clips, 1):
                with np.load(path) as src:
                    if not np.isclose(float(src["fps"]), 50.):
                        raise ValueError(f"Expected 50 Hz source poses in {path}")
                    ref = np.column_stack((src["root_pos"], src["root_quat"][:, [3, 0, 1, 2]], src["dof_pos"]))
                old_q, old_force, old_delays, old_command = trial(previous, path.stem)
                new_q, new_force, new_delays, new_command = trial(selected, path.stem)
                if not (np.allclose(old_q[0], new_q[0]) and np.array_equal(old_delays, new_delays)
                        and np.array_equal(old_command, new_command)):
                    raise ValueError(f"Initial pose, motor delays and command must match for {path.stem}")
                q_by_panel = (ref, old_q, new_q)
                near = [limb_clearances(phys, q[:round(seconds*50)+1])[:, 4:] < .025 for q in q_by_panel]
                # Choose the same three-quarter view from the final reference heading for all three panels.
                w, x, y, z = ref[-1, 3:7]
                yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                cam.azimuth = 143. + np.degrees(yaw)
                clip = path.stem.removeprefix("fallAndGetUp1_subject1_")
                for frame in range(round(seconds * fps)):
                    t = 2 * frame
                    image = Image.new("RGB", canvas_size, "#18212a")
                    draw = ImageDraw.Draw(image)
                    draw.text((13, 5), f"Clip {seq}/11   {clip}      {t/50:.2f} s   ·   1×", fill="#d4e0e6", font=mid)
                    for panel, (poses, force) in enumerate(zip(q_by_panel, (None, old_force, new_force))):
                        x0 = panel * width
                        last = min(t, len(poses)-1)
                        pose = poses[last]
                        cam.lookat[:] = [*pose[:2], .57]
                        data.qpos[:] = pose
                        mujoco.mj_kinematics(model, data)
                        mujoco.mj_camlight(model, data)
                        renderer.update_scene(data, camera=cam)
                        image.paste(Image.fromarray(renderer.render()), (x0, top))
                        draw.text((x0+13, 32), columns[panel], fill="white", font=big)
                        if t >= len(poses):
                            status = "End of demonstration" if panel == 0 else "End of recorded rollout"
                            draw.text((x0+13, top+height+11), status, fill="#d4e0e6", font=mid)
                            continue
                        left, right = near[panel][last]
                        state = "both" if left and right else "left" if left else "right" if right else "neither"
                        draw.text((x0+13, top+height+4), f"Hands within 2.5 cm of floor   {state}",
                                  fill="#82f1cb" if left and right else "#e4ecf0", font=mid)
                        if force is None:
                            line = "Kinematic demonstration"
                        elif t == 0:
                            line = "Hand loads begin at the first physics step"
                        else:
                            l, r = force[t-1]
                            line = f"Hand load (% of body weight)   L {100*l:.0f}%    R {100*r:.0f}%"
                        draw.text((x0+13, top+height+30), line, fill="#c2cdd4", font=small)
                    writer.stdin.write(image.tobytes())
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("ffmpeg could not encode the Zenbot support comparison")
        stage.replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected", type=Path, help="new native evaluation traces recorded with seed 73")
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--previous", type=Path,
                        default=Path("runs/zenbot23_amp_support_10000_to_16000/traces_15250_seed73"))
    parser.add_argument("--reference", type=Path, default=Path("data/zenbot23_walk_run_recovery/recovery"))
    parser.add_argument("--seconds", type=float, default=6., help="seconds per clip at normal speed")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"), help="path to an ffmpeg executable with libx264")
    args = parser.parse_args()
    if not args.ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            args.ffmpeg = get_ffmpeg_exe()
        except ImportError:
            parser.error("Supply --ffmpeg or install imageio-ffmpeg")
    render(args.reference, args.previous, args.selected, args.iteration, args.out, args.ffmpeg, args.seconds)
    print(args.out)
