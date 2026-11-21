"""Render real-time retargeted Zenbot motion beside a recorded unassisted MuJoCo episode."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tools.render import font, visual_model


def render(archive, output, seconds, reference):
    row = json.loads(archive.with_suffix(".json").read_text())
    with np.load(archive) as z:
        poses, loads = z["qpos"], z["support_load_bw"]
    with np.load(reference / row["clip"]) as z:
        target = np.column_stack((z["root_pos"], z["root_quat"][:, [3, 0, 1, 2]], z["dof_pos"]))
    target = target[round(row["offset"] * 50):].copy()
    target[:, :2] += poses[0, :2] - target[0, :2]
    model = visual_model("zenbot23_getup", reference.parent / "meshes")
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.7, 122, -13
    cam.lookat[:] = [*(poses[0, :2] + 0.36 * (poses[min(len(poses) - 1, 240), :2] - poses[0, :2])), 0.51]
    width, height, header = 480, 320, 84
    labels = font(19), font(14)
    rise = None if row["rise"] is None else round(row["rise"] * 50)
    frames = min(len(poses), round(seconds * 50) + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height, width) as renderer, tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        movie = Path(tmp) / "movie.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2*width}x{height+header}",
               "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
            for t in range(0, frames, 2):
                frame = Image.new("RGB", (2 * width, height + header), "#18232e")
                draw = ImageDraw.Draw(frame)
                force = loads[t-1] if t else np.zeros(6)
                for j, pose in enumerate((target[min(t, len(target) - 1)], poses[t])):
                    left = j * width
                    title = "Retargeted reference" if j == 0 else "Zenbot PPO + AMP with physical feedback"
                    caption = "Reference ended, holding last frame" if t >= len(target) else f"Reference at {0.02*t:.2f} s"
                    if j:
                        caption = f"Palm load  L {force[2]:.0%}   R {force[3]:.0%}     Feet {sum(force[:2]):.0%} BW"
                    info = f"Simulation {0.02*t:.2f} s"
                    if j and rise is not None and t >= rise:
                        info += f"     XY from first stand {100*np.linalg.norm(pose[:2]-poses[rise,:2]):.1f} cm"
                    draw.text((left+10, 4), title, font=labels[0], fill="white")
                    draw.text((left+10, 33), caption, font=labels[1], fill="#cddbe9")
                    draw.text((left+10, 55), info, font=labels[1], fill="#cddbe9")
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=cam)
                    frame.paste(Image.fromarray(renderer.render()), (left, header))
                writer.stdin.write(frame.tobytes())
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("ffmpeg could not encode the Zenbot comparison")
        movie.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("data/zenbot23_getup/recovery"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Use a finite positive recording duration")
    render(args.archive, args.out, args.seconds, args.reference)
