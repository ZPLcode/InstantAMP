"""Render the raw clip and CPU rollouts at the same physical speed and world coordinates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.render import visual_model
from tools.render_g1_comparison import center_camera


def render(reports, out, ffmpeg, trial):
    scores = [json.loads(path.read_text()) for path in reports]
    names = [p.stem for p in reports]
    series, loads = [], []
    source = None
    cases = scores[0]["trials"]
    for path, item in zip(reports, scores):
        for key in ("clip_sha256", "support_friction", "assist_n", "policy_hz", "reset_z_lift_m", "duration_s"):
            if item["protocol"][key] != scores[0]["protocol"][key]:
                raise ValueError(f"Evaluation protocol mismatch in {path}: {key}")
        keys = ("source_frame", "initial_world_vx_perturbation_m_s")
        if [[r[k] for k in keys] for r in cases] != [[r[k] for k in keys] for r in item["trials"]]:
            raise ValueError("Evaluations used different starts")
        with np.load(path.with_suffix(".npz")) as data:
            if source is None:
                source = data["reference"]
            elif not np.array_equal(source, data["reference"]):
                raise ValueError("Evaluations used different reference positions")
            series.append(data["qpos"])
            loads.append(data["foot_load_bw"])
    if trial is not None and not 0 <= trial < len(cases):
        raise ValueError(f"trial must be between 0 and {len(cases)-1}")
    width, height, bar = 304, 220, 54
    shape = (width*(1+len(reports)), height+bar)
    model = visual_model()
    data = mujoco.MjData(model)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{shape[0]}x{shape[1]}",
           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)]
    with mujoco.Renderer(model, height, width) as renderer, subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        for i, row in enumerate(cases):
            if trial is not None and i != trial:
                continue
            paths = [source[i], *[q[i].copy() for q in series]]
            for c, score in enumerate(scores, 1):
                n = score["trials"][i]["valid_policy_frames"]
                paths[c][n:] = paths[c][n-1]
            cam = center_camera(model, data, paths)
            for t in range(0, len(paths[0]), 2):
                image = Image.new("RGB", shape, (23, 32, 40))
                draw = ImageDraw.Draw(image)
                for c, (path, label) in enumerate(zip(paths, ["Reference", *names])):
                    x = c*width
                    draw.text((x+6, 2), f"{label}  |  {t*.02:.2f}s", fill="white", font=font)
                    draw.text((x+6, 19), f"start {row['source_seconds']:.2f}s  |  delta vx {row['initial_world_vx_perturbation_m_s']:+.2f}m/s",
                              fill="#c3d3dc", font=font)
                    status = f"pelvis {path[t,2]:.2f} m"
                    if c:
                        if t >= scores[c-1]["trials"][i]["valid_policy_frames"]:
                            status += "  |  episode ended"
                        elif t:
                            status += f"  |  sole load {loads[c-1][i,t-1]:.1f} BW"
                    elif t >= row["reference_frames"]:
                        status += "  |  last source pose"
                    draw.text((x+6, 36), status, fill="#c3d3dc", font=font)
                    data.qpos[:] = path[t]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, cam)
                    image.paste(Image.fromarray(renderer.render()), (x, bar))
                process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Video encoding failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trial", type=int, default=0, help="0 is the canonical, unperturbed start; use --all for every start")
    p.add_argument("--all", action="store_true")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = p.parse_args()
    render(a.reports, a.out, a.ffmpeg, None if a.all else a.trial)
