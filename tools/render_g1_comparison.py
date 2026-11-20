"""Render synchronized reference, baseline and candidate from evaluate_g1_style.py at real speed."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tools.render import font, visual_model


def read_tracks(path, report):
    with np.load(path) as archive:
        refs = archive["reference"]
        cases = json.loads(str(archive["cases_json"])) if "cases_json" in archive else report["baseline"]["trials"]
        if len(refs) != len(cases):
            raise ValueError("Reference and score cases disagree")
        recordings = []
        for label in ("baseline", "candidate"):
            trajectories, endings = archive[label + "_qpos"], archive[label + "_ended"]
            for q, ended in zip(trajectories, endings):
                first = np.flatnonzero(ended)
                if len(first):
                    q[first[0]+1:] = q[first[0]]
            recordings.append((trajectories, endings))
    return refs, cases, recordings


def center_camera(model, data, poses):
    # Identical view and scale in all columns, including the full space covered by the three paths.
    points = []
    for path in poses:
        for q in path[::5]:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            points.append(data.xpos[1:].copy())
    points = np.concatenate(points)
    mid = (points.max(0) + points.min(0)) / 2
    cam = mujoco.MjvCamera()
    w, x, y, z = poses[0][-1, 3:7]
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    cam.azimuth, cam.elevation = np.degrees(yaw) + 110, -12
    cam.lookat[:] = mid
    relative = points-mid
    cam.distance = max(2.2, 2.6*np.linalg.norm(relative[:, :2], axis=1).max()+.45,
                       2.8*np.max(abs(relative[:, 2]))+.45)
    return cam


def render(archive, score_file, output, ffmpeg, trial, show_hold):
    report = json.loads(score_file.read_text())
    refs, cases, recordings = read_tracks(archive, report)
    if trial is not None and not 0 <= trial < len(cases):
        raise ValueError(f"trial must be in [0, {len(cases)-1}]")
    model = visual_model()
    data = mujoco.MjData(model)
    width, height, header = 352, 256, 62
    shape = (3*width, height+header)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{shape[0]}x{shape[1]}",
           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(output)]
    titles = ("Reference", f"G1 baseline | iter {report['baseline']['iteration']}", f"G1 AMP | iter {report['candidate']['iteration']}")
    with mujoco.Renderer(model, height, width) as viewer, subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for i, case in enumerate(cases):
            if trial is not None and i != trial:
                continue
            reference = refs[i]
            if "end_frame" in case:
                end = min(len(reference)-1, case["end_frame"] - case["start_frame"] - 1)
                start_text = str(case["start_frame"]) + " @50Hz"
            else:
                end = min(len(reference)-1, round(case["end_frame_30hz"]/.6) - round(case["source_frame_30hz"]/.6) - 1)
                start_text = str(case["source_frame_30hz"]) + " @30Hz"
            steps = recordings[0][0].shape[1] if show_hold else end+1
            demo = reference[np.minimum(np.arange(steps), end)]
            paths = [demo, *[rec[0][i, :steps] for rec in recordings]]
            cam = center_camera(model, data, paths)
            for frame in range(0, steps, 2):
                image = Image.new("RGB", shape, (22, 32, 39))
                draw = ImageDraw.Draw(image)
                for col, path in enumerate(paths):
                    h = f"{frame*.02:.2f}s    z {path[frame, 2]:.2f}m"
                    if col == 0 and frame > end:
                        h += "    final reference pose"
                    if col and recordings[col-1][1][i,:frame].any():
                        h += "    episode ended"
                    dx = col*width
                    draw.text((dx+8,2), titles[col], font=font(15), fill="white")
                    draw.text((dx+8,22), f"{case['source_take']}    {start_text}", font=font(13), fill="#c6d4df")
                    draw.text((dx+8,41), h, font=font(13), fill="#c6d4df")
                    data.qpos[:] = path[frame]
                    mujoco.mj_forward(model, data)
                    viewer.update_scene(data, camera=cam)
                    image.paste(Image.fromarray(viewer.render()), (dx, header))
                encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Video encoding failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectories", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--trial", type=int, help="zero-based case index; the default renders every case")
    p.add_argument("--show-hold", action="store_true", help="include the final second of the six-second physical rollout")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    args = p.parse_args()
    render(args.trajectories, args.report, args.out, args.ffmpeg, args.trial, args.show_hold)
