"""Render views of the recorded 09_neg_x.npz from fixed cameras at real time."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from tools.render import font, visual_model

root = Path(__file__).resolve().parent
with np.load(root / "09_neg_x.npz") as recording:
    q = recording["qpos"]
    feet = recording["feet_xy"]
model = visual_model("zenbot23_getup")
data = mujoco.MjData(model)
camera = mujoco.MjvCamera()
camera.distance, camera.elevation = 2.45, -12.0
station = 750
camera.lookat[:] = [*q[station, :2], .68]
heading = Rotation.from_quat(q[station, 3:7], scalar_first=True).apply([1, 0, 0])
azimuth = np.rad2deg(np.arctan2(heading[1], heading[0]))
width, height, header = 448, 320, 70
out = root / "trial09_neg_x.mp4"
cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format","rgb24",
       "-video_size", f"{width*2}x{height+header}", "-framerate", "25", "-i", "-",
       "-c:v", "libx264", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
title, small = font(19), font(15)
with mujoco.Renderer(model, height=height, width=width) as renderer, subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
    for i in range(690, 991, 2):
        data.qpos[:] = q[i]
        mujoco.mj_forward(model, data)
        frame = Image.new("RGB", (2 * width, height + header), "#17242b")
        for j, angle in enumerate([azimuth + 180, azimuth + 90]):
            camera.azimuth = angle
            renderer.update_scene(data, camera=camera)
            frame.paste(Image.fromarray(renderer.render()), (j*width, header))
        draw = ImageDraw.Draw(frame)
        event = "PUSH" if station <= i < station+15 else ("Before push" if i < station else "After push")
        draw.text((10, 6), "Zenbot | World -X velocity change of 0.40 m/s at 15.00 s", fill="white", font=title)
        draw.text((10, 40), f"{i * .02:05.2f} s    {event}", fill="#ffcd75" if event == "PUSH" else "#bfcdd4", font=small)
        root_dist = np.linalg.norm(q[i,:2] - q[station,:2]) * 100
        foot_dist = np.linalg.norm(feet[i-1] - feet[station-1], axis=1).max() * 100
        draw.text((280,40), f"From pre-push position   pelvis {root_dist:.1f} cm    max foot {foot_dist:.1f} cm", fill="#dce7eb", font=small)
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg encode failed")
