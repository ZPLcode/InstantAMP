"""Open the mixed Zenbot policy in the native MuJoCo viewer."""
import argparse
import json
import os
from pathlib import Path
from queue import SimpleQueue
import time

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "glfw")

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from instantamp import env, zenbot
from instantamp.native_zenbot import ZenbotNative
from tools.evaluate_zenbot_walk_recovery import starts
from tools.render import visual_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/zenbot23_amp_style"
if not DEFAULT_RUN.is_dir():
    DEFAULT_RUN = ROOT / "runs/zenbot23_walk_run_recovery_20261121_v1"
validated = ROOT / "runs/zenbot23_leggedlab_prior_recovery"
if (validated / "controller/zenbot_walk_run_recovery.npz").is_file():
    DEFAULT_RUN = validated
single_actor = ROOT / "runs/zenbot23_joint_amp_selected"
if (single_actor / "controller/zenbot_walk_run_recovery.npz").is_file():
    DEFAULT_RUN = single_actor
MAX_SPEED = round(float(zenbot.GAIT_COMMANDS[0, 1]), 1)


def speed_key(key, speed):
    if key in (glfw.KEY_UP, glfw.KEY_W):
        return min(MAX_SPEED, round(speed + .1, 1))
    if key in (glfw.KEY_DOWN, glfw.KEY_S):
        return max(0., round(speed - .1, 1))
    if key == glfw.KEY_X:
        return 0.
    return None


def forward_speed(data):
    # Project the pelvis velocity onto its horizontal heading, matching the gait evaluation.
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, help="override the run's controller with a policy export")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--demo", action="store_true", help="use the preset speeds and apply a side push after 6 s")
    parser.add_argument("--speed", type=float, help="initial forward speed in manual mode, in m/s")
    parser.add_argument("--play", action="store_true", help="start the viewer unpaused")
    args = parser.parse_args()
    if args.speed is not None and (args.demo or not 0. <= args.speed <= MAX_SPEED):
        parser.error(f"--speed requires manual mode and a value between 0 and {MAX_SPEED} m/s")
    config = json.loads((args.run / "config.json").read_text())
    if config["robot"] != zenbot.MIXED_TASK:
        parser.error("Expected a Zenbot walk/run/recovery training run")
    walk, floor = (ROOT / config[k] for k in ("walk_dir", "floor_dir"))
    checkpoint = args.checkpoint or args.run / "controller/zenbot_walk_run_recovery.npz"
    native = ZenbotNative(floor, checkpoint, seed=args.seed, walk_dir=walk)
    if config.get("single_actor") and (not isinstance(native.actor, list) or native.native_tracking is not None):
        raise ValueError("The single-policy run requires one feed-forward AMP actor")
    model = visual_model("zenbot23_getup", ROOT / "data/zenbot23_getup/meshes")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    scenes = {}
    for task, _, _, speed, pose, velocity in starts(walk, floor):
        delays = rng.integers(3, 8, 5)
        scenes.setdefault(task, (pose, velocity, delays, speed))
    cases = ("walk_fall_resume", "run_fall_resume", "recover_to_walk")
    labels = ("Walking start", "Running start", "Fallen start")
    if not args.demo:
        stand = np.r_[0., 0., zenbot.DEFAULT_HEIGHT, 1., 0., 0., 0., zenbot.DEFAULT_Q]
        scenes["stand"] = (stand, np.zeros(native.model.nv), rng.integers(3, 8, 5), 0.)
        cases, labels = ("stand", *cases), ("Standing start", *labels)
    keys = SimpleQueue()
    index, paused, push_steps = 0, not args.play, 0
    target_speed = scenes[cases[index]][3] if args.speed is None else args.speed

    def project():
        data.qpos[:] = native.data.qpos
        data.qvel[:] = native.data.qvel
        data.time = native.data.time
        mujoco.mj_forward(model, data)

    def reset():
        pose, velocity, delays, _ = scenes[cases[index]]
        native.reset(pose, velocity, delays, command=(target_speed, 0., 0.))
        mujoco.mj_resetData(model, data)
        project()

    reset()
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False) as view:
        with view.lock():
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 2.65, 115., -14.
        deadline = time.monotonic()
        while view.is_running():
            with view.lock():
                while not keys.empty():
                    key = keys.get()
                    speed = speed_key(key, target_speed)
                    if speed is not None:
                        target_speed = speed
                        native.command = native.command.at[0, 0].set(target_speed)
                    elif key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key in (glfw.KEY_N, glfw.KEY_R):
                        index = (index + int(key == glfw.KEY_N)) % len(cases)
                        if args.demo and key == glfw.KEY_N:
                            target_speed = scenes[cases[index]][3]
                        reset()
                        view.perturb.active = 0
                        push_steps, paused = 0, False
                    elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                        push_steps, paused = 10, False
                if not paused:
                    if args.demo and cases[index].endswith("_fall_resume") and round(native.data.time / env.DT) == 300:
                        push_steps = 10
                    native.data.xfrc_applied[:] = data.xfrc_applied
                    native.data.qfrc_applied[:] = data.qfrc_applied
                    if push_steps:
                        native.data.xfrc_applied[native.sim.torso, 1] += 400.
                        push_steps -= 1
                    native.step()
                    project()
                view.cam.lookat[:] = [*data.qpos[:2], .55]
            status = "Paused" if paused else "Live"
            push = "  |  Side push at 6 s" if args.demo and cases[index].endswith("_fall_resume") else ""
            profile = "Zenbot 23 DoF  |  Single AMP policy" if config.get("single_actor") else "Zenbot 23 DoF"
            view.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            f"{profile}  |  {labels[index]}  |  {status}  {data.time:.1f} s{push}\n"
                            f"Forward speed   Target {target_speed:.1f} m/s   Actual {forward_speed(native.data):+.2f} m/s\n"
                            "Up/W faster   Down/S slower   X zero speed\n"
                            "Space play/pause   N next start   R restart   Enter push", ""))
            view.sync()
            deadline = max(deadline + env.DT, time.monotonic())
            time.sleep(max(0., deadline - time.monotonic()))


if __name__ == "__main__":
    main()
