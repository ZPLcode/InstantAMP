"""Drive a joint Zenbot policy and restart it from fixed fallen poses."""

import argparse
import os
import time
from pathlib import Path
from queue import SimpleQueue

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "glfw")

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from instantamp import env
from instantamp.zenbot_trials import JointTrials, resolve_checkpoint

_LIMITS = ((-1.0, 2.5), (-0.5, 0.5), (-1.5, 1.5))
_KEYS = {
    glfw.KEY_W: (0, 0.1),
    glfw.KEY_UP: (0, 0.1),
    glfw.KEY_S: (0, -0.1),
    glfw.KEY_DOWN: (0, -0.1),
    glfw.KEY_EQUAL: (0, 0.5),
    glfw.KEY_KP_ADD: (0, 0.5),
    glfw.KEY_MINUS: (0, -0.5),
    glfw.KEY_KP_SUBTRACT: (0, -0.5),
    glfw.KEY_A: (1, 0.1),
    glfw.KEY_LEFT: (1, 0.1),
    glfw.KEY_D: (1, -0.1),
    glfw.KEY_RIGHT: (1, -0.1),
    glfw.KEY_Q: (2, 0.2),
    glfw.KEY_E: (2, -0.2),
}
_HELP = (
    "W/S forward    A/D side    Q/E turn    +/- speed    X stop\n"
    "R next fallen start    F push    Space pause"
)


def _display_model(robot, mesh_dir):
    xml = robot.profile.MODEL.xml_path
    spec = mujoco.MjSpec.from_file(str(xml))
    if mesh_dir is None:
        mesh_dir = xml.parent / "meshes" / xml.stem
        if not all((mesh_dir / Path(mesh.file).name).is_file() for mesh in spec.meshes):
            # The source MJCF hides collisions so they do not overlap the visual meshes.
            robot.model.geom_group[:] = 0
            robot.model.geom_rgba[robot.model.geom_bodyid != 0, 3] = 1.
            return robot.model, robot.data
    spec.meshdir = str(mesh_dir.resolve())
    for mesh in spec.meshes:
        mesh.file = Path(mesh.file).name
    model = spec.compile()
    for field, count in (("joint", "njnt"), ("body", "nbody")):
        visual = [
            getattr(model, field)(i).name for i in range(getattr(model, count))
        ]
        physical = [
            getattr(robot.model, field)(i).name.removeprefix("robot_")
            for i in range(getattr(robot.model, count))
        ]
        if physical != visual:
            raise ValueError(f"Zenbot visual and physical {field} order differ")
    return model, mujoco.MjData(model)


def play(checkpoint, reference, mesh_dir, seed):
    """Drive a single policy from an upright start with an optional mesh skin."""
    trials = JointTrials(checkpoint, reference, seed)
    trials.stand()
    robot = trials.native
    model, data = _display_model(robot, mesh_dir)
    pelvis = robot.model.jnt_bodyid[0]
    target = np.zeros(3)
    keys = SimpleQueue()
    paused = False
    trial = 0
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=keys.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                2.7,
                120.0,
                -14.0,
            )
        deadline = time.monotonic()
        while viewer.is_running():
            with viewer.lock():
                while not keys.empty():
                    key = keys.get()
                    if key in _KEYS:
                        axis, step = _KEYS[key]
                        target[axis] = np.clip(
                            round(target[axis] + step, 1), *_LIMITS[axis]
                        )
                    elif key == glfw.KEY_X:
                        target.fill(0.0)
                    elif key == glfw.KEY_R:
                        trials.fallen(trial)
                        trial = (trial + 1) % len(trials.names)
                        target.fill(0.0)
                        paused = False
                    elif key == glfw.KEY_F:
                        robot.data.qvel[:2] += (
                            1.25 * robot.data.xmat[pelvis].reshape(3, 3)[:2, 0]
                        )
                        paused = False
                    elif key == glfw.KEY_SPACE:
                        paused = not paused
                if not paused:
                    if data is not robot.data:
                        robot.data.xfrc_applied[:] = data.xfrc_applied
                        robot.data.qfrc_applied[:] = data.qfrc_applied
                    command = np.asarray(robot.command[0])
                    robot.command = robot.command.at[0].add(
                        np.clip(target - command, -0.04, 0.04)
                    )
                    robot.step()
                if data is not robot.data:
                    data.qpos[:], data.qvel[:], data.time = (
                        robot.data.qpos,
                        robot.data.qvel,
                        robot.data.time,
                    )
                    mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = [*data.qpos[:2], 0.49]
            measured = (
                robot.data.xmat[pelvis].reshape(3, 3).T @ robot.data.qvel[:3]
            )
            label = (
                f"Zenbot {model.nu}-DoF  |  {checkpoint.name}  |  {data.time:.1f} s\n"
                f"Target {target[0]:+.2f}, {target[1]:+.2f} m/s, {target[2]:+.2f} rad/s\n"
                f"Measured {measured[0]:+.2f}, {measured[1]:+.2f} m/s, "
                f"{robot.data.qvel[5]:+.2f} rad/s\n{_HELP}"
            )
            viewer.set_texts(
                (
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    label,
                    "",
                )
            )
            viewer.sync()
            deadline = max(deadline + env.DT, time.monotonic())
            time.sleep(max(0.0, deadline - time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Policy NPZ or run directory",
    )
    parser.add_argument(
        "--reference", type=Path, help="Recovery clips for this articulation"
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        help="Optional directory with matching visual meshes",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    play(
        resolve_checkpoint(args.checkpoint),
        args.reference,
        args.mesh_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
