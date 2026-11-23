"""Interact with the released InstantAMP G1 get-up policy in the native MuJoCo viewer."""

import argparse
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import time

if __name__ == "__main__":
    # A single robot can run locally at 50 Hz without a CUDA runtime.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import glfw
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from instantamp import env, g1_getup
from instantamp.evaluate import fallen_starts
from instantamp.inference import G1StandController, load, policy_memory, policy_step
from instantamp.native_getup import advance, observation_terms
from instantamp.g1_ground_phase import HandContacts
from instantamp.sim import build_scene


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "runs/g1_ground_sequence_release/ckpt/0000.npz"


class GetupPlayer:
    def __init__(self, checkpoint, trial=0):
        self.policy_label = Path(checkpoint).parent.parent.name
        policy, norm, ode = load(checkpoint)
        first = policy.actors[0] if isinstance(policy, G1StandController) else policy
        if (not isinstance(first, list) or ode is not None or first[0]["w"].shape[0] != 570
                or first[-1]["b"].shape != (29,)):
            raise ValueError("Expected a direct-action G1 get-up checkpoint")

        self.model = build_scene(robot_name="g1_getup", getup_visuals=True)
        self.data = mujoco.MjData(self.model)
        self.links = [self.model.body("robot_" + name).id for name in g1_getup.KEY_BODIES]
        self.names, poses, velocities = fallen_starts(ROOT / "data/g1_getup/recovery", g1_getup.RESET_LIFT, "source-play")
        self.poses = np.asarray(poses, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float).copy()
        self.policy = policy
        from instantamp.g1_floating_base import load_native_tracker
        self.measured = load_native_tracker(self.model, policy, checkpoint) if isinstance(policy,G1StandController) else None
        if self.measured is not None:
            self.policy = policy._replace(tracking_capture=None)
            head = policy.tracking_capture
            from instantamp.g1_tracking_capture import residual_features
            self.tracking_residual = jax.jit(lambda h:head.residual_scale * jnp.tanh(
                residual_features(h) @ head.residual_w + head.residual_b))

        @jax.jit
        def infer(history, frame, memory):
            history = g1_getup.push_history(history, frame, g1_getup.ACTOR_TERMS)
            action, memory = policy_step(self.policy, norm, history, memory)
            return history, action, memory

        self.infer = infer
        self.reset(trial)
        # Compile before the first keypress without advancing physics or controller memory.
        self.infer(self.history, self.frame(), self.memory)[1].block_until_ready()

    def frame(self):
        terms = observation_terms(self.data, self.links, self.previous)
        return jnp.asarray(np.concatenate(terms)[None], dtype=jnp.float32)

    def reset(self, trial):
        if not 0 <= trial < len(self.names):
            raise ValueError(f"Trial index must be between 0 and {len(self.names) - 1}")
        self.trial = trial
        if self.measured is not None:
            self.measured.reset()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.poses[trial]
        self.data.qvel[:] = self.velocities[trial]
        # Source linear velocity is at the root COM. MuJoCo integrates the free joint origin.
        mujoco.mj_kinematics(self.model, self.data)
        self.data.qvel[:3] -= self.data.xmat[1].reshape(3, 3) @ np.cross(self.data.qvel[3:6], self.model.body_ipos[1])
        mujoco.mj_forward(self.model, self.data)
        self.previous = np.zeros(self.model.nu, dtype=np.float32)
        self.history = g1_getup.push_history(None, self.frame(), g1_getup.ACTOR_TERMS)
        self.memory = policy_memory(self.policy, self.history)

    def step(self):
        previous = self.previous
        self.history, action, self.memory = self.infer(self.history, self.frame(), self.memory)
        actor = np.asarray(action[0])
        self.previous = actor
        if self.measured is not None:
            self.previous = self.measured.action(self.data, self.previous, previous)
            if self.measured.engaged:
                self.previous = self.previous + np.asarray(self.tracking_residual(self.history)[0])
                self.previous = self.measured.balance_action(self.previous, actor)
        advance(self.model, self.data, self.previous)

    def push(self, angle_deg=0.):
        angle = np.deg2rad(angle_deg)
        self.data.qvel[:2] += 0.20 * np.array([np.cos(angle), np.sin(angle)])


def run(player, palm_load=None, *, play=False, poll_policy=None, measurements=None, handle_key=None):
    keys = SimpleQueue()
    if palm_load is None:
        contacts = HandContacts(player.model)
        def palm_load():
            return contacts.forces(player.data)[:, 0]
    paused = not play
    print("Ready. SPACE play/pause | R restart | N next start | ENTER push (+X)", flush=True)
    print("Mouse: double-click a body, then hold Ctrl and drag with the right button.", flush=True)
    with mujoco.viewer.launch_passive(player.model, player.data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as view:
        def reset_camera():
            view.cam.lookat[:] = [*player.data.qpos[:2], 0.5]
            view.cam.distance = 2.7
            view.cam.azimuth = 120
            view.cam.elevation = -14

        with view.lock():
            reset_camera()
        deadline = time.monotonic()
        next_policy_check = deadline
        while view.is_running():
            if poll_policy is not None and time.monotonic() >= next_policy_check:
                poll_policy()
                next_policy_check = time.monotonic() + .5
            with view.lock():
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key in (glfw.KEY_R, glfw.KEY_N, glfw.KEY_BACKSPACE):
                        trial = (player.trial + int(key == glfw.KEY_N)) % len(player.names)
                        player.reset(trial)
                        view.perturb.active = 0
                        reset_camera()
                        paused = False
                        print(f"Restarted {player.names[trial][0]}", flush=True)
                    elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                        player.push()
                        paused = False
                        print("Push +X (0.20 m/s)", flush=True)
                    elif handle_key is not None:
                        handle_key(key)
                if not paused:
                    player.step()
                measured = measurements() if measurements is not None else ""
            status = f"{'PAUSED' if paused else 'LIVE'}   |   Start {player.trial + 1}/{len(player.names)}   |   {player.data.time:.2f} s"
            hand_load = palm_load()
            status += f"   |   Palm load   L {hand_load[0]:.0%}  R {hand_load[1]:.0%} BW"
            view.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            status + f"   |   {player.policy_label}\n" + (measured + "\n" if measured else "")
                            + "Space play/pause   R restart   N next   Enter push\nDouble-click torso, then Ctrl + right-drag", ""))
            # Sync both renders the new pose and transfers the mouse wrench to the same MjData used by advance().
            view.sync()
            deadline = max(deadline + env.DT, time.monotonic())
            time.sleep(max(0., deadline - time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RELEASE)
    parser.add_argument("--trial", type=int, default=0, help="Zero-based source motion index (default 0)")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Missing policy checkpoint: {args.checkpoint}")
    try:
        player = GetupPlayer(args.checkpoint, args.trial)
    except ValueError as exc:
        parser.error(str(exc))
    run(player)


if __name__ == "__main__":
    main()
