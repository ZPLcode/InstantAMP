"""Walk or run, take a torso shove, land, get up and resume on one continuous native MuJoCo G1."""

import argparse
import copy
from enum import IntEnum
import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if "--viewer" not in sys.argv:
        os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from instantamp import env, g1_getup as robot
from instantamp.inference import policy_memory
from instantamp.g1_ground_phase import HandContacts, knee_extension_frame, sustained_palm
from instantamp.native_gait import G1VelocityPolicy, JogSpeedCalibration, RunGains, ground_contacts
from tools.play_getup import GetupPlayer, RELEASE
from tools.compare_recovery_rollouts import first_upright
from tools.score_getup_in_place import pre_rise_attempts

ROOT = Path(__file__).resolve().parents[1]
RUN_PROFILE = ROOT / "assets/g1_velocity_run.json"


class Phase(IntEnum):
    GAIT = 0
    LANDING = 1
    GETUP = 2
    RESUMED = 3


def landing_step(model, data):
    for _ in range(env.DECIMATION):
        q, v = data.qpos[7:], data.qvel[6:]
        torque = -np.clip(v * robot.KD, -15., 15.)
        data.ctrl[:] = q + (torque + robot.KD * v) / robot.KP
        mujoco.mj_step(model, data)
    mujoco.mj_kinematics(model, data)


def start_getup(player, posture_fraction):
    # The previous target is an actor input. A measured posture gives a warm start
    # when the motor PD targets during landing differ from the nominal standing pose.
    player.previous = np.asarray(posture_fraction * np.clip(
        (player.data.qpos[7:] - robot.DEFAULT_Q) / robot.ACTION_SCALE, -6., 6.), dtype=np.float32)
    player.history = robot.push_history(None, player.frame(), robot.ACTOR_TERMS)
    player.memory = policy_memory(player.policy, player.history)
    if player.measured is not None:
        player.measured.reset()


def ready_for_gait(player):
    data = player.data
    ankles = np.asarray(player.frame()[0, -18:-12]).reshape(2, 3)
    depth = -(ankles @ data.xmat[1].reshape(3, 3)[2])
    return data.xmat[1, 8] > .945 and depth.min() > .628 and np.linalg.norm(data.qvel[3:6]) < .8


def select_ground_entry(player, hands, remaining):
    """Forecast a short landing delay and actor initialization from the measured fall."""
    probe = copy.copy(player)
    probe.data = mujoco.MjData(player.model)
    # The live tracker owns a stance solver, contact workspaces and another MjData
    # for its own lookahead. Only its motion tables and learned weights are immutable.
    measured = player.measured
    if measured is not None:
        probe.measured = type(measured)(player.model, measured.policy, measured.capture,
                                        measured.planning, measured.reference, measured.support_timing)
    best_cost, best = None, (0, 0.)
    options = [(0, 0.), (4, 0.), (9, 0.), (16, 0.), (26, 0.), (40, 0.),
               (0, .5), (6, .5), (14, .5), (0, 1.), (14, 1.)]
    for delay, posture in options:
        mujoco.mj_copyData(probe.data, player.model, player.data)
        probe.data.xfrc_applied[:] = 0.
        probe.data.qfrc_applied[:] = 0.
        for _ in range(min(delay, remaining)):
            landing_step(probe.model, probe.data)
        start_getup(probe, posture)
        poses, palms = [], []
        standing = 0
        for _ in range(min(750, remaining - delay)):
            standing = standing + 1 if ready_for_gait(probe) else 0
            if standing >= 45:
                break
            probe.step()
            poses.append(probe.data.qpos.copy())
            palms.append(hands.forces(probe.data)[:, 0].sum() >= .07)
        if not poses:
            continue
        q = np.asarray(poses)
        up = first_upright(q)
        if up is None or standing < 45:
            continue
        retries = pre_rise_attempts(q[:, 2], env.DT, up * env.DT)["retries"]
        refall = (q[up:, 2] < .45).any() or (1. - 2. * np.square(q[up:, 4:6]).sum(axis=1) < .55).any()
        palm = sustained_palm(palms, knee_extension_frame(q, up)) is not None
        cost = (bool(refall), retries, not palm, delay + up)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, (delay, posture)
        if not refall and retries == 0 and (palm or (delay, posture) == (0, 0.)):
            return delay, posture
    return best


def gait_command(speed, elapsed, resumed, start_profile):
    forward = speed * min(1., max(0., (elapsed - 15) / 60.))
    lateral = yaw = 0.
    if resumed and 15 < elapsed < 180:
        forward += max(0., 1.30 - speed) * min((elapsed - 15) / 28, 1., (180 - elapsed) / 65)
    if resumed and start_profile == 4:
        # Shift weight with a short turning command before accelerating out of quiet stance.
        ramp = np.clip((elapsed - 12) / 20., 0., 1.) * np.clip((100 - elapsed) / 28., 0., 1.)
        forward = min(forward, .70 + max(0, elapsed - 125) * .019)
        yaw = .7 * ramp
    return np.array([forward, lateral, yaw])


def simulate(player, gait, speed, angle, args, gains, viewer=None):
    start_profile = 4 if gains is not None else 0
    model, data = player.model, player.data
    hands = HandContacts(model)
    palm_load, local_tracking = [], []
    chest = model.body("robot_torso_link").id
    mujoco.mj_resetData(model, data)
    data.qpos[2] = .79
    data.qpos[7:] = gait.default
    mujoco.mj_forward(model, data)
    gait.reset()
    phase = Phase.GAIT
    changed = stable = 0
    stalled = 0
    shift_start = None
    turn = -1.
    grounded = False
    landed_this_fall = False
    recovery_tick = None
    recovery_posture = 0.
    ground_entries = []
    events = []
    anchor = None
    steps = round(args.seconds / env.DT)
    qpos, qvel, phases, velocities, contacts, flights, commands = [], [], [], [], [], [], []
    shove = args.force * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    for tick in range(steps):
        t = tick * env.DT
        if viewer is not None:
            viewer.frame(t, phase, data.qpos[:2])
        upright = data.xmat[1, 8]
        foot_mask, body_contact = ground_contacts(model, data, gait.floor, gait.feet)
        if phase in (Phase.GAIT, Phase.RESUMED) and (upright < .60 or body_contact and upright < .82):
            phase, changed, stable = Phase.LANDING, tick, 0
            landed_this_fall = False
            recovery_tick = None
            events.append(("falling", t))
        if phase == Phase.LANDING:
            if body_contact and data.qpos[2] < .47:
                landed_this_fall = grounded = True
            settled = landed_this_fall and body_contact and np.linalg.norm(data.qvel[3:6]) < 2.
            stable = stable + 1 if settled else 0
            ready = stable >= args.landing_confirm or (landed_this_fall and body_contact and tick - changed >= args.landing_timeout)
            if recovery_tick is None and ready:
                delay, recovery_posture = select_ground_entry(player, hands, steps - tick)
                recovery_tick = tick + delay
                ground_entries.append(dict(decision_s=t, damping_s=delay * env.DT,
                                           previous_target_posture_fraction=recovery_posture))
            if recovery_tick == tick:
                start_getup(player, recovery_posture)
                phase, stable = Phase.GETUP, 0
                events.append(("recovering", t))
        if phase == Phase.GETUP:
            stable = stable + 1 if ready_for_gait(player) else 0
            if stable >= 45:
                phase, changed, stable = Phase.RESUMED, tick, 0
                stalled, shift_start, turn = 0, None, -1.
                gait.reset()
                anchor = robot.DEFAULT_Q + robot.ACTION_SCALE * player.previous
                events.append(("upright", t))
        apply_shove = args.push_at <= t < args.push_at + args.duration
        if apply_shove:
            data.xfrc_applied[chest, :2] += shove
        command = np.zeros(3)
        flight = -1
        if phase in (Phase.GAIT, Phase.RESUMED):
            elapsed = tick if phase == Phase.GAIT else tick - changed
            if phase == Phase.RESUMED:
                # When both feet stay planted despite a forward command, first transfer weight with a turn.
                if shift_start is not None:
                    elapsed = tick - shift_start - 70 + 35
                stopped = elapsed >= 60 and foot_mask == 3 and np.linalg.norm(data.qvel[:2]) < .12
                stalled = stalled + 1 if stopped else 0
                if stalled >= 35:
                    if shift_start is not None:
                        turn = -turn
                    shift_start, stalled = tick, 0
                    gait.reset()
                    events.append(("weight_shift", t))
            if phase == Phase.RESUMED and shift_start is not None and tick - shift_start < 70:
                command = np.array([.3, 0., turn])
            else:
                command = gait_command(speed, elapsed, phase == Phase.RESUMED, start_profile)
            target = gait.target(data, command, gains)
            if phase == Phase.RESUMED and tick - changed < 25:
                target = anchor + (target - anchor) * ((tick - changed + 1) / 25.)
            flight = gait.advance(model, data, target)
        elif phase == Phase.LANDING:
            landing_step(model, data)
        else:
            player.step()
        if apply_shove:
            data.xfrc_applied[chest, :2] -= shove
        palm_load.append(hands.forces(data)[:,0])
        local_tracking.append(phase == Phase.GETUP and getattr(getattr(player,"measured",None),"engaged",False))
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        phases.append(phase)
        velocities.append(data.xmat[1].reshape(3, 3).T @ data.qvel[:3])
        contacts.append((foot_mask, body_contact))
        flights.append(flight)
        commands.append(command)
    if abs(data.time - steps * env.DT) > 1e-6 or np.any(data.warning.number):
        raise RuntimeError(f"MuJoCo continuity check failed at {data.time}s, warnings {data.warning.number.tolist()}")
    trace = dict(qpos=np.asarray(qpos), qvel=np.asarray(qvel), phase=np.asarray(phases, np.int8),
                 body_velocity=np.asarray(velocities), floor_contact_before=np.asarray(contacts, np.int8),
                 airborne_substeps=np.asarray(flights, np.int8), command=np.asarray(commands), dt=env.DT)
    trace.update(palm_load_bw=np.asarray(palm_load), tracked_recovery=np.asarray(local_tracking))
    row=summarize(trace,speed,angle,grounded,events,args,gains)
    row["ground_entries"] = ground_entries
    getup=np.flatnonzero(trace["phase"]==Phase.GETUP)
    if len(getup):
        poses=trace["qpos"][getup]
        upright_at=first_upright(poses)
        extension=knee_extension_frame(poses,upright_at)
        palm=sustained_palm(trace["palm_load_bw"][getup].sum(-1)>=.07,extension)
        row.update(recovery_to_first_stand_s=None if upright_at is None else env.DT*upright_at,
                   retries_before_first_stand=pre_rise_attempts(poses[:,2],env.DT,
                       None if upright_at is None else upright_at*env.DT)["retries"],
                   palm_loaded_before_extension=palm is not None,
                   first_palm_load_after_recovery_s=None if palm is None else env.DT*palm,
                   native_tracking_fraction=float(trace["tracked_recovery"][getup].mean()))
    return row,trace


def summarize(trace, speed, angle, grounded, events, args, gains):
    push_at = getattr(args, "push_at", 6.)
    pre = slice(max(0, round((push_at - 3.) / env.DT)), round((push_at - .1) / env.DT))
    post = slice(-150, None)
    v = trace["body_velocity"][:, 0]
    z = trace["qpos"][:, 2]
    phases = trace["phase"]
    flight = trace["airborne_substeps"]
    foot = trace["floor_contact_before"][:, 0]
    def fraction(window):
        counts = flight[window]
        return float(counts[counts >= 0].mean() / env.DECIMATION) if (counts >= 0).any() else 0.
    before, after = fraction(pre), fraction(post)
    resumed = bool(grounded and z[post].min() > .63 and (phases[post] == Phase.RESUMED).all()
                   and np.median(v[post]) > speed * .58)
    running = before > 0 and after > 0
    need_airborne = gains is not None
    return dict(speed=speed, angle=angle, force=args.force, push_at_s=push_at, ground=grounded, events=events, post_speed=float(np.median(v[post])),
                post_min_z=float(z[post].min()), before_speed=float(np.median(v[pre])), min_z=float(z.min()),
                success=resumed and (running or not need_airborne), resumed=resumed, airborne_before_and_after=running,
                before_flight_200hz=before, after_flight_200hz=after, pre_flight=float((foot[pre] == 0).mean()),
                post_flight=float((foot[post] == 0).mean()), post_both=float((foot[post] == 3).mean()),
                run_gains=vars(gains) if gains is not None else None, start_profile=4 if gains is not None else 0)


def video(path, model, trace):
    """Replay saved physical poses on a second MjData while streaming frames to ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        output = Path(temp) / "scene.mp4"
        encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", "640x360", "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt",
                                    "yuv420p", str(output)], stdin=subprocess.PIPE)
        try:
            d = mujoco.MjData(model)
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = 2.7, 120., -14.
            with mujoco.Renderer(model, height=360, width=640) as renderer:
                for tick in range(0, len(trace["qpos"]), 2):
                    d.qpos[:] = trace["qpos"][tick]
                    mujoco.mj_forward(model, d)
                    cam.lookat[:] = [*d.qpos[:2], .5]
                    renderer.update_scene(d, camera=cam)
                    image = Image.fromarray(renderer.render())
                    label = f"{(tick + 1) * env.DT:.2f} s    {Phase(trace['phase'][tick]).name}    {trace['body_velocity'][tick,0]:.2f} m/s"
                    ImageDraw.Draw(image).text((12, 12), label, stroke_width=1, stroke_fill="black", fill="white")
                    encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited with code {code}")
        output.replace(path)


class ViewerClosed(Exception):
    """The user closed a live trial before its scheduled completion."""


class LiveViewer:
    def __init__(self, player):
        import glfw
        import mujoco.viewer
        from queue import SimpleQueue

        self.gl = glfw
        self.keys = SimpleQueue()
        self.view = mujoco.viewer.launch_passive(player.model, player.data, key_callback=self.keys.put,
                                                  show_left_ui=False, show_right_ui=False)
        self.paused = True
        self.deadline = time.monotonic()
        with self.view.lock():
            self.view.cam.distance, self.view.cam.azimuth, self.view.cam.elevation = 2.7, 120., -14.

    def frame(self, seconds, phase, xy):
        while self.view.is_running():
            while not self.keys.empty():
                if self.keys.get() == self.gl.KEY_SPACE:
                    self.paused = not self.paused
            with self.view.lock():
                self.view.cam.lookat[:] = [*xy, .5]
            label = f"{phase.name}   {seconds:.2f}s   |   Space play/pause   |   Select body and Ctrl + drag"
            self.view.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, ""))
            self.view.sync()
            self.deadline = max(self.deadline + env.DT, time.monotonic())
            time.sleep(max(0., self.deadline - time.monotonic()))
            if not self.paused:
                return
        raise ViewerClosed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait", type=Path, default=Path.home() / "supermimic_AMP/deploy/wmrl_66000/policy.onnx")
    parser.add_argument("--recovery", type=Path, default=RELEASE)
    parser.add_argument("--speeds", "--speed", dest="speeds", type=float, nargs="+", default=[.55])
    parser.add_argument("--directions", "--direction", dest="directions", type=float, nargs="+", default=[0.],
                        help="Shove directions in world-frame degrees, zero for +X")
    parser.add_argument("--mode", choices=("auto", "walk", "run"), default="auto",
                        help="Auto uses running once the forward command reaches the full adaptation speed of 1.34 m/s")
    parser.add_argument("--run-profile", type=Path, default=RUN_PROFILE,
                        help="Running adaptation JSON; default is native G1 calibration 108")
    parser.add_argument("--seconds", type=float, default=49.)
    parser.add_argument("--force", type=float, default=410., help="Horizontal torso shove in newtons")
    parser.add_argument("--push-at", type=float, default=6., help="Simulation time of the evaluation shove in seconds")
    parser.add_argument("--duration", type=float, default=.30, help="Shove duration in seconds")
    parser.add_argument("--landing-confirm", type=int, default=16, help="Consecutive settled landing frames at 50 Hz")
    parser.add_argument("--landing-timeout", type=int, default=70, help="Allow recovery after this many landing frames if grounded")
    parser.add_argument("--out", type=Path, help="Save a JSON summary and per-trial NPZ trajectories")
    parser.add_argument("--viewer", action="store_true", help="Run one interactive native scene; Space plays or pauses and Ctrl-drag perturbs a selected body")
    parser.add_argument("--video", type=Path, help="MP4 output for a single speed and direction")
    args = parser.parse_args()
    if (not np.isfinite([*args.speeds, *args.directions, args.seconds, args.force, args.duration, args.push_at]).all()
            or min(args.speeds) <= 0 or args.seconds < 10. or args.force < 0 or args.duration <= 0
            or not .2 <= args.push_at < args.seconds - args.duration
            or args.landing_confirm < 1 or args.landing_timeout < 1):
        parser.error("Use positive speeds, durations and landing intervals, nonnegative force, and at least 10 seconds")
    if (args.video or args.viewer) and len(args.speeds) * len(args.directions) != 1:
        parser.error("--video and --viewer accept one speed and one shove direction")
    gains = RunGains.read(args.run_profile) if args.mode == "run" or (args.mode == "auto" and max(args.speeds) >= 1.34) else None
    player = GetupPlayer(args.recovery)
    gait = G1VelocityPolicy(args.gait, player.model)
    results = []
    for speed in args.speeds:
        use_run = args.mode == "run" or (args.mode == "auto" and speed >= 1.34)
        gait.jog = JogSpeedCalibration.read(args.run_profile, speed) if use_run else None
        for angle in args.directions:
            viewer = LiveViewer(player) if args.viewer else None
            try:
                row, trace = simulate(player, gait, speed, angle, args, gains if use_run else None, viewer)
            except ViewerClosed:
                print(f"Viewer closed at {player.data.time:.2f}s.")
                return
            finally:
                if viewer is not None:
                    viewer.view.close()
            if gait.jog is not None:
                row["jog_speed_calibration"] = {"neural_command_m_s": gait.jog.neural, "hip_swing_multiplier": gait.jog.swing,
                                               "speed_feedback_gain": gait.jog.feedback}
            results.append(row)
            print(json.dumps(row), flush=True)
            if args.out:
                args.out.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(args.out / f"s{speed:g}_a{angle:g}.npz", **trace)
                (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if args.video:
                video(args.video, player.model, trace)


if __name__ == "__main__":
    main()
