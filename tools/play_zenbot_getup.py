"""Interact with the 23-DoF Zenbot get-up policy or record a reference/policy comparison."""

import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np

from instantamp import env, zenbot
from instantamp.evaluate import fallen_starts
from instantamp.native_zenbot import ZenbotNative
from instantamp.zenbot_command_actor import COMMAND_HISTORY_WIDTH
from instantamp.zenbot_motion_train_obs import MOTION_TRAIN_WIDTH
from tools.play_getup import ROOT, run
from tools.restore_zenbot_getup import RELEASE, resolve_checkpoint


REFERENCE = ROOT / "data/zenbot23_getup/recovery"


class ZenbotPlayer:
    def __init__(self, checkpoint, reference=None, trial=0, seed=42, *, use_plan_cache=True, speed=0.,
                 lateral=0., yaw_rate=0., visuals=True):
        checkpoint = resolve_checkpoint(checkpoint)
        if reference is None:
            with np.load(checkpoint, allow_pickle=False) as archive:
                commanded = archive["obs_norm/mean"].size in (COMMAND_HISTORY_WIDTH, MOTION_TRAIN_WIDTH)
            reference = ROOT / "data/zenbot23_walk_run_recovery/recovery" if commanded else REFERENCE
        self.reference = Path(reference)
        self.checkpoint = checkpoint
        self.native = ZenbotNative(reference, checkpoint, seed)
        self.speed = float(speed)
        self.lateral, self.yaw_rate = float(lateral), float(yaw_rate)
        if not (-1. <= self.speed <= 3.8 and abs(self.lateral) <= .5 and abs(self.yaw_rate) <= 2.2):
            raise ValueError("Expected vx in [-1, 3.8] m/s, |vy| ≤ 0.5 m/s, |yaw rate| ≤ 2.2 rad/s")
        if any((self.speed, self.lateral, self.yaw_rate)) and not self.native.command_gait:
            raise ValueError("Velocity commands require a command-conditioned Zenbot policy")
        if not use_plan_cache and self.native.native_tracking is not None:
            self.native.native_tracking.db.pop("cache_state", None)
        self.names, self.poses, self.velocities = fallen_starts(reference, zenbot.RESET_LIFT)
        self.delays = np.random.default_rng(seed).integers(3, 8, size=(len(self.names),5))
        self.data = None
        if not visuals:
            self.reset(trial)
            return
        from tools.render import visual_model
        self.model = visual_model("zenbot23_getup", ROOT / "data/zenbot23_getup/meshes")
        self.data = mujoco.MjData(self.model)
        # The projection has the exact native body/joint ordering; the visual skin never enters the dynamics.
        for field,total in (("joint","njnt"),("body","nbody")):
            screen = [getattr(self.model,field)(i).name for i in range(getattr(self.model,total))]
            physical = [getattr(self.native.model,field)(i).name.removeprefix("robot_")
                        for i in range(getattr(self.native.model,total))]
            if screen != physical:
                raise ValueError(f"Zenbot visual and physical {field} order differ")
        self.chest = self.model.body(zenbot.TORSO).id
        self.reset(trial)

    @property
    def policy_label(self):
        return f"Zenbot | {self.checkpoint.parents[1].name}/{self.checkpoint.name}"

    def follow_checkpoint(self, link):
        """Reload a retargeted checkpoint symlink in the existing simulated world."""
        attempted = None

        def poll():
            nonlocal attempted
            try:
                target = link.resolve(strict=True)
                if target == self.checkpoint:
                    return
                signature = (target, target.stat().st_mtime_ns)
                if signature == attempted:
                    return
                attempted = signature
                self.native.replace_policy(target)
                self.checkpoint = target
                print(f"Loaded {target} at simulation time {self.data.time:.2f} s", flush=True)
            except FileNotFoundError:
                pass  # An updated symlink target can still be in the process of being saved.
            except (OSError, ValueError) as error:
                print(f"Could not load the selected policy: {error}", flush=True)
        return poll

    def update_view(self):
        if self.data is None:
            return
        self.data.qpos[:] = self.native.data.qpos
        self.data.qvel[:] = self.native.data.qvel
        self.data.time = self.native.data.time
        mujoco.mj_forward(self.model,self.data)

    def measurements(self):
        """Live pose at the displayed frame and measured foot loads from the last policy step."""
        x, y, z = self.data.xmat[self.chest].reshape(3, 3)[:, 0]
        pitch = np.degrees(np.arctan2(-z, np.hypot(x, y)))
        joints = self.data.qpos[7:]
        mae = np.degrees(np.abs(joints - zenbot.DEFAULT_Q).mean())
        loads = getattr(self.native, "support_load", None)
        foot = "n/a"
        if self.data.time > 0. and loads is not None:
            feet = loads[:2]
            if len(feet) == 2 and np.isfinite(feet).all():
                foot = f"L {feet[0]:.1%}   R {feet[1]:.1%} BW"
        command = (f"vx {self.speed:+.1f}, vy {self.lateral:+.1f} m/s, yaw {self.yaw_rate:+.1f} rad/s\n"
                   "1/2 walk/run  B back  A/D strafe  Q/E turn  X stop  W/S adjust\n") if self.native.command_gait else ""
        return (command + f"Chest pitch (+forward) {pitch:+.1f} deg   |   Joint MAE vs default {mae:.1f} deg\n"
                f"Pelvis height {self.data.qpos[2]:.3f} m   |   Foot load {foot}")

    def handle_key(self, key):
        import glfw
        commands = {"1": (1., 0., 0.), "2": (2.5, 0., 0.), "B": (-.5, 0., 0.),
                    "A": (0., .4, 0.), "D": (0., -.4, 0.), "Q": (0., 0., .6),
                    "E": (0., 0., -.6), "X": (0., 0., 0.)}
        if key in (glfw.KEY_W, glfw.KEY_UP, glfw.KEY_S, glfw.KEY_DOWN):
            increment = .1 if key in (glfw.KEY_W, glfw.KEY_UP) else -.1
            requested = (np.clip(round(self.speed + increment, 1), -1., 3.8), 0., 0.)
        else:
            requested = commands.get(chr(key)) if 0 <= key <= 127 else None
        if requested is None:
            return
        self.speed, self.lateral, self.yaw_rate = requested
        self.native.command = self.native.command.at[0].set(np.asarray(requested, np.float32))

    def reset(self, trial):
        if not 0 <= trial < len(self.names):
            raise ValueError(f"Trial must be between 0 and {len(self.names)-1}")
        self.trial = trial
        self.native.reset(self.poses[trial], self.velocities[trial], self.delays[trial],
                          command=(self.speed, self.lateral, self.yaw_rate))
        if self.data is None:
            return
        mujoco.mj_resetData(self.model,self.data)
        self.update_view()

    def push(self):
        self.native.data.qvel[0] += .20

    def step(self):
        if self.data is None:
            return self.native.step()
        # MuJoCo's passive viewer supplies mouse perturbation wrenches on its own model.
        self.native.data.xfrc_applied[:] = self.data.xfrc_applied
        self.native.data.qfrc_applied[:] = self.data.qfrc_applied
        foot_load = self.native.step()
        self.update_view()
        return foot_load


def record(player, path, seconds, *, camera_center=None, camera_distance=2.8):
    import subprocess
    from PIL import Image, ImageDraw
    from tools.render import font

    clip,offset = player.names[player.trial]
    with np.load(player.reference/clip,allow_pickle=False) as source:
        ref = np.c_[source["root_pos"], source["root_quat"][:,[3,0,1,2]], source["dof_pos"]].astype(float)
    ref = ref[round(offset/env.DT):]
    ref[:,:2] += player.data.qpos[:2] - ref[0,:2]
    d = mujoco.MjData(player.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = camera_distance, 120., -14.
    camera.lookat[:] = [*player.data.qpos[:2], .49] if camera_center is None else camera_center
    width,height,header = 480,320,56
    path.parent.mkdir(parents=True,exist_ok=True)
    cmd = ["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{2*width}x{height+header}",
           "-r","25","-i","-","-c:v","libx264","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
    with mujoco.Renderer(player.model,height,width) as renderer, subprocess.Popen(cmd,stdin=subprocess.PIPE) as writer:
        fonts = font(18),font(13)
        steps = round(seconds/env.DT)
        for i in range(steps+1):
            if i % 2 == 0:
                frame = Image.new("RGB",(width*2,height+header),"#18232e")
                draw = ImageDraw.Draw(frame)
                pose = player.data.qpos
                reference = ref[min(i,len(ref)-1)]
                for col,(q,label) in enumerate(((reference, "Retargeted reference" if i<len(ref) else "Reference ended | final frame"),
                                                (pose, "Closed-loop Zenbot policy"))):
                    x = col*width
                    draw.text((x+9,4),label,font=fonts[0],fill="white")
                    draw.text((x+9,29),f"{i*env.DT:.2f}s  |  {Path(clip).stem}",font=fonts[1],fill="#cddbe9")
                    d.qpos[:] = q
                    mujoco.mj_forward(player.model,d)
                    renderer.update_scene(d,camera=camera)
                    frame.paste(Image.fromarray(renderer.render()),(x,header))
                writer.stdin.write(frame.tobytes())
            if i < steps:
                player.step()
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError("Could not encode the Zenbot comparison movie")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=RELEASE,help="Policy file or a live symlink to the selected policy")
    p.add_argument("--reference",type=Path,help="Override the recovery motion directory for this policy")
    p.add_argument("--speed",type=float,default=0.,help="Requested forward speed in m/s for a command-conditioned get-up policy")
    p.add_argument("--lateral",type=float,default=0.,help="Leftward speed in m/s")
    p.add_argument("--yaw-rate",type=float,default=0.,help="Leftward yaw rate in rad/s")
    p.add_argument("--trial",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-plan-cache", action="store_true", help="Recompute the motor plan from the current measured state")
    p.add_argument("--video",type=Path,help="Render a real-time two-panel MP4 instead of opening a window")
    p.add_argument("--seconds",type=float,default=12.)
    p.add_argument("--play", action="store_true", help="Start the live viewer in playback mode")
    args=p.parse_args()
    checkpoint_link = args.checkpoint.absolute() if args.checkpoint.is_symlink() else None
    try:
        args.checkpoint = resolve_checkpoint(args.checkpoint)
    except (OSError, ValueError) as error:
        p.error(str(error))
    if not np.isfinite(args.seconds) or args.seconds<=0:
        p.error("--seconds must be positive")
    print(f"Checkpoint {args.checkpoint.resolve()}", flush=True)
    try:
        player=ZenbotPlayer(args.checkpoint,args.reference,args.trial,args.seed,use_plan_cache=not args.no_plan_cache,
                            speed=args.speed, lateral=args.lateral, yaw_rate=args.yaw_rate)
    except ValueError as error:
        p.error(str(error))
    print(player.names[player.trial], "motor delays (physics steps)",player.delays[player.trial].tolist(),flush=True)
    if args.video:
        record(player,args.video,args.seconds)
    else:
        run(player, lambda: player.native.support_load[2:4] if player.data.time else (0.0, 0.0),
            play=args.play, poll_policy=player.follow_checkpoint(checkpoint_link) if checkpoint_link else None,
            measurements=player.measurements, handle_key=player.handle_key if player.native.command_gait else None)


if __name__ == "__main__":
    main()
