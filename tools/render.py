"""GIFs of an InstantAMP checkpoint: one walking clip (commands as in training, pushes on) and one get-up clip (an env that starts down).

  bash tools/download_meshes.sh       (once)
  python tools/render.py runs/instantamp [--ckpt 0014] [--walk_s 10 --getup_s 6] [--out figures/instantamp]

The rollout runs in the training simulator (instantamp/sim.py, averaged policy, no action noise); the frames are drawn with
assets/g1_mjlab.xml and its visual meshes (assets/meshes). Rendering needs OpenGL (MUJOCO_GL=egl by default).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from instantamp import env, nets  # noqa: E402
from instantamp.inference import G1StandController, load, policy_memory, policy_step, reset_policy_memory  # noqa: E402
from instantamp.sim import XML, ZENBOT_XML, Sim, build_scene, unflatten  # noqa: E402

MESH_DIR = os.path.join(os.path.dirname(XML), "meshes")
W, H, FPS = 640, 400, 25


def visual_model(robot="g1", mesh_dir=None):
    if robot == "zenbot23_getup":
        spec = mujoco.MjSpec.from_file(ZENBOT_XML)
        spec.meshdir = os.path.abspath(mesh_dir or "data/zenbot23_getup/meshes")
        for mesh in spec.meshes:
            mesh.file = os.path.basename(mesh.file)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
        return spec.compile()
    assert os.path.isdir(MESH_DIR), "no meshes in assets/meshes: run bash tools/download_meshes.sh"
    spec = mujoco.MjSpec.from_file(XML)
    spec.meshdir = MESH_DIR
    for k in list(spec.keys):
        spec.delete(k)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.84, 0.86], rgb2=[0.72, 0.74, 0.77], width=512, height=512)  # fmt: skip
    mat = spec.add_material(name="grid", texrepeat=[1, 1])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    m = spec.compile()
    assert m.nq == 7 + env.J, m.nq
    return m


def rollout(run, ckpt, n, seconds, seed):
    cfg = json.load(open(os.path.join(run, "config.json")))
    pi, norm, ode = load(ckpt)
    controller = isinstance(pi, G1StandController)
    sim = Sim(n, cfg.get("nconmax", 64), cfg.get("njmax", 640), cfg["walk_dir"], cfg["floor_dir"], not controller, seed, cfg["reset_floor"],
              cfg["prob_shove"], bool(cfg["push"]), robot=cfg.get("robot", "g1"), assist_newtons=0., obs_noise=cfg.get("robot", "g1") == "g1",
              g1_style=bool(cfg.get("g1_style", 0)), getup_rise_weight=cfg.get("getup_rise_weight", -0.25),
              g1_support_friction=bool(cfg.get("g1_support_friction", 0)), getup_reset_frames=cfg.get("getup_reset_frames", 20),
              g1_phase_amp=bool(cfg.get("g1_phase_share", 0.)))  # fmt: skip

    if controller:
        sim.episode_steps = round(seconds / env.DT) + 1

    @jax.jit
    def go(s, key):
        def body(carry, k):
            s, memory = carry
            out, memory = policy_step(pi, norm, s.obs, memory, cfg.get("obs_clip", 10.))
            s1, tr = sim.step(s, nets.ode_step(out, s.a_ode, *ode) if ode is not None else out)
            done = tr.terminated | tr.truncated
            return (s1, reset_policy_memory(memory, done)), (tr.qpos1, tr.qvel1, done, tr.terminated, s.c_obs, s.was_up, s.down)

        return jax.lax.scan(body, (s, policy_memory(pi, s.obs)), jax.random.split(key, int(seconds / env.DT)))

    _, out = go(sim.init(jax.random.PRNGKey(seed)), jax.random.PRNGKey(seed + 1))
    return [np.asarray(x) for x in out]


def body_velocity(q, v):
    S = unflatten(jnp.asarray(q), jnp.asarray(v))
    R, vw = np.asarray(S[1]), np.asarray(S[2])
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    vx = np.cos(yaw) * vw[..., 0] + np.sin(yaw) * vw[..., 1]
    vy = -np.sin(yaw) * vw[..., 0] + np.cos(yaw) * vw[..., 1]
    return np.stack([vx, vy], -1), np.asarray(S[3])[..., 2]


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(m, qpos, lines, title, path):
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 3.2, -15.0, 135.0
    look = qpos[0, :3].copy()
    f_big, f_small = font(20), font(16)
    frames = []
    step = int(round(1 / (FPS * env.DT)))
    for i in range(0, len(qpos), step):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        look = 0.85 * look + 0.15 * qpos[i, :3]  # a smoothed follow camera
        cam.lookat[:] = [look[0], look[1], 0.6]
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        dr.text((14, 10), title, fill=(255, 255, 255), font=f_big)
        for j, t in enumerate(lines[i]):
            dr.text((14, 40 + 22 * j), t, fill=(225, 230, 235), font=f_small)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    r.close()
    print(f"wrote {path} ({len(frames)} frames)")


def render_evaluated_trial(archive, trial, out):
    """Replay the exact post-step states saved by the fixed-start evaluator."""
    with np.load(archive, allow_pickle=False) as data:
        robot = str(data["robot"])
        if not 0 <= trial < len(data["clips"]):
            raise ValueError(f"trial must be in [0, {len(data['clips'])-1}]")
        if not np.isclose(data["dt"], env.DT):
            raise ValueError("Unexpected trajectory control rate")
        clip = str(data["clips"][trial])
        offset = float(data["offsets_s"][trial])
        status = "passed" if data["success"][trial] else "failed"
        title = f"Get-up trial {trial} | {data['checkpoint']} ({data['which']})"
        qpos = data["qpos"][:, trial]
        loads = data["foot_load_bw"][:, trial]
        ended = np.flatnonzero(data["terminated"][:, trial])
        if len(ended):
            qpos = qpos[:ended[0]+2]  # include the terminal physical pose
        lines = [
            [f"{clip} from {offset:g} s | assistance {float(data['assist_n']):g} N",
             f"{i * env.DT:.2f} s | pelvis {pose[2]:.2f} m" + (f" | foot load {loads[i-1]:.2f} BW" if i else ""),
             f"Final standing check: {status}"]
            for i, pose in enumerate(qpos)
        ]
    if robot == "g1_getup" and not os.path.isdir(MESH_DIR):
        model = build_scene(robot_name=robot)
        # The physics model's collision geoms belong to a hidden viewer group.
        model.geom_group[:] = 0
        model.geom_rgba[:, :] = [0.45, 0.70, 0.89, 1.0]
        model.geom_rgba[model.geom("floor").id] = [0.24, 0.27, 0.30, 1.0]
        model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    else:
        model = visual_model(robot)
    path = Path(out) if out else Path(archive).with_name(f"{Path(archive).stem}_trial{trial:02}.gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    render(model, qpos, lines, title, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="training run directory or a recovery .npz from evaluate --trajectories")
    p.add_argument("--trial", type=int, default=0, help="zero-based fixed trial to render from an evaluation archive")
    p.add_argument("--mesh_dir", help="visual mesh directory for Zenbot")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--walk_s", type=float, default=10.0)
    p.add_argument("--getup_s", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="")
    a = p.parse_args()
    if Path(a.run).is_file():
        return render_evaluated_trial(a.run, a.trial, a.out)
    ckpt = os.path.join(a.run, "ckpt", a.ckpt + ".npz") if a.ckpt else sorted(glob.glob(os.path.join(a.run, "ckpt", "*.npz")))[-1]
    name = os.path.basename(ckpt)[:-4]
    meta = json.load(open(ckpt[:-4] + ".json"))
    out = a.out or os.path.join("figures", os.path.basename(os.path.normpath(a.run)))
    T = int(max(a.walk_s, a.getup_s) / env.DT)
    q, v, done, term, cobs, was_up, down = rollout(a.run, ckpt, a.n, max(a.walk_s, a.getup_s), a.seed)
    vel, wz = body_velocity(q, v)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    m = visual_model(cfg.get("robot", "g1"), a.mesh_dir)
    tag = f"{os.path.basename(os.path.normpath(a.run))} ckpt {name}"
    if "train_s" in meta:
        tag += f" ({meta['train_s'] / 60:.0f} min)"

    def lines_for(e, n_steps):
        return [[f"cmd   vx {cobs[i, e, 0]:+.2f}  vy {cobs[i, e, 1]:+.2f}  wz {cobs[i, e, 2]:+.2f}",
                 f"robot vx {vel[i, e, 0]:+.2f}  vy {vel[i, e, 1]:+.2f}  wz {wz[i, e]:+.2f}",
                 "up" if was_up[i, e] else "getting up"] for i in range(n_steps)]  # fmt: skip

    if cfg.get("robot") in ("zenbot23_getup", "g1_getup"):
        n = int(a.getup_s / env.DT)
        e = 0
        labels = [[f"pelvis height {q[i, e, 2]:.2f} m", "assistance 0 N"] for i in range(n)]
        render(m, q[:n, e], labels, f"get-up | {tag}", f"{out}_getup.gif")
        return

    # walking: an env that starts up and is not reset during the clip, with the largest commanded speed
    n_w = int(a.walk_s / env.DT)
    ok = (~down[0]) & (done[:n_w].sum(0) == 0)
    speed = np.linalg.norm(cobs[:n_w, :, :2], axis=-1).mean(0)
    e = int(np.argmax(np.where(ok, speed, -1)))
    print(f"walking: env {e}, mean commanded speed {speed[e]:.2f} m/s")
    render(m, q[:n_w, e], lines_for(e, n_w), f"walking | {tag}", f"{out}_walk.gif")

    # get-up: an env that starts down and rises without being reset
    n_g = int(a.getup_s / env.DT)
    rose = down[0] & (done[:n_g].sum(0) == 0) & was_up[n_g - 1]
    if rose.any():
        e = int(np.nonzero(rose)[0][0])
        print(f"get-up: env {e}, up after {np.argmax(was_up[:, e]) * env.DT:.2f} s")
        render(m, q[:n_g, e], lines_for(e, n_g), f"get-up | {tag}", f"{out}_getup.gif")
    else:
        print(f"no env started down and rose within {a.getup_s} s (down starts: {int(down[0].sum())})")


if __name__ == "__main__":
    main()
