"""Reset states from motion clips: a reset can start the robot in a frame of a clip (reference state initialization).

A bank is a dict of arrays, one row per frame, made from the npz clips of tools/convert_motions.py: z (pelvis height over the floor),
quat (w, x, y, z), v and w (world), q and qd (the joints in env.DOF order), cmd (the clip's own velocity in the pelvis's heading frame,
a centered 25-frame mean). It lives in the simulator's World, so the jitted programs take it as an argument.

  kind "walk"   every frame of the clips
  kind "floor"  the frames in which the robot is not up (lying, sitting, getting up): the starts of a get-up

`states` draws n starts from a bank: a random frame, a random heading, and reset noise: roll and pitch +-0.1 rad, joints +-0.1 rad,
root velocities +-0.5 m/s and +-0.52 / 0.52 / 0.78 rad/s.
"""

from __future__ import annotations

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

CMD_WINDOW = 25  # frames of the centered mean that gives a frame's command (0.5 s at 50 Hz)
NOISE_RP = 0.1  # roll / pitch noise [rad]
NOISE_Q = 0.1  # joint noise [rad]
NOISE_V = (0.5, 0.5, 0.2)  # root linear velocity noise [m/s]
NOISE_W = (0.52, 0.52, 0.78)  # root angular velocity noise [rad/s]
Z_MARGIN = {"walk": 0.01, "floor": 0.03}  # lift above the clip's pelvis height, so that a pose from a mocap that sinks a little starts free


def empty(J: int):
    """A bank of one frame, for a reset mode that is off (its rows are never drawn)."""
    return {"z": jnp.full((1,), 0.8), "quat": jnp.asarray([[1.0, 0.0, 0.0, 0.0]]), "v": jnp.zeros((1, 3)), "w": jnp.zeros((1, 3)),
            "q": jnp.zeros((1, J)), "qd": jnp.zeros((1, J)), "cmd": jnp.zeros((1, 3))}  # fmt: skip


def _wxyz(q):  # the clips store (x, y, z, w)
    return np.concatenate([q[:, 3:], q[:, :3]], -1)


def _heading_velocity(quat_xyzw, v, w):
    """Linear velocity in the pelvis's heading frame and the world yaw rate."""
    x, y, z, wq = quat_xyzw.T
    yaw = np.arctan2(2 * (wq * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], w[:, 2]], -1)


def _centered_mean(x, n=CMD_WINDOW):
    pad = np.concatenate([np.repeat(x[:1], n // 2, 0), x, np.repeat(x[-1:], n // 2, 0)], 0)
    c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1])), pad], 0), 0)
    return (c[n:] - c[:-n]) / n


def load(directory: str, kind: str, up_height: float, up_tilt: float, joints: int | None = None, first_frames: int | None = None, source_indices=False):
    """The bank of the clips in `directory`. kind "walk": every frame; "floor": the frames that are not up (pelvis at or below
    up_height, or tilted by up_tilt or more)."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    assert files, f"no clips in {directory} (bash tools/download_data.sh)"
    cols = {k: [] for k in ("z", "quat", "v", "w", "q", "qd", "cmd")}
    if source_indices:
        cols.update(clip=[], frame=[])
    for clip_index, f in enumerate(files):
        d = np.load(f)
        assert int(d["fps"]) == 50, f
        if joints is not None and d["dof_pos"].shape[1] != joints:
            raise ValueError(f"{f}: expected {joints} joint columns")
        quat, z = d["root_quat"], d["root_pos"][:, 2]
        keep = np.ones(len(z), bool)
        if kind == "floor":
            x, y = quat[:, 0], quat[:, 1]
            keep = (z <= up_height) | (2 * (x * x + y * y) >= up_tilt)
            if first_frames is not None:
                keep[first_frames:] = False
        if source_indices:
            frames = np.flatnonzero(keep)
            cols["clip"].append(np.full(len(frames), clip_index, np.int32))
            cols["frame"].append(frames)
        cmd = _centered_mean(_heading_velocity(quat, d["root_lin_vel"], d["root_ang_vel"]))
        for k, a in (("z", z), ("quat", _wxyz(quat)), ("v", d["root_lin_vel"]), ("w", d["root_ang_vel"]), ("q", d["dof_pos"]), ("qd", d["dof_vel"]), ("cmd", cmd)):
            cols[k].append(a[keep])
    if not any(len(z) for z in cols["z"]):
        raise ValueError(f"No {kind} reset frames in {directory}")
    return {k: jnp.asarray(np.concatenate(v, 0), jnp.int32 if k in ("clip", "frame") else jnp.float32) for k, v in cols.items()}


def qmul(a, b):
    """Product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = jnp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jnp.moveaxis(b, -1, 0)
    return jnp.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)  # fmt: skip


def rotate_z(yaw, v):
    """Rotate vectors v (n, 3) about the world z axis by yaw (n,)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


def states(key, bank, n: int, kind: str, q_lo, q_hi, noise: bool = True, return_source=False):
    """n starts from a bank: pelvis height over the floor, quaternion (w, x, y, z), world velocities, joints q, qd, and the clip's
    command. q is clipped to [q_lo, q_hi]."""
    ki, kh, kr, kq, kv, kw = jax.random.split(key, 6)
    i = jax.random.randint(ki, (n,), 0, bank["z"].shape[0])
    yaw = jax.random.uniform(kh, (n,), minval=-jnp.pi, maxval=jnp.pi)
    half = yaw / 2
    quat = qmul(jnp.stack([jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)], -1), bank["quat"][i])
    v, w, q, qd, z, cmd = rotate_z(yaw, bank["v"][i]), rotate_z(yaw, bank["w"][i]), bank["q"][i], bank["qd"][i], bank["z"][i] + Z_MARGIN[kind], bank["cmd"][i]
    if noise:
        rp = jax.random.uniform(kr, (n, 2), minval=-NOISE_RP, maxval=NOISE_RP) / 2  # half angles
        dq = jnp.stack([jnp.ones(n), jnp.sin(rp[:, 0]), jnp.sin(rp[:, 1]), jnp.zeros(n)], -1)
        quat = qmul(quat, dq / jnp.linalg.norm(dq, axis=-1, keepdims=True))  # a small roll and pitch in the pelvis frame
        q = q + jax.random.uniform(kq, q.shape, minval=-NOISE_Q, maxval=NOISE_Q)
        v = v + jax.random.uniform(kv, (n, 3), minval=-1.0, maxval=1.0) * jnp.asarray(NOISE_V)
        w = w + jax.random.uniform(kw, (n, 3), minval=-1.0, maxval=1.0) * jnp.asarray(NOISE_W)
    q = jnp.clip(q, q_lo, q_hi)
    result = z, quat / jnp.linalg.norm(quat, axis=-1, keepdims=True), v, w, q, qd, cmd
    if return_source:
        return result, (bank["clip"][i], bank["frame"][i], yaw)
    return result
