"""Convert the motion clips of the AMP and recovery work to 50 Hz npz files in the G1's joint order (instantamp.env.DOF, 29 hinges).

Two sources, both for the Unitree G1:

  amp   legged_lab's walk and run clips (github.com/zitongbai/legged_lab, Apache-2.0; GMR retargets of AMASS ACCAD, Git LFS pickles
        source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/*.pkl). The pickles hold fps, root_pos, root_rot (w, x,
        y, z) and dof_pos in Isaac Lab's joint order, which is the `lab_dof_names` of scripts/tools/retarget/config/g1_29dof.yaml; the
        script reads that list and puts the joints in the G1's order. The pickles are read with an unpickler that only rebuilds numpy
        arrays.
  csv   Unitree's LAFAN1 retarget (HuggingFace lvhaidong/LAFAN1_Retargeting_Dataset, g1/*.csv): 30 fps, one row per frame: root
        position (3), root quaternion (x, y, z, w), the 29 joints in the G1's order. Used for the get-up starts (fallAndGetUp*).

Each output npz holds, at 50 Hz: root_pos (T, 3), root_quat (T, 4; x, y, z, w), root_lin_vel and root_ang_vel (T, 3; world
frame), dof_pos and dof_vel (T, 29), fps (50). Velocities are central differences of the interpolated signals (positions linearly, the
quaternion by slerp).

  python tools/convert_motions.py amp --src data/raw/legged_lab --dst data/amp/walk_run
  python tools/convert_motions.py csv --src data/raw/lafan1_g1 --dst data/recovery/lafan1 --pattern 'fallAndGetUp*.csv'

The clips are not part of the repository: AMASS's and LAFAN1's licenses apply to them.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

OUT_FPS = 50
CLIPS = "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
JOINT_ORDER = "scripts/tools/retarget/config/g1_29dof.yaml"
G1_DOF = ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch",
          "left_wrist_yaw", "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
          "right_wrist_pitch", "right_wrist_yaw")  # fmt: skip. instantamp.env.DOF, without importing JAX


class NumpyUnpickler(pickle.Unpickler):
    """Rebuilds numpy arrays and plain Python values, nothing else."""

    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"not allowed: {module}.{name}")


def slerp(q0, q1, t):
    """Spherical interpolation of unit quaternions (..., 4) at fractions t (...,)."""
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    s = np.sin(th)
    t = t[..., None]
    w0 = np.where(s > 1e-6, np.sin((1 - t) * th) / np.where(s > 1e-6, s, 1), 1 - t)
    w1 = np.where(s > 1e-6, np.sin(t * th) / np.where(s > 1e-6, s, 1), t)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul_xyzw(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def resample(pos, quat, q, fps_in, fps_out=OUT_FPS):
    """Resample root position, root quaternion (xyzw) and joints from fps_in to fps_out (same duration)."""
    n_in = len(pos)
    n_out = int(np.floor((n_in - 1) / fps_in * fps_out)) + 1
    t = np.arange(n_out) * fps_in / fps_out
    i0 = np.minimum(np.floor(t).astype(int), n_in - 2)
    f = t - i0
    lerp = lambda x: x[i0] * (1 - f[:, None]) + x[i0 + 1] * f[:, None]  # noqa: E731
    return lerp(pos), slerp(quat[i0], quat[i0 + 1], f), lerp(q)


def velocities(pos, quat, q, fps=OUT_FPS):
    """Central-difference velocities: root linear (world), root angular (world), joints."""
    dt = 1.0 / fps
    d = lambda x: np.gradient(x, dt, axis=0)  # noqa: E731
    conj = quat * np.asarray([-1.0, -1.0, -1.0, 1.0])
    # world angular velocity from dq/dt * q^-1 = (0, w/2); central difference of q
    dq = d(quat)
    w = 2.0 * qmul_xyzw(dq, conj)[..., :3]
    return d(pos), w, d(q)


def save(path, pos, quat, q, fps_in):
    pos, quat, q = resample(pos, quat, q, fps_in)
    v, w, qd = velocities(pos, quat, q)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, fps=OUT_FPS, root_pos=pos.astype(np.float32), root_quat=quat.astype(np.float32), root_lin_vel=v.astype(np.float32),
             root_ang_vel=w.astype(np.float32), dof_pos=q.astype(np.float32), dof_vel=qd.astype(np.float32))
    return len(pos)


def lab_dof_names(path):
    """The `lab_dof_names` list of legged_lab's retarget config (a plain YAML list; no YAML library needed), without "_joint"."""
    names, on = [], False
    for line in open(path):
        if line.strip() == "lab_dof_names:":
            on = True
        elif on and line.strip().startswith("- "):
            names.append(line.strip()[2:].strip().removesuffix("_joint"))
        elif on and line.strip() and not line.strip().startswith("#"):
            break
    return names


def convert_amp(src, dst):
    lab = lab_dof_names(os.path.join(src, JOINT_ORDER))
    assert sorted(lab) == sorted(G1_DOF), "the lab joint list is not the G1's 29 joints"
    perm = [lab.index(n) for n in G1_DOF]  # joint i is lab column perm[i]
    total = 0
    for path in sorted(glob.glob(os.path.join(src, CLIPS, "*.pkl"))):
        with open(path, "rb") as f:
            d = NumpyUnpickler(f).load()
        fps = float(d["fps"])
        pos = np.asarray(d["root_pos"], np.float64)
        wxyz = np.asarray(d["root_rot"], np.float64)
        quat = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], -1)
        q = np.asarray(d["dof_pos"], np.float64)[:, perm]
        assert q.shape[1] == 29 and len(pos) == len(q), path
        n = save(os.path.join(dst, os.path.basename(path).replace(".pkl", ".npz")), pos, quat, q, fps)
        total += n
        print(f"{os.path.basename(path):55s} {fps:5.1f} fps -> {n} frames ({n / OUT_FPS:5.1f} s)")
    print(f"{total} frames, {total / OUT_FPS:.0f} s")


def convert_csv(src, dst, pattern, fps_in):
    total = 0
    for path in sorted(glob.glob(os.path.join(src, pattern))):
        a = np.loadtxt(path, delimiter=",")
        assert a.shape[1] == 36, f"{path}: {a.shape[1]} columns"
        n = save(os.path.join(dst, os.path.basename(path).replace(".csv", ".npz")), a[:, :3], a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=1, keepdims=True), a[:, 7:], fps_in)
        total += n
        print(f"{os.path.basename(path):40s} {fps_in:4.0f} fps -> {n} frames ({n / OUT_FPS:5.1f} s)")
    print(f"{total} frames, {total / OUT_FPS:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["amp", "csv"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--fps", type=float, default=30.0, help="csv: frame rate of the source (LAFAN1 retarget: 30)")
    a = ap.parse_args()
    if a.kind == "amp":
        convert_amp(a.src, a.dst)
    else:
        convert_csv(a.src, a.dst, a.pattern, a.fps)
    sys.exit(0)
