"""Prepare native 43 kg Zenbot data from an existing motion_train checkout."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp.zenbot import DOF, KEY_BODIES, MASS
from tools.convert_motions import NumpyUnpickler, OUT_FPS, qmul_xyzw, resample


def permutation(names):
    joints = [name + "_joint" for name in DOF]
    if sorted(names) != sorted(joints):
        raise ValueError("Expected the 23 named Zenbot joints in the source manifest")
    return [names.index(name) for name in joints]


def convert(path, pos, wxyz, joints, order, fps, landmarks=None):
    if joints.shape != (len(pos), len(DOF)) or pos.shape[1:] != (3,) or wxyz.shape != (len(pos), 4):
        raise ValueError(f"Invalid motion dimensions: {path}")
    if len(pos) < 2 or fps <= 0 or not all(np.isfinite(a).all() for a in (pos, wxyz, joints)):
        raise ValueError(f"Invalid motion values: {path}")
    norm = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if (norm < 1e-6).any():
        raise ValueError(f"Zero quaternion: {path}")
    quat = (wxyz / norm)[:, [1, 2, 3, 0]]
    # Maintain a continuous quaternion lift for the finite-difference angular velocities.
    signs = np.where((quat[1:] * quat[:-1]).sum(1) < 0, -1, 1)
    quat[1:] *= np.cumprod(signs)[:, None]
    joints = joints[:, order]
    # legged_lab differentiates at the native rate, repeats the last velocity,
    # then interpolates velocities when querying the motion at simulation time.
    delta = qmul_xyzw(quat[1:], quat[:-1] * [-1, -1, -1, 1])
    sin_half = np.linalg.norm(delta[:, :3], axis=1, keepdims=True)
    angle = 2 * np.arctan2(sin_half, delta[:, 3:4])
    ang = delta[:, :3] * (fps * angle / np.maximum(sin_half, 1e-12))
    vel = np.concatenate((np.diff(pos, axis=0) * fps, ang, np.diff(joints, axis=0) * fps), -1)
    features = [joints, np.concatenate((vel, vel[-1:]))]
    if landmarks is not None:
        if landmarks.shape != (len(pos), len(KEY_BODIES), 3) or not np.isfinite(landmarks).all():
            raise ValueError(f"Invalid recovery landmarks: {path}")
        features.append(landmarks.reshape(len(pos), -1))
    pos, quat, values = resample(pos, quat, np.concatenate(features, -1), fps)
    j = len(DOF)
    motion = dict(fps=OUT_FPS, root_pos=pos, root_quat=quat, dof_pos=values[:, :j],
                  root_lin_vel=values[:, j:j+3], root_ang_vel=values[:, j+3:j+6], dof_vel=values[:, j+6:2*j+6])
    if landmarks is not None:
        motion["key_body_pos"] = values[:, 2*j+6:].reshape(len(pos), len(KEY_BODIES), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            np.savez(tmp, **{k: np.asarray(v, np.float32) for k, v in motion.items()})
            tmp.flush()
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
    return len(pos)


def prepare(source, legged_lab, dest, meshes=False):
    robot = source / "robolab/data/robots/roboparty/zenbot"
    bank = legged_lab / "source/legged_lab/legged_lab/data/MotionData/zenbot_23dof_43kg/amp/get_up_frax"
    manifest = json.loads((bank / "manifest.json").read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    xml = robot / "mjcf/zenbot.xml"
    bundled = Path(__file__).resolve().parents[1] / "assets/zenbot_23dof_43kg.xml"
    if digest(xml) != digest(bundled) or digest(xml) != manifest["native_frax_run"]["mjcf_sha256"]:
        raise ValueError("The source robot and the recovery clips must use the bundled 23-DoF MJCF")
    if (manifest["root_quaternion_order"] != "wxyz" or tuple(manifest["key_body_names"]) != KEY_BODIES
            or not np.isclose(manifest["robot_mass_kg"], MASS)):
        raise ValueError("Unexpected Zenbot recovery manifest")
    order = permutation(manifest["joint_names"])
    frames = 0
    for info in manifest["clips"]:
        path = bank / info["file"]
        if digest(path) != info["sha256"]:
            raise ValueError(f"Clip checksum mismatch: {path}")
        with path.open("rb") as stream:
            clip = NumpyUnpickler(stream).load()
        frames += convert(dest / "recovery" / path.with_suffix(".npz").name, clip["root_pos"], clip["root_rot"],
                          clip["dof_pos"], order, float(clip["fps"]), clip["key_body_pos"])
    if meshes:
        shutil.copytree(robot / "meshes", dest / "meshes", dirs_exist_ok=True)
    print(f"Prepared {len(manifest['clips'])} equally weighted get-up clips, {frames} frames at 50 Hz, in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local motion_train checkout")
    parser.add_argument("--legged-lab", type=Path, required=True, help="checkout containing the native get_up_frax bank")
    parser.add_argument("--dest", type=Path, default=Path("data/zenbot23_getup"))
    parser.add_argument("--meshes", action="store_true", help="also prepare visual meshes")
    args = parser.parse_args()
    prepare(args.source, args.legged_lab, args.dest, args.meshes)
