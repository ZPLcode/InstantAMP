"""Prepare native 43 kg Zenbot data from an existing motion_train checkout."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp.zenbot import DOF, KEY_BODIES, MASS
from tools.convert_motions import NumpyUnpickler, convert_recovery_clip as convert


def permutation(names):
    joints = [name + "_joint" for name in DOF]
    if sorted(names) != sorted(joints):
        raise ValueError("Expected the 23 named Zenbot joints in the source manifest")
    return [names.index(name) for name in joints]


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
