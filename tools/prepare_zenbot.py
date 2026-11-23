"""Prepare native 43 kg Zenbot data from an existing motion_train checkout."""
import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp.zenbot import DOF, KEY_BODIES, MASS
from tools.convert_motions import NumpyUnpickler, convert_recovery_clip as convert, lab_dof_names


def permutation(names):
    joints = [name + "_joint" for name in DOF]
    if sorted(names) != sorted(joints):
        raise ValueError("Expected the 23 named Zenbot joints in the source manifest")
    return [names.index(name) for name in joints]


def prepare_locomotion(source, dest):
    """Read the active AMP recipe without importing Isaac Lab or executing its config."""
    cfg = source / "robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py"
    config = ast.parse(cfg.read_text())
    weights, = [ast.literal_eval(node.value) for node in config.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MOTION_WEIGHTS"
                                                        for t in node.targets)]
    weights = {name: weight for name, weight in weights.items() if weight > 0}
    if not weights or not np.isfinite(list(weights.values())).all():
        raise ValueError(f"No usable positive motion weights in {cfg}")
    # motion_train revisions use either the original Lab retargets or the newer Frax retargets.
    # Discover the literal directory named by the same config that supplies the sampling weights.
    bank_name = "zenbot_lab"
    for node in ast.walk(config):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "motion_data_dir"
                                                for t in node.targets):
            if not (isinstance(node.value, ast.Call) and node.value.args and
                    isinstance(node.value.args[-1], ast.Constant) and isinstance(node.value.args[-1].value, str)):
                raise ValueError(f"Expected a literal locomotion directory in {cfg}")
            bank_name = node.value.args[-1].value
    bank = source / "robolab/data/motions" / bank_name
    source_manifest = bank / "manifest.json"
    expected_hashes = {}
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text())
        if manifest["quaternion_order"] != "wxyz":
            raise ValueError(f"Unexpected quaternion order in {source_manifest}")
        order = permutation(manifest["joint_names"])
        expected_hashes = {c["name"]: c["output_sha256"] for c in manifest["clips"]}
    else:
        order = permutation([name + "_joint" for name in lab_dof_names(source / "robolab/scripts/tools/retarget/config/zenbot.yaml")])
    directory = dest / "walk_run"
    extra = {p.stem for p in directory.glob("*.npz")} - weights.keys()
    if extra:
        raise ValueError(f"Use a fresh motion directory. Extra clips in {directory}: {sorted(extra)}")
    frames = 0
    hashes = {}
    for name in weights:
        path = bank / (name + ".pkl")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_manifest.is_file() and hashes[name] != expected_hashes.get(name):
            raise ValueError(f"Source hash does not match the manifest: {path}")
        with path.open("rb") as stream:
            clip = NumpyUnpickler(stream).load()
        frames += convert(directory / (name + ".npz"), clip["root_pos"], clip["root_rot"],
                          clip["dof_pos"], order, float(clip["fps"]))
    (directory / "sampling.json").write_text(json.dumps(dict(weights=weights, source_bank=bank_name,
                                                            source_sha256=hashes), indent=2) + "\n")
    print(f"Prepared {len(weights)} weighted locomotion clips, {frames} frames at 50 Hz, in {directory}")


def prepare(source, legged_lab, dest, meshes=False, locomotion=False):
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
    if locomotion:
        extra = {p.stem for p in (dest / "recovery").glob("*.npz")} - {Path(c["file"]).stem for c in manifest["clips"]}
        if extra:
            raise ValueError(f"Use a fresh recovery directory. Unexpected clips: {sorted(extra)}")
        prepare_locomotion(source, dest)
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
    parser.add_argument("--legged-lab", type=Path, help="checkout containing the native get_up_frax bank")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--meshes", action="store_true", help="also prepare visual meshes")
    parser.add_argument("--locomotion", action="store_true", help="include the active motion_train walk/run AMP bank and weights")
    parser.add_argument("--locomotion-only", action="store_true", help="prepare only walk_run using the motion_train checkout")
    args = parser.parse_args()
    dest = args.dest or Path("data/zenbot23_walk_run_recovery" if args.locomotion or args.locomotion_only else "data/zenbot23_getup")
    if args.locomotion_only:
        if args.meshes or args.locomotion:
            parser.error("--locomotion-only prepares only the walk_run bank")
        prepare_locomotion(args.source, dest)
    elif args.legged_lab is None:
        parser.error("--legged-lab is required to prepare recovery motions")
    else:
        prepare(args.source, args.legged_lab, dest, args.meshes, args.locomotion)
