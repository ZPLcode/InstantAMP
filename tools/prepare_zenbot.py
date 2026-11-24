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
from instantamp import zenbot
from instantamp.zenbot_models import ZENBOT23
from tools.convert_motions import (
    NumpyUnpickler,
    convert_recovery_clip as convert,
    lab_dof_names,
)
from tools.zenbot_step_width import MAX_FOOT_SHIFT, TARGET_WIDTH, StepWidth

LOCOMOTION_BANK = "zenbot_amp_frax_walk_sole8"


def permutation(names):
    """Map source joint columns to the 23-DoF articulation used for retargeting."""
    joints = [name + "_joint" for name in ZENBOT23.dof]
    if sorted(names) != sorted(joints):
        raise ValueError(
            "Expected the 23 named Zenbot joints in the source manifest"
        )
    return [names.index(name) for name in joints]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _motion_weights(config_path):
    """Read the source experiment's positive sampling weights without importing it."""
    assignments = [
        node.value
        for node in ast.parse(config_path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MOTION_WEIGHTS"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError(
            f"Expected one MOTION_WEIGHTS assignment in {config_path}"
        )
    weights = ast.literal_eval(assignments[0])
    if not weights or not np.isfinite(list(weights.values())).all():
        raise ValueError(f"Invalid motion weights in {config_path}")
    positive = {name: weight for name, weight in weights.items() if weight > 0}
    if not positive:
        raise ValueError(f"No positive motion weights in {config_path}")
    return positive


def prepare_locomotion(source, dest, bank_name=LOCOMOTION_BANK):
    """Convert a named gait bank with Motion Train's active clip weights."""
    cfg = (
        source / "robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py"
    )
    weights = _motion_weights(cfg)
    bank = source / "robolab/data/motions" / bank_name
    source_manifest = bank / "manifest.json"
    expected_hashes = None
    if bank_name == LOCOMOTION_BANK or source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text())
        if manifest["quaternion_order"] != "wxyz":
            raise ValueError(
                f"Unexpected quaternion order in {source_manifest}"
            )
        order = permutation(manifest["joint_names"])
        expected_hashes = {
            c["name"]: c["output_sha256"] for c in manifest["clips"]
        }
    else:
        order = permutation(
            [
                name + "_joint"
                for name in lab_dof_names(
                    source / "robolab/scripts/tools/retarget/config/zenbot.yaml"
                )
            ]
        )
    directory = dest / "walk_run"
    extra = {p.stem for p in directory.glob("*.npz")} - weights.keys()
    if extra:
        raise ValueError(
            f"Use a fresh motion directory. Extra clips in {directory}: {sorted(extra)}"
        )
    frames = 0
    hashes = {}
    widener = StepWidth() if bank_name == LOCOMOTION_BANK else None
    shifts = {}
    for name in weights:
        path = bank / (name + ".pkl")
        hashes[name] = _digest(path)
        if expected_hashes is not None and hashes[name] != expected_hashes.get(
            name
        ):
            raise ValueError(f"Source hash does not match the manifest: {path}")
        with path.open("rb") as stream:
            clip = NumpyUnpickler(stream).load()
        joints = clip["dof_pos"]
        pos = clip["root_pos"]
        if widener is not None:
            adjusted, shift = widener.adjust(
                pos, clip["root_rot"], joints[:, order], float(clip["fps"])
            )
            pos, adjusted = widener.correct_footfalls(
                pos, clip["root_rot"], adjusted, float(clip["fps"])
            )
            joints = joints.copy()
            joints[:, order] = adjusted
            if shift:
                shifts[name] = round(shift, 6)
        frames += convert(
            directory / (name + ".npz"),
            pos,
            clip["root_rot"],
            joints,
            order,
            float(clip["fps"]),
        )
    sampling = {
        "weights": weights,
        "source_bank": bank_name,
        "source_sha256": hashes,
    }
    if widener is not None:
        sampling["step_width"] = {
            "target_m": TARGET_WIDTH,
            "max_per_foot_m": MAX_FOOT_SHIFT,
            "per_foot_shift_m": shifts,
        }
        sampling["footfall_correction"] = (
            "level only in settled walking stance and preserve heel strike and toe-off"
        )
    (directory / "sampling.json").write_text(
        json.dumps(sampling, indent=2) + "\n"
    )
    print(
        f"Prepared {len(weights)} weighted locomotion clips, {frames} frames at 50 Hz, in {directory}"
    )


def prepare(
    source,
    legged_lab,
    dest,
    meshes=False,
    locomotion=False,
    locomotion_bank=LOCOMOTION_BANK,
):
    """Convert the licensed LeggedLab recovery bank in the matching joint order."""
    robot = source / "robolab/data/robots/roboparty/zenbot"
    bank = (
        legged_lab
        / "source/legged_lab/legged_lab/data/MotionData/zenbot_23dof_43kg/amp/get_up_frax"
    )
    manifest = json.loads((bank / "manifest.json").read_text())
    xml = robot / "mjcf/zenbot.xml"
    xml_hash = _digest(xml)
    if (
        xml_hash != _digest(ZENBOT23.xml_path)
        or xml_hash != manifest["native_frax_run"]["mjcf_sha256"]
    ):
        raise ValueError(
            "The source robot and the recovery clips must use the bundled 23-DoF MJCF"
        )
    if (
        manifest["root_quaternion_order"] != "wxyz"
        or tuple(manifest["key_body_names"]) != zenbot.KEY_BODIES
        or not np.isclose(manifest["robot_mass_kg"], ZENBOT23.mass)
    ):
        raise ValueError("Unexpected Zenbot recovery manifest")
    order = permutation(manifest["joint_names"])
    frames = 0
    if locomotion:
        extra = {p.stem for p in (dest / "recovery").glob("*.npz")} - {
            Path(c["file"]).stem for c in manifest["clips"]
        }
        if extra:
            raise ValueError(
                f"Use a fresh recovery directory. Unexpected clips: {sorted(extra)}"
            )
        prepare_locomotion(source, dest, locomotion_bank)
    for info in manifest["clips"]:
        path = bank / info["file"]
        if _digest(path) != info["sha256"]:
            raise ValueError(f"Clip checksum mismatch: {path}")
        with path.open("rb") as stream:
            clip = NumpyUnpickler(stream).load()
        frames += convert(
            dest / "recovery" / path.with_suffix(".npz").name,
            clip["root_pos"],
            clip["root_rot"],
            clip["dof_pos"],
            order,
            float(clip["fps"]),
            clip["key_body_pos"],
        )
    if meshes:
        shutil.copytree(robot / "meshes", dest / "meshes", dirs_exist_ok=True)
    print(
        f"Prepared {len(manifest['clips'])} get-up clips, {frames} frames at 50 Hz, in {dest}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local motion_train checkout")
    parser.add_argument(
        "--legged-lab",
        type=Path,
        help="checkout containing the native get_up_frax bank",
    )
    parser.add_argument("--dest", type=Path)
    parser.add_argument(
        "--meshes", action="store_true", help="also prepare visual meshes"
    )
    parser.add_argument(
        "--locomotion",
        action="store_true",
        help="include walk/run AMP clips and weights",
    )
    parser.add_argument(
        "--locomotion-bank",
        default=LOCOMOTION_BANK,
        help=f"walk/run source directory (default: {LOCOMOTION_BANK})",
    )
    parser.add_argument(
        "--locomotion-only",
        action="store_true",
        help="prepare only walk_run using the motion_train checkout",
    )
    args = parser.parse_args()
    dest = args.dest or Path("data") / ZENBOT23.mixed_task
    if args.locomotion_only:
        if args.meshes or args.locomotion:
            parser.error("--locomotion-only prepares only the walk_run bank")
        prepare_locomotion(args.source, dest, args.locomotion_bank)
    elif args.legged_lab is None:
        parser.error("--legged-lab is required to prepare recovery motions")
    else:
        prepare(
            args.source,
            args.legged_lab,
            dest,
            args.meshes,
            args.locomotion,
            args.locomotion_bank,
        )
