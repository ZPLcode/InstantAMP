"""Prepare the ten equally weighted demonstrations from LeggedLab's G1 get-up task."""
import argparse
import sys
import tempfile
from pathlib import Path

from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.convert_motions import G1_DOF, JOINT_ORDER, convert_recovery_clip, lab_dof_names

ACTIVE_CLIPS = (
    "fallAndGetUp1_subject1_1060_1150", "fallAndGetUp1_subject1_1400_1480",
    "fallAndGetUp1_subject1_2100_2200", "fallAndGetUp1_subject5_2500_2600",
    "fallAndGetUp2_subject2_850_1050", "fallAndGetUp2_subject3_900_1000",
    "fallAndGetUp6_subject1_530_600", "fallAndGetUp6_subject1_650_700",
    "fallAndGetUp6_subject1_1080_1180", "fallAndGetUp6_subject1_1630_1690",
)


def prepare(source, dest):
    bank = source / "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/get_up"
    names = lab_dof_names(source / JOINT_ORDER)
    if sorted(names) != sorted(G1_DOF):
        raise ValueError("Unexpected G1 joint names in LeggedLab's retarget config")
    order = [names.index(joint) for joint in G1_DOF]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "recovery"
        frames = 0
        for name in ACTIVE_CLIPS:
            clip = load(bank / f"{name}.pkl")
            frames += convert_recovery_clip(staged / f"{name}.npz", clip["root_pos"], clip["root_rot"],
                                            clip["dof_pos"], order, float(clip["fps"]), clip["key_body_pos"])
        if dest.exists():
            existing = {p.name for p in dest.glob("*.npz")}
            expected = {p.name for p in staged.glob("*.npz")}
            if expected != existing or any((staged / f).read_bytes() != (dest / f).read_bytes() for f in expected):
                raise FileExistsError(f"Different recovery data already exists in {dest}. Choose a new --dest.")
        else:
            staged.rename(dest)
    print(f"Prepared {len(ACTIVE_CLIPS)} clips, {frames} frames at 50 Hz in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeggedLab checkout containing the get_up motion files")
    parser.add_argument("--dest", type=Path, default=Path("data/g1_getup/recovery"))
    args = parser.parse_args()
    prepare(args.source, args.dest)
