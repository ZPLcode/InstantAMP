"""Package a trained G1 PPO+AMP motor residual with measured feedback trajectories."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from instantamp import env
from instantamp.g1_ground_phase import HandContacts, knee_extension_frame
from instantamp.sim import build_scene
from instantamp.stability import contiguous
from tools.compare_recovery_rollouts import first_upright
from tools.score_g1_tracking import load_native_reference


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("observations", "qpos", "qvel", "action", "gain", "support_jacobian", "stance_parameters", "start", "end")


def source_phases(starts, motions):
    hands = HandContacts(build_scene(robot_name="g1_getup"))
    phases = {}
    for row in starts:
        poses, velocities = load_native_reference(motions / row["clip"])
        offset = round(row["offset_s"] / env.DT)
        if not 0 <= offset < len(poses):
            raise ValueError(f"Offset outside reference: {row['clip']} at {row['offset_s']} s")
        q, v = poses[offset:], velocities[offset:]
        # The planner's original reference annotation uses 4.5 cm clearance. Physical
        # palm force is measured separately on every simulated step of the controller.
        near = hands.reference_clearances(q)[:, :, 0] < 0.045
        rise = first_upright(q, 3)
        if rise is None:
            rise = len(q) - 1
        held = first_upright(q, 10)
        stop = len(q) if held is None else held + 1
        palm_required = any(b - a >= 5 for a, b in contiguous(near[:stop].any(axis=1)))
        sides = [[(a, b) for a, b in contiguous(mask) if b - a >= 3] for mask in near[:rise + 1].T]
        trial = row["trial"]
        if trial in phases:
            raise ValueError(f"Duplicate teacher trial {trial}")
        phases[trial] = (q, v, rise, sides, palm_required, knee_extension_frame(q, held), row["first"])
    return phases


def align_start(source, poses, capture, slow, retime):
    """Build the planner's reference and support_timing arrays for one local trajectory."""
    q, v, rise, sides, required, knee, teacher_rise_s = source
    if not np.allclose(poses[0, 7:], q[0, 7:], rtol=0, atol=1e-5):
        raise ValueError("Feedback bank initial joint configuration does not match the declared reference")
    ticks = np.arange(len(poses))
    bank_rise = first_upright(poses)
    if bank_rise is None:
        bank_rise = capture
    source_ticks = np.rint(ticks * rise / max(bank_rise, 1)).astype(int) if retime else ticks
    aligned = source_ticks.clip(0, len(q) - 1)
    active = ticks <= max(bank_rise if retime else rise, 9)
    reference = np.column_stack((q[aligned, 7:], v[aligned, 6:], active)).astype(np.float32)

    support = np.zeros((len(ticks), 6), dtype=np.float32)
    for side, intervals in enumerate(sides):
        for start, end in intervals:
            # The source marks half-open intervals. Pad both ends for palm transitions.
            support[max(start - 2, 0):min(end + 3, len(ticks)), side] = 1
    extension = knee if knee is not None else max(1, 0.65 * rise)
    phase_gate = np.clip(0.95 * extension / max(rise, 1), 0.25, 0.83)
    desired_frames = max(1.31 * rise, 1.28 * (teacher_rise_s / env.DT))
    support[:, 2:] = (required, desired_frames, slow, phase_gate)
    if retime:
        support[:, :2] = support[source_ticks.clip(0, len(ticks) - 1), :2]
        support = np.column_stack((support, np.full(len(ticks), bank_rise)))
    return reference, support


def capture_steps(archive, spec, trials, root):
    if "floating_base/capture_offsets" in archive:
        return archive["floating_base/capture_offsets"]
    steps = []
    for trial in trials:
        with np.load(root / spec["gain_dir"] / f"{trial:02d}.npz") as gain:
            steps.append(int(gain["capture_step"]))
    return steps


def assemble(manifest, recipe, motor, root):
    starts = json.loads((root / manifest["teacher_starts"]).read_text())
    phases = source_phases(starts, root / manifest["motions"])
    fields = {name: [] for name in FIELDS}
    references, timings, captures = [], [], []
    offset = 0
    for name in recipe["banks"]:
        bank = manifest["banks"][name]
        trials = bank.get("trials", list(phases))
        with np.load(root / bank["checkpoint"]) as archive:
            data = {key: archive["tracking_capture/" + key] for key in FIELDS}
            steps = capture_steps(archive, bank, trials, root)
        begin = np.flatnonzero(data["start"] == np.arange(len(data["start"])))
        for trial, first, capture in zip(trials, begin, steps, strict=True):
            stop = int(data["end"][first]) + 1
            if not 0 <= capture < stop - first:
                raise ValueError(f"Capture frame {capture} outside {name} trial {trial}")
            ref, timing = align_start(phases[trial], data["qpos"][first:stop], capture,
                                      name in recipe["slow_banks"], recipe["planning"] >= 6)
            references.append(ref)
            timings.append(timing)
        captures.extend(steps)
        for key, array in data.items():
            fields[key].append(array + offset if key in ("start", "end") else array)
        offset += len(data["qpos"])

    with np.load(motor) as archive:
        weights = dict(archive)
    if not (float(weights["tracking_capture/residual_scale"]) > 0
            and np.any(weights["tracking_capture/residual_w"])):
        raise ValueError("Motor checkpoint needs a trained PPO motor correction")
    weights.update({"tracking_capture/" + key: np.concatenate(parts) for key, parts in fields.items()})
    weights["floating_base/sensor"] = np.array("mujoco_root_and_foot_poses")
    weights["floating_base/capture_offsets"] = np.asarray(captures)
    weights["floating_base/reference"] = np.concatenate(references)
    weights["floating_base/support_timing"] = np.concatenate(timings)
    weights["floating_base/planning"] = np.array(recipe["planning"])
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="Name in the manifest's recipes table")
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/g1_ground_sequence_recipes.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="Base for all data paths in the manifest")
    parser.add_argument("--motor", type=Path, help="Override with a compatible trained motor checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--compare", type=Path, help="Require exact array and dtype equality to this checkpoint")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.recipe not in manifest["recipes"] or args.out.exists():
        parser.error("Choose a recipe in the manifest and an unused output directory")
    recipe = manifest["recipes"][args.recipe]
    if "source" in recipe:
        from tools.export_g1_supported_stance import export

        switches = {9: {}, 10: {"absorb_impulse": True}, 11: {"reactive_actor": True},
                    12: {"contact_balance": True}, 13: {"separate_impulse": True}}
        if args.motor or recipe["planning"] not in switches:
            parser.error("A standing-feedback export needs a source policy and a supported planning mode")
        path = export(args.root / recipe["source"], args.out, compare=args.compare, **switches[recipe["planning"]])
        print(path)
        return
    motor = args.motor or args.root / recipe["motor"]
    weights = assemble(manifest, recipe, motor, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        run = Path(tmp) / "run"
        directory = run / "ckpt"
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "0000.npz", **weights)
        if args.compare:
            with np.load(args.compare) as expected, np.load(directory / "0000.npz") as actual:
                if set(expected.files) != set(actual.files):
                    raise ValueError("Checkpoint fields differ")
                for key in expected.files:
                    a, b = actual[key], expected[key]
                    if a.dtype != b.dtype or not np.array_equal(a, b):
                        raise ValueError(f"Checkpoint differs at {key}")
            print(f"Matched {len(weights)} checkpoint fields, including motor weights and gains.")
        shutil.copyfile(motor.parents[1] / "config.json", run / "config.json")
        inputs = dict(recipe=args.recipe, motor=str(motor), manifest=manifest, dt_s=env.DT)
        (directory / "0000.json").write_text(json.dumps(inputs, indent=2) + "\n")
        run.rename(args.out)
    print(args.out / "ckpt/0000.npz")


if __name__ == "__main__":
    main()
