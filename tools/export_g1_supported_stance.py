"""Export a palm-supported G1 recovery controller with post-rise support prediction."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np


def checkpoint_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(checkpoint, out, absorb_impulse=False, compare=None, reactive_actor=False, contact_balance=False,
           separate_impulse=False):
    config = checkpoint.parents[1] / "config.json"
    if out.exists():
        raise FileExistsError(out)
    with np.load(checkpoint) as z:
        stages = [(13, separate_impulse), (12, contact_balance), (11, reactive_actor),
                  (10, absorb_impulse), (9, True)]
        mode = next(stage for stage, enabled in stages if enabled)
        source_modes = range(8, mode)
        if (int(z.get("floating_base/planning", -1)) not in source_modes or
                z["floating_base/support_timing"].shape[1] != 7):
            raise ValueError("Expected a trained G1 controller with palm support before extension")
        fields = dict(z)
    fields["floating_base/planning"] = np.asarray(mode, dtype=fields["floating_base/planning"].dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch:
        run = Path(scratch) / "run"
        folder = run / "ckpt"
        folder.mkdir(parents=True)
        target = folder / "0000.npz"
        np.savez_compressed(target, **fields)
        if compare is not None and checkpoint_hash(target) != checkpoint_hash(compare):
            raise ValueError(f"Export differs from {compare}")
        shutil.copyfile(config, run / "config.json")
        metadata = dict(source=str(checkpoint), source_sha256=checkpoint_hash(checkpoint),
                        planning=mode, foot_support_weight_fraction=0.5,
                        required_loaded_samples=45, support_window_samples=50)
        (folder / "0000.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run.rename(out)
    return out / "ckpt/0000.npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--absorb-impulse", action="store_true",
                        help="Predict a bounded motor correction when quiet standing is disturbed")
    parser.add_argument("--reactive-actor", action="store_true",
                        help="Use the AMP balance actor when local impulse prediction cannot hold the stance")
    parser.add_argument("--contact-balance", action="store_true",
                        help="Follow slow ground reactions and reacquire the loaded feet after an AMP balance step")
    parser.add_argument("--separate-impulse", action="store_true",
                        help="Wait for a settled stance after an impulse before estimating sustained loads")
    parser.add_argument("--compare", type=Path, help="Require the exported checkpoint to match this file byte for byte")
    args = parser.parse_args()
    print(export(args.checkpoint, args.out, args.absorb_impulse, args.compare,
                 args.reactive_actor, args.contact_balance, args.separate_impulse))
