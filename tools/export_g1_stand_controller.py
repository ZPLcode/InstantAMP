"""Package the named task's direct-action source policies as a state-triggered stand controller.

The rise actor first controls the robot. The next actor takes over when the torso and ankles indicate
a sufficient rise. An optional third actor can take over when both legs extend and the trunk slows.
An optional continuous sensor confirmation delays the final hand-off and returns control to the
balancing actor when upright posture or bilateral support breaks down. At a 50 Hz policy rate,
--confirm-seconds .20 requires ten consecutive qualified observations. The exported controller
contains the policy parameters and calibration of all its actors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from instantamp.inference import load


def export(rise, balance, stand, gates, out, *, filter_stance=False, confirm_seconds=None, release_limits=None, rise_library=None):
    paths = [("rise", rise), ("balance", balance)]
    if stand:
        paths.append(("stand", stand))
    limits = np.asarray(gates, np.float32)
    expected = 7 if stand else 3
    if len(limits) != expected or not np.isfinite(limits).all():
        raise ValueError(f"Expected {expected} finite gate parameters")
    archive = {"stand_thresholds": limits}
    if confirm_seconds is not None or release_limits is not None:
        if stand is None or confirm_seconds is None or release_limits is None:
            raise ValueError("Stance confirmation requires a third policy, --confirm-seconds and --release")
        from instantamp.env import DT
        frames = round(confirm_seconds / DT)
        if frames < 1 or not np.isclose(frames * DT, confirm_seconds):
            raise ValueError(f"Confirmation duration must be a positive multiple of {DT} s")
        archive["stand_guard"] = np.asarray([frames, *release_limits], dtype=np.float32)
    if filter_stance:
        archive["stand_observation_filter"] = "encoder_fk_linear5"
    sources = {}
    if rise_library is not None:
        from instantamp.g1_rise import LocalRise
        with np.load(rise_library, allow_pickle=False) as library:
            LocalRise.read(library)
            archive.update({"local_rise/" + name: library[name] for name in library if name in LocalRise._fields or name.startswith("residual/")})
        sources["local_rise"] = {"library": rise_library.name, "sha256": hashlib.sha256(rise_library.read_bytes()).hexdigest()}
    for role, path in paths:
        actor, _, ode = load(path)
        if not isinstance(actor, list) or ode is not None or actor[0]["w"].shape[0] != 570 or actor[-1]["b"].shape != (29,):
            raise ValueError(f"{path} must contain a direct 570-input, 29-output G1 actor")
        sources[role] = {"checkpoint": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with np.load(path, allow_pickle=False) as checkpoint:
            archive.update({role + "/" + key: checkpoint[key] for key in checkpoint
                            if key.startswith(("avg/", "live/", "obs_norm/"))})
    # Validate both weight selections before publishing the new run atomically.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        run = Path(tmp) / "run"
        folder = run / "ckpt"
        folder.mkdir(parents=True)
        np.savez(folder / "0000.npz", **archive)
        for which in ("avg", "live"):
            load(folder / "0000.npz", which)
        meta = {"kind": "g1_getup_stand_controller", "sources": sources, "thresholds": limits.tolist()}
        if "stand_guard" in archive:
            meta["confirm_seconds"] = confirm_seconds
            meta["release_depth_m"], meta["release_upright_cosine"], meta["release_gyro_rad_s"] = map(float, release_limits)
        if filter_stance:
            meta["stand_observation_filter"] = archive["stand_observation_filter"]
        (folder / "0000.json").write_text(json.dumps(meta, indent=2) + "\n")
        config = {"robot": "g1_getup", "nconmax": 64, "njmax": 640, "floor_dir": "data/g1_getup/recovery",
                  "walk_dir": None, "reset_floor": 1., "prob_shove": 0., "push": False}
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if out.exists():
            raise FileExistsError(out)
        run.rename(out)
    return out / "ckpt/0000.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rise", required=True, type=Path)
    parser.add_argument("--balance", required=True, type=Path)
    parser.add_argument("--stand", type=Path)
    parser.add_argument("--filter-stance", action="store_true", help="Use encoder-derived landmarks and five-frame sensor denoising for balance.")
    parser.add_argument("--rise-library", type=Path, help="Optional proprioceptive local rise policy fitted to physical demonstrations")
    parser.add_argument("--gates", nargs="+", type=float, required=True,
                        help="Ankle depth (m), torso cosine, blend increment. For --stand append bilateral "
                             "depth (m), torso cosine, maximum gyro norm (rad/s), and second blend increment.")
    parser.add_argument("--confirm-seconds", type=float, help="Require continuous stable samples before quiet stance")
    parser.add_argument("--release", nargs=3, type=float, metavar=("ANKLE_M", "UPRIGHT_COS", "GYRO_RAD_S"),
                        help="Return to the balancing actor when either bilateral depth or upright cosine"
                             " falls below its limit, or trunk angular speed exceeds its limit")
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    args = parser.parse_args()
    print(export(args.rise, args.balance, args.stand, args.gates, args.out, filter_stance=args.filter_stance,
                 confirm_seconds=args.confirm_seconds, release_limits=args.release, rise_library=args.rise_library))


if __name__ == "__main__":
    main()
