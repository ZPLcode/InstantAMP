"""Export the LeggedLab Zenbot 23-DoF get-up discriminator in InstantAMP frame order."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


MODEL = Path(__file__).resolve().parents[1] / "assets/zenbot_23dof_43kg.xml"
LANDMARKS = tuple(f"{side}_{link}" for link in ("ankle_roll_Link", "knee_Link", "wrist_roll_rubber_hand")
                  for side in ("left", "right"))
LAYERS = ("disc_trunk.0", "disc_trunk.2", "disc_linear")


def feature_order(manifest):
    """Indices into one source frame that give the corresponding native frame."""
    joints = [a.attrib["joint"] for a in ET.parse(MODEL).findall("./actuator/motor")]
    source_joints, source_points = manifest["joint_names"], manifest["key_body_names"]
    if len(joints) != 23 or sorted(source_joints) != sorted(joints) or sorted(source_points) != sorted(LANDMARKS):
        raise ValueError("Expected the named joints and six landmarks of the Zenbot 23-DoF get-up bank")
    joint_order = np.array([source_joints.index(j) for j in joints])
    point_order = np.array([source_points.index(b) for b in LANDMARKS])
    return np.r_[np.arange(9), 9 + joint_order, 32 + joint_order, (55 + point_order[:, None]*3 + np.arange(3)).ravel()]


def convert(checkpoint, manifest_path, dest):
    import torch

    manifest = json.loads(manifest_path.read_text())
    order = feature_order(manifest)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = {k: v.detach().numpy() for k, v in data["amp_discriminator_state_dict"].items()}
    norm = {k: source["disc_obs_normalizer." + k] for k in ("_mean", "_var", "_std", "count")}
    sizes = (730, 1024, 512, 1)
    if any(norm[k].shape != (1, 73) for k in ("_mean", "_var", "_std")) or norm["count"].size != 1:
        raise ValueError("Expected the 73-feature LeggedLab discriminator normalizer")
    np.testing.assert_allclose(np.sqrt(norm["_var"]), norm["_std"], rtol=1e-5, atol=1e-6)
    fields = {"norm/mean": norm["_mean"][0, order], "norm/var": norm["_var"][0, order],
              "norm/count": np.asarray(norm["count"], np.float32).reshape(())}
    for i, key in enumerate(LAYERS):
        w, b = source[key + ".weight"], source[key + ".bias"]
        if w.shape != (sizes[i+1], sizes[i]) or b.shape != (sizes[i+1],):
            raise ValueError(f"Unexpected Zenbot discriminator layer dimensions at {key}")
        if i == 0:
            w = w.reshape(len(b), 10, 73)[:, :, order].reshape(w.shape)
        fields[f"net/{i}/w"], fields[f"net/{i}/b"] = w.T, b
    if not all(np.isfinite(a).all() for a in fields.values()) or fields["norm/count"] < 0:
        raise ValueError("Invalid discriminator parameters in source checkpoint")
    # Check both normalizer and the frame order repeated across the ten-step history.
    windows = np.random.default_rng(73).normal(size=(8, 10, 73)).astype(np.float32)
    expected = torch.from_numpy(np.clip((windows - norm["_mean"]) / (norm["_std"] + .01), -10., 10.).reshape(-1, 730))
    actual = np.clip((windows[..., order] - fields["norm/mean"]) / (np.sqrt(fields["norm/var"]) + .01), -10., 10.).reshape(-1, 730)
    for i, key in enumerate(LAYERS):
        expected = torch.nn.functional.linear(expected, torch.from_numpy(source[key + ".weight"]), torch.from_numpy(source[key + ".bias"]))
        actual = actual @ fields[f"net/{i}/w"] + fields[f"net/{i}/b"]
        if i < 2:
            expected = torch.nn.functional.leaky_relu(expected)
            actual = np.maximum(actual, .01 * actual)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-5)
    fields["checkpoint_sha256"] = np.asarray(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    fields["source_iteration"] = np.int64(data["iter"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        np.savez_compressed(out, **fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "motion_manifest", "dest"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    convert(args.checkpoint, args.motion_manifest, args.dest)
