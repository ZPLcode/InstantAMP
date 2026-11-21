"""Export a LeggedLab Zenbot get-up AMP actor in native MuJoCo observation and motor order."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.convert_zenbot_amp_discriminator import feature_order


TERMS = ("base_ang_vel", "root_local_rot_tan_norm", "joint_pos", "joint_vel", "actions", "key_body_pos_b")
WIDTHS = (3, 6, 23, 23, 23, 18)


def convert(checkpoint, manifest_file, destination):
    import torch
    import yaml

    params = checkpoint.parent / "params"
    config = yaml.load((params / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
    environment = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    policy, actions = environment["observations"]["policy"], environment["actions"]["joint_pos"]
    manifest = json.loads(manifest_file.read_text())
    source_terms = tuple(k for k, v in policy.items() if isinstance(v, dict) and "func" in v)
    if (config["experiment_name"] != "zenbot_23dof_43kg_amp_get_up" or
            config["algorithm"]["class_name"] != "PPOAMP" or config["policy"]["activation"] != "elu" or
            config["policy"]["actor_obs_normalization"] != "false" or source_terms != TERMS or
            policy["history_length"] != "5" or policy["flatten_history_dim"] != "true" or
            policy["key_body_pos_b"]["params"]["asset_cfg"]["body_names"] != manifest["key_body_names"] or
            actions["use_default_offset"] != "true" or float(actions["scale"]) != 0.25):
        raise ValueError("Source settings differ from the standalone 23-joint Zenbot get-up AMP actor")

    frame_order = feature_order(manifest)
    joint_order = frame_order[9:32] - 9
    point_order = frame_order[55:] - 55
    axes = (np.arange(3), np.arange(6), joint_order, joint_order, joint_order, point_order)
    # IsaacLab flattens each term's five chronological samples before concatenation.
    # Native observations concatenate all six terms within each chronological frame.
    ids = np.arange(480)
    blocks = np.split(ids, 5 * np.cumsum(WIDTHS)[:-1])
    first_layer_order = np.concatenate([b.reshape(5, -1)[:, ax] for b, ax in zip(blocks, axes)], axis=1).ravel()
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = bundle["model_state_dict"]
    if "amp_discriminator_state_dict" not in bundle or any(k.startswith("actor_obs_normalizer.") for k in state):
        raise ValueError("Expected a raw-observation actor from an adversarial motion prior checkpoint")
    dimensions = (480, 512, 256, 128, 23)
    fields = {"obs_norm/mean": np.zeros(480, np.float32), "obs_norm/var": np.ones(480, np.float32),
              "obs_norm/raw": np.bool_(True)}
    for i, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
        w = state[f"actor.{2*i}.weight"].numpy()
        b = state[f"actor.{2*i}.bias"].numpy()
        if (w.shape != (outputs, inputs) or b.shape != (outputs,) or
                not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError(f"Invalid parameters for actor layer {i}")
        if i == 0:
            w = w[:, first_layer_order]
        if i == 3:
            w, b = w[joint_order], b[joint_order]
        for name in ("live", "avg"):
            fields[f"{name}/{i}/w"], fields[f"{name}/{i}/b"] = w.T, b

    # Build source inputs directly from independent native sensor histories and compare motor predictions in FP32.
    rng = np.random.default_rng(43)
    histories = rng.normal(size=(32, 5, 96)).astype(np.float32)
    histories[:, :, 32:55] *= 4.
    native = histories.reshape(len(histories), -1)
    terms = np.split(histories, np.cumsum(WIDTHS)[:-1], axis=-1)
    source = torch.from_numpy(np.concatenate(
        [term[:, :, np.argsort(ax)].reshape(len(term), -1) for term, ax in zip(terms, axes)], axis=1))
    for i in range(4):
        source = torch.nn.functional.linear(source, state[f"actor.{2*i}.weight"], state[f"actor.{2*i}.bias"])
        native = native @ fields[f"avg/{i}/w"] + fields[f"avg/{i}/b"]
        if i < 3:
            source = torch.nn.functional.elu(source)
            native = np.maximum(native, 0.) + np.expm1(np.minimum(native, 0.))
    expected = source.numpy()[:, joint_order]
    np.testing.assert_allclose(native, expected, atol=5e-5, rtol=2e-5)
    metadata = dict(source_checkpoint=str(checkpoint.resolve()), source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_iteration=int(bundle["iter"]), fp32_max_action_difference=float(np.max(np.abs(native - expected))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "motion_manifest", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.motion_manifest, args.destination)))
