"""Export the averaged Zenbot actor and verify its Motion Train ordered actions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from instantamp import env, nets, zenbot_models
from instantamp.inference import load
from instantamp.zenbot_motion_train_obs import (
    joint_layout,
    motion_train_actor_mean,
)


def export(checkpoint, destination):
    actor, norm, ode = load(checkpoint, which="avg")
    robot = zenbot_models.for_joint_count(actor[-1]["b"].size)
    joints = len(robot.dof)
    width = len(joint_layout(joints).history_index)
    if ode is not None or actor[0]["w"].shape[0] != width:
        raise ValueError("Expected a three-frame, direct-action Zenbot policy")
    nodes, constants = [], []

    def tensor(name, value):
        constants.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    def node(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], **attributes))
        return output

    current = "obs"
    mean = np.asarray(norm["mean"])
    std = np.asarray(nets.rn_std(norm))
    if "raw" not in norm:
        current = node("Sub", [current, tensor("mean", mean)], "centered")
        current = node("Div", [current, tensor("std", std)], "scaled")
        current = node("Clip", [current, tensor("clip_min", -10.), tensor("clip_max", 10.)], "normalized")
    for index, layer in enumerate(actor):
        prefix = f"layer{index}_"
        last = index == len(actor) - 1
        weight, bias = np.asarray(layer["w"]), np.asarray(layer["b"])
        if last:
            weight, bias = weight[:, robot.lab_order], bias[robot.lab_order]
        current = node("MatMul", [current, tensor(prefix + "weight", weight)], prefix + "mul")
        current = node("Add", [current, tensor(prefix + "bias", bias)], "actions" if last else prefix + "linear")
        if not last:
            if "ln_g" in layer:
                current = node("LayerNormalization", [
                    current, tensor(prefix + "ln_scale", layer["ln_g"]), tensor(prefix + "ln_bias", layer["ln_b"]),
                ], prefix + "normalized", axis=-1, epsilon=1e-6)
            current = node("Elu", [current], prefix + "elu")

    graph = helper.make_graph(nodes, f"zenbot{joints}_instantamp",
                              [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, joints])], constants)
    model = helper.make_model(graph, producer_name="InstantAMP", ir_version=9,
                              opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)

    # Exercise ordinary observations and the observation normalizer's clipping boundaries.
    rng = np.random.default_rng(42)
    standardized = np.concatenate((rng.standard_normal((96, width)), rng.uniform(-15., 15., (96, width)))).astype(np.float32)
    probes = standardized if "raw" in norm else mean + std * standardized
    observations = np.concatenate((np.zeros((1, width), np.float32), mean[None], probes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    actual = np.concatenate([session.run(["actions"], {"obs": row[None]})[0] for row in observations])
    expected = np.asarray(motion_train_actor_mean(actor, norm, observations))[:, robot.lab_order]
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("Nonfinite actions during ONNX verification")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    error = float(np.max(np.abs(actual - expected)))
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {
        "robot": robot.mixed_task, "actor": "avg", "checkpoint": checkpoint.name,
        "checkpoint_sha256": source_hash, "history_frames": "3", "history_order": "term",
        "control_hz": str(round(1 / env.DT)), "action_order": "Motion Train lab_dof_names",
        "action_joint_names": json.dumps(robot.lab_joints),
        "default_joint_pos_rad": json.dumps(robot.default_q[robot.lab_order].tolist()),
        "observation_normalization": "raw" if "raw" in norm else "included",
        "validation_max_abs_error": str(error), "validation_samples": str(len(observations)),
    }
    sidecar = checkpoint.with_suffix(".json")
    if sidecar.is_file():
        iteration = json.loads(sidecar.read_text()).get("it")
        if iteration is not None:
            metadata["iteration"] = str(iteration)
    helper.set_model_props(model, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    return {"onnx_checker": "passed", "onnxruntime_provider": session.get_providers()[0],
            "input": {"name": "obs", "shape": [1, width]}, "output": {"name": "actions", "shape": [1, joints]},
            "parity_samples": len(observations), "parity_max_abs_error": error, "checkpoint_sha256": source_hash}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--validation", type=Path, help="Write the action parity report as JSON")
    args = parser.parse_args()
    report = json.dumps(export(args.checkpoint, args.destination), indent=2)
    if args.validation:
        args.validation.parent.mkdir(parents=True, exist_ok=True)
        args.validation.write_text(report + "\n")
    print(report)
