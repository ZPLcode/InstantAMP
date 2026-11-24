"""Initialize a joint Zenbot policy from a Motion Train locomotion ONNX export."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from instantamp.zenbot_models import for_joint_count
from instantamp.zenbot_motion_train_obs import joint_layout


def convert(source: Path, dest: Path) -> None:
    """Convert the actor to MJCF action order and compare against ONNX Runtime."""
    graph = onnx.load(source).graph
    expected_ops = [
        "Sub",
        "Div",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
    ]
    if [node.op_type for node in graph.node] != expected_ops:
        raise ValueError("Expected a normalized four-layer Motion Train actor")

    arrays = {
        value.name: numpy_helper.to_array(value) for value in graph.initializer
    }
    mean, std = (arrays[node.input[1]].ravel() for node in graph.node[:2])
    joint_count = len(arrays[graph.node[-1].input[2]])
    model = for_joint_count(joint_count)
    width = len(joint_layout(joint_count).history_index)
    if (
        mean.shape != (width,)
        or std.shape != (width,)
        or np.any(std < 0.01)
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError(f"Invalid parameters for a {width}-input Zenbot actor")

    result = {"obs_norm/mean": mean, "obs_norm/var": std**2 - 0.01**2}
    normalized = (
        np.random.default_rng(42)
        .uniform(-3.0, 3.0, (64, width))
        .astype(np.float32)
    )
    output = normalized
    for index, node in enumerate(graph.node[2::2]):
        weight, bias = arrays[node.input[1]].T, arrays[node.input[2]]
        if index == 3:
            native_order = np.argsort(model.lab_order)
            weight, bias = weight[:, native_order], bias[native_order]
        for label in ("avg", "live"):
            result[f"{label}/{index}/w"] = weight
            result[f"{label}/{index}/b"] = bias
        output = output @ weight + bias
        if index < 3:
            output = np.maximum(output, 0.0) + np.expm1(np.minimum(output, 0.0))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    expected = np.concatenate(
        [
            session.run(["actions"], {"obs": row[None]})[0]
            for row in mean + std * normalized
        ]
    )
    lab_output = output[:, model.lab_order]
    np.testing.assert_allclose(lab_output, expected, rtol=1e-5, atol=1e-5)
    metadata = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "observations_checked": len(normalized),
        "max_absolute_action_error": float(np.max(abs(lab_output - expected))),
        "input": width,
        "output": joint_count,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as stream:
        np.savez(stream, **result)
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Saved {joint_count}-DoF actor to {dest} after checking {len(normalized)} observations"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
