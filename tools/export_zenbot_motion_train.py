"""Export a 234-input Zenbot actor for Motion Train's original Zenbot deployment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from instantamp.zenbot_motion_train_obs import LAB_ORDER, MOTION_TRAIN_WIDTH


class Actor(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        self.raw = "obs_norm/raw" in arrays
        self.register_buffer("mean", torch.from_numpy(arrays["obs_norm/mean"]))
        self.register_buffer("std", torch.from_numpy(np.sqrt(arrays["obs_norm/var"] + 0.01**2)))
        modules = []
        width = MOTION_TRAIN_WIDTH
        if self.mean.shape != (width,) or self.std.shape != (width,):
            raise ValueError("Only Motion Train 234-input Zenbot actors can be exported")
        if any(k.startswith(("ode/", "gait/", "zenbot_tracking/")) for k in arrays):
            raise ValueError("Expected a standalone direct-action actor")
        i = 0
        while f"avg/{i}/w" in arrays:
            prefix = f"avg/{i}/"
            weight, bias = arrays[prefix + "w"], arrays[prefix + "b"]
            if weight.shape != (width, len(bias)) or bias.ndim != 1:
                raise ValueError(f"Inconsistent actor layer {i}")
            layer = nn.Linear(width, len(bias))
            width = len(bias)
            last = f"avg/{i + 1}/w" not in arrays
            if last:
                if width != len(LAB_ORDER):
                    raise ValueError("Expected 23 Zenbot motor actions")
                weight, bias = weight[:, LAB_ORDER], bias[LAB_ORDER]
            layer.weight.data.copy_(torch.from_numpy(weight.T.copy()))
            layer.bias.data.copy_(torch.from_numpy(bias.copy()))
            modules.append(layer)
            if not last:
                if prefix + "ln_g" in arrays:
                    norm = nn.LayerNorm(width, eps=1e-6)
                    norm.weight.data.copy_(torch.from_numpy(arrays[prefix + "ln_g"]))
                    norm.bias.data.copy_(torch.from_numpy(arrays[prefix + "ln_b"]))
                    modules.append(norm)
                modules.append(nn.ELU())
            i += 1
        if not modules:
            raise ValueError("Missing actor layers")
        self.network = nn.Sequential(*modules)

    def forward(self, obs):
        if not self.raw:
            obs = ((obs - self.mean) / self.std).clamp(-10.0, 10.0)
        return self.network(obs)


def export(checkpoint, out):
    with np.load(checkpoint, allow_pickle=False) as arrays:
        actor = Actor(arrays).eval()
    out.mkdir(exist_ok=True, parents=True)
    example = torch.zeros(1, MOTION_TRAIN_WIDTH)
    pt, graph = out / "policy.pt", out / "policy.onnx"
    torch.jit.script(actor).save(str(pt))
    torch.onnx.export(
        actor,
        example,
        str(graph),
        input_names=["obs"],
        output_names=["actions"],
        dynamo=False,
        opset_version=18,
    )
    model = onnx.load(graph)
    onnx.helper.set_model_props(
        model,
        {
            "robot": "zenbot",
            "history_frames": "3",
            "history_order": "term",
            "control_hz": "50",
            "action_order": "Motion Train lab_dof_names",
        },
    )
    onnx.checker.check_model(model)
    onnx.save(model, graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    scripted = torch.jit.load(str(pt)).eval()
    rng = np.random.default_rng(2026)
    error = 0.0
    with torch.inference_mode():
        for row in rng.normal(size=(64, 1, MOTION_TRAIN_WIDTH)).astype(np.float32):
            expected = scripted(torch.from_numpy(row)).numpy()
            actual = session.run(["actions"], {"obs": row})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            error = max(error, float(np.max(abs(actual - expected))))
    report = {
        "input": "float32 obs[1,234]",
        "output": "float32 actions[1,23]",
        "history": "3, term-major",
        "max_torchscript_onnx_abs_error": error,
        "comparisons": 64,
        "sha256": {
            k: hashlib.sha256(v.read_bytes()).hexdigest()
            for k, v in (("checkpoint", checkpoint), ("policy.pt", pt), ("policy.onnx", graph))
        },
    }
    (out / "export.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.out)
