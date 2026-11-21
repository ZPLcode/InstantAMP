"""Bundle a joint InstantAMP Zenbot actor and a motion_train three-frame AMP gait actor."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from tools.convert_motions import lab_dof_names


def convert_gait(path, source, *, joint_policy=False):
    model = onnx.load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    frames = meta.get("history_frames", meta.get("history_length"))
    order_name = meta.get("history_order")
    if joint_policy and meta.get("task") == "ZENBOT-Dual-AMP-Flat":
        manifest = json.loads(path.with_name("export_metadata.json").read_text())
        if manifest["onnx_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("The joint AMP export and observation manifest differ")
        frames, order_name = str(manifest["history_length"]), manifest["history_order"]
    tasks = ("ZENBOT-AMP-Loco-Flat", "ZENBOT-Dual-AMP-Flat") if joint_policy else ("ZENBOT-AMP",)
    if meta.get("task") not in tasks or frames not in ("3", "4") or order_name != "term":
        raise ValueError("Expected a motion_train AMP export with a supported term-major sensor history")
    if [n.op_type for n in model.graph.node] != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError("Unsupported gait network")
    robot = Path(__file__).resolve().parents[1] / "assets/zenbot_23dof_43kg.xml"
    src_robot = source / "robolab/data/robots/roboparty/zenbot/mjcf/zenbot.xml"
    if src_robot.read_bytes() != robot.read_bytes():
        raise ValueError("The gait and recovery profiles require the same 43 kg robot")
    hinges = [m.attrib["joint"].removesuffix("_joint") for m in ET.parse(robot).findall("actuator/motor")]
    names = lab_dof_names(source / "robolab/scripts/tools/retarget/config/zenbot.yaml")
    if len(hinges) != 23 or sorted(hinges) != sorted(names):
        raise ValueError("Expected the same 23 named joints")
    order = np.asarray([hinges.index(n) for n in names])
    last = np.arange(390).reshape(5, 78)[-int(frames):]
    blocks = [last[:, i:i+3] for i in (0, 3, 6)] + [last[:, i+order] for i in (9, 32, 55)]
    index = np.concatenate([block.ravel() for block in blocks])
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    mean = np.zeros(390, np.float32)
    var = np.ones(390, np.float32)
    mean[index] = arrays[model.graph.node[0].input[1]].ravel()
    std = arrays[model.graph.node[1].input[1]].ravel()
    if not np.isfinite(std).all() or np.any(std < .01):
        raise ValueError("Invalid gait normalization")
    var[index] = std**2 - .01**2
    out = {"obs_norm/mean": mean, "obs_norm/var": var}
    sample = np.random.default_rng(41).normal(0., .5, (12, 390)).astype(np.float32)
    x = (sample - mean) / np.sqrt(var + .01**2)
    for i, node in enumerate(n for n in model.graph.node if n.op_type == "Gemm"):
        w, b = arrays[node.input[1]].T, arrays[node.input[2]]
        if i == 0:
            wide = np.zeros((390, w.shape[1]), np.float32)
            wide[index] = w
            w = wide
        if i == 3:
            inverse = np.argsort(order)
            w, b = w[:, inverse], b[inverse]
        for which in ("pi", "avg"):
            out.update({f"{which}/{i}/w": w, f"{which}/{i}/b": b})
        x = x @ w + b
        if i < 3:
            x = np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
    ref = ReferenceEvaluator(model)
    expected = np.concatenate([ref.run(None, {model.graph.input[0].name: row[None, index]})[0] for row in sample])
    np.testing.assert_allclose(x[:, order], expected, rtol=3e-5, atol=3e-5)
    return out


def pack(rise, gait, source, dest, handoff_frames=0, fall_upright=.6):
    if not 0 <= handoff_frames <= 250 or not 0. < fall_upright < .9:
        raise ValueError("Invalid Zenbot gait handoff parameters")
    mapped = convert_gait(gait, source)
    with np.load(rise, allow_pickle=False) as actor:
        if "ode/dmax" not in actor or actor["avg/0/w"].shape[0] != 390 or actor["avg/3/b"].shape != (23,):
            raise ValueError("Expected a locomotion/recovery Zenbot ODE checkpoint")
        fields = {k: actor[k] for k in actor.files}
    fields.update({"gait/" + key: value for key, value in mapped.items()})
    fields.update({"gait/handoff_frames": handoff_frames, "gait/fall_upright": fall_upright})
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **fields)
    meta = json.loads(rise.with_suffix(".json").read_text())
    meta.update(handoff_frames=handoff_frames, fall_upright=fall_upright)
    for key, path in (("joint_actor", rise), ("gait_actor", gait)):
        meta[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rise", "gait", "motion_train", "dest"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--handoff-frames", type=int, default=0)
    parser.add_argument("--fall-upright", type=float, default=.6)
    args = parser.parse_args()
    pack(args.rise, args.gait, args.motion_train, args.dest, args.handoff_frames, args.fall_upright)
