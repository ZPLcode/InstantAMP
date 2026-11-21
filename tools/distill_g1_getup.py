"""Initialize the 570-input G1 get-up policy by distilling a SuperMimic AMP Flat ONNX teacher.

From the repository root, with the optional onnxruntime package installed:

  python -m tools.distill_g1_getup --teacher /path/to/deploy/wmrl_66000/policy.onnx

The teacher's metadata JSON must be next to the ONNX file. Only this training
tool uses the teacher. The exported student uses the standard InstantAMP policy,
normalizer, action integration and evaluator.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from scipy.spatial.transform import Rotation

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from instantamp import env, g1_getup as robot, nets, optim
from instantamp.inference import load, policy_output
from instantamp.native_getup import advance, observation_terms
from instantamp.sim import build_scene
from instantamp.train import G1GetUpConfig, save_policy


class AmpTeacher:
    """Translate the source policy's joint command to the get-up task's PD drives."""

    def __init__(self, path):
        import onnxruntime as ort

        meta = json.loads(path.with_suffix(".json").read_text())
        names = [name.removesuffix("_joint") for name in meta["joint_names"]]
        if sorted(names) != sorted(robot.DOF) or not meta["normalization_in_onnx"]:
            raise ValueError("Expected a G1 29-DoF AMP Flat ONNX export with embedded normalization")
        self.order = np.array([robot.DOF.index(name) for name in names])
        self.default = np.asarray(meta["default_q"])
        self.scale = np.asarray(meta["action_scale"])
        self.kp = np.array([m["stiffness"] for m in meta["motors"]])
        self.kd = np.array([m["damping"] for m in meta["motors"]])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape != [1, 384] or self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("Expected a 384-input, 29-output AMP Flat teacher")

    def reset(self):
        self.raw = np.zeros(29)
        self.history = None

    def target(self, gyro, rotation, q, qd):
        q, qd = q[self.order], qd[self.order]
        x, z = rotation[:3], rotation[3:]
        gravity = -np.array([x[2], np.cross(z, x)[2], z[2]])
        # Zero velocity command requests stationary recovery.
        frame = np.r_[gyro, gravity, [0., 0., 0.], q - self.default, qd, self.raw]
        if self.history is None:
            self.history = np.tile(frame, (4, 1))
        else:
            self.history = np.concatenate([self.history[1:], frame[None]])
        self.raw = self.session.run(None, {self.input: self.history.reshape(1, 384).astype(np.float32)})[0][0]
        reference = self.default + self.scale * self.raw
        # Kp_new * (target - q) - Kd_new * qd = Kp_source * (reference - q) - Kd_source * qd.
        target = np.empty(29)
        target[self.order] = q + (self.kp * (reference - q) + (robot.KD[self.order] - self.kd) * qd) / robot.KP[self.order]
        return (target - robot.DEFAULT_Q) / robot.ACTION_SCALE


def collect(model, teacher, motions, cfg, count, seed, student=None, student_share=0.):
    rng = np.random.default_rng(seed)
    if student is not None:
        policy, stats, limits = load(student)
        student_action = jax.jit(lambda history, old: nets.ode_step(policy_output(policy, stats, history), old, *limits))
    links = [model.body("robot_" + name).id for name in robot.KEY_BODIES]
    observations, increments, ticks = [], [], []
    for episode in range(count):
        clip = motions[episode % len(motions)]
        heights = clip["root_pos"][:, 2]
        pool = np.flatnonzero(heights < (.48 if rng.random() < .85 else .88))
        # Keep deterministic evaluation times out of the training reset sampler.
        pool = pool[(pool != 0) & (pool != round(.5 / env.DT))]
        if not len(pool):
            pool = np.arange(1, len(heights) - 1)
        start = rng.choice(pool)
        if start in (0, round(.5 / env.DT)) and heights[start] + robot.RESET_LIFT <= .5:
            raise ValueError("Training reset coincides with a fixed fallen evaluation pose")
        lift = rng.uniform(.02, .15)
        data = mujoco.MjData(model)
        quat = clip["root_quat"][start]
        data.qpos[:] = np.r_[0., 0., heights[start] + lift, quat[[3, 0, 1, 2]], clip["dof_pos"][start]]
        rot = Rotation.from_quat(quat).as_matrix()
        gyro = rot.T @ clip["root_ang_vel"][start]
        free_vel = clip["root_lin_vel"][start] - rot @ np.cross(gyro, model.body_ipos[1])
        data.qvel[:] = np.r_[free_vel, gyro, clip["dof_vel"][start]]
        mujoco.mj_forward(model, data)
        teacher.reset()
        action, history = np.zeros(29), []
        for tick in range(145 if episode % 3 else 280):
            terms = observation_terms(data, links, action)
            fields = [term.astype(np.float32) for term in terms]
            if tick == 0:
                history = [np.tile(field, (robot.HIST, 1)) for field in fields]
            else:
                history = [np.concatenate((old[1:], field[None])) for old, field in zip(history, fields)]
            obs = np.concatenate([field.flatten() for field in history])
            delta = np.clip(teacher.target(*terms[:4]) - action, -cfg.ode_dmax, cfg.ode_dmax)
            expert = np.clip(action + delta, -cfg.ode_amax, cfg.ode_amax)
            observations.append(obs)
            increments.append((expert - action).astype("f4"))
            ticks.append(tick)
            if student is not None and rng.random() < student_share:
                applied = np.asarray(student_action(obs, action.astype("f4")))
            else:
                applied = expert
            # Motor target perturbations broaden the state distribution.
            noisy = applied + rng.normal(0, .15 if episode % 5 else .6, 29)
            action = np.clip(action + np.clip(noisy - action, -cfg.ode_dmax, cfg.ode_dmax), -cfg.ode_amax, cfg.ode_amax)
            advance(model, data, action)
    return np.stack(observations), np.stack(increments), np.asarray(ticks)



def fit(samples, output, updates, cfg, prior=None, seed=31, *, increment_labels=True):
    obs, goal, tick = (jnp.asarray(np.concatenate(field)) for field in zip(*samples))
    early = jnp.asarray(np.flatnonzero(np.asarray(tick) < 70))
    key = jax.random.key(seed)
    if prior:
        actor, norm, _ = load(prior)
    else:
        actor = nets.pi_init(key, obs.shape[-1], goal.shape[-1])
        norm = nets.rn_update(nets.rn_init(obs.shape[-1]), obs)
        norm["var"] = jnp.maximum(norm["var"], .03**2)
    normalized = jnp.clip(nets.rn_apply(norm, obs), -cfg.obs_clip, cfg.obs_clip)
    mask = jax.tree.map(lambda _: False, actor)
    optimizer = optim.init(actor, mask)

    def loss(policy, inputs, labels):
        return jnp.mean((nets.pi_apply(policy, inputs) - labels)**2)

    @jax.jit
    def update(rng, policy, state, x, y, rises, rate):
        k1, k2, k3 = jax.random.split(rng, 3)
        rows = jnp.r_[jax.random.randint(k1, (128,), 0, len(x)), rises[jax.random.randint(k2, (128,), 0, len(rises))]]
        inputs = x[rows] + .012 * jax.random.normal(k3, (256, x.shape[-1]))
        error, gradient = jax.value_and_grad(loss)(policy, inputs, y[rows])
        policy, state = optim.step(policy, gradient, state, mask, rate)
        return policy, state, error

    for i in range(1, updates + 1):
        key, draw = jax.random.split(key)
        rate = (.0004 if prior else .0007) * (1. - .75 * i / updates)
        actor, optimizer, error = update(draw, actor, optimizer, normalized, goal, early, rate)
        if i % 500 == 0 or i == updates:
            print(f"Update {i}/{updates}, supervised increment MSE {float(error):.4f}", flush=True)
    save_policy(str(output), {"pi": actor, "avg": actor, "obs_norm": norm},
                {"it": 0, "source": "teacher_distillation", "updates": updates, "training_transitions": len(obs)},
                ode=(cfg.ode_dmax, cfg.ode_amax) if increment_labels else None)
    return output.with_suffix(".npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="AMP Flat policy.onnx with sibling policy.json")
    parser.add_argument("--motions", type=Path, default=Path("data/g1_getup/recovery"))
    parser.add_argument("--out", type=Path, default=Path("runs/getup_demo_init"))
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--dagger-episodes", type=int, default=240)
    parser.add_argument("--dagger-updates", type=int, default=3500)
    args = parser.parse_args()
    if min(args.episodes, args.updates, args.dagger_episodes, args.dagger_updates) < 1:
        parser.error("Episode and update counts must be positive")
    cfg = G1GetUpConfig(walk_dir=str(args.motions.resolve()), floor_dir=str(args.motions.resolve()), log_dir=str(args.out))
    teacher = AmpTeacher(args.teacher)
    model = build_scene(robot_name=cfg.robot)
    motions = []
    for f in sorted(args.motions.glob("*.npz")):
        with np.load(f) as clip:
            motions.append(dict(clip))
    if len(motions) != 10:
        parser.error("Expected the ten positive-weight LeggedLab G1 recovery demonstrations")
    checkpoint_dir = args.out / "ckpt"
    if (checkpoint_dir / "0000.npz").exists():
        parser.error(f"Checkpoint already exists in {checkpoint_dir}. Choose a new --out to reproduce training.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "expert_fit").mkdir(exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metadata = {**vars(args), "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                "expert_rollout_seed": 42, "dagger_rollout_seed": 314, "optimizer_seed": 31, "dagger_student_share": .8,
                "teacher_metadata_sha256": hashlib.sha256(args.teacher.with_suffix(".json").read_bytes()).hexdigest()}
    (args.out / "distillation.json").write_text(json.dumps(metadata, default=str, indent=2))
    samples = [collect(model, teacher, motions, cfg, args.episodes, 42)]
    prior = fit(samples, args.out / "expert_fit/0000", args.updates, cfg)
    samples.append(collect(model, teacher, motions, cfg, args.dagger_episodes, 314, prior, .8))
    fit(samples, args.out / "ckpt/0000", args.dagger_updates, cfg, prior)
    metadata["transitions_per_stage"] = [len(sample[0]) for sample in samples]
    (args.out / "distillation.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
