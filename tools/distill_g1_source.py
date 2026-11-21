"""Fit the optional single-policy student from the recorded LeggedLab G1 get-up rollouts.

    python -m tools.distill_g1_source --bundle runs/g1_getup_source_recipe --out runs/source_refit

The bundle contains the four banks of physical rollouts, the rise and balance
policies that supplied training labels, and a manifest with their checksums.
The actor exported under ckpt/5400.npz uses the named task's direct action and
570 raw observation values. All three supervised stages use seed 97.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import nets, optim
from instantamp.inference import load, policy_output
from instantamp.train import save_policy


@dataclass(frozen=True)
class Stage:
    updates: int
    horizon: int
    lr: float
    correction_bank: str | None = None
    feedback: bool = False


STAGES = (
    Stage(3800, 40, 0.000095),
    Stage(4500, 32, 0.00016, "preferred_train_states"),
    Stage(5400, 36, 0.00021, "preferred_state_mix", True),
)
ACTION_HISTORY = slice(335, 480)


def read_bank(directory, horizon):
    observations, labels, flat_observations, flat_labels = [], [], [], []
    starts, early, all_rows = [], [], []
    offset = 0
    for path in sorted(directory.glob("*/rollout.npz")):
        with np.load(path, allow_pickle=False) as f:
            q, v, ref = (f[name] for name in ("qpos", "qvel", "source_qpos"))
            actions, targets, obs = (f[name] for name in ("action", "expert", "actor_obs"))
            if obs.shape != (len(actions), 570) or targets.shape != (len(actions), 29):
                raise ValueError(f"Invalid named-task policy dimensions in {path}")
            if not all(np.isfinite(a).all() for a in (obs, targets, actions, q, v, ref)):
                raise ValueError(f"Nonfinite physical transition in {path}")
            fallen = np.flatnonzero((q[1:, 2] < 0.50) & (ref[1:, 2] > 0.71))
            holds = ((q[-50:, 2] > 0.70) & (1 - 2 * np.sum(q[-50:, 4:6] ** 2, axis=1) > 0.75)).all()
            n = len(actions)
            if not holds and "complete_supervision" not in f and len(fallen):
                n = int(fallen[0])
            if n >= horizon:
                observations.append(obs[:n])
                labels.append(targets[:n])
                starts.append(offset)
                early.extend(range(offset, offset + min(n - horizon + 1, 32)))
                all_rows.extend(range(offset, offset + n - horizon + 1))
                offset += n
            # This supplemental one-step loss only uses the part of the recording
            # before a fall or large discontinuity in the native simulator.
            bad = (q[1:, 2] < 0.45) & (ref[1:, 2] > 0.69)
            bad |= (abs(v[1:]).max(1) > 160) | (abs(np.diff(q[:, :3], axis=0)).max(1) > 0.45)
            bad |= abs(np.diff(q[:, 7:], axis=0)).max(1) > 2.8
            cutoff = int(np.argmax(bad)) if bad.any() else len(actions)
            flat_observations.append(obs[:cutoff])
            flat_labels.append(targets[:cutoff])

    if not starts:
        raise ValueError(f"No eligible complete trajectories in {directory}")

    def stack(arrays):
        return jnp.asarray(np.concatenate(arrays).astype(np.float32))

    return (
        (stack(observations), stack(labels)),
        (starts, early, all_rows),
        (stack(flat_observations), stack(flat_labels)),
    )


def huber(error, threshold):
    return jnp.where(abs(error) < threshold, 0.5 * error**2, threshold * (abs(error) - 0.5 * threshold))


def fit(bundle, output, prior, stage, updates, seed):
    canonical = bundle / "preferred_bank"
    directories = [canonical]
    if stage.correction_bank:
        directories += [bundle / stage.correction_bank, bundle / "sampled_stance6500"]
    observations, labels, indices = [], [], []
    offset = 0
    extra = None
    for directory in directories:
        (x, y), starts, flat = read_bank(directory, stage.horizon)
        observations.append(x)
        labels.append(y)
        indices.extend(jnp.asarray(group) + offset for group in starts)
        offset += len(x)
        if directory.name == stage.correction_bank:
            extra = flat
    if extra is None:
        _, _, extra = read_bank(bundle / "sampled_stance6500", stage.horizon)
    data = jnp.concatenate(observations), jnp.concatenate(labels)
    counts = (6, 4, 8, 7, 8, 10, 0, 0, 5) if stage.correction_bank else (12, 12, 24)
    print(
        f"Loaded {offset} sequence rows, {len(indices[0])} canonical starts and {len(extra[0])} additional observations",
        flush=True,
    )

    actor, norm, ode = load(prior)
    if ode is not None or "raw" not in norm:
        raise ValueError("The source rise checkpoint must be a raw-observation direct actor")
    if stage.correction_bank:
        rise, rise_norm, _ = load(bundle / "teachers/rise.npz")
        with np.load(canonical / "c0_s0_p0/rollout.npz") as f:
            q = f["source_qpos"]
            up = (q[:, 2] > 0.66) & (1 - 2 * np.sum(q[:, 4:6] ** 2, axis=1) > 0.67)
            first_obs = jnp.asarray(f["actor_obs"][: np.flatnonzero(up)[0]])
    term_widths = np.array([3, 6, 29, 29, 29, 18]) * 5
    first_noise = jnp.asarray(np.repeat([0.10, 0.015, 0.021, 0.22, 0.13, 0.008], term_widths), dtype="f4")
    sensor_noise = jnp.asarray(np.repeat([0.25, 0.045, 0.055, 0.65, 0.65, 0.014], term_widths), dtype="f4")
    if stage.feedback:
        balance, balance_norm, _ = load(bundle / "teachers/balance.npz")
        mean = data[0].mean(axis=0)
        scale = jnp.maximum(data[0].std(axis=0), 0.075)
        first = actor[0]
        actor[0] = {**first, "w": first["w"] * scale[:, None], "b": first["b"] + mean @ first["w"]}

    mask = jax.tree.map(lambda _: False, actor)
    opt = optim.init(actor, mask)

    def act(policy, obs):
        return nets.pi_apply(policy, (obs - mean) / scale if stage.feedback else obs)

    @jax.jit
    def update(actor, opt, key, data, indices, extra, lr):
        x, y = data
        keys = jax.random.split(key, len(indices) + 3)
        starts = jnp.concatenate(
            [rows[jax.random.randint(k, (n,), 0, len(rows))] for rows, k, n in zip(indices, keys, counts)]
        )
        ids = jax.random.randint(keys[-3], (96,), 0, len(extra[0]))
        batch = len(starts)
        mem0 = x[starts, ACTION_HISTORY].reshape(batch, 5, 29)
        mem0 += jax.random.normal(keys[-2], mem0.shape) * 0.25 * (jnp.arange(batch) >= 12)[:, None, None]

        def objective(policy):
            def advance(memory, t):
                observed = x[starts + t]
                state = jnp.concatenate(
                    (observed[:, :335], memory.reshape(batch, 145), observed[:, 480:]), axis=-1
                )
                pred = act(policy, state)
                error = pred - y[starts + t]
                # All five action-history slots eventually contain student outputs.
                # Dampen derivatives through time while keeping the forward actions.
                detached = jax.lax.stop_gradient(pred)
                action = detached + 0.7 * (pred - detached)
                action += 0.045 * jax.random.normal(jax.random.fold_in(keys[-1], t), pred.shape)
                loss = jnp.mean(jnp.mean(huber(error, 1.6), axis=1))
                return jnp.concatenate((memory[:, 1:], action[:, None]), axis=1), loss

            _, losses = jax.lax.scan(advance, mem0, jnp.arange(stage.horizon))
            obs, target = extra[0][ids], extra[1][ids]
            pred = act(policy, obs)
            aux = jnp.mean(huber(pred - target, 1.5))

            if stage.feedback:
                noise = jax.random.normal(jax.random.fold_in(keys[-3], 81), obs.shape) * sensor_noise
                rise_base = policy_output(rise, rise_norm, obs)
                balance_base = policy_output(balance, balance_norm, obs)
                axis = balance_base - rise_base
                weight = jnp.clip(
                    jnp.sum((target - rise_base) * axis, axis=-1, keepdims=True)
                    / (1e-5 + jnp.sum(axis**2, axis=-1, keepdims=True)),
                    0,
                    1,
                )
                delta = (1 - weight) * (policy_output(rise, rise_norm, obs + noise) - rise_base)
                delta += weight * (policy_output(balance, balance_norm, obs + noise) - balance_base)
                # Differentiate the two feedback terms independently. With bf16
                # actor matmuls, reusing a single output rounds their combined adjoint.
                response_delta = act(policy, obs + noise) - pred
                perturbed_error = act(policy, obs + noise) - target - delta
                aux += 1.7 * jnp.mean((response_delta - delta) ** 2) + 0.6 * jnp.mean(
                    jnp.minimum(3.0, perturbed_error**2)
                )

            anchor = 0.0
            if stage.correction_bank:
                k = jax.random.fold_in(keys[-2], 369)
                idx = jax.random.randint(k, (128,), 0, len(first_obs))
                noise = jax.random.normal(jax.random.fold_in(k, 1), (128, 570)) * first_noise
                src = first_obs[idx] + noise * (jnp.arange(128) >= 80)[:, None]
                anchor = 1.8 * jnp.mean((act(policy, src) - policy_output(rise, rise_norm, src)) ** 2)
            return losses.mean() + 0.35 * losses[:4].mean() + 0.16 * aux + anchor

        loss, grad = jax.value_and_grad(objective)(actor)
        grad, norm_grad = optim.clip_by_global_norm(grad, 3.0)
        actor, opt = optim.step(actor, grad, opt, mask, lr, ok=jnp.isfinite(norm_grad))
        return actor, opt, loss

    key = jax.random.key(seed)
    for iteration in range(1, updates + 1):
        key, k = jax.random.split(key)
        lr = stage.lr * (1 - 0.65 * iteration / stage.updates)
        actor, opt, loss = update(actor, opt, k, data, indices, extra, lr)
        if iteration % 200 == 0 or iteration == updates:
            print(f"{iteration:5d}  loss {float(loss):.7f}", flush=True)
        if iteration in (1600, updates):
            deploy = actor
            if stage.feedback:
                w = actor[0]["w"] / scale[:, None]
                deploy = [{**actor[0], "w": w, "b": actor[0]["b"] - mean @ w}, *actor[1:]]
            output.mkdir(exist_ok=True, parents=True)
            path = output / f"{iteration:04d}"
            save_policy(
                str(path),
                {"pi": deploy, "avg": deploy, "obs_norm": norm},
                {
                    "it": iteration,
                    "rows": offset,
                    "horizon": stage.horizon,
                    "seed": seed,
                    "source": "LeggedLab G1 get-up physical distillation",
                },
            )
    return path.with_suffix(".npz")


def verify(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for record in manifest["files"]:
        path = bundle / record["path"]
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Checksum mismatch in {path}")
    print(f"Verified {len(manifest['files'])} bundle files", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", type=Path, help="A new run directory for the three fits")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run one optimizer update at each stage")
    args = p.parse_args()
    verify(args.bundle)
    if args.verify_only:
        return
    if args.out is None:
        p.error("--out is required when fitting")
    args.out.mkdir(exist_ok=False, parents=True)
    config = {
        "robot": "g1_getup",
        "nconmax": 64,
        "njmax": 640,
        "floor_dir": "data/g1_getup/recovery",
        "source": "physical LeggedLab reference distillation",
        "smoke": args.smoke,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    prior = args.bundle / "teachers/rise.npz"
    for i, stage in enumerate(STAGES):
        output = args.out / ("ckpt" if stage.feedback else f"stage{i + 1}/ckpt")
        prior = fit(args.bundle, output, prior, stage, 1 if args.smoke else stage.updates, 97)
    print(f"Final checkpoint {prior}")


if __name__ == "__main__":
    main()
