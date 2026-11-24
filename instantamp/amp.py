"""Adversarial motion prior (AMP, Peng et al. 2021) for the G1, in JAX and on the world model's imagination.

A discriminator tells windows of `frames` consecutive 50 Hz frames of the robot's motion from windows of the human walking and running
clips (data/amp/walk_run, tools/download_data.sh). Its score gives a style reward for the imagined motion, which the policy follows
through the world model like every other term of the reward: the style reward is differentiable in the imagined states.

A frame holds the joint positions and velocities, the pelvis's angular velocity in the pelvis frame, the direction of gravity in the
pelvis frame, and the positions of the feet and hands relative to the pelvis in its heading frame (the ankle-roll and wrist-yaw links):
29 + 29 + 3 + 3 + 12 = 76 values (64 with use_wrists(False)).

The critic is a WGAN-GP critic (Gulrajani et al. 2017): the loss mean D(robot) - mean D(demo) + 1e-3 mean D(demo)^2 plus 10 times the squared
deviation of the gradient norm from 1 at random points between a demo window and a robot window. The style reward is
(1 + tanh((d - mu_robot) / scale)) / 2, centered at the robot's running mean score with a fixed scale (see `style`). Why WGAN-GP and not
LSGAN: the policy follows the reward's gradient through the world model, so the critic's slope matters.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import nets, optim
from .rot import mv3, quat_to_mat, tr3, yaw_mat

WRIST = (19, 20, 21, 26, 27, 28)  # the G1's wrist roll, pitch and yaw in action order
KEPT = np.arange(29)
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
FRAME_DIM = 2 * len(KEPT) + 3 + 3 + 3 * len(KEY_BODIES)


def use_wrists(on: bool):
    """Keep the wrists in the frame features (on, the default) or leave them out (off). Left out, nothing in the reward held the
    wrists, and a policy turned them to about 1.5 rad, each side another way. Call before Demos and init: the critic's input width
    follows."""
    global KEPT, FRAME_DIM
    KEPT = np.arange(29) if on else np.asarray([j for j in range(29) if j not in WRIST])
    FRAME_DIM = 2 * len(KEPT) + 3 + 3 + 3 * len(KEY_BODIES)


EMA = 0.99  # running mean scores of the critic
GP_WEIGHT = 10.0
DRIFT = 1e-3
SCORE_FLOOR = 0.5  # smallest half-gap in the style reward


def frame(S, key_pos, joints=None):
    """Frame features (.., FRAME_DIM) of states S = (p, R, v, w, q, qd) (world-frame w) and the key bodies' world positions (.., 4, 3)."""
    joints = KEPT if joints is None else joints
    p, R, _, w, q, qd = S
    return jnp.concatenate((q[..., joints], qd[..., joints], mv3(tr3(R), w), -R[..., 2, :], mv3(tr3(yaw_mat(R))[..., None, :, :], key_pos - p[..., None, :]).reshape(*p.shape[:-1], -1)), -1)


SUPPORT_FRAME_DIM = 22


def gait_window(window, joints, stride, dt):
    """Compute joint motion from sampled poses identically for mocap and simulated trajectories."""
    samples = window[..., ::stride, :]
    q = samples[..., :joints]
    velocity = jnp.diff(q, axis=-2) / (stride * dt)
    velocity = jnp.concatenate((velocity[..., :1, :], velocity), axis=-2)
    return samples.at[..., joints:2*joints].set(velocity)


SUPPORT_FEATURE_DIM = 52
ORIENTED_SUPPORT_DIM = SUPPORT_FEATURE_DIM + 9


def support_window(raw, dt, oriented=False):
    """Embed a physical Zenbot pose sequence for the ground-support discriminator.

    A raw frame contains pelvis height, body-frame gravity, six pelvis-relative
    horizontal landmark positions in the heading frame, then six collision
    clearances from forward kinematics. The order is feet, knees, hands.
    The same finite differences and contact geometry apply to every window.
    """
    if raw.shape[-1] != SUPPORT_FRAME_DIM or raw.shape[-2] == 0 or dt <= 0:
        raise ValueError("Expected a nonempty Zenbot support window with positive frame spacing")
    pelvis = raw[..., :1]
    gravity = raw[..., 1:4]
    points = raw[..., 4:16].reshape(*raw.shape[:-1], 6, 2)
    # The native motion clips can intersect the plane. Group contact and small
    # penetrations together so depth below the ground cannot identify the source.
    above = jnp.maximum(raw[..., 16:22] - .025, 0.)
    clearance = jnp.tanh(above / .40)
    proximity = 1. / (1. + jnp.square(above / .07))
    bilateral = proximity[..., ::2] * proximity[..., 1::2]

    def difference(sequence):
        return jnp.diff(sequence, axis=-2, prepend=sequence[..., :1, :])

    def distance(xy):
        return jnp.sqrt(jnp.sum(jnp.square(xy), axis=-1) + 1e-6)

    # Pelvis heading is ill-defined when its forward axis is vertical.
    # Distances remain valid in those poses and across heading wraparounds.
    heading_confidence = jnp.sqrt(jnp.maximum(1. - gravity[..., :1]**2, 1e-6))
    xy = jnp.tanh(points.reshape(*raw.shape[:-1], 12) / .50) * heading_confidence
    radius = distance(points)
    hand_to_foot = distance(points[..., 4:6, :] - points[..., :2, :])
    posture = jnp.concatenate((pelvis / .75, gravity), axis=-1)
    rates = jnp.tanh(difference(jnp.concatenate((posture, clearance, radius, hand_to_foot), axis=-1)) / dt)

    # Record what changed while the palms were planted. A stationary prone
    # hold and raising the pelvis over planted hands have different windows.
    hands = bilateral[..., 2:3]
    preceding = jnp.concatenate((hands[..., :1, :], hands[..., :-1, :]), axis=-2)
    planted = .5 * (preceding + hands)
    elapsed = jnp.broadcast_to(jnp.arange(raw.shape[-2])[:, None] * dt, hands.shape)
    changes = difference(jnp.concatenate((elapsed, pelvis, radius[..., :2].mean(-1, keepdims=True)), axis=-1))
    during_support = jnp.tanh(jnp.cumsum(planted * changes, axis=-2) / .35)
    features = jnp.concatenate((posture, xy, clearance, proximity, bilateral, rates, during_support), axis=-1)
    if oriented:
        # A palm behind a supine torso and one in front of a prone torso can
        # have the same clearance. Let the discriminator learn this joint distribution.
        palms = jnp.concatenate((proximity[..., 4:6], hands), -1)
        palm_orientation = (palms[..., :, None] * gravity[..., None, :]).reshape(*raw.shape[:-1], 9)
        features = jnp.concatenate((features, palm_orientation), -1)
    return features


class Demos:
    """The frame features of all demo clips, and the start of every window of `frames` frames that stays inside a clip."""

    def __init__(self, directory: str, sim, frames: int, *, per_clip=False, frame_fn=None, extra_clip=None):
        from . import g1_getup
        self.g1_bank = g1_getup.load_reference(directory) if sim.robot is g1_getup else None
        self.amp_frame = sim.amp_frame
        self.frame_fn = frame_fn
        files = [str(directory)] if Path(directory).is_file() else sorted(glob.glob(os.path.join(directory, "*.npz")))
        assert files, f"no clips in {directory} (bash tools/download_data.sh)"
        source_names = [Path(f).stem for f in files]
        if extra_clip is not None:
            extra_clip = Path(extra_clip)
            if not extra_clip.is_file():
                raise FileNotFoundError(f"Missing additional AMP reference: {extra_clip}")
            if any(Path(f).resolve() == extra_clip.resolve() for f in files):
                raise ValueError(f"Additional AMP reference already in {directory}: {extra_clip}")
            files.append(str(extra_clip))
        feats, starts, n = [], [], 0
        for f in files:
            d = np.load(f)
            assert int(d["fps"]) == 50, f
            x, y, z, w = d["root_quat"].T  # (x, y, z, w) -> the rotation matrix
            R = quat_to_mat(jnp.stack([w, x, y, z], -1))
            assert d["dof_pos"].shape[1] == sim.J, f
            q, qd = jnp.asarray(d["dof_pos"]), jnp.asarray(d["dof_vel"])
            S = (jnp.asarray(d["root_pos"]), R, jnp.asarray(d["root_lin_vel"]) if frame_fn is not None or getattr(sim, "mixed", False) else None,
                 jnp.asarray(d["root_ang_vel"]), q, qd)
            feats.append(frame_fn(S) if frame_fn is not None else sim.amp_frame(S, d.get("key_body_pos")))
            starts.append(np.arange(n, n + len(x) - frames + 1))
            n += len(x)
        self.frames = jnp.concatenate(feats, 0)
        self.starts = jnp.asarray(np.concatenate(starts), jnp.int32)
        self.window = frames
        self.clip_lengths = jnp.array([len(a) for a in starts]) if sim.recovery or per_clip else None
        self.clip_probabilities = None
        if self.clip_lengths is not None:
            if min(map(len, starts)) < 1:
                raise ValueError("A get-up clip is shorter than the discriminator window")
            self.clip_offsets = jnp.array([a[0] for a in starts])
            sampling = Path(directory) / "sampling.json"
            if per_clip and sampling.exists():
                weights = json.loads(sampling.read_text())["weights"]
                if set(weights) != set(source_names):
                    raise ValueError(f"Sampling manifest and motion files differ in {directory}")
                values = np.asarray([weights[name] for name in source_names], np.float32)
                if extra_clip is not None:
                    values = np.append(values, values.mean())
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"Invalid clip weights in {sampling}")
                self.clip_probabilities = jnp.asarray(values / values.sum())

    def sample(self, key, B):
        """Sample recovery windows uniformly by clip, or locomotion windows uniformly by start frame."""
        if self.g1_bank is not None:
            from .g1_getup import sample_reference
            f = sample_reference(self.g1_bank, key, B, self.window)
            state = (f["pos"], quat_to_mat(f["quat"]), f["v"], f["w"], f["q"], f["qd"])
            return self.frame_fn(state) if self.frame_fn is not None else self.amp_frame(state, f.get("points"))
        i = self.starts[jax.random.randint(key, (B,), 0, len(self.starts))]
        if self.clip_lengths is not None:
            kc, kt = jax.random.split(key)
            clip = jax.random.randint(kc, (B,), 0, len(self.clip_lengths))
            if self.clip_probabilities is not None:
                clip = jax.random.choice(kc, len(self.clip_lengths), (B,), p=self.clip_probabilities)
            i = self.clip_offsets[clip] + jax.random.randint(kt, (B,), 0, self.clip_lengths[clip])
        return self.frames[i[:, None] + jnp.arange(self.window)]

    def sample_opening_windows(self, key, B):
        """Draw independent clip/end-frame samples, including short histories at clip start."""
        if self.clip_lengths is None:
            raise ValueError("Opening-window sampling requires per-clip motion data")
        kc, kt = jax.random.split(key)
        clip = jax.random.choice(kc, len(self.clip_lengths), (B,), p=self.clip_probabilities)
        frame_counts = self.clip_lengths[clip] + self.window - 1
        end = jax.random.randint(kt, (B,), 0, frame_counts)
        history = end[:, None] + jnp.arange(1 - self.window, 1)
        return self.frames[self.clip_offsets[clip, None] + jnp.maximum(history, 0)]

    def sample_wrong_tempo(self, key, batch, frames, stride, joints, *, freeze_start=False):
        """Negative Zenbot examples from moving expert segments, frozen or played at twice the recorded rate."""
        kc, kt, ks = jax.random.split(key, 3)
        clip = jax.random.choice(kc, len(self.clip_lengths), (batch,), p=self.clip_probabilities)
        lengths = self.clip_lengths[clip] + self.window - 1
        span = 2 * stride * (frames - 1) + 1
        start = jax.random.randint(kt, (batch,), 0, jnp.maximum(lengths - span + 1, 1))
        frame = jnp.minimum(start[:, None] + 2 * stride * jnp.arange(frames), lengths[:, None] - 1)
        fast = self.frames[self.clip_offsets[clip, None] + frame]
        # Keep stationary intervals as valid demonstrations of resting or standing.
        trim = frames // 4
        joint_range = jnp.ptp(fast[:, trim:frames-trim, :joints], axis=1)
        moving = (lengths >= span) & (joint_range.max(-1) > np.deg2rad(12.))
        speed = 2. * jax.random.bernoulli(ks, .5, (batch, 1, 1))
        # Freezing the beginning also covers prone poses that normally start moving within this segment.
        anchor = 0 if freeze_start else frames // 2
        raw = jnp.where(speed > 0, fast, fast[:, anchor:anchor+1])
        # Scale joint, pelvis angular and pelvis linear velocities together with the time base.
        raw = raw.at[..., joints:2*joints+3].multiply(speed)
        return raw.at[..., -3:].multiply(speed), moving

    def prepare_ground_heights(self):
        """Group recovery windows by coarse pelvis height, retaining equal clip weights within each group."""
        starts = np.asarray(self.starts)
        height = np.asarray(self.frames)[starts + self.window - 1, -4]
        groups = np.searchsorted([.35, .55], height)
        clip_id = np.repeat(np.arange(len(self.clip_lengths)), np.asarray(self.clip_lengths))
        cdfs = []
        for group in range(3):
            chosen = groups == group
            if not chosen.any():
                chosen = np.ones(len(starts), dtype=bool)
            counts = np.bincount(clip_id[chosen], minlength=len(self.clip_lengths))
            weight = chosen / np.maximum(counts[clip_id], 1)
            cdfs.append(np.cumsum(weight) / weight.sum())
        self.height_cdf = jnp.asarray(np.stack(cdfs), jnp.float32)

    def sample_ground_heights(self, key, heights):
        # Each independent expert draw can come from any clip in the height group.
        group = (heights > .35).astype(jnp.int32) + (heights > .55).astype(jnp.int32)
        u = jax.random.uniform(key, heights.shape)
        indices = jax.vmap(lambda cdf: jnp.searchsorted(cdf, u, side="right"))(self.height_cdf)
        start = self.starts[jnp.minimum(indices[group, jnp.arange(len(u))], len(self.starts)-1)]
        return self.frames[start[:, None] + jnp.arange(self.window)]


# ---------------------------------------------------------------- the critic
def init(key, frames: int, hidden=(1024, 512), frame_dim=None):
    """The critic's parameters (an MLP with ELU on the flattened window, one output), the running normalizer of the frames, the running
    mean scores of the demos and the robot, and the optimizer state."""
    frame_dim = FRAME_DIM if frame_dim is None else frame_dim
    layers = nets.mlp_init(key, [frames * frame_dim, *hidden, 1])
    mask = jax.tree.map(lambda _: False, layers)
    return {"net": layers, "norm": nets.rn_init(frame_dim), "mu": jnp.zeros((2,)), "opt": optim.init(layers, mask), "ramp": jnp.zeros(())}


NORM_CLIP = 10.0  # the normalized frames are clipped, so that an extreme imagined state cannot overflow the critic or its gradient


def normalize(norm, window):
    return jnp.clip(nets.rn_apply(norm, window, eps=1e-2), -NORM_CLIP, NORM_CLIP)


def score(net, norm, window):
    """The critic's score of windows (.., frames, FRAME_DIM) of raw frame features."""
    x = normalize(norm, window).reshape(*window.shape[:-2], -1)
    return nets.mlp_apply(net, x, jax.nn.elu)[..., 0]


def style(st, d, scale: float = 0.0):
    """The style reward in [0, 1] of scores d, from the running mean scores of the demos and the robot.

    scale = 0: centered at the midpoint of the two means, with their half-gap as the scale. The robot's mean then always gets about
    0.12, and the slope is 1 / (2 half-gap): with a gap of about 140 the reward is nearly flat and its gradient through the world
    model nearly 0.
    scale > 0: centered at the robot's mean score, with this fixed scale: the robot's mean gets 0.5, and the slope where the robot
    is stays 1 / (2 scale) whatever the gap. (Not centered at the midpoint with a capped scale: the robot would sit deep in the
    saturated tail.)"""
    mu_d, mu_r = st["mu"][0], st["mu"][1]
    if scale > 0:
        return 0.5 * (1 + jnp.tanh((d - mu_r) / scale))
    m, s = (mu_d + mu_r) / 2, jnp.maximum((mu_d - mu_r) / 2, SCORE_FLOOR)
    return 0.5 * (1 + jnp.tanh((d - m) / s))


def loss(net, norm, demo, robot, wt, key):
    """WGAN-GP loss of the critic on windows of the demos and of the robot (B, frames, FRAME_DIM) and the statistics; wt (B,) weighs the
    robot windows (0: not usable) and, by pairs, the interpolates."""
    xd, xr = normalize(norm, demo), normalize(norm, robot)
    f = lambda x: nets.mlp_apply(net, x.reshape(x.shape[0], -1), jax.nn.elu)[:, 0]  # noqa: E731
    d_demo, d_robot = f(xd), f(xr)
    eps = jax.random.uniform(key, (xd.shape[0], 1, 1))
    xh = (eps * xd + (1 - eps) * xr).reshape(xd.shape[0], -1)
    g = jax.vmap(jax.grad(lambda x: nets.mlp_apply(net, x[None], jax.nn.elu)[0, 0]))(xh)
    n = jnp.maximum(wt.sum(), 1.0)
    gp = GP_WEIGHT * jnp.sum(wt * (jnp.linalg.norm(g, axis=-1) - 1.0) ** 2) / n
    d_robot = jnp.sum(wt * d_robot) / n
    total = d_robot - d_demo.mean() + DRIFT * jnp.mean(d_demo**2) + gp
    return total, {"d_demo": d_demo.mean(), "d_robot": d_robot, "gp": gp}


def recovery_features(norm, window, leggedlab):
    if leggedlab:
        # rsl_rl EmpiricalNormalization normalizes by std + eps.
        return (window - norm["mean"]) / (jnp.sqrt(norm["var"]) + 1e-2)
    return normalize(norm, window)


def recovery_style(st, window, scale, leggedlab=False, *, soft_floor=False, motion_train=False):
    x = recovery_features(st["norm"], window, leggedlab or motion_train).reshape(*window.shape[:-2], -1)
    d = nets.mlp_apply(st["net"], x, jax.nn.elu if motion_train else jax.nn.leaky_relu)[..., 0]
    quality = 1. - 0.25 * (d - 1.)**2
    if soft_floor:
        # Keep a slope for imagined gait windows just below the least-squares cutoff.
        return scale * jax.nn.softplus(10. * quality) / jax.nn.softplus(10.)
    return scale * jnp.maximum(quality, 0.)


def recovery_loss(net, norm, demo, robot, wt, key, leggedlab=False, motion_train=False):
    def predict(row):
        return nets.mlp_apply(net, row, jax.nn.elu if motion_train else jax.nn.leaky_relu)[..., 0]

    xd = recovery_features(norm, demo, leggedlab or motion_train).reshape(len(demo), -1)
    xr = recovery_features(norm, robot, leggedlab or motion_train).reshape(len(robot), -1)
    dd, dr = predict(xd), predict(xr)
    n = jnp.maximum(wt.sum(), 1.)
    gp = 10. * jnp.square(jax.vmap(jax.grad(predict))(xd)).sum(-1).mean()
    loss = 0.5 * (jnp.square(dd - 1).mean() + (wt * (dr + 1)**2).sum() / n) + gp
    return loss, {"d_demo": dd.mean(), "d_robot": (wt * dr).sum() / n, "gp": gp}


def train_step_with_robot(st, demo, robot, wt, key, lr, **kwargs):
    """Keep the prior and its normalization fixed until the collector supplies a valid negative."""
    def update():
        return train_step(st, demo, robot, wt, key, lr, **kwargs)

    def skip():
        zero = jnp.zeros((), jnp.float32)
        return st, {"d_demo": zero, "d_robot": zero, "gp": zero}

    return jax.lax.cond((wt > 0).any(), update, skip)


def train_step(st, demo, robot, wt, key, lr: float, recovery=False, leggedlab=False, include_demo_norm=False, motion_train=False):
    """One critic step on a batch of demo and robot windows; the robot windows with weight 0 take no part."""
    args = (st["net"], st["norm"], demo, robot, wt, key)
    if recovery:
        (_, aux), grads = jax.value_and_grad(recovery_loss, has_aux=True)(*args, leggedlab=leggedlab, motion_train=motion_train)
    else:
        (_, aux), grads = jax.value_and_grad(loss, has_aux=True)(*args)
    if not (leggedlab or motion_train):
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
    if recovery:
        # Coupled Adam weight decay, following the reference trunk and head parameter groups.
        grads = [{k: g + (0.1 if i == len(grads)-1 else 0.001) * st["net"][i][k] for k, g in layer.items()}
                 for i, layer in enumerate(grads)]
    net, opt = optim.step(st["net"], grads, st["opt"], jax.tree.map(lambda _: False, st["net"]), lr, betas=(0.9, 0.999) if recovery else (0.5, 0.9))
    # In dev/get_up, PPOAMP.update feeds only the replayed policy windows into
    # update_normalization and applies Adam directly to the discriminator.
    max_count = 1e8 if leggedlab or motion_train else 2e8
    norm = nets.rn_update(st["norm"], robot.reshape(-1, robot.shape[-1]), weights=jnp.repeat(wt, robot.shape[1]),
                          max_count=max_count)
    if (recovery and not (leggedlab or motion_train)) or include_demo_norm:
        norm = nets.rn_update(norm, demo.reshape(-1, demo.shape[-1]), max_count=max_count)
    first = (st["mu"][0] == 0) & (st["mu"][1] == 0)
    now = jnp.stack([aux["d_demo"], aux["d_robot"]])
    live = wt.sum() > 0  # no usable robot window: the scores of this step say nothing about the robot
    mu = jnp.where(live, jnp.where(first, now, EMA * st["mu"] + (1 - EMA) * now), st["mu"])
    return {**st, "net": net, "opt": opt, "norm": norm, "mu": mu}, aux
