"""Load an exported InstantAMP policy and evaluate its raw action before ODE integration."""

from functools import lru_cache
from typing import NamedTuple

from jax.tree_util import register_static

import jax.numpy as jnp
import numpy as np

from . import env, nets
from .g1_rise import LocalRise
from .g1_capture import CaptureGate
from . import g1_inplace
from .zenbot_rise import ZenbotLibrary, ZenbotRisePolicy, residual_weight


class G1LimbBlend(NamedTuple):
    base: list
    style: list
    weights: object
    gated: object


@register_static
class G1StanceObservationFilter:
    """Estimate landmark locations from encoders and denoise five past sensor samples."""

    def __init__(self):
        from . import g1_getup
        from .kin import Kinematics
        from .sim import build_scene

        self.widths = g1_getup.ACTOR_TERMS
        model = build_scene(robot_name="g1_getup")
        self.kin = Kinematics(model, ["robot_" + name for name in g1_getup.KEY_BODIES])

    def __call__(self, obs):
        batch_shape = obs.shape[:-1]
        history = 5
        slices = jnp.split(obs, history * np.cumsum(self.widths)[:-1], axis=-1)
        terms = [block.reshape(*batch_shape, history, width) for block, width in zip(slices, self.widths)]
        q = terms[2]
        root = jnp.zeros((*q.shape[:-1], 3), q.dtype)
        axes = jnp.broadcast_to(jnp.eye(3, dtype=q.dtype), (*q.shape[:-1], 3, 3))
        points, _ = self.kin.forward(root, axes, q)
        terms[5] = points.reshape(*batch_shape, history, self.widths[5])
        # Least-squares straight line across the five *past* frames. Preserve recent motion
        # and suppress high-frequency jitter. Encoders and applied actions stay at their own timestamps.
        time = jnp.arange(-2, 3, dtype=q.dtype)[:, None]
        for i in (0, 1, 3, 5):
            block = terms[i]
            slope = (time * block).sum(axis=-2, keepdims=True) / 10.
            terms[i] = block.mean(axis=-2, keepdims=True) + time * slope
        return jnp.concatenate([block.reshape(*batch_shape, -1) for block in terms], axis=-1)


@lru_cache(maxsize=1)
def stance_observation_filter():
    return G1StanceObservationFilter()


class G1StandController(NamedTuple):
    """Direct get-up actors in the order they acquire control."""
    actors: tuple
    thresholds: object
    filtered_stance: object = None
    quiet_guard: object = None
    local_rise: object = None
    capture_gate: object = None
    quiet_stance: object = None
    in_place_residual: object = None
    in_place_demo: object = None
    momentum_capture: object = None
    tracking_capture: object = None


def _actor(archive, which, prefix=""):
    root = prefix + which + "/"
    layers = []
    while root + f"{len(layers)}/w" in archive:
        layer = root + f"{len(layers)}/"
        layers.append({key[len(layer):]: jnp.asarray(archive[key]) for key in archive.files if key.startswith(layer)})
    if not layers:
        raise ValueError(f"Missing {root} policy weights")
    norm = {key: jnp.asarray(archive[prefix + "obs_norm/" + key]) for key in ("mean", "var")}
    if prefix + "obs_norm/raw" in archive:
        norm["raw"] = jnp.asarray(True)
    ode = None
    if prefix + "ode/dmax" in archive:
        ode = tuple(float(archive[prefix + "ode/" + key]) or None for key in ("dmax", "amax"))
    return layers, norm, ode


def load(path, which="avg"):
    """Return (policy, observation normalizer, ODE limits) for an NPZ export."""
    with np.load(path, allow_pickle=False) as archive:
        if "zenbot_library/keys" in archive:
            layers, norm, ode = _actor(archive, which)
            scale = archive["zenbot_library/residual_scale"]
            if (ode is not None or layers[0]["w"].shape[0] != 480 or layers[-1]["b"].shape != (23,)
                    or scale.shape or not np.isfinite(scale) or scale < 0):
                raise ValueError(f"Invalid Zenbot recovery residual in {path}")
            from .zenbot_stance import ZenbotStance

            gate = archive.get("zenbot_library/residual_gate")
            if gate is not None and (gate.shape != (2,) or not np.isfinite(gate).all() or gate[0] < 0 or gate[1] <= 0):
                raise ValueError(f"Invalid Zenbot residual distance gate in {path}")
            policy = ZenbotRisePolicy(ZenbotLibrary.from_archive(archive), layers, jnp.asarray(scale),
                                      gate=None if gate is None else jnp.asarray(gate),
                                      stance=ZenbotStance.from_archive(archive))
            return policy, norm, None
        if "stand_thresholds" in archive:
            limits = archive["stand_thresholds"]
            if limits.shape not in ((3,), (7,)) or not np.isfinite(limits).all():
                raise ValueError(f"Invalid G1 stand controller thresholds in {path}")
            prefixes = ("rise/", "balance/") if len(limits) == 3 else ("rise/", "balance/", "stand/")
            parts = [_actor(archive, which, prefix) for prefix in prefixes]
            if any(ode is not None or pi[0]["w"].shape[0] != 570 or pi[-1]["b"].shape != (29,)
                   for pi, _, ode in parts):
                raise ValueError(f"Expected direct 570-input, 29-joint G1 actors in {path}")
            if not (0 < limits[2] <= 1) or (len(limits) == 7 and not (0 < limits[6] <= 1)):
                raise ValueError(f"Invalid G1 stand controller hand-off increment in {path}")
            filtered = None
            if "stand_observation_filter" in archive:
                if str(archive["stand_observation_filter"]) != "encoder_fk_linear5":
                    raise ValueError(f"Unknown stand observation filter in {path}")
                filtered = stance_observation_filter()
            guard = None
            if "stand_guard" in archive:
                cfg = archive["stand_guard"]
                if (len(parts) != 3 or cfg.shape != (4,) or not np.isfinite(cfg).all()
                        or cfg[0] < 1 or cfg[0] != int(cfg[0]) or cfg[1] >= limits[3]
                        or not -1 <= cfg[2] < limits[4] or cfg[3] <= limits[5]):
                    raise ValueError(f"Invalid G1 stance confirmation or release thresholds in {path}")
                guard = jnp.asarray(cfg)
            local_rise = LocalRise.read(archive, "local_rise/") if "local_rise/keys" in archive else None
            capture, quiet = None, None
            if "quiet_stance" in archive:
                from .g1_sensor_stand import SensorStance
                if str(archive["quiet_stance"]) != "sensor_wbc_v1" or guard is None:
                    raise ValueError(f"Invalid G1 quiet stance controller in {path}")
                capture, quiet = CaptureGate.read(archive), SensorStance.read(archive)
            tracking = None
            if any(name.startswith("tracking_capture/") for name in archive):
                from .g1_tracking_capture import TrackingCapture
                tracking = TrackingCapture.from_archive(archive)
            from .g1_momentum import MomentumCapture
            return G1StandController(tuple(pi for pi, _, _ in parts), jnp.asarray(limits), filtered, guard,
                                     local_rise, capture, quiet, g1_inplace.read(archive),
                                     g1_inplace.LandingLibrary.read(archive), momentum_capture=MomentumCapture.read(archive), tracking_capture=tracking), tuple(
                norm for _, norm, _ in parts), None
        if "blend" not in archive:
            return _actor(archive, which)
        base, base_norm, ode = _actor(archive, which, "base/")
        style, style_norm, style_ode = _actor(archive, which, "style/")
        weights = archive["blend"]
        if ode != style_ode or weights.shape != (env.J,) or any(
            actor[0]["w"].shape[0] != env.OBS_DIM or actor[-1]["b"].shape != weights.shape
            for actor in (base, style)
        ):
            raise ValueError(f"Incompatible G1 actors or joint weights in {path}")
        return G1LimbBlend(base, style, jnp.asarray(weights), jnp.asarray(archive["gate"])), (base_norm, style_norm), ode


def g1_recovery_gate(obs):
    """Weight from projected gravity and bilateral knee flexion in the latest 96-value G1 frame."""
    frame = obs[..., -env.FRAME_DIM:]
    knees = np.array([env.DOF.index("left_knee"), env.DOF.index("right_knee")])
    tilt = 1 + frame[..., 5]
    knee_angles = frame[..., 9 + knees] + env.DEFAULT_Q[knees]
    tilt_gate = jnp.clip((tilt - .08) / .35, 0., 1.)
    kneel_gate = jnp.clip((knee_angles.min(-1) - 1.) / .65, 0., 1.)
    return jnp.maximum(tilt_gate, kneel_gate)


def policy_output(policy, norm, obs, obs_clip=10.):
    """Return the raw output from unnormalized observation history, batched or unbatched."""
    def apply(layers, stats):
        if "raw" in stats:
            return nets.pi_apply(layers, obs)
        return nets.pi_apply(layers, jnp.clip(nets.rn_apply(stats, obs), -obs_clip, obs_clip))

    if isinstance(policy, G1StandController):
        raise ValueError("A G1 stand controller needs per-environment memory. Use policy_step().")
    if isinstance(policy, ZenbotRisePolicy):
        if policy.stance is not None:
            raise ValueError("A Zenbot stance controller needs per-environment memory. Use policy_step().")
        base, distance = policy.library.action_with_distance(obs)
        w = residual_weight(distance, policy.gate)
        return base + (w * policy.scale)[..., None] * jnp.tanh(apply(policy.layers, norm))
    if isinstance(policy, G1LimbBlend):
        base = apply(policy.base, norm[0])
        style = apply(policy.style, norm[1])
        gate = jnp.where(policy.gated, g1_recovery_gate(obs), 1.)
        return base + (gate[..., None] * policy.weights) * (style - base)
    return apply(policy, norm)

def policy_memory(policy, obs):
    """Initial per-environment controller state at an episode boundary."""
    if isinstance(policy, ZenbotRisePolicy) and policy.stance is not None:
        return jnp.zeros((*obs.shape[:-1], 3), dtype=obs.dtype)
    if isinstance(policy, G1StandController):
        size = len(policy.actors) - 1 + int(policy.quiet_guard is not None)
        if policy.capture_gate is not None:
            size += 31  # Blend fraction, continuous confirmation count and the 29 pre-handoff actions.
        if policy.momentum_capture is not None:
            size += policy.momentum_capture.memory_size
        memory = jnp.zeros((*obs.shape[:-1], size), dtype=obs.dtype)
        if policy.tracking_capture is not None:
            memory = jnp.concatenate((memory, policy.tracking_capture.memory(obs)), axis=-1)
        return memory
    return None


def reset_policy_memory(memory, done):
    return None if memory is None else jnp.where(done[..., None], 0., memory)


def _posture_features(obs):
    """Ankle depths below the root and its upright cosine in the latest sensor frame."""
    x, z = obs[..., 39:42], obs[..., 42:45]
    vertical = jnp.stack((x[..., 2], jnp.cross(z, x)[..., 2], z[..., 2]), axis=-1)
    feet = obs[..., -18:-12].reshape((*obs.shape[:-1], 2, 3))
    return -jnp.sum(feet * vertical[..., None, :], axis=-1), z[..., 2]


def policy_step(policy, norm, obs, memory, obs_clip=10.):
    """Return the raw action and the controller memory for the next physical step."""
    if isinstance(policy, ZenbotRisePolicy) and policy.stance is not None:
        action = policy_output(policy._replace(stance=None), norm, obs, obs_clip)
        return policy.stance.step(obs, action, memory)
    if not isinstance(policy, G1StandController):
        return policy_output(policy, norm, obs, obs_clip), None
    if policy.tracking_capture is not None:
        size = policy.tracking_capture.memory(obs).shape[-1]
        memory, tracking_memory = memory[..., :-size], memory[..., -size:]
    if policy.momentum_capture is not None:
        size = policy.momentum_capture.memory_size
        memory, momentum_memory = memory[..., :-size], memory[..., -size:]
    depths, upright = _posture_features(obs)
    limit = policy.thresholds
    rise_gate = limit[:3] if policy.local_rise is None else policy.local_rise.handoff_thresholds(obs, limit[:3])
    if policy.local_rise is not None and policy.local_rise.aligned_handoff is not None:
        min_upright, max_spread, depth, increment = policy.local_rise.aligned_handoff
        aligned = (upright >= min_upright) & (jnp.abs(depths[..., 0] - depths[..., 1]) <= max_spread)
        rise_gate = jnp.where(aligned, depth, rise_gate[0]), rise_gate[1], jnp.where(aligned, increment, rise_gate[2])
    reached = (jnp.max(depths, axis=-1) >= rise_gate[0]) & (upright >= rise_gate[1])
    def advance(share, condition, increment):
        return jnp.where(condition | (share > 0), jnp.minimum(1., share + increment), 0.)
    def handoff(action, incoming, share):
        weight = share[..., None]
        # Exact endpoints preserve the original source action on either side of a hand-off.
        return jnp.where(weight >= 1., incoming, action + weight * (incoming - action))
    share = advance(memory[..., 0], reached, rise_gate[2])
    action = policy_output(policy.actors[0], norm[0], obs, obs_clip)
    if policy.local_rise is not None:
        action = policy.local_rise.apply(obs, action)
    stance_obs = policy.filtered_stance(obs) if policy.filtered_stance is not None else obs
    balance = policy_output(policy.actors[1], norm[1], stance_obs, obs_clip)
    action = handoff(action, balance, share)
    if len(policy.actors) == 2:
        return action, share[..., None]
    if policy.quiet_guard is not None:
        # The rise transition above always uses the original signals. Confirmation and
        # release of quiet stance use the same optional encoder/IMU filter as its policy.
        depths, upright = _posture_features(stance_obs)
        gyro_obs = stance_obs
    else:
        gyro_obs = obs
    gyro = jnp.linalg.norm(gyro_obs[..., 12:15], axis=-1)
    ankle_depth = jnp.min(depths, axis=-1)
    settled = (share >= 1.) & (ankle_depth >= limit[3]) & (upright >= limit[4]) & (gyro <= limit[5])
    if policy.quiet_guard is None:
        stand_share = advance(memory[..., 1], settled, limit[6])
        next_memory = jnp.stack((share, stand_share), axis=-1)
    else:
        frames, release_depth, release_cosine, release_gyro = policy.quiet_guard
        unstable = (ankle_depth < release_depth) | (upright < release_cosine) | (gyro > release_gyro)
        consecutive = jnp.where(settled & ~unstable, jnp.minimum(memory[..., 2] + 1., frames), 0.)
        stand_share = jnp.where(unstable, 0., advance(memory[..., 1], consecutive >= frames, limit[6]))
        next_memory = jnp.stack((share, stand_share, consecutive), axis=-1)
    stand = policy_output(policy.actors[2], norm[2], stance_obs, obs_clip)
    action = handoff(action, stand, stand_share)
    if policy.capture_gate is not None:
        capture = policy.capture_gate.transition(obs, memory[..., 3:], share)
        fraction = capture[..., 0]
        quiet = policy.quiet_stance(obs, fraction > 0, capture[..., 2:])
        action = jnp.where((fraction > 0)[..., None], handoff(capture[..., 2:], quiet, fraction), action)
        next_memory = jnp.concatenate((next_memory, capture), axis=-1)
    demo_gate = 1.
    if policy.in_place_demo is not None:
        target, confidence, distance = policy.in_place_demo.target(obs)
        if policy.capture_gate is not None:
            confidence *= 1. - next_memory[..., 3]
        action = handoff(action,target,confidence)
        w = jnp.clip((distance-.015)/.35,0.,1.)
        demo_gate = w*w*(3.-2.*w)
    if policy.in_place_residual is not None:
        _, gate = g1_inplace.stance_inputs(obs)
        if policy.capture_gate is not None:
            gate = gate * (1. - next_memory[..., 3])
        action = action + (demo_gate*gate)[..., None] * g1_inplace.residual(policy.in_place_residual, obs)
    if policy.momentum_capture is not None:
        quiet_fraction = next_memory[..., 3] if policy.capture_gate is not None else jnp.zeros_like(share)
        action, momentum_memory = policy.momentum_capture.step(obs, action, momentum_memory, quiet_fraction)
        next_memory = jnp.concatenate((next_memory, momentum_memory), axis=-1)
    if policy.tracking_capture is not None:
        action, tracking_memory = policy.tracking_capture.step(obs, action, tracking_memory)
        next_memory = jnp.concatenate((next_memory, tracking_memory), axis=-1)
    return action, next_memory
