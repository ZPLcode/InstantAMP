"""Training-only feedback targets on the unmodified states visited by the sampled actor."""

import hashlib
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import zenbot
from .rot import quat_to_mat, so3_log, tr3, yaw_mat


class GroundBank(NamedTuple):
    q: object
    v: object
    raw: object
    gain: object
    lengths: object
    settled: object
    delay_groups: object
    stance_quat: object
    stance_torque: object
    opening_hold: object = None


def load_bank(path, floor_dir):
    floor = sorted(Path(floor_dir).glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        if not floor or [p.name for p in floor] != data["clip_names"].tolist():
            raise ValueError("Physical recovery bank does not match the ordered floor clips")
        if [hashlib.sha256(p.read_bytes()).hexdigest() for p in floor] != data["source_sha256"].tolist():
            raise ValueError("A source floor clip differs from the one used to plan the physical trajectory")
        bank = GroundBank(
            *(data[key] for key in GroundBank._fields[:-1]),
            data["opening_hold"] if "opening_hold" in data else np.zeros(len(floor), np.int32),
        )
    n = len(floor)
    if bank.raw.ndim != 3:
        raise ValueError("Expected clip-by-time-by-joint raw action targets")
    frames = bank.raw.shape[1]
    expected = (
        (n, frames, 30),
        (n, frames, 29),
        (n, frames, 23),
        (n, frames, 23, 104),
        (n,),
        (n,),
        (n, 5),
        (4,),
        (23,),
        (n,),
    )
    if any(x.shape != shape or not np.isfinite(x).all() for x, shape in zip(bank, expected)):
        raise ValueError("Invalid online feedback table shape or nonfinite parameter")
    if (
        not np.issubdtype(bank.lengths.dtype, np.integer)
        or not np.issubdtype(bank.settled.dtype, np.integer)
        or not np.all((bank.lengths >= 1) & (bank.lengths <= frames))
        or not np.all((bank.settled >= 0) & (bank.settled < bank.lengths))
        or not np.all((bank.delay_groups >= 3) & (bank.delay_groups <= 7))
    ):
        raise ValueError("Invalid physical feedback prefix length or grouped delay")
    return GroundBank(*(jnp.asarray(x) for x in bank))


def quaternion_tangent(reference, current):
    """MuJoCo's body-frame shortest rotation vector from two wxyz free-joint quaternions."""
    a, b = reference[..., 1:], current[..., 1:]
    w = jnp.sum(reference * current, axis=-1)
    v = reference[..., :1] * b - current[..., :1] * a - jnp.cross(a, b)
    v = jnp.where(w[..., None] < 0.0, -v, v)
    length = jnp.linalg.norm(v, axis=-1)
    angle = 2.0 * jnp.arctan2(length, jnp.abs(w))
    scale = jnp.where(length > 1e-7, angle / jnp.maximum(length, 1e-7), 2.0)
    return v * scale[..., None]


def align_ground_heading(reference, actual):
    """Express actual tilt in the reference heading and return the planar velocity rotation."""
    w = jnp.sum(reference * actual, axis=-1)
    z = (reference[..., 0] * actual[..., 3] - actual[..., 0] * reference[..., 3]
         - jnp.cross(actual[..., 1:], reference[..., 1:])[..., 2])
    half = jnp.arctan2(jnp.where(w < 0.0, -z, z), jnp.abs(w))
    c, s = jnp.cos(half), jnp.sin(half)
    aw, ax, ay, az = jnp.moveaxis(actual, -1, 0)
    aligned = jnp.stack((c * aw + s * az, c * ax + s * ay, c * ay - s * ax, c * az - s * aw), axis=-1)
    return aligned, c * c - s * s, 2.0 * s * c


def stance_target(q, v, equilibrium_quat, nominal_torque):
    """Gravity compensation and body-frame ankle feedback, aligned to the observed yaw."""
    rotation = quat_to_mat(q[..., 3:7])
    rest = quat_to_mat(equilibrium_quat)
    reference = yaw_mat(rotation) @ tr3(yaw_mat(rest)) @ rest
    angle = so3_log(tr3(reference) @ rotation)
    torque = jnp.broadcast_to(nominal_torque, q.shape[:-1] + (23,))
    for joint, axis in ((4, 1), (10, 1), (5, 0), (11, 0)):
        torque = torque.at[..., joint].add(100.0 * angle[..., axis] + 20.0 * v[..., 3 + axis])
    return torque / jnp.asarray(zenbot.KP * zenbot.ACTION_SCALE)


def aligned_reference_actions(bank, clip, phase, actual_delay_groups):
    """Advance or retard each joint request by its physical delivery difference, in 20 ms control frames."""
    blend = (actual_delay_groups[:, zenbot.GROUP] - bank.delay_groups[clip][:, zenbot.GROUP]) / 4.0
    # Zero-delay motors replay recorded requests up to 1.75 frames earlier in the clip.
    # Use the preceding interval only beyond the original ±1-frame domain.
    offset = jnp.where(blend < -1., -1, 0)
    blend = blend - offset
    joint = jnp.arange(len(zenbot.GROUP))[None, :]

    def read(t):
        t = jnp.minimum(t, bank.lengths[clip, None] - 1)
        return jnp.where(t >= 0, bank.raw[clip[:, None], jnp.maximum(0, t), joint], 0.0)

    def at(t):
        base = read(t[:, None] + offset)
        neighbor = read(t[:, None] + offset + jnp.sign(blend).astype(jnp.int32))
        return jnp.where((t >= 0)[:, None], base + jnp.abs(blend) * (neighbor - base), 0.0)

    return at(phase), at(phase - 1), at(phase - 2)


def labels(
    bank,
    q,
    v,
    previous,
    older,
    clip,
    start,
    elapsed,
    command,
    gait_ready,
    floor_load_bw,
    actual_delay_groups=None,
    *, physical_reset_phase=False,
):
    """Return MJCF-order raw target, class (0 skip, 1 recover, 2 stand), and matched phase.

    The episode starts at world x=y=0, including random reference-phase resets. Match a fixed
    ±8-frame neighborhood of its recording clock. Actuator memory remains the two actual requests,
    in their original order, independent of which phase the pose matcher selects.
    """
    # Each gain models its trajectory. Group-specific phase alignment applies only to its feedforward
    # requests and reference motor history. Sim executes the real worker delay without modification.
    c = jnp.clip(clip, 0, len(bank.lengths) - 1)
    length = bank.lengths[c]
    opening_hold = getattr(bank, "opening_hold", None)
    if opening_hold is not None and not physical_reset_phase:
        start = jnp.where(start > 0, start + opening_hold[c], start)
    clock = start + elapsed
    near = clock[:, None] + jnp.array([0, -2, 2, -4, 4, -8, 8])
    # A robot that is still near its opening may need another request to start a push.
    opening = start[:, None] + jnp.array([0, 4, 8])
    phase = jnp.clip(jnp.concatenate((near, opening), axis=1), start[:, None], length[:, None] - 1)
    poses = bank.q[c[:, None], phase]
    aligned, heading_cos, heading_sin = align_ground_heading(poses[..., 3:7], q[:, None, 3:7])
    rotation = quaternion_tangent(poses[..., 3:7], aligned)
    z_error = q[:, None, 2] - poses[..., 2]
    joints = q[:, None, 7:] - poses[..., 7:]
    distance = (
        (z_error / 0.18) ** 2 + jnp.sum(rotation**2, axis=-1) / 0.72**2 + jnp.mean(joints**2, axis=-1) / 0.36**2
    )
    distance += 0.04 * jnp.abs(phase - clock[:, None]).clip(0, 10)
    which = jnp.argmin(distance, axis=1)
    row = jnp.arange(len(c))
    k = phase[row, which]
    pose = poses[row, which]
    xyz = q[:, :3] - pose[:, :3]
    xyz = xyz.at[:, :2].set(0.0)
    cy, sy = heading_cos[row, which], heading_sin[row, which]
    local_v = jnp.asarray(v).at[:, 0].set(cy * v[:, 0] + sy * v[:, 1]).at[:, 1].set(cy * v[:, 1] - sy * v[:, 0])
    angle = rotation[row, which]
    dq = joints[row, which]
    if actual_delay_groups is None:
        current = bank.raw[c, k]
        last = jnp.where((k >= 1)[:, None], bank.raw[c, jnp.maximum(k - 1, 0)], 0.0)
        old = jnp.where((k >= 2)[:, None], bank.raw[c, jnp.maximum(k - 2, 0)], 0.0)
    else:
        current, last, old = aligned_reference_actions(bank, c, k, actual_delay_groups)
    dx = jnp.concatenate(
        (
            xyz,
            angle,
            dq,
            local_v - bank.v[c, k],
            zenbot.ACTION_SCALE * (previous - last),
            zenbot.ACTION_SCALE * (older - old),
        ),
        axis=-1,
    )
    # The terminal finite-horizon gain decays by construction. Keep the measured
    # settled feedback when the same zero command continues past the recorded trajectory.
    gain_phase = jnp.where(length - bank.settled[c] > 51, jnp.minimum(k, length - 51), k)
    correction = jnp.einsum("bij,bj->bi", bank.gain[c, gain_phase], dx)
    recover_target = current + jnp.clip(correction, -0.65, 0.65) / zenbot.ACTION_SCALE

    speed = jnp.linalg.norm(v[:, :2], axis=1)
    angular = jnp.linalg.norm(v[:, 3:5], axis=1)
    loaded_upright = (
        (elapsed > 0)
        & (floor_load_bw[:, :2].sum(axis=1) >= 0.5)
        & (floor_load_bw[:, 2] < 0.1)
        & (q[:, 2] >= 0.60)
        & (q[:, 2] <= 0.78)
        & (1.0 - 2.0 * jnp.sum(q[:, 4:6] ** 2, axis=1) >= np.cos(np.deg2rad(13.0)))
    )
    quiet = loaded_upright & (jnp.min(floor_load_bw[:, :2], axis=1) >= 0.06) & (speed < 0.12) & (angular < 0.5)
    retain = loaded_upright & (speed < 0.22) & (angular < 0.8)
    supported = quiet | (gait_ready & retain)
    stationary = jnp.all(jnp.abs(command) < 1e-6, axis=-1)
    # Each floor plan settles over its own foot placements. Use the nominal standing
    # equilibrium only in worlds reset to the corresponding free-standing geometry.
    near_stance_pose = jnp.mean(jnp.abs(q[:, 7:] - zenbot.DEFAULT_Q), axis=1) < 0.08
    stand = (clip < 0) & supported & near_stance_pose & stationary
    close = (
        (jnp.linalg.norm(angle, axis=1) <= 0.72)
        & (jnp.abs(xyz[:, 2]) <= 0.18)
        & (jnp.mean(jnp.abs(dq), axis=1) <= 0.36)
    )
    recover = (
        (clip >= 0)
        & (clip < len(bank.lengths))
        & (start >= 0)
        & (start < length)
        & ((clock < length + 8) | (which >= near.shape[1]) | stationary)
        & close
        & (~gait_ready | stationary)
    )
    kind = jnp.where(stand, 2, jnp.where(recover, 1, 0))
    static = stance_target(q, v, bank.stance_quat, bank.stance_torque)
    target = jnp.where(stand[:, None], static, recover_target)
    valid = (
        jnp.all(jnp.isfinite(target), axis=1)
        & jnp.all(jnp.isfinite(q), axis=1)
        & jnp.all(jnp.isfinite(v), axis=1)
        & jnp.all(jnp.isfinite(floor_load_bw), axis=1)
    )
    kind = jnp.where(valid, kind, 0)
    return jax.lax.stop_gradient(jnp.where(kind[:, None] > 0, target, 0.0)), kind, k


def measured_floor_loads(sim, state):
    """Current MJX-Warp left foot, right foot and all other floor loads in body weights."""
    contacts = state.data._impl
    pairs = contacts.contact__geom
    world = jnp.clip(contacts.contact__worldid, 0, sim.N - 1)
    rows = contacts.contact__efc_address.reshape(len(world), -1)
    force = (contacts.efc__force[world[:, None], jnp.maximum(rows, 0)] * (rows >= 0)).sum(-1)
    floor = (jnp.arange(len(world)) < contacts.nacon[0]) & jnp.any(pairs == sim.floor_id, axis=-1)
    side = jnp.max(jnp.asarray(sim.foot_side)[pairs], axis=-1)
    body = jnp.where(side < 0, 2, side)
    loads = jnp.zeros((sim.N, 3)).at[world, body].add(jnp.where(floor, jnp.maximum(force, 0.0), 0.0))
    return loads / (state.world.body_mass.sum(-1, keepdims=True) * abs(sim.m.opt.gravity[2]))


def rollout_control(kind, command, iteration):
    """Fade physical motor guidance over the first 2,500 updates, then sample the actor alone."""
    stationary = jnp.all(jnp.abs(command) < 1e-6, axis=-1)
    ground = ((kind == 1) | (kind == 2)) & stationary
    gait = (kind == 3) & ~stationary
    guide = jnp.clip((2500 - iteration) / 2499., 0., 1.) * (.98 * ground + .5 * gait)
    # Quiet support is sensitive to white motor noise. Locomotion retains the full exploration scale.
    noise = jnp.where(stationary, .24, 1.)
    return guide, noise


def behavior_gaussian(actor_mean, base_std, label, guide_fraction, noise_scale):
    """The sampled distribution used again when evaluating PPO's likelihood ratio."""
    mean = actor_mean + guide_fraction[:, None] * (label - actor_mean)
    return mean, base_std * noise_scale[:, None]
