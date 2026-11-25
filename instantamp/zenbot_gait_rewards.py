"""Motion Train Zenbot reward terms for joint floor recovery and locomotion.

Weights are per second. The simulator applies the policy timestep once, along
with the original task rewards. Foot slip, contact impact and joint costs also apply during recovery.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import env, zenbot_models
from .rot import mv3, tr3, yaw_mat

WEIGHTS = {
    "foot_slip": -.25,
    "sound_suppression": -.001,
    "feet_effective_landing": .5,
    "feet_crossing": -1.,
    "feet_yaw_l2": -2.,
    "feet_swing_clearance": -.15,
    "undesired_contacts": -10.,
    "joint_vel_l2": -2e-4,
    "joint_acc_l2": -2.5e-7,
    "joint_torques_l2": -1e-5,
    "joint_energy": -1e-4,
    "joint_pos_limits": -1.,
    "ankle_roll_l2": -.10,
}
_ANKLE_JOINTS = ("left_ankle_roll", "right_ankle_roll")
_ANKLE_INDICES = {
    len(model.dof): np.array([model.dof.index(name) for name in _ANKLE_JOINTS])
    for model in zenbot_models.MODELS
}
ANKLE_ROLL = _ANKLE_INDICES[len(zenbot_models.ZENBOT23.dof)]

SOURCE_WEIGHTS = {
    "velocity_progress_tanh": .5, "alive": .15, "lin_vel_z_l2": -.2, "ang_vel_xy_l2": -.1,
    "feet_slide": -.1, "feet_double_flight": -1.,
}
POSTURE_WEIGHTS = {
    "flat_orientation_l2": -1.2,
    "slow_speed_base_height_l2": -1.,
}


def base_velocity(state, root_com):
    """Pelvis COM planar velocity and yaw rate, expressed in the pelvis frame."""
    _, rotation, linear, angular = state[:4]
    inverse = tr3(rotation)
    gyro = mv3(inverse, angular)
    vel = mv3(inverse, linear) + jnp.cross(gyro, root_com)
    return jnp.concatenate((vel[..., :2], gyro[..., 2:3]), axis=-1)


def posture_terms(state, command, gait, actual):
    pos, rotation = state[:2]
    speed_sq = jnp.maximum(jnp.sum(actual**2, -1), jnp.sum(command**2, -1))
    # The gate saturates below .1; keep its gradient finite at rest.
    speed = jnp.sqrt(jnp.maximum(speed_sq, .1**2))
    slow = jnp.clip((1.50 - speed) / 1.40, 0., 1.)
    crouch = jax.nn.relu(.65 - pos[..., 2]) / .10
    return {
        "flat_orientation_l2": gait * jnp.sum(rotation[..., 2, :2]**2, -1),
        "slow_speed_base_height_l2": gait * slow * crouch**2,
    }


class FootGeometry(NamedTuple):
    positions: jax.Array
    rotations: jax.Array
    bottom: jax.Array  # sole boxes first, then the non-ankle collision bodies
    predicted_slip: jax.Array
    com_velocity_z: jax.Array
    com_velocity_xy: jax.Array | None = None


def slip_cost(velocity_xy):
    """Squared tangential speed in units of 0.20 m/s, with a linear tail for impacts."""
    squared = jnp.sum(jnp.square(velocity_xy / .20), axis=-1)
    return jnp.where(squared <= 1., squared, 2. * jnp.sqrt(jnp.maximum(squared, 1.)) - 1.)


def predicted_foot_slip(pos, rotation, velocity, angular, local_corners):
    """Approximate loaded material points on each sole box using their height above the plane."""
    offsets = mv3(rotation[..., None, :, :], local_corners)
    heights = pos[..., 2:3] + offsets[..., 2]
    low = heights.min(axis=-1)
    contact = jnp.clip((env.FOOT_CONTACT_H - low) / env.FOOT_CONTACT_SOFT, 0., 1.)
    shares = jax.nn.softmax(-heights / env.FOOT_CONTACT_SOFT, axis=-1)
    # No force signal in imagination. Differentiate the slip speed with the support fixed.
    weights = jax.lax.stop_gradient(contact[..., None] * shares)
    point_velocity = velocity[..., None, :] + jnp.cross(angular[..., None, :], offsets)
    return (weights * slip_cost(point_velocity[..., :2])).sum(axis=(-2, -1))


class StepHistory(NamedTuple):
    """Contact timers and sole heights for landing and clearance rewards."""

    air: jax.Array
    stance: jax.Array
    seen: jax.Array
    stance_z: jax.Array
    peak_z: jax.Array
    supported_swing: jax.Array

    @classmethod
    def empty(cls, worlds):
        ticks = jnp.zeros((worlds, 2), jnp.int32)
        z = jnp.zeros_like(ticks, jnp.float32)
        return cls(ticks, ticks, ticks > 0, z, z, ticks > 0)

    def reset(self, done):
        return jax.tree.map(lambda x: jnp.where(done[..., None], jnp.zeros_like(x), x), self)

    def advance(self, contact, sole_z, command):
        """Score a foot on its first supported ground contact after a swing."""
        air = jnp.where(contact, 0, self.air + 1)
        stance = jnp.where(contact, self.stance + 1, 0)
        supported = stance[..., ::-1] >= round(.20 / env.DT)
        landed = contact & (self.stance == 0) & self.seen & supported
        swing = ~contact & self.seen & supported
        peak = jnp.where(swing, jnp.where(self.supported_swing, jnp.maximum(self.peak_z, sole_z), sole_z), self.peak_z)
        had_swing = self.supported_swing | swing
        speed = jnp.linalg.norm(command[..., :2], axis=-1)
        turning = (speed <= .10) & (jnp.abs(command[..., 2]) > .10)
        sideways = (speed > .10) & (jnp.abs(command[..., 1]) > jnp.abs(command[..., 0]))
        target = jnp.where(turning, .035, jnp.where(sideways, .025, .040))[..., None]
        gate = ((speed > .10) | turning) * jnp.clip((1.5 - speed) / .5, 0., 1.)
        credit = jnp.clip((self.air * env.DT - .20) / .30, 0., 1.)
        landing = gate * jnp.sum(jnp.where(landed, credit, 0.), -1)
        deficit = jnp.clip(1. - (peak - jnp.maximum(self.stance_z, sole_z)) / target, 0., 1.)
        valid = landed & had_swing & (self.air >= round(.20 / env.DT))
        clearance = gate * jnp.sum(jnp.where(valid, deficit**2, 0.), -1)
        history = self._replace(air=air, stance=stance, seen=self.seen | contact,
                                stance_z=jnp.where(contact, sole_z, self.stance_z),
                                peak_z=jnp.where(contact, 0., peak), supported_swing=had_swing & ~contact)
        return history, landing, clearance


def reward_terms(state, feet, command, torque, acceleration, q_soft, foot_contact, undesired, foot_slip,
                 foot_acc_z, history, gait, *, min_ankle_spacing=.10):
    """Use measured motor/contact signals in physics and predicted signals in imagination."""
    _, rotation, _, _, q, qd = state
    history, landing, clearance = history.advance(foot_contact, feet.bottom[..., :2], command)
    history = history.reset(gait <= 0.)
    separation = feet.positions[..., 0, :] - feet.positions[..., 1, :]
    lateral = jnp.sum(separation * yaw_mat(rotation)[..., :, 1], -1)
    # First column of each foot rotation in the full pelvis frame.
    axis = jnp.einsum("...ij,...fj->...fi", tr3(rotation), feet.rotations[..., :, 0])
    horizontal = jnp.sum(axis[..., :2]**2, axis=-1) > 1e-10
    yaw = jnp.arctan2(jnp.where(horizontal, axis[..., 1], 0.), jnp.where(horizontal, axis[..., 0], 1.))
    straight = jnp.clip(1. - jnp.abs(command[..., 2]) / .05, 0., 1.)
    model = zenbot_models.for_joint_count(q.shape[-1])
    ankle = _ANKLE_INDICES[len(model.dof)]
    ankle_error = jnp.abs(q[..., ankle] - model.default_q[ankle])
    ankle_excess = jax.nn.relu(ankle_error - np.deg2rad(5.)) / np.deg2rad(10.)
    speed = jnp.linalg.norm(command[..., :2], axis=-1)
    walking = (speed > .10) & (jnp.abs(command[..., 0]) >= jnp.abs(command[..., 1]))
    ankle_gate = walking * jnp.clip((1.5 - speed) / .5, 0., 1.)
    terms = {
        "foot_slip": foot_slip,
        "sound_suppression": (speed < 1.5) * jnp.where(foot_contact, jnp.minimum(foot_acc_z**2, 50.), 0.).sum(-1),
        "feet_effective_landing": gait * landing,
        "feet_swing_clearance": gait * clearance,
        "feet_crossing": gait * jax.nn.relu(min_ankle_spacing - lateral),
        "feet_yaw_l2": gait * straight * jnp.sum(jax.nn.relu(jnp.abs(yaw) - np.deg2rad(2.))**2, -1),
        "undesired_contacts": gait * undesired,
        "ankle_roll_l2": gait * ankle_gate * jnp.sum(ankle_excess**2, -1),
        "joint_vel_l2": jnp.sum(qd**2, -1),
        "joint_acc_l2": jnp.sum(acceleration**2, -1),
        "joint_torques_l2": jnp.sum(torque**2, -1),
        "joint_energy": jnp.sum(jnp.abs(qd * torque), -1),
        "joint_pos_limits": jnp.sum(jax.nn.relu(q_soft[:, 0] - q) + jax.nn.relu(q - q_soft[:, 1]), -1),
    }
    if feet.com_velocity_xy is not None:
        # The source penalizes the loaded foot body's world speed. Keep the original slip cost on the floor.
        speed_xy = jnp.sqrt(jnp.maximum(jnp.sum(feet.com_velocity_xy**2, -1), 1e-12))
        terms["feet_slide"] = gait * jnp.sum(foot_contact * speed_xy, -1)
        terms["foot_slip"] = (1. - gait) * foot_slip
        command_speed = jnp.linalg.norm(command, axis=-1)
        terms["feet_double_flight"] = gait * ~foot_contact.any(-1) * jnp.clip((1.5 - command_speed) / .5, 0., 1.)
    return history, terms


def measured_foot_acc_z(data, foot_bodies, post_velocity_z):
    """Difference foot COM velocities across the final MuJoCo physics substep."""
    # mj_step integrates qpos/qvel after computing cvel and xipos. These still
    # describe the start of the last substep, about the root subtree COM.
    twist = data.cvel[:, foot_bodies]
    offset = data.xipos[:, foot_bodies] - data.subtree_com[:, 1:2]
    pre_velocity_z = twist[..., 5] + jnp.cross(twist[..., :3], offset)[..., 2]
    return (post_velocity_z - pre_velocity_z) / env.SIM_DT


def measured_contacts(data, geom_body, foot_bodies, other_bodies, foot_dofs, body_weight):
    """Ground loads and force-weighted foot slip at the actual MuJoCo contact points."""
    impl = data._impl
    worlds, bodies = data.subtree_com.shape[:2]
    world = jnp.clip(impl.contact__worldid, 0, worlds - 1)
    rows = impl.contact__efc_address
    valid = (jnp.arange(len(rows)) < impl.nacon[0]) & (rows[:, 0] >= 0)
    in_range = (rows >= 0) & (rows < impl.efc__force.shape[1])
    lambdas = jnp.where(in_range, impl.efc__force[world[:, None], jnp.maximum(0, rows)], 0.)
    normal = lambdas.sum(-1)
    body0, body1 = geom_body[impl.contact__geom].T
    ground = valid & ((body0 == 0) | (body1 == 0))
    # All normal forces from the single flat plane point along the same axis.
    body = jnp.maximum(body0, body1)
    loads = jnp.zeros((worlds, bodies), normal.dtype).at[world, body].add(jnp.where(ground, normal, 0.))
    foot_load = loads[:, foot_bodies]

    # cdof is the spatial Jacobian about the root subtree COM. Use post-solver qvel
    # so a sticking foot is credited immediately, including on the landing substep.
    twist = jnp.einsum("fd,ndk,nd->nfk", foot_dofs, data.cdof, data.qvel)
    side = (body == foot_bodies[1]).astype(jnp.int32)
    on_foot = ground & ((body == foot_bodies[0]) | (body == foot_bodies[1])) & (normal > 0.)
    point = jnp.where(on_foot[:, None], impl.contact__pos - data.subtree_com[world, 1], 0.)
    foot = twist[world, side]
    velocity_xy = (foot[:, 3:] + jnp.cross(foot[:, :3], point))[:, :2]
    weighted = jnp.zeros_like(foot_load).at[world, side].add(jnp.where(on_foot, normal * slip_cost(velocity_xy), 0.))
    support = jnp.clip((foot_load / body_weight[:, None] - .05) / .10, 0., 1.)
    slip = (support * weighted / jnp.maximum(foot_load, 1.)).sum(axis=-1)
    return foot_load > 1., loads[:, other_bodies] > 1., slip
