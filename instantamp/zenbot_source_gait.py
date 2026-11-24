"""Motion Train's locomotion reset, command curriculum and tracking signal for a mixed Zenbot policy."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import env, zenbot
from .rot import mv3, tr3
from .zenbot_motion_train_obs import LAB_ORDER

GAIT_STEPS = round(20. / env.DT)
AMP_STEPS = 10
AMP_DIM = 3 + 2 * len(LAB_ORDER) + 3 * len(zenbot.KEY_BODIES)


def amp_frame(rotation, angular, joints, rates, offsets):
    """Source discriminator order. Angular velocity and landmark offsets enter in world axes."""
    inverse = tr3(rotation)
    points = mv3(inverse[..., None, :, :], offsets)
    return jnp.concatenate((mv3(inverse, angular), joints[..., LAB_ORDER], rates[..., LAB_ORDER],
                            points.reshape(*joints.shape[:-1], -1)), axis=-1)


def stand_reset(key, n):
    """Source standing pose with its real randomized root velocity and scaled joint coordinates."""
    kp, kh, kv, kq = jax.random.split(key, 4)
    xy = jax.random.uniform(kp, (n, 2), minval=-.5, maxval=.5)
    yaw = jax.random.uniform(kh, (n,), minval=-jnp.pi, maxval=jnp.pi)
    zeros = jnp.zeros(n)
    quat = jnp.stack((jnp.cos(yaw / 2), zeros, zeros, jnp.sin(yaw / 2)), axis=1)
    joints = zenbot.DEFAULT_Q * jax.random.uniform(kq, (n, len(zenbot.DOF)), minval=.8, maxval=1.2)
    q = jnp.concatenate((xy, jnp.full((n, 1), .69), quat, joints), axis=1)
    velocity = jax.random.uniform(kv, (n, 6), minval=-.2, maxval=.2)
    # The source reset specifies world velocities. MuJoCo stores free-root angular velocity locally.
    wx, wy = velocity[:, 3], velocity[:, 4]
    velocity = velocity.at[:, 3].set(jnp.cos(yaw) * wx + jnp.sin(yaw) * wy)
    velocity = velocity.at[:, 4].set(-jnp.sin(yaw) * wx + jnp.cos(yaw) * wy)
    return q, jnp.pad(velocity, ((0, 0), (0, len(zenbot.DOF))))


def sample_command(key, n, stage, *, stand_share=.05, side_share=0.):
    mode, x, y, heading, turn = jax.random.uniform(key, (5, n))
    stand = mode < stand_share
    translate = mode >= stand_share + .20
    cap = jnp.minimum(3, stage + 1)
    vx = jnp.where(translate, -1. + (cap + 1.) * x, 0.)
    vy = jnp.where(translate, y - .5, 0.)
    wz = jnp.where(~translate & ~stand, 2.2 * (2. * turn - 1.), 0.)
    # Reserve equal probabilities from the translation budget. Pure sidesteps bypass heading control.
    side = translate & (mode < stand_share + .20 + 2 * side_share)
    left = mode < stand_share + .20 + side_share
    vx = jnp.where(side, 0., vx)
    vy = jnp.where(side, jnp.where(left, 1., -1.) * (.15 + .25 * y), vy)
    translate = translate & ~side
    return env.Cmd(jnp.stack((vx, vy, wz), -1), jnp.pi * (2. * heading - 1.), translate, stand, jnp.full(n, 10.))


def command(cmd, rot):
    heading_error = cmd.heading - env.heading(rot)
    rate = jnp.clip(.5 * jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error)), -1.5, 1.5)
    velocity = cmd.vel.at[:, 2].set(jnp.where(cmd.is_heading, rate, cmd.vel[:, 2]))
    return jnp.where(cmd.is_standing[:, None], 0., velocity)


def velocity_progress(cmd, actual):
    speed = jnp.linalg.norm(cmd[..., :2], axis=-1)
    aligned = (cmd[..., :2] * actual[..., :2]).sum(-1) / jnp.maximum(speed, 1e-6)
    move = (speed >= .1) * jnp.tanh(aligned / .5)
    turn = (jnp.abs(cmd[..., 2]) >= .1) * jnp.clip(1. - speed / .1, 0., 1.)
    return move + turn * jnp.tanh(jnp.sign(cmd[..., 2]) * actual[..., 2] / .5)


def tracking(state, cmd):
    _, rotation, velocity, angular, _, _ = state
    body = mv3(tr3(rotation), velocity)
    yaw_rate = mv3(tr3(rotation), angular)[:, 2]
    upright = jnp.clip(rotation[:, 2, 2], 0., .7) / .7
    difference = body[:, :2] - cmd[:, :2]
    xy = jnp.exp(-jnp.square(difference).sum(-1) / .25)
    yaw = jnp.exp(-jnp.square(yaw_rate - cmd[:, 2]) / .25)
    actual = jnp.column_stack((body[:, :2], yaw_rate))
    return upright * (2. * xy + 1.5 * yaw + .5 * velocity_progress(cmd, actual)), jnp.linalg.norm(difference, axis=-1)


class Curriculum(NamedTuple):
    stage: jax.Array
    control_ticks: jax.Array
    last_check: jax.Array
    partial: jax.Array
    completed: jax.Array
    last_metrics: jax.Array

    @classmethod
    def empty(cls, n):
        # The collector donates each counter to JAX independently.
        return cls(jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.zeros((n, 6)), jnp.zeros(8), jnp.zeros(4))

    def advance(self, command_x, error, active, terminated, timeout):
        # Only completed gait episodes enter the source's assessment, including failed ones.
        cap = self.stage + 1.
        forward, backward = command_x >= .75 * cap, command_x <= -.75
        error = jnp.nan_to_num(error, nan=4., posinf=4., neginf=4.)
        row = jnp.stack((jnp.ones_like(error), error, forward, forward * error, backward, backward * error), axis=-1)
        partial = self.partial + jnp.where(active[:, None], row, 0.)
        done = terminated | timeout
        finished = jnp.where(done[:, None], partial, 0.).sum(axis=0)
        valid = done & (partial[:, 0] > 0)
        window = self.completed + jnp.concatenate((finished, jnp.stack(((valid & terminated).sum(), valid.sum()))))
        partial = jnp.where(done[:, None], 0., partial)
        ticks = self.control_ticks + 1
        n, total, n_f, e_f, n_b, e_b, falls, episodes = window
        assess = ((ticks - self.last_check >= 1000) & (episodes >= 1024) &
                  (n_f >= 10000) & (n_b >= 10000) & jnp.any(done))
        metrics = jnp.stack((total / jnp.maximum(n, 1), e_f / jnp.maximum(n_f, 1),
                             e_b / jnp.maximum(n_b, 1), falls / jnp.maximum(episodes, 1)))
        promoted = assess & (self.stage < 2) & jnp.all(metrics <= jnp.array([.35, .40, .40, .10]))
        return Curriculum(self.stage + promoted.astype(jnp.int32), ticks, jnp.where(assess, ticks, self.last_check),
                          jnp.where(promoted, 0., partial), jnp.where(assess, 0., window),
                          jnp.where(assess, metrics, self.last_metrics))


def failed(height, up_cos, illegal_contact, diverged):
    return illegal_contact | (height < .20) | (up_cos < .5) | diverged


def illegal_contact_geoms(model):
    """The source Zenbot's illegal contacts include any hip, knee, shoulder, elbow or hand and the trunk."""
    import numpy as np

    names = [model.body(int(b)).name.removeprefix("robot_") for b in model.geom_bodyid]
    return np.array([n in ("pelvis", "torso_Link") or any(tag in n for tag in
                     ("_hip_", "_knee_", "_shoulder_", "_elbow_", "_rubber_hand")) for n in names])
