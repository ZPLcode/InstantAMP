"""Startup dynamics for Motion Train's Zenbot AMP tasks.

Ranges and paired Coulomb friction follow Motion Train ad923704. Its active
Newton material backend samples one continuous sliding coefficient per shape;
the PhysX-only dynamic-friction range and 64 material buckets do not apply.
With equal geom priorities and ground friction 1, both MuJoCo contact paths use
max(robot_mu, 1). No restitution is invented: the Newton-to-MJWarp contact
converter does not consume shape_material_restitution.

Construct once from a compiled model, then sample once per training startup.
Resetting episodes must reuse the sampled values. The model must have equal
collision-geom priorities (including the floor) to match Motion Train.
"""

import re

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .zenbot_models import ZENBOT26, for_joint_count


MODEL_FIELDS = (
    "body_ipos", "body_mass", "body_inertia", "body_subtreemass",
    "dof_armature", "dof_frictionloss", "geom_friction",
    "body_invweight0", "dof_invweight0", "dof_M0", "actuator_acc0",
)
PAIR_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
NOMINAL_PAIRED_FRICTION = np.asarray(
    ((.405, .390), (.425, .325), (.250, .345), (.4125, .4125), (.55, .55), (.35, .35)),
    dtype=np.float32,
)


class DomainRandomization:
    """JIT-compatible sampling with recomputed reference inertial constants.

``mujoco.mjx`` does not expose a JAX ``set_const`` operation. For this fixed
articulation, reference-pose body Jacobians are computed once by MuJoCo. New
COMs translate their linear Jacobians, and J.T I J gives the exact new mass
matrix at qpos0. This refreshes the same inverse weights used by mj_setConst,
without host callbacks or a different startup random seed.
    """

    def __init__(self, model, profile=None):
        articulation = for_joint_count(model.nu)
        if profile is not None and getattr(profile, "MODEL", profile) is not articulation:
            raise ValueError("Zenbot actuator count does not match its dynamics profile")
        if model.nv != model.nu + 6 or model.ntendon or model.neq:
            raise ValueError("Expected a free-root Zenbot without tendons/equalities")
        self.nv = model.nv
        self.joint_dofs = np.asarray([
            model.jnt_dofadr[model.joint("robot_" + name + "_joint").id]
            for name in articulation.dof
        ])
        self.pair_dofs = np.asarray([
            [self.joint_dofs[articulation.dof.index(side + "_" + name)] for side in ("left", "right")]
            for name in PAIR_NAMES
        ])
        names = [model.body(i).name.removeprefix("robot_") for i in range(model.nbody)]
        self.pelvis = names.index("pelvis")
        self.com_bodies = np.asarray([names.index("torso_Link"), self.pelvis])
        self.limb_bodies = np.asarray([
            i for i, name in enumerate(names)
            if re.fullmatch(r"left_.*_Link|right_.*_[Ll]ink|.*_rubber_hand", name)
        ])
        self.robot_geoms = np.flatnonzero(
            (model.geom_bodyid != 0) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        # The caller creates mjx.Model after this constructor, so static contact
        # priorities and nominal friction agree with the sampled model fields.
        model.dof_frictionloss[self.pair_dofs] = NOMINAL_PAIRED_FRICTION
        if articulation is ZENBOT26:
            for joint, friction in (("waist_roll", .350), ("waist_pitch", .550)):
                model.dof_frictionloss[self.joint_dofs[articulation.dof.index(joint)]] = friction
        model.geom_friction[:, 0] = 1.
        model.geom_priority[:] = 0
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        self.nominal = {name: jnp.asarray(getattr(model, name), dtype=jnp.float32) for name in MODEL_FIELDS}
        self.nominal["kp"] = jnp.asarray(articulation.kp)
        self.nominal["kd"] = jnp.asarray(articulation.kd)

        descendants = np.zeros((model.nbody, model.nbody), dtype=np.float32)
        for child in range(model.nbody):
            body = child
            while True:
                descendants[body, child] = 1
                if not body:
                    break
                body = model.body_parentid[body]
        self.descendants = jnp.asarray(descendants)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        jacv = np.zeros((model.nbody, 3, model.nv))
        jacw = np.zeros_like(jacv)
        for body in range(1, model.nbody):
            mujoco.mj_jacBodyCom(model, data, jacv[body], jacw[body], body)
        self.jacv, self.jacw = jnp.asarray(jacv, jnp.float32), jnp.asarray(jacw, jnp.float32)
        self.body_rotation = jnp.asarray(data.xmat.reshape(-1, 3, 3), jnp.float32)
        principal_rotation = data.ximat.reshape(-1, 3, 3)
        self.principal_jacw = jnp.asarray(np.einsum("bik,bij->bkj", principal_rotation, jacw), jnp.float32)
        self.movable = jnp.asarray(model.body_weldid != 0)
        self.actuator_moment = np.zeros((model.nu, model.nv), dtype=np.float32)
        for actuator in range(model.nu):
            if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("Expected joint torque motors")
            dof = model.jnt_dofadr[model.actuator_trnid[actuator, 0]]
            self.actuator_moment[actuator, dof] = model.actuator_gear[actuator, 0]
        self.actuator_moment = jnp.asarray(self.actuator_moment)

    def sample(self, key, num_envs, enabled=True):
        """Return per-environment model fields plus joint-ordered ``kp``/``kd``.

        Disabling randomization retains the Motion Train nominal joint friction.
        ``num_envs`` and ``enabled`` are static arguments when directly jitting.
        """
        values = {name: jnp.broadcast_to(value, (num_envs,) + value.shape) for name, value in self.nominal.items()}
        if not enabled:
            return values
        keys = jax.random.split(key, 9)
        uniform = lambda k, shape, lo, hi: jax.random.uniform(k, shape, minval=lo, maxval=hi)
        joint_shape = values["kp"].shape
        values["kp"] *= uniform(keys[0], joint_shape, .8, 1.2)
        values["kd"] *= uniform(keys[1], joint_shape, .8, 1.2)
        values["dof_armature"] = values["dof_armature"].at[:, self.joint_dofs].multiply(
            uniform(keys[2], joint_shape, .8, 1.2)
        )
        pair_mean = jnp.asarray(NOMINAL_PAIRED_FRICTION.mean(axis=1)) * uniform(keys[3], (num_envs, 6), .5, 1.5)
        half_mismatch = .5 * uniform(keys[4], (num_envs, 6), -.1, .1)
        paired = pair_mean[..., None] * jnp.stack((1 + half_mismatch, 1 - half_mismatch), axis=-1)
        values["dof_frictionloss"] = values["dof_frictionloss"].at[:, self.pair_dofs].set(paired)
        values["body_mass"] = values["body_mass"].at[:, self.pelvis].add(uniform(keys[5], (num_envs,), -3., 3.))
        values["body_mass"] = values["body_mass"].at[:, self.limb_bodies].multiply(
            uniform(keys[6], (num_envs, len(self.limb_bodies)), .8, 1.2)
        )
        # Isaac Lab scales all principal inertias by new_mass/default_mass.
        ratio = values["body_mass"] / jnp.where(self.nominal["body_mass"] > 0, self.nominal["body_mass"], 1.)
        values["body_inertia"] *= ratio[..., None]
        # The reference event shares one xyz offset across torso and pelvis.
        values["body_ipos"] = values["body_ipos"].at[:, self.com_bodies].add(
            uniform(keys[7], (num_envs, 1, 3), -.03, .03)
        )
        values["geom_friction"] = values["geom_friction"].at[:, self.robot_geoms, 0].set(
            uniform(keys[8], (num_envs, len(self.robot_geoms)), .3, 1.6)
        )
        return self.refresh_inertial_constants(values)

    def refresh_inertial_constants(self, values):
        """Update qpos0 constants after mass, COM, inertia and armature changes."""
        values = dict(values)
        delta = jnp.einsum("bij,nbj->nbi", self.body_rotation, values["body_ipos"] - self.nominal["body_ipos"])
        jacv = self.jacv + jnp.cross(self.jacw.transpose(0, 2, 1)[None], delta[:, :, None]).transpose(0, 1, 3, 2)
        matrix = jnp.einsum("nbki,nb,nbkj->nij", jacv, values["body_mass"], jacv)
        matrix += jnp.einsum("bki,nbk,bkj->nij", self.principal_jacw, values["body_inertia"], self.principal_jacw)
        matrix += jnp.eye(self.nv) * values["dof_armature"][:, None, :]
        inverse = jnp.linalg.inv(matrix)
        diagonal = jnp.diagonal(inverse, axis1=-2, axis2=-1)
        # MuJoCo averages the free joint's translation and rotation blocks.
        invweight = diagonal.at[:, :3].set(diagonal[:, :3].mean(axis=-1, keepdims=True))
        invweight = invweight.at[:, 3:6].set(diagonal[:, 3:6].mean(axis=-1, keepdims=True))
        linear = jnp.einsum("nbki,nij,nbkj->nbk", jacv, inverse, jacv).mean(axis=-1)
        angular = jnp.einsum("bki,nij,bkj->nbk", self.jacw, inverse, self.jacw).mean(axis=-1)
        values["body_invweight0"] = jnp.stack((linear, angular), axis=-1) * self.movable[None, :, None]
        values["dof_invweight0"] = invweight
        values["dof_M0"] = jnp.diagonal(matrix, axis1=-2, axis2=-1)
        values["body_subtreemass"] = values["body_mass"] @ self.descendants.T
        acceleration = jnp.einsum("nij,aj->nai", inverse, self.actuator_moment)
        values["actuator_acc0"] = jnp.linalg.norm(acceleration, axis=-1)
        return values
