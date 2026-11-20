"""The simulator of the G1 AMP-Flat task (env.py) on MuJoCo Warp through MJX, on a flat floor: N worlds, one jitted function per policy
step.

The scene is mjlab's G1 (assets/g1_mjlab.xml) as mjlab builds it: full collision (every `*_collision` geom collides, self-collisions
included; the 14 foot capsules condim 3, priority 1, friction 0.6, the rest condim 1), a position actuator per joint (kp, kd, force range
+- effort, joint armature), a ground plane, and the solver options: timestep 5 ms, implicitfast, Newton with 10 iterations and 20
line-search iterations, pyramidal cone. Visual geoms and meshes are removed: training needs none.

A step:
  1. joint target q0 + scale a - encoder bias; 4 substeps, each with the Unitree actuator's torque recomputed from the current q, qd
  2. counter + 1; terminations: a robot that has been up in this episode ends when a termination body touches the floor with more than
     1 N; one that has not been up ends when it is still not up after 5 s; time-out after 20 s; rewards of the final state with the
     command the policy saw
  3. finished envs reset: a share `reset_floor` in a not-up frame of a get-up clip (random heading, noise), the rest in a frame of the
     walking clips; `prob_shove` of the upright resets get a kick of the root and joint velocities
  4. the command term ticks (resampling every 3 to 8 s, heading control); a push overwrites the root's world xy velocity every 5 to 10 s;
     the 4-frame observation history appends the new frame
Start-up randomization: foot friction, encoder bias, torso CoM (env.py).
"""

from __future__ import annotations

import glob
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import warp as wp
from mujoco import mjx

from . import amp, env, resets, zenbot
from .kin import Kinematics
from .rot import axis_angle_to_mat, mm3, mv3, quat_to_mat, tr3, yaw_mat

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "g1_mjlab.xml")
wp.config.log_level = wp.LOG_WARNING
WORLD_FIELDS = ("body_ipos", "geom_friction")  # the model fields that differ between envs
DIVERGED = 1e3  # |qvel| beyond this counts as a physics failure
NCONMAX, NJMAX = 64, 640  # contact and constraint pools per world
WALK_DIR = "data/amp/walk_run"
FLOOR_DIR = "data/recovery/lafan1"


ZENBOT_XML = os.path.join(os.path.dirname(XML), "zenbot_23dof_43kg.xml")


def robot_profile(name):
    if name == "g1":
        return env
    if name == "zenbot23_getup":
        return zenbot
    raise ValueError(f"Unknown robot {name!r}; expected g1 or zenbot23_getup")


def motion_dirs(robot):
    if robot == "zenbot23_getup":
        return ("data/zenbot23_getup/recovery",) * 2
    robot_profile(robot)
    return WALK_DIR, FLOOR_DIR


def build_scene(iterations: int = 10, ls_iterations: int = 20, ccd_iterations: int = 50, robot_name: str = "g1",
                g1_support_friction: bool = False) -> mujoco.MjModel:
    """The compiled scene: the floor and the robot (prefix "robot_"), without visual geoms, meshes or sensors."""
    profile = robot_profile(robot_name)
    if g1_support_friction and robot_name != "g1":
        raise ValueError("g1_support_friction requires the G1 model")
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    o = world.option
    o.timestep, o.iterations, o.ls_iterations, o.ccd_iterations = env.SIM_DT, iterations, ls_iterations, ccd_iterations
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    ground = world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    robot = mujoco.MjSpec.from_file(ZENBOT_XML if profile is zenbot else XML)
    if profile is zenbot:
        # Attach the articulation with the host task's solver and pool sizes.
        robot.option.timestep, robot.option.iterations, robot.option.solver = o.timestep, o.iterations, o.solver
        robot.njmax, robot.nconmax = world.njmax, world.nconmax
        robot.visual.map.znear = world.visual.map.znear
        # motion_train enables self-collision across the URDF's physical links.
        src_ground = robot.geom("ground")
        ground.contype, ground.conaffinity = src_ground.contype, src_ground.conaffinity
        robot.delete(src_ground)
        for body in robot.bodies:
            for geom in body.geoms:
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.contype = geom.conaffinity = 1
                geom.name = body.name + "_collision"
                if body.name in zenbot.CAPSULES:
                    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                    geom.size[:2] = zenbot.CAPSULES[body.name]
        for side, body in zip(("left", "right"), zenbot.FEET):
            foot = robot.geom(body + "_collision")
            foot.priority = 1  # randomized sole friction takes precedence on floor contact
            robot.body(body).add_site(name=side + "_foot", pos=foot.pos)
    for g in [g for g in robot.geoms if g.contype == 0 and g.conaffinity == 0]:
        robot.delete(g)
    for m in list(robot.meshes):
        robot.delete(m)
    for s in list(robot.sensors):
        robot.delete(s)
    if profile is zenbot:
        # The source MJCF supplies the torque motors, joint limits and inertias.
        world.attach(robot, frame=world.worldbody.add_frame(), prefix="robot_")
        return world.compile()
    for g in robot.geoms:  # full collision
        if not g.name.endswith("_collision"):
            continue
        if "_foot" in g.name:
            g.condim, g.priority, g.friction = 3, 1, [0.6, g.friction[1], g.friction[2]]
        else:
            g.condim = 1
            if g1_support_friction:
                # Match mjlab's full-collision G1. With equal priority the ground's
                # three contact dimensions also allow a hand or knee to push tangentially.
                g.priority = ground.priority
    for i, name in enumerate(env.DOF):
        j = robot.joint(f"{name}_joint")
        j.armature = float(env.ARMATURE[i])
        a = robot.add_actuator(name=name, target=f"{name}_joint")
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype, a.gaintype, a.biastype = mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = float(env.KP[i]), -float(env.KP[i]), -float(env.KD[i])
        a.inheritrange, a.ctrllimited = 0.0, False
        a.forcelimited, a.forcerange = True, [-float(env.EFFORT[i]), float(env.EFFORT[i])]
    world.attach(robot, frame=world.worldbody.add_frame(), prefix="robot_")
    return world.compile()


def collider_points(m: mujoco.MjModel, geoms, frames):
    """Reduce geoms to points with a radius, so that a point's height over the floor minus its radius tells how far the geom is above the
    floor: a sphere is its center, a capsule or cylinder its two end centers, a box its four bottom corners (radius 0). Returns the
    points in their body frames, the index of each point's body in `frames`, and the radii."""
    local, frame, radius = [], [], []
    for g in geoms:
        R = np.asarray(quat_to_mat(np.asarray(m.geom_quat[g], np.float32)))
        kind, size, pos = int(m.geom_type[g]), m.geom_size[g], m.geom_pos[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts, r = [pos], size[0]
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):  # a cylinder as the capsule of its axis
            pts, r = [pos + s * size[1] * R[:, 2] for s in (1, -1)], size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            pts, r = [pos + R @ (np.asarray([sx, sy, -1.0]) * size) for sx in (1, -1) for sy in (1, -1)], 0.0
        else:
            raise ValueError(f"collider geom {m.geom(g).name} has unsupported type {kind}")
        local += pts
        frame += [frames.index(int(m.geom_bodyid[g]))] * len(pts)
        radius += [float(r)] * len(pts)
    return np.asarray(local, np.float32), np.asarray(frame), np.asarray(radius, np.float32)


class VelKinematics(Kinematics):
    """Forward kinematics with velocities: world poses, the world linear velocities of the body origins and the angular velocities of
    the requested bodies (every hinge sits at its body's origin, which Kinematics checks)."""

    def forward_vel(self, p, R, v, w, q, qd):
        P, Rs, V, W = {1: p}, {1: R}, {1: v}, {1: w}
        for b in self.order:
            par = self.parent[b]
            P[b] = P[par] + mv3(Rs[par], self.pos[b])
            Rb = mm3(Rs[par], self.rot[b])
            V[b] = V[par] + jnp.cross(W[par], P[b] - P[par])
            if self.col[b] < 0:
                Rs[b], W[b] = Rb, W[par]
            else:
                Rs[b] = mm3(Rb, axis_angle_to_mat(self.axis[b], q[..., self.col[b]]))
                W[b] = W[par] + mv3(Rb, self.axis[b]) * qd[..., self.col[b], None]
        st = lambda d, ax: jnp.stack([d[b] for b in self.out], ax)  # noqa: E731
        return st(P, -2), st(Rs, -3), st(V, -2), st(W, -2)


class World(NamedTuple):
    body_ipos: jax.Array  # (N, nbody, 3)
    geom_friction: jax.Array  # (N, ngeom, 3)
    bias: jax.Array  # (N, J) encoder bias
    model: mjx.Model
    walk: dict  # reset frames of the walking and running clips
    body_mass: jax.Array
    body_subtreemass: jax.Array
    floor: dict  # reset frames of the get-up clips in which the robot is not up


class State(NamedTuple):
    data: mjx.Data
    world: World
    obs: jax.Array  # (N, OBS_DIM) actor observation history (with noise)
    a_last: jax.Array  # (N, J) last raw action (zero after a reset)
    a_ode: jax.Array  # (N, J) last action without exploration noise (the ODE actor integrates from it)
    cmd: env.Cmd
    c_obs: jax.Array  # (N, 3) the command in the current observation
    t: jax.Array  # (N,) episode counter
    was_up: jax.Array  # (N,) the robot has been up in this episode (it started up, or has risen since)
    down: jax.Array  # (N,) the episode started with the robot not up (a get-up start)
    push_t: jax.Array  # (N,) steps since the last push or reset
    push_at: jax.Array  # (N,) push interval in steps
    ep_sums: jax.Array  # (N, len(TERMS)) running episode sums of the weighted terms
    key: jax.Array
    assist: jax.Array  # per-world curriculum force in newtons
    delay_steps: jax.Array  # per actuator group, in physics steps
    reference_clip: jax.Array | None = None  # selected reset clip for phase-aligned Zenbot rewards


class Transition(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    target: jax.Array  # PD targets in joint order, flattened across substeps for delayed motors
    qpos1: jax.Array
    qvel1: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    diverged: jax.Array
    parts: jax.Array  # (N, len(TERMS)) the reward's weighted terms
    contact: jax.Array  # (N, 2) foot-ground contact after the last substep
    clearance: jax.Array  # (N, k) termination colliders over the floor (imagination's termination geometry)
    stats: dict


class View(NamedTuple):
    kin: env.Kin
    height: jax.Array  # (N,) pelvis over the floor
    soles: jax.Array  # (N, k) sole collider points over the floor (radius included)
    clearance: jax.Array  # (N, k) termination collider points over the floor (radius included); negative when penetrating


def unflatten(qpos, qvel):
    """MuJoCo's (qpos, qvel) -> S = (p, R, v, w, q, qd) with world-frame w (MuJoCo's free-joint angular velocity is in the body frame)."""
    R = quat_to_mat(qpos[..., 3:7])
    return qpos[..., :3], R, qvel[..., :3], mv3(R, qvel[..., 3:6]), qpos[..., 7:], qvel[..., 6:]


def load_bank(directory: str, joints: int = env.J, first_only=False):
    """Every frame of the walking clips in `directory` (npz of tools/convert_motions.py, quaternions x, y, z, w): pelvis height, quaternion
    (w, x, y, z), world velocities, joints. Resets draw uniformly over all frames of all clips."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    assert files, f"no locomotion clips in {directory}; prepare the motion banks for the selected robot"
    cols = {k: [] for k in ("z", "quat", "v", "w", "q", "qd")}
    for f in files:
        d = np.load(f)
        assert int(d["fps"]) == 50, f
        assert d["dof_pos"].shape[1] == joints, f
        q = d["root_quat"]
        for k, a in (("z", d["root_pos"][:, 2]), ("quat", np.concatenate([q[:, 3:], q[:, :3]], -1)), ("v", d["root_lin_vel"]), ("w", d["root_ang_vel"]),
                     ("q", d["dof_pos"]), ("qd", d["dof_vel"])):  # fmt: skip
            cols[k].append(a[:1] if first_only else a)
    return {k: jnp.asarray(np.concatenate(v, 0), jnp.float32) for k, v in cols.items()}


class Sim:
    def __init__(self, num_envs: int = 4096, nconmax: int = NCONMAX, njmax: int = NJMAX, walk_dir: str | None = None,
                 floor_dir: str | None = None, random_start: bool = True, seed: int = 0, reset_floor: float = 0.2,
                 prob_shove: float = 0.0, push: bool = True, physics: bool = True, robot: str = "g1", amp_wrists: bool = True, assist_newtons=None, obs_noise=True, randomize_world=True, reference_weight=0., g1_style=False, getup_rise_weight=-0.25, g1_support_friction=False):  # fmt: skip
        """physics=False: everything but MuJoCo Warp's model and data (no GPU): the scene, kinematics, reset banks and `_fresh`, for
        checks on the CPU."""
        self.N, self.random_start, self.seed = num_envs, random_start, seed
        self.reset_floor, self.prob_shove, self.push = reset_floor, prob_shove, push
        self.robot = profile = robot_profile(robot)
        self.J = len(profile.DOF)
        self.recovery = profile is zenbot
        self.history = zenbot.HIST if self.recovery else env.HIST
        self.assist_newtons = zenbot.INITIAL_ASSIST if assist_newtons is None else assist_newtons
        self.randomize_world = randomize_world
        self.noisy_obs = obs_noise
        self.weights = zenbot.WEIGHTS if self.recovery else env.WEIGHTS
        if g1_style:
            self.weights = {**self.weights, "rise": getup_rise_weight}
        if reference_weight:
            if not self.recovery or not np.isfinite(reference_weight) or reference_weight < 0:
                raise ValueError("reference_weight requires a finite positive weight for Zenbot recovery")
            self.weights = {**self.weights, "motion_reference": reference_weight}
        self.terms = tuple(self.weights) if self.recovery else env.TERMS
        self.episode_steps = 500 if self.recovery else env.MAX_EPISODE
        self.world_fields = WORLD_FIELDS + (("body_mass", "body_subtreemass") if self.recovery else ())
        self.frame_dim = 9 + 3 * self.J + (18 if self.recovery else 0)
        self.obs_dim = self.history * self.frame_dim
        self.obs_noise = env.observation_noise(self.J)
        self.amp_joints = np.asarray([i for i in range(self.J) if amp_wrists or i not in profile.WRIST_IDX])
        self.amp_dim = 2 * len(self.amp_joints) + 6 + 3 * len(amp.KEY_BODIES)
        if self.recovery:
            self.amp_dim = 9 + 2 * len(self.amp_joints) + 18
            self.obs_noise = np.repeat([0.2, 0.05, 0.01, 1.5, 0.0, 0.08], [3, 6, self.J, self.J, self.J, 18]).astype(np.float32)
        self.m = m = build_scene(robot_name=robot, g1_support_friction=g1_support_friction)
        assert m.nu == self.J and m.nq == 7 + self.J
        assert tuple(m.joint(m.actuator_trnid[i, 0]).name.removeprefix("robot_").removesuffix("_joint") for i in range(m.nu)) == profile.DOF
        joint_ids = m.actuator_trnid[:, 0]
        self.qa, self.va = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
        if not np.array_equal(self.qa, np.arange(7, m.nq)) or not np.array_equal(self.va, np.arange(6, m.nv)):
            raise ValueError("Motion columns must match the model hinge order")
        rng = m.jnt_range[1:]
        mid, half = rng.mean(-1), (rng[:, 1] - rng[:, 0]) / 2 * env.SOFT_LIMIT
        self.q_soft = np.stack([mid - half, mid + half], -1).astype(np.float32)
        torso = "torso_link" if profile is env else zenbot.TORSO
        keys = amp.KEY_BODIES if profile is env else zenbot.KEY_BODIES
        self.torso = m.body("robot_" + torso).id
        feet = [m.body("robot_" + name).id for name in keys[:2]]
        self.foot_geoms = np.asarray([g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_contype[g] > 0])
        assert len(self.foot_geoms) == (env.N_FOOT_GEOMS if profile is env else 2)
        tb = [m.body(f"robot_{b}").id for b in profile.TERMINATION_BODIES]
        self.term_geoms = np.asarray([g for g in range(m.ngeom) if m.geom_bodyid[g] in tb and m.geom_contype[g] > 0])
        assert len(set(int(m.geom_bodyid[g]) for g in self.term_geoms)) == len(tb), "every termination body carries a collision geom"
        # frames of the forward kinematics: the torso, the two ankle-roll links, then the termination bodies
        frames = [self.torso] + feet + tb
        self.kin = VelKinematics(m, [m.body(b).name for b in frames])
        self.site_off = np.stack([m.site_pos[m.site(f"robot_{s}_foot").id] for s in ("left", "right")]).astype(np.float32)
        self.sole_local, self.sole_frame, self.sole_radius = collider_points(m, self.foot_geoms, frames)
        self.coll_local, self.coll_frame, self.coll_radius = collider_points(m, self.term_geoms, frames)
        self.kin_amp = Kinematics(m, [f"robot_{b}" for b in keys])
        # contact decoding: which foot (0 / 1) a geom belongs to, the termination geoms, the floor, the robot's geoms
        self.foot_side = np.full(m.ngeom, -1, np.int32)
        for g in self.foot_geoms:
            self.foot_side[g] = feet.index(int(m.geom_bodyid[g]))
        self.is_term = np.isin(np.arange(m.ngeom), self.term_geoms)
        supports = [m.body("robot_" + b).id for b in keys]
        self.support_link = np.array([supports.index(b) if b in supports else -1 for b in m.geom_bodyid])
        self.floor_id = m.geom("floor").id
        self.robot_geom = np.asarray([m.body(int(m.geom_bodyid[g])).name.startswith("robot_") for g in range(m.ngeom)])
        walk_default, floor_default = motion_dirs(robot)
        self.walk = load_bank((floor_dir or floor_default) if self.recovery else (walk_dir or walk_default), self.J, first_only=self.recovery)
        self.reference = zenbot.RecoveryReference(floor_dir or floor_default, self.key_positions) if reference_weight else None
        self.floor = resets.load(floor_dir or floor_default, "floor", profile.UP_HEIGHT, env.UP_TILT, self.J,
                                 first_frames=20 if g1_style else None) if reset_floor > 0 and not self.recovery else resets.empty(self.J)
        if not physics:
            return
        self.mx = mjx.put_model(m, impl="warp")
        # the solver budget (10 iterations, 20 line-search iterations) often runs out; MuJoCo Warp would print a warning per step. Those
        # two warnings are off; the overflow warnings of the contact and constraint pools stay on.
        import mujoco_warp as mjw

        quiet = int(self.mx.opt._impl.warn_overflow) & ~int(mjw.OverflowType.ITERATIONS) & ~int(mjw.OverflowType.LS_ITERATIONS)
        self.mx = self.mx.tree_replace({"opt._impl.warn_overflow": quiet})
        self.dx = mjx.make_data(m, impl="warp", naconmax=num_envs * nconmax, njmax=njmax)
        self.axes = jax.tree.map(lambda _: None, self.mx).tree_replace({k: 0 for k in self.world_fields})

    # ------------------------------------------------------------------ geometry shared with imagination
    def view_state(self, S) -> View:
        p, R, v, w, q, qd = S
        P, Rs, V, W = self.kin.forward_vel(p, R, v, w, q, qd)
        foot_v = V[..., 1:3, :] + jnp.cross(W[..., 1:3, :], mv3(Rs[..., 1:3, :, :], self.site_off))

        def clearance(local, frame, radius):
            pts = P[..., frame, :] + mv3(Rs[..., frame, :, :], local)
            return pts[..., 2] - radius

        return View(env.Kin(Rs[..., 0, :, :], V[..., 0, :], W[..., 0, :], foot_v), p[..., 2],
                    clearance(self.sole_local, self.sole_frame, self.sole_radius), clearance(self.coll_local, self.coll_frame, self.coll_radius))  # fmt: skip

    def soft_contact(self, soles):
        """(N, 2) soft foot-ground contact of the imagined states: the lowest sole point of each foot near the floor."""
        n = len(self.sole_frame) // 2
        low = jnp.stack([soles[..., :n].min(-1), soles[..., n:].min(-1)], -1)
        return jax.nn.sigmoid((env.FOOT_CONTACT_H - low) / env.FOOT_CONTACT_SOFT)

    def key_positions(self, p, R, q):
        """World positions (.., 4, 3) of the motion prior's key bodies (amp.KEY_BODIES)."""
        return self.kin_amp.forward(p, R, q)[0]

    def command(self, command, rot):
        return jnp.zeros_like(command.vel) if self.recovery else env.command(command, rot)

    def obs_frame(self, state, action, command, key):
        if self.recovery:
            rot, ang, key_pos = self.recovery_features(state)
            x = jnp.concatenate((ang, rot, state[4], state[5], action, key_pos), -1)
            if self.noisy_obs and key is not None:
                x += self.obs_noise * jax.random.uniform(key, x.shape, minval=-1., maxval=1.)
            return x
        return env.obs_frame(state, action, command, key, self.robot.DEFAULT_Q, self.obs_noise)

    def is_up(self, height, rotation):
        return env.is_up(height, rotation, self.robot.UP_HEIGHT)

    def recovery_features(self, state, key_pos=None):
        pos, rot, _, ang, q, _ = state
        local_rot = mm3(tr3(yaw_mat(rot)), rot)
        points = (self.key_positions(pos, rot, q) if key_pos is None else key_pos) - pos[..., None, :]
        return (jnp.concatenate((local_rot[..., :, 0], local_rot[..., :, 2]), -1), mv3(tr3(rot), ang),
                mv3(tr3(rot)[..., None, :, :], points).reshape(*pos.shape[:-1], -1))

    def weighted(self, terms):
        return {k: env.DT * self.weights[k] * terms[k] for k in self.terms}

    def reference_pose(self, state):
        pos, rot, _, _, q, _ = state
        return zenbot.RecoveryReference.pose(pos, rot, q, self.key_positions(pos, rot, q))

    def reward_terms(self, *args, torque=None, clip=None, step=None):
        if self.recovery:
            terms = zenbot.reward_terms(args[0], args[1], args[3], args[4], torque, self.q_soft, env.DT)
            if self.reference is not None:
                terms["motion_reference"] = self.reference.score(self.reference_pose(args[1]), clip, step)
            return terms
        return env.reward_terms(*args, wrist_idx=self.robot.WRIST_IDX, default_q=self.robot.DEFAULT_Q,
                                rise_height=self.robot.RISE_HEIGHT)

    def amp_frame(self, state, key_pos=None):
        if self.recovery:
            rot, ang, points = self.recovery_features(state, key_pos)
            return jnp.concatenate((rot, ang, state[4][..., self.amp_joints], state[5][..., self.amp_joints], points), -1)
        return amp.frame(state, self.key_positions(state[0], state[1], state[4]), self.amp_joints)

    def control_targets(self, action, bias, history, context):
        if self.robot is env:
            return env.joint_target(action, bias)
        # A 3–7 physics-step delay can reach either of the previous two policy
        # commands. The previous actions precede the 18 key-body coordinates.
        frames = history.reshape(*history.shape[:-1], self.history, self.frame_dim)
        previous, older = frames[..., -1, -18 - self.J:-18], frames[..., -2, -18 - self.J:-18]
        delay = context[..., 2 + zenbot.GROUP].astype(jnp.int32)
        age = jnp.arange(env.DECIMATION)[:, None] - delay[..., None, :]
        applied = jnp.where(age >= 0, action[..., None, :],
                            jnp.where(age >= -env.DECIMATION, previous[..., None, :], older[..., None, :]))
        targets = self.robot.DEFAULT_Q + self.robot.ACTION_SCALE * applied - bias[..., None, :]
        return targets.reshape(*action.shape[:-1], -1)

    def context(self, s: State):
        """Foot friction, torso CoM offset and (for Zenbot) group delays and assistance. Bias enters via the target."""
        m, w = self.m, s.world
        base = jnp.concatenate((w.geom_friction[:, self.foot_geoms[0], :1], w.body_ipos[:, self.torso] - m.body_ipos[self.torso]), -1)
        if self.recovery:
            return jnp.concatenate((base[:, :1], (w.body_mass[:, self.torso] - m.body_mass[self.torso])[:, None],
                                    s.delay_steps, s.assist[:, None]), -1)
        return base

    # ------------------------------------------------------------------ setup
    def world(self, key) -> World:
        m, N = self.m, self.N
        k_mu, k_b, k_com = jax.random.split(key, 3)
        tile = lambda x: jnp.tile(jnp.asarray(x, jnp.float32)[None], (N,) + (1,) * np.ndim(x))  # noqa: E731
        mu = jax.random.uniform(k_mu, (N,), minval=env.FOOT_FRICTION[0], maxval=env.FOOT_FRICTION[1])
        fr = tile(m.geom_friction).at[:, self.foot_geoms, 0].set(mu[:, None])
        com = jax.random.uniform(k_com, (N, 3), minval=-1.0, maxval=1.0) * jnp.asarray(env.TORSO_COM)
        ipos = tile(m.body_ipos).at[:, self.torso].add(com)
        bias = jax.random.uniform(k_b, (N, self.J), minval=-env.ENCODER_BIAS, maxval=env.ENCODER_BIAS)
        mass, subtree = m.body_mass, m.body_subtreemass
        if self.recovery:
            delta = jax.random.uniform(k_com, (N,), minval=-1., maxval=1.) if self.randomize_world else jnp.zeros(N)
            mass = tile(mass).at[:, self.torso].add(delta)
            subtree = tile(subtree)
            ancestor = self.torso
            while ancestor:
                subtree = subtree.at[:, ancestor].add(delta)
                ancestor = m.body_parentid[ancestor]
            subtree = subtree.at[:, 0].add(delta)
            ipos, bias = tile(m.body_ipos), jnp.zeros_like(bias)
            friction = jax.random.randint(k_mu, (N, self.m.ngeom), 0, 64) * (0.7 / 63) + 0.3
            fr = tile(m.geom_friction).at[:, :, 0].set(friction) if self.randomize_world else tile(m.geom_friction)
        return World(ipos, fr, bias, getattr(self, "mx", None), self.walk, mass, subtree, self.floor)  # (no model: built with physics=False)

    def _push_interval(self, key):
        u = jax.random.uniform(key, (self.N,), minval=env.PUSH_INTERVAL[0], maxval=env.PUSH_INTERVAL[1])
        return (u / env.DT).astype(jnp.int32)

    def _fresh(self, key, world: World, clip=None):
        """Fresh start states for every env (`_reset` applies them where an episode ended): a walking frame (no noise, the clip's
        heading), or with probability reset_floor a not-up frame of a get-up clip (random heading, roll / pitch / joint / velocity
        noise); prob_shove of the starts that are up get a kick of the velocities."""
        N, J = self.N, self.J
        kw, kf, ku, ks, kk1, kk2 = jax.random.split(key, 6)
        i = jax.random.randint(kw, (N,), 0, world.walk["z"].shape[0]) if clip is None else clip
        wb = world.walk
        q = jnp.clip(wb["q"][i], self.q_soft[:, 0], self.q_soft[:, 1])
        qpos = jnp.concatenate((jnp.zeros((N, 2)), wb["z"][i][:, None], wb["quat"][i], q), -1)
        qvel = jnp.concatenate((wb["v"][i], mv3(tr3(quat_to_mat(wb["quat"][i])), wb["w"][i]), wb["qd"][i]), -1)
        if self.recovery:
            qpos = qpos.at[:, 2].add(0.02).at[:, 7:].set(wb["q"][i])
            return qpos, qvel, self.is_up(qpos[:, 2], quat_to_mat(qpos[:, 3:7]))
        if self.reset_floor > 0:
            z, quat, v, w, fq, fqd, _ = resets.states(kf, world.floor, N, "floor", self.q_soft[:, 0], self.q_soft[:, 1])
            fpos = jnp.concatenate((jnp.zeros((N, 2)), z[:, None], quat, fq), -1)
            fvel = jnp.concatenate((v, mv3(tr3(quat_to_mat(quat)), w), fqd), -1)
            floor = jax.random.uniform(ku, (N,)) < self.reset_floor
            qpos, qvel = jnp.where(floor[:, None], fpos, qpos), jnp.where(floor[:, None], fvel, qvel)
        was_up = self.is_up(qpos[:, 2], quat_to_mat(qpos[:, 3:7]))
        if self.prob_shove > 0:
            shove = was_up & (jax.random.uniform(ks, (N,)) < self.prob_shove)
            kick = jnp.concatenate((jax.random.uniform(kk1, (N, 6), minval=-1.0, maxval=1.0) * jnp.asarray(env.SHOVE_ROOT),
                                    env.SHOVE_JOINT * jax.random.uniform(kk2, (N, J), minval=-1.0, maxval=1.0)), -1)  # fmt: skip
            qvel = qvel + jnp.where(shove[:, None], kick, 0.0)  # (the root's angular part is in the body frame, as MuJoCo's)
        return qpos, qvel, was_up

    def _reset(self, s: State, done, key) -> State:
        """Reset the envs where `done`: start state (`_fresh`), command drawn anew, last action zero, push schedule restarted."""
        kf, kc, kp = jax.random.split(key, 3)
        clip = None
        if self.reference is not None:
            clip = jax.random.randint(jax.random.split(kf, 6)[0], (self.N,), 0, len(self.reference.files))
            s = s._replace(reference_clip=jnp.where(done, clip, s.reference_clip))
        qpos, qvel, was_up = self._fresh(kf, s.world, clip)
        on = done[:, None]
        d = s.data.replace(qpos=jnp.where(on, qpos, s.data.qpos), qvel=jnp.where(on, qvel, s.data.qvel), qacc_warmstart=jnp.where(on, 0.0, s.data.qacc_warmstart))
        cmd = env.where_cmd(done, env.sample_cmd(kc, self.N), s.cmd)
        if self.robot is zenbot:
            delays = jax.random.randint(jax.random.fold_in(key, 9), s.delay_steps.shape, 3, 8)
            s = s._replace(delay_steps=jnp.where(on, delays, s.delay_steps))
        return s._replace(data=d, a_last=jnp.where(on, 0.0, s.a_last), a_ode=jnp.where(on, 0.0, s.a_ode), cmd=cmd, t=jnp.where(done, 0, s.t),
                          was_up=jnp.where(done, was_up, s.was_up), down=jnp.where(done, ~was_up, s.down), push_t=jnp.where(done, 0, s.push_t),
                          push_at=jnp.where(done, self._push_interval(kp), s.push_at), ep_sums=jnp.where(on, 0.0, s.ep_sums))  # fmt: skip

    def init(self, key) -> State:
        N, J = self.N, self.J
        k_world, k_reset, k_t, k_obs, key = jax.random.split(key, 5)
        world = self.world(k_world)
        data = jax.vmap(lambda q: self.dx.replace(qpos=q))(jnp.zeros((N, self.m.nq), jnp.float32))
        z, zi, zb = jnp.zeros((N,), jnp.float32), jnp.zeros((N,), jnp.int32), jnp.zeros((N,), bool)
        cmd = env.Cmd(jnp.zeros((N, 3)), z, zb, zb, z)
        s = State(data, world, jnp.zeros((N, self.obs_dim)), jnp.zeros((N, J)), jnp.zeros((N, J)), cmd, jnp.zeros((N, 3)), zi, jnp.ones((N,), bool),
                  jnp.zeros((N,), bool), zi, zi, jnp.zeros((N, len(self.terms))), key, jnp.full((N,), self.assist_newtons, jnp.float32), jnp.zeros((N, 5 if self.recovery else 0), jnp.int32))  # fmt: skip
        if self.reference is not None:
            s = s._replace(reference_clip=zi)
        s = self._reset(s, jnp.ones((N,), bool), k_reset)
        if self.random_start and not self.recovery:  # episodes then end at different times (a get-up start keeps its full time to rise)
            s = s._replace(t=jnp.where(s.down, 0, jax.random.randint(k_t, (N,), 0, env.MAX_EPISODE)))
        S = unflatten(s.data.qpos, s.data.qvel)
        c_obs = self.command(s.cmd, S[1])
        frame = self.obs_frame(S, s.a_last, c_obs, k_obs)
        return s._replace(c_obs=c_obs, obs=jnp.tile(frame, (1, self.history)))

    # ------------------------------------------------------------------ physics
    def contact_stats(self, data):
        """Of the solver's current contacts: foot-ground contact (N, 2), self-collision above 10 N (N,), and a floor contact above 1 N on a
        termination body (N,)."""
        im, N = data._impl, self.N
        valid = jnp.arange(im.contact__dist.shape[0]) < im.nacon[0]
        w, rows = jnp.clip(im.contact__worldid, 0, N - 1), im.contact__efc_address
        rows = rows.reshape(rows.shape[0], -1)
        normal = (im.efc__force[w[:, None], jnp.maximum(rows, 0)] * (rows >= 0)).sum(-1)  # the normal force: the sum of a contact's rows
        g0, g1 = im.contact__geom[:, 0], im.contact__geom[:, 1]
        robot, side, term = jnp.asarray(self.robot_geom), jnp.asarray(self.foot_side), jnp.asarray(self.is_term)
        floor = (g0 == self.floor_id) | (g1 == self.floor_id)
        fside = jnp.maximum(side[g0], side[g1])
        foot = valid & floor & (fside >= 0)
        found = jnp.zeros((N, 2)).at[w, jnp.maximum(fside, 0)].max(foot.astype(jnp.float32))
        selfc = valid & robot[g0] & robot[g1] & (jnp.abs(normal) > env.SELF_FORCE)
        hit = jnp.zeros((N,)).at[w].max(selfc.astype(jnp.float32))
        tc = valid & floor & (term[g0] | term[g1]) & (jnp.abs(normal) > env.CONTACT_FORCE)
        touch = jnp.zeros((N,)).at[w].max(tc.astype(jnp.float32)) > 0
        loads = None
        if self.recovery:
            support = jnp.asarray(self.support_link)
            link = jnp.maximum(support[g0], support[g1])
            loads = jnp.zeros((N, len(zenbot.KEY_BODIES))).at[w, jnp.maximum(link, 0)].add(
                jnp.where(valid & floor & (link >= 0), normal, 0))
        return found, hit, touch, loads

    def physics(self, world: World, data, target, assist, fresh):
        """4 substeps with the actuator's control recomputed before each; the feet's contact and the termination contact after the last
        substep, and the number of substeps with a self-collision."""
        mx = world.model.tree_replace({k: getattr(world, k) for k in self.world_fields})
        step = jax.vmap(lambda m_, d, c: mjx.step(m_, d.replace(ctrl=c)), in_axes=(self.axes, 0, 0))

        def sub(carry, i):
            d, hits = carry
            tgt = target.reshape(self.N, env.DECIMATION, self.J)[:, i] if self.robot is zenbot else target
            if self.recovery:
                # Only upward ground reaction on ankles, knees or hands enables assistance.
                loads = self.contact_stats(d)[3]
                scale = jnp.clip((zenbot.RISE_HEIGHT - d.qpos[:, 2]) / (zenbot.RISE_HEIGHT - 0.48), 0., 1.)
                force = assist * scale * (loads > 1.).any(-1) * (~fresh | (i > 0))
                d = d.replace(xfrc_applied=d.xfrc_applied.at[:, self.torso, 2].set(force))
            d = step(mx, d, self.robot.actuator_ctrl(d.qpos[:, self.qa], d.qvel[:, self.va], tgt))
            found, hit, touch, loads = self.contact_stats(d)
            return (d, hits + hit), (found, touch, loads)

        (data, hits), (found, touch, loads) = jax.lax.scan(sub, (data, jnp.zeros((self.N,))), jnp.arange(env.DECIMATION))
        return data, found[-1], hits, touch[-1], loads[-1] if self.recovery else None

    # ------------------------------------------------------------------ one policy step
    def step(self, s: State, action, clean=None):
        N = self.N
        key, k_reset, k_cmd, k_obs, k_push, k_push_at = jax.random.split(s.key, 6)
        a = action.astype(jnp.float32)
        target = self.control_targets(a, s.world.bias, s.obs, self.context(s))
        data, contact, hits, touch, loads = self.physics(s.world, s.data, target, s.assist, s.t == 0)
        S0, S1 = unflatten(s.data.qpos, s.data.qvel), unflatten(data.qpos, data.qvel)
        t = s.t + 1
        v = self.view_state(S1)
        diverged = ~(jnp.isfinite(data.qpos).all(-1) & jnp.isfinite(data.qvel).all(-1)) | (jnp.abs(data.qvel) > DIVERGED).any(-1)
        up1 = self.is_up(v.height, S1[1])
        terminated = jnp.where(s.was_up, touch, (t >= env.GETUP_STEPS) & ~up1) | diverged
        was_up1 = s.was_up | up1
        if self.recovery:
            terminated = diverged
        timeout = t >= self.episode_steps
        terms = self.reward_terms(S0, S1, v.kin, a, s.a_last, s.c_obs, ~was_up1, terminated.astype(jnp.float32), contact, hits, self.q_soft,
                                  torque=data.ctrl, clip=s.reference_clip, step=t)
        parts = self.weighted(terms)
        parts = jnp.where(diverged[:, None], 0.0, jnp.stack([parts[k] for k in self.terms], -1))
        reward = parts.sum(-1)
        done = terminated | timeout
        ep_sums = s.ep_sums + parts
        fin = done.astype(jnp.float32)
        stats = {"reward": reward.mean(), "done": fin.sum(), "terminated": terminated.sum().astype(jnp.float32), "len_sum": (fin * t).sum(),
                 "rew_sums": (fin[:, None] * ep_sums).sum(0) / (self.episode_steps * env.DT), "diverged": diverged.sum().astype(jnp.float32),
                 "nacon": data._impl.nacon[0] / data._impl.contact__dist.shape[0],
                 "term_geom": (((v.clearance.min(-1) < 0) == terminated) | diverged | ~s.was_up).mean(),  # imagination's geometric termination agrees
                 "down_done": (fin * s.down).sum(), "down_up": (fin * s.down * was_up1).sum(), "up_frac": was_up1.mean(),
                 "self_hits": hits.mean(), "foot_contact": contact.mean()}  # fmt: skip
        s1 = s._replace(data=data, a_last=a, a_ode=a if clean is None else clean.astype(jnp.float32), t=t, was_up=was_up1, ep_sums=ep_sums)
        if self.reference is not None:
            valid = (t < self.reference.lengths[s.reference_clip]) & ~diverged
            stats["reference_score_sum"] = jnp.where(valid, terms["motion_reference"], 0.).sum()
            stats["reference_frames"] = valid.sum()
        if self.recovery:
            height_score = ep_sums[:, self.terms.index("target_base_height")] / (self.episode_steps * env.DT)
            avg = (height_score * fin).sum() / jnp.maximum(fin.sum(), 1)
            dec = done & (avg > 0.6 * self.weights["target_base_height"])
            s1 = s1._replace(assist=jnp.maximum(0., s.assist - dec * zenbot.ASSIST_DECREMENT))
            stats.update(assist_n=s.assist.mean(), foot_load=loads[:, :2].sum(-1))
        s1 = self._reset(s1, done, k_reset)
        cmd = env.tick(s1.cmd, k_cmd)
        push_t = s1.push_t + 1
        push = (push_t == s1.push_at) & (self.push and not self.recovery)
        qvel = s1.data.qvel  # a push overwrites the root's world xy velocity
        qvel = qvel.at[:, :2].set(jnp.where(push[:, None], env.PUSH_VEL * jax.random.uniform(k_push, (N, 2), minval=-1.0, maxval=1.0), qvel[:, :2]))
        s1 = s1._replace(data=s1.data.replace(qvel=qvel), push_t=jnp.where(push, 0, push_t), push_at=jnp.where(push, self._push_interval(k_push_at), s1.push_at))
        stats["pushes"] = push.sum().astype(jnp.float32)
        S2 = unflatten(s1.data.qpos, s1.data.qvel)
        c_obs = self.command(cmd, S2[1])
        obs = env.push_frame(s.obs, self.obs_frame(S2, s1.a_last, c_obs, k_obs), done)
        s2 = s1._replace(obs=obs, cmd=cmd, c_obs=c_obs, key=key)
        return s2, Transition(s.data.qpos, s.data.qvel, target, data.qpos, data.qvel, reward, terminated, timeout & ~terminated, diverged, parts,
                              contact, v.clearance, stats)  # fmt: skip
