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

from .zenbot_motion_train_obs import MotionTrainSensors, advance_sensors
from . import zenbot_source_gait as source_gait
from .zenbot_gait_demos import reference_command_index
from . import zenbot_gait_rewards as gait_rewards
from . import amp, env, g1_getup, resets, zenbot
from .kin import Kinematics
from .rot import axis_angle_to_mat, mm3, mv3, quat_to_mat, tr3, yaw_mat

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "g1_mjlab.xml")
wp.config.log_level = wp.LOG_WARNING
WORLD_FIELDS = ("body_ipos", "geom_friction")  # the model fields that differ between envs
DIVERGED = 1e3  # |qvel| beyond this counts as a physics failure
MIXED_STAY_HEIGHT = .48
MIXED_STAY_UP_COS = .50
GAIT_TRANSITION_SHARE = .25
NCONMAX, NJMAX = 64, 640  # contact and constraint pools per world
WALK_DIR = "data/amp/walk_run"
FLOOR_DIR = "data/recovery/lafan1"


ZENBOT_XML = os.path.join(os.path.dirname(XML), "zenbot_23dof_43kg.xml")
G1_GETUP_XML = os.path.join(os.path.dirname(XML), "g1_leggedlab.xml")


def robot_profile(name):
    if name == "g1":
        return env
    if name == "g1_getup":
        return g1_getup
    if name in ("zenbot23_getup", zenbot.MIXED_TASK):
        return zenbot
    raise ValueError(f"Unknown robot profile {name!r}")


def motion_dirs(robot):
    if robot == zenbot.MIXED_TASK:
        return f"data/{robot}/walk_run", f"data/{robot}/recovery"
    if robot in ("g1_getup", "zenbot23_getup"):
        return (f"data/{robot}/recovery",) * 2
    robot_profile(robot)
    return WALK_DIR, FLOOR_DIR


def _add_getup_visuals(world, ground, robot):
    """Attach the G1 surface meshes to the corresponding bodies of the physical articulation."""
    appearance = mujoco.MjSpec.from_file(XML)
    mesh_dir = os.path.join(os.path.dirname(XML), "meshes")
    for geom in robot.geoms:
        geom.group = 3
    for mesh in appearance.meshes:
        robot.add_mesh(name="surface_" + mesh.name, file=os.path.join(mesh_dir, mesh.file), scale=mesh.scale)
    for mat in appearance.materials:
        robot.add_material(name=mat.name, rgba=mat.rgba, specular=mat.specular, shininess=mat.shininess)
    for source_body in appearance.bodies:
        body = robot.body(source_body.name)
        for geom in source_body.geoms:
            if geom.contype or geom.conaffinity:
                continue
            body.add_geom(type=geom.type, meshname="surface_" + geom.meshname,
                          pos=geom.pos, quat=geom.quat, material=geom.material, group=2,
                          contype=0, conaffinity=0, mass=0.)

    texture = world.add_texture(name="floor_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                                rgb1=[.42, .46, .50], rgb2=[.34, .38, .42])
    material = world.add_material(name="floor_grid", texrepeat=[2, 2], texuniform=True)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    ground.material = material.name
    world.worldbody.add_light(pos=[0, -2, 3], dir=[0, .5, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


def build_scene(iterations: int = 10, ls_iterations: int = 20, ccd_iterations: int = 50, robot_name: str = "g1",
                g1_support_friction: bool = False, *, getup_visuals: bool = False) -> mujoco.MjModel:
    """Compile the physical articulation and floor, optionally adding surface meshes for the G1 get-up viewer."""
    profile = robot_profile(robot_name)
    if g1_support_friction and robot_name != "g1":
        raise ValueError("g1_support_friction requires the G1 model")
    if getup_visuals and profile is not g1_getup:
        raise ValueError("getup_visuals requires the G1 get-up model")
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    o = world.option
    o.timestep, o.iterations, o.ls_iterations, o.ccd_iterations = env.SIM_DT, iterations, ls_iterations, ccd_iterations
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    ground = world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    robot = mujoco.MjSpec.from_file(ZENBOT_XML if profile is zenbot else G1_GETUP_XML if profile is g1_getup else XML)
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
    if profile is not g1_getup:  # the named task's meshes are collision hulls
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
        if profile is g1_getup:
            g.condim, g.priority = 3, 1
            g.friction[0] = 1.0
        elif "_foot" in g.name:
            g.condim, g.priority, g.friction = 3, 1, [0.6, g.friction[1], g.friction[2]]
        else:
            g.condim = 1
            if g1_support_friction:
                # Match mjlab's full-collision G1. With equal priority the ground's
                # three contact dimensions also allow a hand or knee to push tangentially.
                g.priority = ground.priority
    for i, name in enumerate(profile.DOF):
        j = robot.joint(f"{name}_joint")
        j.armature = float(profile.ARMATURE[i])
        a = robot.add_actuator(name=name, target=f"{name}_joint")
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype, a.gaintype, a.biastype = mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = float(profile.KP[i]), -float(profile.KP[i]), -float(profile.KD[i])
        a.inheritrange, a.ctrllimited = 0.0, False
        a.forcelimited, a.forcerange = True, [-float(profile.EFFORT[i]), float(profile.EFFORT[i])]
    if getup_visuals:
        _add_getup_visuals(world, ground, robot)
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
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            first, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            pts, r = pos + m.mesh_vert[first:first + count] @ R.T, 0.0
        else:
            raise ValueError(f"collider geom {m.geom(g).name} has unsupported type {kind}")
        local.extend(pts)
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
    body_inertia: jax.Array | None = None


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
    reference_clip: jax.Array | None = None  # training-only reference metadata, -1 for G1 walking resets
    reference_start: jax.Array | None = None
    reference_yaw: jax.Array | None = None
    critic_obs: jax.Array | None = None
    episode_limit: jax.Array | None = None  # first timeout can be staggered while the reference starts at elapsed time zero
    recovery_left: jax.Array | None = None  # post-fall AMP scoring interval in physical control steps
    supported_hold: zenbot.SupportedStand | None = None
    gait_ready: jax.Array | None = None  # can follow a travel command after rising onto the feet
    motion_train_sensors: MotionTrainSensors | None = None
    first_supported_stand: jax.Array | None = None  # first successful floor recovery step in the assisted scratch trial
    gait_curriculum: source_gait.Curriculum | None = None
    gait_window: jax.Array | None = None  # physical gait start row and exclusive end; -1 for ordinary episodes
    step_history: gait_rewards.StepHistory | None = None


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
    feet: gait_rewards.FootGeometry | None = None


def unflatten(qpos, qvel):
    """MuJoCo's (qpos, qvel) -> S = (p, R, v, w, q, qd) with world-frame w (MuJoCo's free-joint angular velocity is in the body frame)."""
    R = quat_to_mat(qpos[..., 3:7])
    return qpos[..., :3], R, qvel[..., :3], mv3(R, qvel[..., 3:6]), qpos[..., 7:], qvel[..., 6:]


def load_bank(directory: str, joints: int = env.J, first_only=False, sample_clip_times=False):
    """Load 50 Hz reset states in MJCF order. G1 recovery retains clip boundaries and root XY for reference resets."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    assert files, f"no locomotion clips in {directory}; prepare the motion banks for the selected robot"
    cols = {k: [] for k in ("z", "quat", "v", "w", "q", "qd")}
    lengths = []
    if sample_clip_times:
        cols["xy"] = []
    for f in files:
        with np.load(f) as d:
            assert int(d["fps"]) == 50, f
            assert d["dof_pos"].shape[1] == joints, f
            if sample_clip_times:
                cols["xy"].append(d["root_pos"][:, :2])
                lengths.append(len(d["root_pos"]))
            q = d["root_quat"]
            for k, a in (("z", d["root_pos"][:, 2]), ("quat", np.concatenate([q[:, 3:], q[:, :3]], -1)), ("v", d["root_lin_vel"]), ("w", d["root_ang_vel"]),
                         ("q", d["dof_pos"]), ("qd", d["dof_vel"])):  # fmt: skip
                cols[k].append(a[:1] if first_only else a)
    bank = {k: jnp.asarray(np.concatenate(v, 0), jnp.float32) for k, v in cols.items()}
    if sample_clip_times:
        if min(lengths) <= 10:
            raise ValueError("G1 get-up resets need at least 0.2 seconds of animation")
        bank["offsets"] = jnp.asarray(np.cumsum([0, *lengths[:-1]]), jnp.int32)
        bank["lengths"] = jnp.asarray(lengths, jnp.int32)
    return bank


class Sim:
    zenbot_source_gait = False
    zenbot_motion_train_wm = False
    zenbot_wm_reference_training = False
    zenbot_wm_getup_only = False
    wm_min_rise_score = 0.
    wm_floor_amp = False
    wm_floor_locomotion = False
    wm_gait_rewards = False
    wm_posture_and_tracking = False
    wm_range_commands = False
    wm_lift_gate = 0.
    floor_stop_command_share = 0.
    gait_source_commands = False
    wm_command_speeds = zenbot.DIRECTIONAL_COMMANDS

    def __init__(self, num_envs: int = 4096, nconmax: int = NCONMAX, njmax: int = NJMAX, walk_dir: str | None = None,
                 floor_dir: str | None = None, random_start: bool = True, seed: int = 0, reset_floor: float = 0.2,
                 prob_shove: float = 0.0, push: bool = True, physics: bool = True, robot: str = "g1", amp_wrists: bool = True, assist_newtons=None, obs_noise=True, randomize_world=True, reference_weight=0., g1_style=False, getup_rise_weight=-0.25, g1_support_friction=False, getup_reset_frames=20, g1_phase_amp=False, reference_hold_scale=1., action_rate_l2_weight=None, zenbot_phase_resets=0., zenbot_episode_steps=500, recovery_amp_seconds=None, fall_push_speed=0., legged_support=False, support_rsi=0., zenbot_getup_stand=False, zenbot_support_weight=0., zenbot_default_stand=False, zenbot_standing_amp=False, zenbot_command_gait=False, zenbot_directional_gait=False, zenbot_motion_train_obs=False, zenbot_scratch_resets=False, zenbot_scratch_curriculum=False, zenbot_source_gait=False,
                 zero_motor_delay_probability=None, zenbot_motion_train_wm=False, zenbot_wm_reference_training=False, zenbot_wm_getup_only=False,
                 wm_stance_start_share=0., wm_min_rise_score=0., wm_floor_amp=False, wm_stand_reference=None, wm_lift_gate=0.,
                 floor_stop_command_share=0., gait_source_commands=False, wm_gait_rewards=False, wm_posture_and_tracking=False, wm_range_commands=False):  # fmt: skip
        """physics=False: everything but MuJoCo Warp's model and data (no GPU): the scene, kinematics, reset banks and `_fresh`, for
        checks on the CPU."""
        self.N, self.random_start, self.seed = num_envs, random_start, seed
        self.robot = profile = robot_profile(robot)
        self.zenbot_motion_train_wm = bool(zenbot_motion_train_wm)
        self.zenbot_wm_reference_training = bool(zenbot_wm_reference_training)
        self.zenbot_wm_getup_only = bool(zenbot_wm_getup_only)
        self.wm_stance_start_share = wm_stance_start_share
        self.wm_min_rise_score = wm_min_rise_score
        self.wm_floor_amp = bool(wm_floor_amp)
        self.wm_floor_locomotion = self.wm_floor_amp and not self.zenbot_wm_getup_only
        self.wm_range_commands = wm_range_commands
        self.wm_posture_and_tracking = wm_posture_and_tracking
        self.wm_gait_rewards = wm_gait_rewards
        if (wm_range_commands or wm_posture_and_tracking or wm_gait_rewards) and not self.wm_floor_locomotion:
            raise ValueError("Motion Train command ranges, posture and gait rewards require joint floor/locomotion training")
        self.gait_source_commands = bool(gait_source_commands)
        if self.gait_source_commands and not self.wm_floor_locomotion:
            raise ValueError("Recorded command speeds require the joint floor and locomotion task")
        if not 0. <= floor_stop_command_share <= 1. or (floor_stop_command_share and not self.wm_floor_locomotion):
            raise ValueError("Floor stop command share must be in [0, 1] and requires joint locomotion training")
        self.floor_stop_command_share = floor_stop_command_share
        if not 0. <= wm_lift_gate < zenbot.UP_HEIGHT or (wm_lift_gate and not self.wm_floor_amp):
            raise ValueError("The late lift reward needs a floor AMP task and a gate below upright pelvis height")
        self.wm_lift_gate = wm_lift_gate
        if self.wm_floor_amp and (not zenbot_motion_train_wm or zenbot_wm_reference_training or wm_stance_start_share
                                  or (self.wm_floor_locomotion and (reference_weight or not 0. < reset_floor < 1.))):
            raise ValueError("Floor-opening AMP needs independent motion windows and both floor and gait starts for locomotion")
        if not 0. <= wm_min_rise_score < 1. or (wm_min_rise_score and not (self.zenbot_wm_getup_only and self.zenbot_wm_reference_training)):
            raise ValueError("A minimum rise score requires paired get-up training and a threshold in [0, 1)")
        if not 0. <= wm_stance_start_share <= .5 or (wm_stance_start_share and not self.zenbot_wm_getup_only):
            raise ValueError("Supported stance starts require the zero-command get-up stage and a share in [0, .5]")
        self.wm_start_quantiles = None
        if wm_min_rise_score and wm_stance_start_share:
            # Failed rises reset sooner than stable stands. Reserve worlds for each kind so long stands cannot fill the buffer.
            # Adjacent deterministic and noisy collectors share a kind, including the worlds providing real PPO data.
            pairs = (num_envs + 1) // 2
            ranks = np.random.default_rng(seed).permutation(pairs)
            self.wm_start_quantiles = jnp.asarray(np.repeat((ranks + .5) / pairs, 2)[:num_envs], jnp.float32)
        if self.zenbot_wm_getup_only:
            if not self.zenbot_motion_train_wm:
                raise ValueError("Zenbot get-up-only WM requires the Motion Train world-model task")
            reset_floor, prob_shove, push = 1., 0., False
        self.reset_floor, self.prob_shove, self.push = reset_floor, prob_shove, push
        self.wm_physical_resets = (self.zenbot_wm_reference_training or self.zenbot_wm_getup_only) and not self.wm_floor_amp
        if self.zenbot_wm_reference_training and not self.zenbot_motion_train_wm:
            raise ValueError("Paired Zenbot reference resets require the Motion Train world-model task")
        if self.zenbot_motion_train_wm and robot != zenbot.MIXED_TASK:
            raise ValueError("Motion Train world-model control requires the mixed Zenbot task")
        if zero_motor_delay_probability is None:
            zero_motor_delay_probability = .5 if self.zenbot_motion_train_wm else 0.
        if recovery_amp_seconds is None:
            recovery_amp_seconds = 4. if self.zenbot_motion_train_wm else 0.
        if not 0. <= zero_motor_delay_probability <= 1. or (zero_motor_delay_probability and profile is not zenbot):
            raise ValueError("Motor delay mixing requires a Zenbot and probability in [0, 1]")
        self.zero_motor_delay_probability = float(zero_motor_delay_probability)
        self.J = len(profile.DOF)
        self.mixed = robot == zenbot.MIXED_TASK
        self.zenbot_getup_stand = bool(zenbot_getup_stand)
        self.physical_floor_starts = None
        self.physical_gait_starts = None
        self.zenbot_scratch_resets = bool(zenbot_scratch_resets)
        self.zenbot_source_gait = bool(zenbot_source_gait)
        if self.zenbot_source_gait and not (self.zenbot_scratch_resets and zenbot_motion_train_obs and zenbot_command_gait and zenbot_support_weight > 0):
            raise ValueError("Source locomotion requires the mixed scratch task and 234-value observations")
        self.zenbot_scratch_curriculum = bool(zenbot_scratch_curriculum)
        if self.zenbot_scratch_curriculum and not self.zenbot_scratch_resets:
            raise ValueError("The training assistance curriculum requires scratch floor resets")
        if self.zenbot_scratch_resets and not (zenbot_command_gait and self.zenbot_getup_stand):
            raise ValueError("Scratch reference resets require the commanded Zenbot get-up task")
        if self.zenbot_getup_stand and (robot != "zenbot23_getup" or (reference_weight and not self.zenbot_scratch_resets) or g1_style):
            raise ValueError("Get-up and stand AMP requires the standalone Zenbot task; tracking requires scratch resets")
        if not np.isfinite(zenbot_support_weight) or zenbot_support_weight < 0 or (zenbot_support_weight and not self.zenbot_getup_stand):
            raise ValueError("zenbot_support_weight requires a finite nonnegative weight on the standalone get-up and stand task")
        self.zenbot_support_weight = float(zenbot_support_weight)
        self.zenbot_default_stand = bool(zenbot_default_stand)
        if self.zenbot_default_stand and not (self.zenbot_getup_stand and self.zenbot_support_weight > 0.):
            raise ValueError("zenbot_default_stand requires the standalone get-up and stand task and positive zenbot_support_weight")
        self.zenbot_standing_amp = bool(zenbot_standing_amp)
        if self.zenbot_standing_amp and (not self.zenbot_support_weight or self.zenbot_default_stand):
            raise ValueError("Standing AMP requires foot support and exclusive control of the standing style reward")
        self.zenbot_command_gait = bool(zenbot_command_gait)
        self.zenbot_directional_gait = bool(zenbot_directional_gait)
        self.zenbot_motion_train_obs = bool(zenbot_motion_train_obs or self.zenbot_motion_train_wm)
        if self.zenbot_motion_train_obs and not (self.zenbot_command_gait or self.zenbot_motion_train_wm):
            raise ValueError("Motion Train observations require the Zenbot command actor")
        if self.zenbot_directional_gait and not self.zenbot_command_gait:
            raise ValueError("Directional locomotion requires the Zenbot command actor")
        if self.zenbot_command_gait and not self.zenbot_standing_amp:
            raise ValueError("Velocity commands on the get-up actor require the standing AMP task")
        if self.zenbot_command_gait and (not walk_dir or not 0. <= reset_floor <= 1.):
            raise ValueError("Commanded gait requires a walk_dir and reset_floor probability in [0, 1]")
        if not np.isfinite(recovery_amp_seconds) or recovery_amp_seconds < 0 or (recovery_amp_seconds and not self.mixed):
            raise ValueError("Recovery AMP interval requires the mixed Zenbot task and a finite nonnegative duration")
        self.recovery_amp_steps = round(recovery_amp_seconds / env.DT)
        self.legged_support = bool(legged_support)
        if self.legged_support and not self.recovery_amp_steps:
            raise ValueError("LeggedLab support rewards require a mixed Zenbot recovery interval")
        if not 0. <= support_rsi <= 1. or (support_rsi and not (self.legged_support or (self.wm_floor_amp and not reference_weight))):
            raise ValueError("Intermediate ground starts require a probability in [0,1] and unpaired floor or support AMP")
        self.support_rsi = float(support_rsi)
        if not np.isfinite(fall_push_speed) or fall_push_speed < 0 or (fall_push_speed and not self.recovery_amp_steps):
            raise ValueError("Fall perturbations require the mixed recovery AMP interval")
        self.fall_push_speed = fall_push_speed
        self.recovery = profile is not env and not self.mixed
        self.cmd_ranges = zenbot.GAIT_COMMANDS if self.mixed else env.CMD_RANGE
        self.history = profile.HIST
        self.assist_newtons = (profile.INITIAL_ASSIST if self.recovery else 0.) if assist_newtons is None else assist_newtons
        self.randomize_world = randomize_world
        self.noisy_obs = obs_noise
        self.weights = zenbot.WEIGHTS if self.recovery else env.WEIGHTS
        if self.zenbot_getup_stand:
            if assist_newtons not in (0., None):
                raise ValueError("Standalone get-up and stand uses zero assistance")
            self.assist_newtons = 0.
            self.weights = {"joint_torques_l2": zenbot.WEIGHTS["joint_torques_l2"],
                            "joint_pos_limits": zenbot.WEIGHTS["joint_pos_limits"]}
            if not self.zenbot_standing_amp:
                self.weights["stand_pose"] = 15.
            if self.zenbot_support_weight:
                self.weights["supported_hold"] = self.zenbot_support_weight
        if self.recovery_amp_steps:
            self.weights = {**self.weights, "recovery_timeout_cost": -100.}
        if self.legged_support:
            self.weights = {**self.weights, **zenbot.WEIGHTS}
        if profile is g1_getup and action_rate_l2_weight is None:
            action_rate_l2_weight = g1_getup.ACTION_RATE_L2_WEIGHT
        if self.zenbot_command_gait and action_rate_l2_weight is None:
            action_rate_l2_weight = 3. * zenbot.WEIGHTS["action_rate_l2"]
        if self.zenbot_motion_train_wm:
            rate_weight = 3. * zenbot.WEIGHTS["action_rate_l2"] if action_rate_l2_weight is None else action_rate_l2_weight
            if not np.isfinite(rate_weight) or rate_weight > 0.:
                raise ValueError("The Motion Train action-rate weight must be finite and nonpositive")
            self.weights = {name: env.WEIGHTS[name] for name in ("track_lin", "track_ang", "rise", "termination")}
            self.weights.update(action_rate=rate_weight, recovery_timeout_cost=-100.)
            if self.zenbot_wm_reference_training:
                self.weights["motion_reference"] = 5.
            if self.wm_floor_amp:
                self.weights = {"stand": 6., "action_rate": rate_weight}
                if self.wm_lift_gate:
                    self.weights["lift"] = .5
                if self.wm_floor_locomotion:
                    self.weights.update(track_lin_vel_xy_exp=2., track_ang_vel_z_exp=1.5)
                if self.wm_posture_and_tracking:
                    self.weights.update(gait_rewards.POSTURE_WEIGHTS)
                if self.wm_gait_rewards:
                    self.weights.update(gait_rewards.WEIGHTS)
        elif action_rate_l2_weight is not None:
            if not self.recovery or not np.isfinite(action_rate_l2_weight) or action_rate_l2_weight > 0:
                raise ValueError("Expected a nonpositive action-rate weight for a named get-up task")
            self.weights = {**self.weights, "action_rate_l2": action_rate_l2_weight}
        if g1_style:
            self.weights = {**self.weights, "rise": getup_rise_weight}
        self.g1_phase_amp = bool(g1_phase_amp)
        if self.g1_phase_amp and (profile is not env or not g1_style or reset_floor <= 0):
            raise ValueError("G1 phase AMP requires g1_style and floor resets on the G1")
        self.reference_reward = bool(reference_weight)
        self.g1_reference = bool(g1_style and (self.reference_reward or self.g1_phase_amp) and profile is env)
        if not 0. <= zenbot_phase_resets <= 1. or zenbot_episode_steps <= 0:
            raise ValueError("Zenbot reference reset probability and episode length must be valid")
        if zenbot_phase_resets and (profile is not zenbot or not (self.reference_reward or self.zenbot_scratch_resets)):
            raise ValueError("Zenbot phase resets require its tracking reward")
        self.zenbot_phase_resets = zenbot_phase_resets
        self.track_getup = self.zenbot_scratch_resets or ((profile is g1_getup or bool(zenbot_phase_resets)) and self.reference_reward)
        if not 0. <= reference_hold_scale <= 1.:
            raise ValueError("reference_hold_scale must be between 0 and 1")
        self.reference_hold_scale = reference_hold_scale
        if reference_weight:
            if not (self.recovery or self.g1_reference or self.wm_floor_amp) or not np.isfinite(reference_weight) or reference_weight < 0:
                raise ValueError("reference_weight requires a finite positive weight and a recovery task")
            if self.g1_reference and self.reset_floor <= 0:
                raise ValueError("G1 reference rewards require recovery resets")
            self.weights = {**self.weights, "motion_reference": reference_weight}
        self.terms = tuple(self.weights)
        self.episode_steps = 500 if self.recovery else env.MAX_EPISODE
        if profile is zenbot:
            self.episode_steps = zenbot_episode_steps
        self.world_fields = WORLD_FIELDS + (("body_mass", "body_subtreemass") if profile is not env else ())
        if profile is g1_getup:
            self.world_fields += ("body_inertia",)
        self.frame_dim = 9 + 3 * self.J + (18 if self.recovery else 0)
        if self.zenbot_command_gait:
            self.frame_dim += 3  # append commands after the existing 96 sensor and action channels
        self.obs_dim = self.history * self.frame_dim
        self.obs_noise = env.observation_noise(self.J)
        self.amp_joints = np.asarray([i for i in range(self.J) if amp_wrists or i not in profile.WRIST_IDX])
        self.amp_dim = 2 * len(self.amp_joints) + 6 + 3 * len(amp.KEY_BODIES)
        if self.mixed:
            self.amp_dim = 2 * len(self.amp_joints) + 6 + 3 * len(profile.KEY_BODIES) + 4
        if self.recovery:
            self.amp_dim = 9 + 2 * len(self.amp_joints) + 18
            if self.zenbot_getup_stand:
                self.amp_dim += amp.SUPPORT_FRAME_DIM
            self.obs_noise = np.repeat([0.2, 0.05, 0.01, 1.5, 0.0, 0.08], [3, 6, self.J, self.J, self.J, 18]).astype(np.float32)
        if self.zenbot_command_gait:
            self.obs_noise = np.pad(self.obs_noise, (0, 3))
        self.m = m = build_scene(robot_name=robot, g1_support_friction=g1_support_friction)
        assert m.nu == self.J and m.nq == 7 + self.J
        assert tuple(m.joint(m.actuator_trnid[i, 0]).name.removeprefix("robot_").removesuffix("_joint") for i in range(m.nu)) == profile.DOF
        joint_ids = m.actuator_trnid[:, 0]
        self.qa, self.va = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
        if not np.array_equal(self.qa, np.arange(7, m.nq)) or not np.array_equal(self.va, np.arange(6, m.nv)):
            raise ValueError("Motion columns must match the model hinge order")
        rng = m.jnt_range[joint_ids]
        self.joint_range = rng.astype(np.float32)
        mid, half = rng.mean(-1), (rng[:, 1] - rng[:, 0]) / 2 * env.SOFT_LIMIT
        self.q_soft = np.stack([mid - half, mid + half], -1).astype(np.float32)
        torso = profile.TORSO if profile is not env else "torso_link"
        keys = profile.KEY_BODIES if profile is not env else amp.KEY_BODIES
        self.torso = m.body("robot_" + torso).id
        self.root_com = m.body_ipos[1].astype(np.float32)
        feet = [m.body("robot_" + name).id for name in keys[:2]]
        self.foot_geoms = np.asarray([g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_contype[g] > 0])
        assert len(self.foot_geoms) == (2 if profile is zenbot else 12 if profile is g1_getup else env.N_FOOT_GEOMS)
        tb = [m.body(f"robot_{b}").id for b in profile.TERMINATION_BODIES]
        self.term_geoms = np.asarray([g for g in range(m.ngeom) if m.geom_bodyid[g] in tb and m.geom_contype[g] > 0])
        assert len(set(int(m.geom_bodyid[g]) for g in self.term_geoms)) == len(tb), "every termination body carries a collision geom"
        # frames of the forward kinematics: the torso, the two ankle-roll links, then the termination bodies
        frames = [self.torso] + feet + tb
        self.kin = VelKinematics(m, [m.body(b).name for b in frames])
        if self.wm_gait_rewards:
            geoms = np.r_[self.foot_geoms, self.term_geoms]
            bodies = m.geom_bodyid[geoms]
            if set(geoms) != set(np.flatnonzero((m.geom_bodyid != 0) & (m.geom_contype != 0))) or len(set(bodies)) != len(geoms):
                raise ValueError("Expected one collider per Zenbot contact body")
            self.gait_bodies = bodies
            self.foot_com = m.body_ipos[feet].astype(np.float32)
            self.gait_frames = np.asarray([frames.index(b) for b in bodies])
            self.gait_geom_pos = m.geom_pos[geoms].astype(np.float32)
            self.gait_geom_rot = np.asarray(quat_to_mat(m.geom_quat[geoms].astype(np.float32)))
            self.gait_geom_size = m.geom_size[geoms].astype(np.float32)
            self.gait_boxes = m.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_BOX
            if not np.all(self.gait_boxes | (m.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_CAPSULE)):
                raise ValueError("Expected Zenbot sole boxes and box or capsule collision bodies")
            if not self.gait_boxes[:2].all():
                raise ValueError("Zenbot foot slip requires a box collider on each foot")
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
            corners = signs * self.gait_geom_size[:2, None, :]
            self.foot_corners = self.gait_geom_pos[:2, None, :] + corners @ self.gait_geom_rot[:2].transpose(0, 2, 1)
            self.foot_dofs = np.zeros((2, m.nv), dtype=np.float32)
            for side, foot in enumerate(feet):
                while foot != 0:
                    self.foot_dofs[side, m.dof_bodyid == foot] = 1.
                    foot = m.body_parentid[foot]
        self.site_off = np.stack([m.site_pos[m.site(f"robot_{s}_foot").id] for s in ("left", "right")]).astype(np.float32)
        self.sole_local, self.sole_frame, self.sole_radius = collider_points(m, self.foot_geoms, frames)
        self.coll_local, self.coll_frame, self.coll_radius = collider_points(m, self.term_geoms, frames)
        self.kin_amp = Kinematics(m, [f"robot_{b}" for b in keys])
        if self.wm_physical_resets:
            from .zenbot_wm_reference import PHYSICAL_FLOOR, PhysicalRecoveryReference
            self.wm_reference = PhysicalRecoveryReference(PHYSICAL_FLOOR, floor_dir or motion_dirs(robot)[1], self.key_positions)
        if self.mixed or self.zenbot_getup_stand:
            # In the Zenbot model the two soles are boxes and the knees and palms are capsules.
            ids = [m.geom("robot_" + body + "_collision").id for body in keys]
            if list(m.geom_type[ids]) != [mujoco.mjtGeom.mjGEOM_BOX] * 2 + [mujoco.mjtGeom.mjGEOM_CAPSULE] * 4:
                raise ValueError("Unexpected Zenbot support collision geometry")
            self.support_pos = np.asarray(m.geom_pos[ids], np.float32)
            self.support_rot = np.asarray(quat_to_mat(np.asarray(m.geom_quat[ids], np.float32)))
            self.support_size = np.asarray(m.geom_size[ids], np.float32)
        # contact decoding: which foot (0 / 1) a geom belongs to, the termination geoms, the floor, the robot's geoms
        self.foot_side = np.full(m.ngeom, -1, np.int32)
        for g in self.foot_geoms:
            self.foot_side[g] = feet.index(int(m.geom_bodyid[g]))
        self.is_term = (source_gait.illegal_contact_geoms(m) if self.zenbot_source_gait else
                        np.isin(np.arange(m.ngeom), self.term_geoms))
        supports = [m.body("robot_" + b).id for b in (profile.SUPPORT_BODIES if self.recovery else keys)]
        self.support_link = np.array([supports.index(b) if b in supports else -1 for b in m.geom_bodyid])
        self.floor_id = m.geom("floor").id
        self.robot_geom = np.asarray([m.body(int(m.geom_bodyid[g])).name.startswith("robot_") for g in range(m.ngeom)])
        walk_default, floor_default = motion_dirs(robot)
        if self.gait_source_commands:
            from .zenbot_gait_demos import source_command_speeds
            self.wm_command_speeds = source_command_speeds(walk_dir or walk_default)
        bank_dir = (floor_dir or floor_default) if self.recovery else (walk_dir or walk_default)
        if self.zenbot_wm_getup_only:
            self.walk = {}
            self.wm_gait_worlds = jnp.zeros(self.N, bool)
        elif self.zenbot_command_gait:
            bank = load_bank(walk_dir, self.J)
            upright = self.is_up(bank["z"], quat_to_mat(bank["quat"]))
            if not np.any(upright):
                raise ValueError(f"No upright gait reset frames in {walk_dir}")
            if self.zenbot_scratch_resets:
                from .zenbot_gait_demos import command_reset_bank
                self.walk = command_reset_bank(walk_dir, bank, upright, self.zenbot_directional_gait)
            else:
                self.walk = {name: value[upright] for name, value in bank.items()}
        else:
            self.walk = (g1_getup.load_reference(bank_dir) if profile is g1_getup and not self.track_getup else
                         load_bank(bank_dir, self.J, first_only=robot == "zenbot23_getup" and not self.track_getup, sample_clip_times=self.track_getup))
        if self.wm_floor_locomotion or (self.zenbot_wm_reference_training and not self.zenbot_wm_getup_only):
            from .zenbot_gait_demos import command_reset_bank
            self.wm_gait_worlds = (jnp.arange(self.N) // 2) % 4 == 0
            upright = self.is_up(self.walk["z"], quat_to_mat(self.walk["quat"]))
            self.walk["command_rows"] = command_reset_bank(
                bank_dir, {"row": jnp.arange(len(upright))}, upright, directional=True,
                command_speeds=self.wm_command_speeds if self.gait_source_commands else None)
        self.reference = None
        if self.g1_reference:
            from .g1_reference_reward import G1RecoveryReference
            self.reference = G1RecoveryReference(floor_dir or floor_default, self.key_positions)
        elif reference_weight:
            self.reference = zenbot.RecoveryReference(floor_dir or floor_default, self.key_positions, self.J,
                                                      coupled_pose=self.track_getup or self.wm_floor_amp,
                                                      with_rates=self.wm_floor_amp)
            if self.wm_floor_amp:
                # The recovery clips often end in mid-step. Use the upright AMP clip for quiet standing.
                self.standing_reference = zenbot.RecoveryReference(
                    wm_stand_reference or os.path.join(walk_dir or walk_default, "stand_4s_120f.npz"),
                    self.key_positions, self.J, coupled_pose=True, with_rates=True)
        if self.wm_lift_gate:
            stand_clip = wm_stand_reference or os.path.join(walk_dir or walk_default, "stand_4s_120f.npz")
            with np.load(stand_clip, allow_pickle=False) as data:
                joints = data["dof_pos"]
                self.floor_stand_joints = jnp.asarray(joints[len(joints) // 2])
        floor_frames = getup_reset_frames if g1_style or (self.mixed and not self.recovery_amp_steps) else None
        if self.legged_support:
            floor_frames = 1
        self.floor = resets.load(floor_dir or floor_default, "floor", profile.UP_HEIGHT, env.UP_TILT, self.J,
                                 first_frames=None if self.zenbot_motion_train_wm else floor_frames,
                                 source_indices=self.g1_reference or self.zenbot_motion_train_wm) if reset_floor > 0 and not self.recovery else resets.empty(self.J)
        if self.zenbot_motion_train_wm and reset_floor > 0:
            frame = np.asarray(self.floor["frame"])
            opening = (frame == 0) & (np.asarray(self.floor["z"]) + zenbot.RESET_LIFT <= .5)
            if not opening.any() or not (frame > 0).any():
                raise ValueError("Mixed Zenbot world-model floor starts need fallen opening poses and later recovery frames")
            phases = frame > 0
            if self.wm_floor_amp and self.support_rsi:
                height = np.asarray(self.floor["z"])
                phases &= height + zenbot.RESET_LIFT <= zenbot.UP_HEIGHT
                if not self.wm_floor_locomotion:
                    phases &= height > height[opening].max()
                if not phases.any():
                    raise ValueError("Floor support practice needs low intermediate recovery poses")
            self.floor = {"openings": jax.tree.map(lambda x: x[opening], self.floor),
                          "phases": jax.tree.map(lambda x: x[phases], self.floor)}
            if self.wm_floor_amp and self.support_rsi:
                self.floor["phases"] = jax.tree.map(lambda a, b: jnp.concatenate((a, b)),
                                                    self.floor["openings"], self.floor["phases"])
            if self.wm_physical_resets:
                ref = self.wm_reference
                files = sorted(glob.glob(os.path.join(floor_dir or floor_default, "*.npz")))
                if tuple(map(os.path.basename, files)) != tuple(ref.clip_names):
                    raise ValueError("Cold opening files must match the physical recovery reference clips")
                up = np.asarray(self.is_up(ref.qpos[:, 2], quat_to_mat(ref.qpos[:, 3:7])))
                limits = []
                for first, count, support in zip(np.asarray(ref.offsets), np.asarray(ref.lengths), np.asarray(ref.first_support)):
                    before_support = min(int(count), int(support) + 1)
                    upright = np.flatnonzero(up[first:first + before_support])
                    limits.append(int(upright[0]) if upright.size else before_support)
                if min(limits) <= 1:
                    raise ValueError("Each paired recovery needs fallen physical frames after its opening")
                self.floor["phases"] = dict(qpos=ref.qpos, qvel=ref.qvel, offsets=ref.offsets, limits=jnp.asarray(limits, jnp.int32))
        elif self.zenbot_scratch_resets:
            self.floor = load_bank(floor_dir or floor_default, self.J, sample_clip_times=True)
            up = np.asarray(self.is_up(self.floor["z"] + zenbot.RESET_LIFT, quat_to_mat(self.floor["quat"])))
            down_counts = []
            for start, count in zip(np.asarray(self.floor["offsets"]), np.asarray(self.floor["lengths"])):
                first_up = np.flatnonzero(up[start:start + count])
                down_counts.append(first_up[0] if first_up.size else count)
            if min(down_counts) == 0:
                raise ValueError("Each recovery clip must open with a pose on the floor")
            self.floor["down_counts"] = jnp.asarray(down_counts, jnp.int32)
        elif self.zenbot_command_gait and reset_floor > 0:
            self.floor = load_bank(floor_dir or floor_default, self.J, first_only=True)
        if self.support_rsi and reset_floor > 0 and not self.zenbot_motion_train_wm:
            self.floor_openings = len(self.floor["z"])
            # Keep each sampled state down after the ordinary 2 cm reset clearance is applied.
            states = resets.load(floor_dir or floor_default, "floor", profile.UP_HEIGHT - profile.RESET_LIFT, env.UP_TILT, self.J)
            self.floor = jax.tree.map(lambda openings, all_down: jnp.concatenate((openings, all_down)), self.floor, states)
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

        feet = None
        if self.wm_gait_rewards:
            geom_up = Rs[..., self.gait_frames, 2, :]
            center = P[..., self.gait_frames, 2] + (geom_up * self.gait_geom_pos).sum(-1)
            up = jnp.einsum("...bi,bij->...bj", geom_up, self.gait_geom_rot)
            size = self.gait_geom_size
            extent = jnp.where(self.gait_boxes, (jnp.abs(up) * size).sum(-1), size[:, 0] + size[:, 1] * jnp.abs(up[..., 2]))
            foot_pos, foot_rot = P[..., 1:3, :], Rs[..., 1:3, :, :]
            slip = gait_rewards.predicted_foot_slip(foot_pos, foot_rot, V[..., 1:3, :], W[..., 1:3, :], self.foot_corners)
            com_velocity = V[..., 1:3, :] + jnp.cross(W[..., 1:3, :], mv3(foot_rot, self.foot_com))
            feet = gait_rewards.FootGeometry(foot_pos, foot_rot, center - extent, slip, com_velocity[..., 2])
        return View(env.Kin(Rs[..., 0, :, :], V[..., 0, :], W[..., 0, :], foot_v), p[..., 2],
                    clearance(self.sole_local, self.sole_frame, self.sole_radius), clearance(self.coll_local, self.coll_frame, self.coll_radius), feet)  # fmt: skip

    def soft_contact(self, soles):
        """(N, 2) soft foot-ground contact of the imagined states: the lowest sole point of each foot near the floor."""
        n = len(self.sole_frame) // 2
        low = jnp.stack([soles[..., :n].min(-1), soles[..., n:].min(-1)], -1)
        return jax.nn.sigmoid((env.FOOT_CONTACT_H - low) / env.FOOT_CONTACT_SOFT)

    def key_positions(self, p, R, q):
        """World positions (.., 4, 3) of the motion prior's key bodies (amp.KEY_BODIES)."""
        return self.kin_amp.forward(p, R, q)[0]

    def sample_wm_command(self, key, n, *, stand_share=.05):
        """Draw a command and its holding time for a physical or imagined environment."""
        interval = (jax.random.uniform(jax.random.fold_in(key, 1), (n,), minval=2., maxval=4.)
                    if self.wm_floor_locomotion else jnp.full(n, 10.))
        if self.wm_range_commands:
            # Use the final walk/run range of Motion Train's Zenbot sampler.
            return source_gait.sample_command(key, n, stage=2, stand_share=stand_share)._replace(time_left=interval)
        velocity = (jnp.zeros((n, 3), jnp.float32) if self.zenbot_wm_getup_only else
                    jax.random.choice(key, jnp.asarray(self.wm_command_speeds), (n,)))
        return env.Cmd(velocity, jnp.zeros(n), jnp.zeros(n, bool), jnp.all(velocity == 0., -1), interval)

    def tick_wm_command(self, cmd, key, state, upright, recovery_left):
        """At the regular command boundary, give an already walking robot more chances to stop."""
        stand_share = .05
        if self.wm_range_commands:
            forward_speed = gait_rewards.base_velocity(state, self.root_com)[:, 0]
            walking = upright & (recovery_left == 0) & (cmd.vel[:, 0] > 0.) & (forward_speed > .45)
            stand_share = jnp.where(walking, GAIT_TRANSITION_SHARE, stand_share)
        return env.tick(cmd, key, sample=lambda k: self.sample_wm_command(k, cmd.vel.shape[0], stand_share=stand_share))

    def wm_velocity_command(self, cmd, rotation, upright, recovery_left=None):
        """Return the velocity request visible to this task's actor."""
        if self.zenbot_wm_getup_only:
            return jnp.zeros_like(cmd.vel)
        if self.wm_range_commands:
            return source_gait.command(cmd, rotation)
        if self.wm_floor_locomotion:
            # cmd.vel and c_obs are separate donated leaves of the simulator state.
            return jnp.copy(cmd.vel)
        if not self.zenbot_motion_train_wm:
            return self.command(cmd, rotation)
        ready = upright if recovery_left is None else upright & (recovery_left <= 0)
        return jnp.where(ready[..., None], cmd.vel, 0.)

    def sample_command(self, key, stage=0):
        if self.zenbot_motion_train_wm:
            return self.sample_wm_command(key, self.N)
        if self.zenbot_source_gait:
            return source_gait.sample_command(key, self.N, stage)
        if self.zenbot_command_gait:
            return zenbot.sample_gait_command(key, self.N, self.zenbot_directional_gait)
        return env.sample_cmd(key, self.N, self.cmd_ranges)

    def command(self, command, rot):
        if self.zenbot_source_gait:
            return source_gait.command(command, rot)
        if self.zenbot_command_gait:
            return command.vel
        return jnp.zeros_like(command.vel) if self.recovery else env.command(command, rot)

    def recovery_command(self, cmd, down_start):
        """Use a standing command throughout a G1 phase-AMP episode drawn from the floor bank."""
        if not self.g1_phase_amp:
            return cmd
        return cmd._replace(vel=jnp.where(down_start[:, None], 0., cmd.vel),
                            is_heading=cmd.is_heading & ~down_start, is_standing=cmd.is_standing | down_start)

    def recovery_timeout(self, clip, start, elapsed):
        """End floor-born phase-AMP rollouts one second after the native clip finishes."""
        if not self.g1_phase_amp:
            return jnp.zeros_like(elapsed, dtype=bool)
        length = self.reference.lengths[jnp.maximum(clip, 0)]
        return (clip >= 0) & (start >= 0) & (start < length) & (elapsed >= length - start + round(1. / env.DT))

    def obs_frame(self, state, action, command, key):
        if self.recovery:
            rot, ang, key_pos = self.recovery_features(state)
            x = jnp.concatenate((ang, rot, state[4], state[5], action, key_pos), -1)
            if self.zenbot_command_gait:
                x = jnp.concatenate((x, command), axis=-1)
            if self.noisy_obs and key is not None:
                x += self.obs_noise * jax.random.uniform(key, x.shape, minval=-1., maxval=1.)
            return x
        return env.obs_frame(state, action, command, key if self.noisy_obs else None, self.robot.DEFAULT_Q, self.obs_noise)

    def motion_train_sensors(self, state, rotation, command, fresh, key):
        # The free joint stores angular velocity in the body frame. Gravity is world -Z in that frame.
        qvel = state.data.qvel
        imu = jnp.concatenate((qvel[:, 3:6], -rotation[:, 2, :]), axis=-1)
        return advance_sensors(state.motion_train_sensors, imu, command, state.data.qpos[:, 7:],
                               qvel[:, 6:], state.a_last, fresh, jax.random.fold_in(key, 234), self.noisy_obs)

    def push_actor(self, history, frame, fresh=None):
        if self.robot is g1_getup:
            return g1_getup.push_history(history, frame, g1_getup.ACTOR_TERMS, fresh)
        if history is None:
            return jnp.tile(frame, (1, self.history))
        return env.push_frame(history, frame, fresh)

    def reference_velocity(self, qpos, qvel):
        """IsaacLab writes the reference linear velocity at the pelvis COM. MuJoCo stores it at the free joint."""
        if self.robot is not g1_getup:
            return qvel
        rot = quat_to_mat(qpos[..., 3:7])
        shift = mv3(rot, jnp.cross(qvel[..., 3:6], self.root_com))
        return qvel.at[..., :3].add(-shift)

    def push_critic(self, history, state, action, fresh=None):
        pos, rotation, velocity, _, q, qd = state
        rot6, gyro, points = self.recovery_features(state)
        velocity_b = mv3(tr3(rotation), velocity) + jnp.cross(gyro, self.root_com)
        frame = jnp.concatenate((velocity_b, gyro, rot6, pos[:, 2:3], q, qd, action, points), -1)
        return g1_getup.push_history(history, frame, g1_getup.CRITIC_TERMS, fresh)

    def estimated_torque(self, state, target):
        if self.robot is g1_getup:
            return g1_getup.actuator_torque(state[4], state[5], target)
        return zenbot.actuator_ctrl(state[4], state[5], target[..., -self.J:])

    def is_up(self, height, rotation):
        return env.is_up(height, rotation, self.robot.UP_HEIGHT)

    def next_upright(self, previous, height, rotation):
        if self.mixed:
            # A crouch or single low step should not switch the motion prior.
            still_up = (height > MIXED_STAY_HEIGHT) & (rotation[..., 2, 2] > MIXED_STAY_UP_COS)
            return (previous & still_up) | self.is_up(height, rotation)
        return previous | self.is_up(height, rotation)

    def wm_fall(self, previous_up, up, height, rotation):
        """Hard stance-loss event with a local gradient at the actual hysteresis boundaries."""
        hard = (previous_up & ~up).astype(height.dtype)
        above = jax.nn.sigmoid((height - MIXED_STAY_HEIGHT) / .02)
        aligned = jax.nn.sigmoid((rotation[..., 2, 2] - MIXED_STAY_UP_COS) / .04)
        soft = previous_up * (1. - above * aligned)
        return hard + (soft - jax.lax.stop_gradient(soft))

    def next_recovery_interval(self, remaining, previous_up, up):
        # Bouncing through the upright threshold during a rise does not restart the clock.
        fall = previous_up & ~up & (remaining == 0)
        return jnp.where(fall, self.recovery_amp_steps, jnp.maximum(remaining - 1, 0))

    def floor_gait_interval(self, previous, upright, support):
        """Release the gait reward after standing on the feet; retain it through running flight until a fall."""
        ready = upright & ((previous == 0) | (support >= 1.))
        return jnp.where(ready, 0, self.recovery_amp_steps)

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

    def advance_floor_stand(self, hold, posture, loaded, clip, step):
        if self.reference_reward:
            source = jnp.maximum(clip, 0)
            pose = self.reference.at(source, step)
            # Start counting foot support once the reference has also reached standing.
            ready = (clip < 0) | (step >= self.reference.lengths[source]) | zenbot.standing_posture(pose[..., 0], pose[..., 3])
            posture, loaded = posture & ready, loaded & ready
        return hold.advance(posture, loaded)

    def reward_terms(self, *args, torque=None, joint_acc=None, clip=None, step=None, reference_start=None, reference_yaw=None,
                     recovery_active=False, recovery_deadline=False, stand_blend=1., reference_rate_weight=1.):
        if self.recovery:
            terms = zenbot.reward_terms(args[0], args[1], args[3], args[4], torque, self.q_soft, env.DT,
                                        up_height=self.robot.UP_HEIGHT, target_height=self.robot.RISE_HEIGHT,
                                        default_q=self.robot.DEFAULT_Q, joint_acc=joint_acc,
                                        com_offset=self.root_com if self.robot is g1_getup else None)
            if self.reference is not None:
                # The named G1 task resets at an arbitrary frame in the clip. Keep
                # the final standing pose as the target for the rest of the episode.
                if self.zenbot_scratch_resets:
                    frame = reference_start + step
                    score = self.reference.score(self.reference_pose(args[1]), jnp.maximum(clip, 0), frame)
                    terms["motion_reference"] = jnp.where(clip >= 0, score, 0.)
                elif self.track_getup:
                    frame = reference_start + step
                    end = self.reference.lengths[clip] - 1
                    score = self.reference.score(self.reference_pose(args[1]), clip, jnp.minimum(frame, end))
                    terms["reference_pose_score"] = score
                    terms["motion_reference"] = score * jnp.where(frame <= end, 1., self.reference_hold_scale)
                else:
                    terms["motion_reference"] = self.reference.score(self.reference_pose(args[1]), clip, step)
            return terms
        terms = env.reward_terms(*args, wrist_idx=self.robot.WRIST_IDX, default_q=self.robot.DEFAULT_Q,
                                 rise_height=self.robot.RISE_HEIGHT)
        if self.wm_floor_amp:
            floor_terms = {"stand": stand_blend * zenbot.floor_stand_score(args[1], args[8]),
                           "action_rate": terms["action_rate"]}
            if self.wm_lift_gate:
                before = zenbot.lift_progress(args[0][0][..., 2], self.wm_lift_gate)
                after = zenbot.lift_progress(args[1][0][..., 2], self.wm_lift_gate)
                floor_terms["lift"] = (after - before) / env.DT
                # Compare the quiet stance with the standing AMP clip once the feet support it.
                joint_error = jnp.square((args[1][4] - self.floor_stand_joints) / .5).mean(axis=-1)
                floor_terms["stand"] /= 1. + joint_error
            if self.wm_floor_locomotion:
                cmd = args[5]
                ready = ~jnp.asarray(recovery_active, dtype=bool)
                moving = ready & jnp.any(cmd != 0., axis=-1)
                actual = gait_rewards.base_velocity(args[1], self.root_com)
                error_sq = (actual - cmd)**2
                upright = ready * jnp.clip(args[1][1][..., 2, 2], 0., .7) / .7
                floor_terms["track_lin_vel_xy_exp"] = upright * jnp.exp(-error_sq[..., :2].sum(-1) / .5**2)
                floor_terms["track_ang_vel_z_exp"] = upright * jnp.exp(-error_sq[..., 2] / .5**2)
                if self.wm_posture_and_tracking:
                    floor_terms.update(gait_rewards.posture_terms(args[1], cmd, ready, actual))
                floor_terms["stand"] = jnp.where(moving, 0., floor_terms["stand"])
                if self.wm_lift_gate:
                    floor_terms["lift"] = jnp.where(recovery_active, floor_terms["lift"], 0.)
            if self.reference_reward:
                pose = self.reference_pose(args[1])
                source = jnp.maximum(clip, 0)
                active = (clip >= 0) & (step < self.reference.lengths[source])
                prev, curr = args[:2]
                rates = self.reference.motion_rate(curr[0], curr[1], curr[4], prev[0], prev[4])
                moving = self.reference.score(pose, source, step, smooth_l1=True, rates=rates,
                                              rate_weight=reference_rate_weight)
                stand = self.standing_reference
                # A late recovery still needs to move after the reference has ended.
                stationary = stand.score(pose, jnp.zeros_like(clip), stand.lengths[0] // 2,
                                         smooth_l1=True, rates=rates, rate_weight=reference_rate_weight * stand_blend)
                # Once the source has finished, keep its neutral standing target even for a late rise.
                floor_terms["motion_reference"] = jnp.where(clip >= 0, jnp.where(active, moving, stationary), 0.)
            return floor_terms
        if self.zenbot_motion_train_wm:
            pos, rotation, velocity, angular = args[1][:4]
            command, recovering = args[5:7]
            body_velocity = mv3(tr3(yaw_mat(rotation)), velocity)
            track = ~jnp.asarray(recovering) & ~jnp.asarray(recovery_active)
            lin_error = jnp.square(body_velocity[..., :2] - command[..., :2]).sum(axis=-1)
            yaw_error = jnp.square(angular[..., 2] - command[..., 2])
            terms["track_lin"] = jnp.where(track, jnp.exp(-lin_error / .5**2), 0.)
            terms["track_ang"] = jnp.where(track, jnp.exp(-yaw_error / .5**2), 0.)
            terms["rise"] *= .25
            unbalanced = 1. - env.soft_up(pos[..., 2], rotation, self.robot.UP_HEIGHT)
            terms["recovery_timeout_cost"] = jnp.where(recovery_deadline, unbalanced, 0.)
            if self.zenbot_wm_reference_training:
                terms["motion_reference"] = self.wm_reference.tracked_score(
                    self.reference_pose(args[1]), clip, reference_start, step, recovery_active, args[7])
            return {name: terms[name] for name in self.terms}
        if self.recovery_amp_steps:
            # Keep the one-off gait fall penalty. Posture and timing during recovery come from AMP.
            terms["rise"] = jnp.where(recovery_active, 0., .25 * terms["rise"])
            pos, rotation = args[1][:2]
            unbalanced = 1. - env.soft_up(pos[..., 2], rotation, self.robot.UP_HEIGHT)
            terms["recovery_timeout_cost"] = jnp.where(recovery_deadline, unbalanced, 0.)
            for name in ("track_lin", "track_ang", "body_ang_vel_xy"):
                terms[name] = jnp.where(recovery_active, 0., terms[name])
        if self.legged_support:
            native = zenbot.reward_terms(args[0], args[1], args[3], args[4], torque, self.q_soft, env.DT,
                                         joint_acc=joint_acc, com_offset=self.root_com)
            for name in env.TERMS:
                if name != "termination":
                    terms[name] = jnp.where(recovery_active, 0., terms[name])
            terms.update({name: jnp.where(recovery_active, value, 0.) for name, value in native.items()})
        if self.g1_phase_amp:
            active = self.reference.active(clip, reference_start, step)
            for name in ("track_lin", "track_ang", "body_ang_vel_xy"):
                terms[name] = jnp.where(active, 0., terms[name])
        if self.g1_reference and self.reference_reward:
            state = args[1]
            points = self.key_positions(state[0], state[1], state[4])
            terms["motion_reference"] = self.reference.score(state, points, clip, reference_start, reference_yaw, step)
        return terms

    def amp_support_frame(self, state, *, with_gait=False, with_native=False):
        """Exact collider clearances for unpaired expert and policy windows, including inverted poses."""
        p, R, _, _, q, _ = state
        origins, rotations = self.kin_amp.forward(p, R, q)
        body_up = rotations[..., 2, :]
        center_z = origins[..., 2] + (body_up * self.support_pos).sum(-1)
        geom_up = jnp.einsum("...bi,bij->...bj", body_up, self.support_rot)
        sole_extent = (jnp.abs(geom_up[..., :2, :]) * self.support_size[:2]).sum(-1)
        capsule_extent = self.support_size[2:, 0] + self.support_size[2:, 1] * jnp.abs(geom_up[..., 2:, 2])
        clearance = center_z - jnp.concatenate((sole_extent, capsule_extent), axis=-1)
        local = mv3(tr3(yaw_mat(R))[..., None, :, :], origins - p[..., None, :])
        support = jnp.concatenate((p[..., 2:3], -R[..., 2, :], local[..., :2].reshape(*p.shape[:-1], 12), clearance), -1)
        if with_native:
            rot, gyro, body_points = self.recovery_features(state, origins)
            joints = self.amp_joints if self.zenbot_getup_stand else slice(None)
            support = jnp.concatenate((rot, gyro, state[4][..., joints], state[5][..., joints], body_points, support), -1)
        if with_gait:
            return jnp.concatenate((self._gait_amp_frame(state, origins), support), -1)
        return support

    def amp_frame(self, state, key_pos=None):
        if self.zenbot_getup_stand:
            frame = self.amp_support_frame(state, with_native=True)
            # Retargeted clips can penetrate the plane. Treat contact and small
            # clearance identically for both mocap and simulated motion.
            return frame.at[..., -6:].set(jnp.maximum(frame[..., -6:] - .025, 0.))
        if self.recovery:
            rot, ang, points = self.recovery_features(state, key_pos)
            return jnp.concatenate((rot, ang, state[4][..., self.amp_joints], state[5][..., self.amp_joints], points), -1)
        return self._gait_amp_frame(state, self.key_positions(state[0], state[1], state[4]))

    def _gait_amp_frame(self, state, points):
        frame = amp.frame(state, points, self.amp_joints)
        if self.mixed:
            return jnp.concatenate((frame, state[0][..., 2:3], mv3(tr3(yaw_mat(state[1])), state[2])), -1)
        return frame

    def control_targets(self, action, bias, history, context, *, elapsed=None):
        if self.robot is not zenbot:
            return self.robot.DEFAULT_Q + self.robot.ACTION_SCALE * action - bias
        # A 3–7 physics-step delay can reach either of the previous two policy
        # commands. The previous actions precede the 18 key-body coordinates.
        frames = history.reshape(*history.shape[:-1], self.history, self.frame_dim)
        start = 9 + 2 * self.J
        previous, older = frames[..., -1, start:start + self.J], frames[..., -2, start:start + self.J]
        from .zenbot_wm_sensors import pd_targets
        return pd_targets(action, previous, older, bias, context, elapsed=elapsed)

    def context(self, s: State):
        """Foot friction, torso CoM offset and (for Zenbot) group delays and assistance. Bias enters via the target."""
        m, w = self.m, s.world
        base = jnp.concatenate((w.geom_friction[:, self.foot_geoms[0], :1], w.body_ipos[:, self.torso] - m.body_ipos[self.torso]), -1)
        if self.robot is not env:
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
        mass, subtree, inertia = m.body_mass, m.body_subtreemass, None
        if self.robot is not env:
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
            if self.robot is g1_getup:
                # IsaacLab scales the randomized body's inertia by its new-to-old mass ratio.
                ratio = mass[:, self.torso] / m.body_mass[self.torso]
                inertia = tile(m.body_inertia).at[:, self.torso].multiply(ratio[:, None])
                # Only the robot is randomized. All robot shapes draw from one table of 64 materials.
                if self.randomize_world:
                    kb, ks = jax.random.split(k_mu)
                    materials = jax.random.uniform(kb, (64,), minval=.3, maxval=1.)
                    ids = jax.random.randint(ks, (N, m.ngeom - 1), 0, 64)
                    fr = tile(m.geom_friction).at[:, 1:, 0].set(materials[ids])
        if self.physical_floor_starts is not None:
            # The physical source trajectories were solved on the nominal Zenbot. Give most
            # recovery worlds that same contact model, alongside randomized recovery worlds.
            number = jnp.arange(N)
            nominal = (number < round(self.reset_floor * N)) & (number % 4 != 3)
            fr = jnp.where(nominal[:, None, None], m.geom_friction, fr)
            mass = jnp.where(nominal[:, None], m.body_mass, mass)
            subtree = jnp.where(nominal[:, None], m.body_subtreemass, subtree)
        return World(ipos, fr, bias, getattr(self, "mx", None), self.walk, mass, subtree, self.floor, inertia)

    def _push_interval(self, key):
        interval = (2., 4.) if self.fall_push_speed else env.PUSH_INTERVAL
        u = jax.random.uniform(key, (self.N,), minval=interval[0], maxval=interval[1])
        return (u / env.DT).astype(jnp.int32)

    def _scratch_start(self, key, world, command):
        """Stratify parallel worlds so early falls cannot starve the gait training batch."""
        gait_key, clip_key, phase_key, opening_key, noise_key = jax.random.split(key, 5)
        walk, recovery = world.walk, world.floor
        targets = jnp.asarray(zenbot.DIRECTIONAL_COMMANDS[:len(walk["counts"])])
        group = jnp.sum((command[:, None] - targets) ** 2, axis=-1).argmin(axis=-1)
        row = walk["offsets"][group] + jax.random.randint(gait_key, (self.N,), 0, walk["counts"][group])
        clip = jax.random.randint(clip_key, (self.N,), 0, len(recovery["lengths"]))
        minimum_phase = recovery["minimum_phase"][clip] if "minimum_phase" in recovery else 0
        phase = jax.random.randint(phase_key, (self.N,), minimum_phase, recovery["down_counts"][clip])
        phase = jnp.where(jax.random.uniform(opening_key, (self.N,)) < self.zenbot_phase_resets, phase, 0)
        frame = recovery["offsets"][clip] + phase
        floor = jnp.arange(self.N) < round(self.reset_floor * self.N)
        z = jnp.where(floor, recovery["z"][frame] + zenbot.RESET_LIFT, walk["z"][row])
        quat, v, w, q, qd = (jnp.where(floor[:, None], recovery[name][frame], walk[name][row])
                             for name in ("quat", "v", "w", "q", "qd"))
        # Match the native get-up evaluation's slight joint, roll/pitch and horizontal velocity perturbations.
        noise = jax.random.uniform(noise_key, (self.N, self.J + 4), minval=-1., maxval=1.) * floor[:, None]
        q = q + np.deg2rad(2.) * noise[:, :self.J]
        roll, pitch = (np.deg2rad(1.) * noise[:, self.J:self.J+2]).T
        tilt = jnp.stack((jnp.cos(roll)*jnp.cos(pitch), jnp.sin(roll)*jnp.cos(pitch),
                          jnp.cos(roll)*jnp.sin(pitch), -jnp.sin(roll)*jnp.sin(pitch)), axis=-1)
        quat = resets.qmul(quat, tilt)
        v = v.at[:, :2].add(.07 * noise[:, -2:])
        qpos = jnp.concatenate((jnp.zeros((self.N, 2)), z[:, None], quat, q), axis=-1)
        qvel = jnp.concatenate((v, mv3(tr3(quat_to_mat(quat)), w), qd), axis=-1)
        return (qpos, qvel, ~floor), (jnp.where(floor, clip, -1), jnp.where(floor, phase, 0))

    def _wm_floor_start(self, key, bank):
        """Draw fallen poses and optional intermediate recovery states for a physical reset."""
        phase_key, opening_key, noise_key, mix_key = jax.random.split(key, 4)
        opening = bank["openings"]
        row = jax.random.randint(opening_key, (self.N,), 0, len(opening["z"]))
        if self.wm_floor_amp and self.support_rsi:
            first_count = len(opening["z"])
            opening = bank["phases"]
            phase_row = jax.random.randint(phase_key, (self.N,), first_count, len(opening["z"]))
            row = jnp.where(jax.random.bernoulli(mix_key, self.support_rsi, (self.N,)), phase_row, row)
        noise = jax.random.uniform(noise_key, (self.N, self.J + 4), minval=-1., maxval=1.)
        # Same roll/pitch composition, joint perturbation, clearance and horizontal drift as source_getup.
        roll, pitch = (np.deg2rad(1.) * noise[:, self.J:self.J + 2]).T
        tilt = jnp.stack((jnp.cos(roll)*jnp.cos(pitch), jnp.sin(roll)*jnp.cos(pitch),
                          jnp.cos(roll)*jnp.sin(pitch), -jnp.sin(roll)*jnp.sin(pitch)), axis=-1)
        quat = resets.qmul(opening["quat"][row], tilt)
        vel = jnp.zeros((self.N, 3)).at[:, :2].set(.07 * noise[:, -2:])
        q = opening["q"][row] + np.deg2rad(2.) * noise[:, :self.J]
        xy = jnp.zeros((self.N, 2))
        cold_pos = jnp.concatenate((xy, opening["z"][row, None] + zenbot.RESET_LIFT,
                                    quat / jnp.linalg.norm(quat, axis=-1, keepdims=True), q), axis=-1)
        cold_vel = jnp.concatenate((vel, jnp.zeros_like(vel), jnp.zeros_like(q)), axis=-1)
        if self.wm_floor_amp:
            if self.wm_floor_locomotion and self.support_rsi:
                # Resume intermediate motion from its recorded rates, including the early hand support.
                body_gyro = mv3(tr3(quat_to_mat(quat)), opening["w"][row])
                rates = jnp.concatenate((opening["v"][row], body_gyro, opening["qd"][row]), -1)
                cold_vel += jnp.where(opening["frame"][row, None] > 0, rates, 0.)
            return (cold_pos, cold_vel), (opening["clip"][row], opening["frame"][row], jnp.zeros(self.N))
        mask = (self.wm_start_quantiles < .5 if self.wm_start_quantiles is not None else
                jax.random.bernoulli(mix_key, .5, (self.N,)))
        phases = bank["phases"]
        if self.wm_physical_resets:
            clip = opening["clip"][row]
            frame = jax.random.randint(phase_key, (self.N,), 1, phases["limits"][clip])
            if self.wm_stance_start_share:
                # Half of all resets are still cold openings. Draw supported poses from the verified final two seconds.
                stance_key, time_key = jax.random.split(jax.random.fold_in(phase_key, 1))
                stance = (self.wm_start_quantiles >= 1. - self.wm_stance_start_share if self.wm_start_quantiles is not None else
                          jax.random.bernoulli(stance_key, 2. * self.wm_stance_start_share, (self.N,)))
                last = self.wm_reference.lengths[clip] - 1
                standing_frame = last - jax.random.randint(time_key, (self.N,), 0, round(2. / env.DT))
                frame = jnp.where(stance, standing_frame, frame)
            index = phases["offsets"][clip] + frame
            phase_pos, phase_vel = phases["qpos"][index], phases["qvel"][index]
            yaw = jnp.zeros(self.N)
        else:
            phase, (clip, frame, yaw) = resets.states(phase_key, phases, self.N, "floor",
                                                      self.q_soft[:, 0], self.q_soft[:, 1], return_source=True)
            z, quat, v, w, joints, joint_velocity, _ = phase
            phase_pos = jnp.concatenate((xy, z[:, None], quat, joints), -1)
            phase_vel = jnp.concatenate((v, mv3(tr3(quat_to_mat(quat)), w), joint_velocity), -1)
        state = jnp.where(mask[:, None], cold_pos, phase_pos), jnp.where(mask[:, None], cold_vel, phase_vel)
        source = (jnp.where(mask, opening["clip"][row], clip), jnp.where(mask, 0, frame), jnp.where(mask, 0., yaw))
        return state, source

    def _fresh(self, key, world: World, clip=None, return_source=False, command=None):
        """Fresh start states for every env (`_reset` applies them where an episode ended): a walking frame (no noise, the clip's
        heading), or with probability reset_floor a not-up frame of a get-up clip (random heading, roll / pitch / joint / velocity
        noise); prob_shove of the starts that are up get a kick of the velocities."""
        if self.zenbot_wm_getup_only:
            (qpos, qvel), source = self._wm_floor_start(key, world.floor)
            state = qpos, qvel, self.is_up(qpos[:, 2], quat_to_mat(qpos[:, 3:7]))
            return (state, source[:2]) if return_source else state
        if self.zenbot_scratch_resets:
            if command is None:
                command = self.sample_command(jax.random.fold_in(key, 10)).vel
            state, source = self._scratch_start(key, world, command)
            if self.zenbot_source_gait:
                gait = source[0] < 0
                q, v = source_gait.stand_reset(jax.random.fold_in(key, 508), self.N)
                state = (jnp.where(gait[:, None], q, state[0]), jnp.where(gait[:, None], v, state[1]), state[2])
            return (state, source) if return_source else state
        N, J = self.N, self.J
        kw, kf, ku, ks, kk1, kk2 = jax.random.split(key, 6)
        wb = world.walk
        cold_forward = False
        if self.robot is g1_getup and not self.track_getup:
            frame = {k: v[:, 0] for k, v in g1_getup.sample_reference(wb, kw, N, window=1, fetch_steps=10).items()}
            qpos = jnp.concatenate((frame["pos"], frame["quat"], frame["q"]), -1)
            qvel = jnp.concatenate((frame["v"], mv3(tr3(quat_to_mat(frame["quat"])), frame["w"]), frame["qd"]), -1)
        else:
            if self.track_getup:  # the opt-in dense reward uses the stored 50 Hz clip/frame indices
                track = jax.random.randint(kw, (N,), 0, len(wb["lengths"]))
                start = jax.random.randint(kf, (N,), 0, wb["lengths"][track] - 10)
                if self.zenbot_phase_resets:
                    # Balance transitions across the motion with complete get-ups from the floor.
                    start = jnp.where(jax.random.uniform(ku, (N,)) < self.zenbot_phase_resets, start, 0)
                i = wb["offsets"][track] + start
            else:
                i = jax.random.randint(kw, (N,), 0, len(wb["z"])) if clip is None else clip
                if self.zenbot_wm_reference_training or self.wm_floor_locomotion:
                    if command is None:
                        command = self.sample_wm_command(jax.random.fold_in(key, 10), N).vel
                    group = reference_command_index(command, self.wm_command_speeds)
                    if self.wm_range_commands:
                        forward = command[:, 0] > jnp.abs(command[:, 1])
                        cold_forward = forward & jax.random.bernoulli(jax.random.fold_in(kw, 2), GAIT_TRANSITION_SHARE, (N,))
                        group = jnp.where(cold_forward, 0, group)  # standing reference, keep the forward command
                    by_command = wb["command_rows"]
                    row = by_command["offsets"][group] + jax.random.randint(
                        jax.random.fold_in(kw, 1), (N,), 0, by_command["counts"][group])
                    i = by_command["row"][row] if self.wm_floor_locomotion else jnp.where(self.wm_gait_worlds, by_command["row"][row], i)
            q = wb["q"][i] if self.recovery or self.wm_floor_locomotion else jnp.clip(wb["q"][i], self.q_soft[:, 0], self.q_soft[:, 1])
            if self.zenbot_wm_reference_training:
                q = jnp.where(self.wm_gait_worlds[:, None], wb["q"][i], q)
            xy = wb["xy"][i] if self.track_getup else jnp.zeros((N, 2))
            qpos = jnp.concatenate((xy, wb["z"][i][:, None], wb["quat"][i], q), -1)
            qvel = jnp.concatenate((wb["v"][i], mv3(tr3(quat_to_mat(wb["quat"][i])), wb["w"][i]), wb["qd"][i]), -1)
        if self.recovery and not self.zenbot_command_gait:
            qpos = qpos.at[:, 2].add(self.robot.RESET_LIFT)
            state = qpos, self.reference_velocity(qpos, qvel), self.is_up(qpos[:, 2], quat_to_mat(qpos[:, 3:7]))
            if self.track_getup and return_source:
                return state, (track, i - wb["offsets"][track])
            return state
        if self.reset_floor > 0:
            if self.zenbot_motion_train_wm:
                (fpos, fvel), (source_clip, source_frame, source_yaw) = self._wm_floor_start(kf, world.floor)
            elif self.legged_support or self.zenbot_command_gait:
                # LeggedLab centers the opening pose in XY and preserves its orientation and velocities.
                bank = world.floor
                if self.support_rsi:
                    opening_key, state_key, choose_key = jax.random.split(kf, 3)
                    opening = jax.random.randint(opening_key, (N,), 0, self.floor_openings)
                    support = jax.random.randint(state_key, (N,), self.floor_openings, len(bank["z"]))
                    i = jnp.where(jax.random.bernoulli(choose_key, self.support_rsi, (N,)), support, opening)
                else:
                    i = jax.random.randint(kf, (N,), 0, len(bank["z"]))
                z, quat = bank["z"][i] + zenbot.RESET_LIFT, bank["quat"][i]
                v, w, fq, fqd = (bank[name][i] for name in ("v", "w", "q", "qd"))
            else:
                sample = resets.states(kf, world.floor, N, "floor", self.q_soft[:, 0], self.q_soft[:, 1], return_source=return_source)
                if return_source:
                    sample, (source_clip, source_frame, source_yaw) = sample
                z, quat, v, w, fq, fqd, _ = sample
            if not self.zenbot_motion_train_wm:
                fpos = jnp.concatenate((jnp.zeros((N, 2)), z[:, None], quat, fq), -1)
                fvel = jnp.concatenate((v, mv3(tr3(quat_to_mat(quat)), w), fqd), -1)
            floor = jax.random.uniform(ku, (N,)) < self.reset_floor
            if self.zenbot_wm_reference_training or self.wm_floor_locomotion:
                floor &= ~self.wm_gait_worlds
            qpos, qvel = jnp.where(floor[:, None], fpos, qpos), jnp.where(floor[:, None], fvel, qvel)
        was_up = self.is_up(qpos[:, 2], quat_to_mat(qpos[:, 3:7]))
        if self.prob_shove > 0:
            shove = was_up & (jax.random.uniform(ks, (N,)) < self.prob_shove)
            if self.g1_phase_amp:
                shove &= ~floor
            if self.zenbot_wm_reference_training:
                shove &= ~self.wm_gait_worlds
            kick = jnp.concatenate((jax.random.uniform(kk1, (N, 6), minval=-1.0, maxval=1.0) * jnp.asarray(env.SHOVE_ROOT),
                                    env.SHOVE_JOINT * jax.random.uniform(kk2, (N, J), minval=-1.0, maxval=1.0)), -1)  # fmt: skip
            qvel = qvel + jnp.where(shove[:, None], kick, 0.0)  # (the root's angular part is in the body frame, as MuJoCo's)
        if self.wm_range_commands:
            qvel = jnp.where((cold_forward & ~floor)[:, None], 0., qvel)
        if return_source:
            if self.zenbot_wm_reference_training:
                empty = jnp.zeros(N, jnp.int32)
                source = ((jnp.where(floor, source_clip, -1), jnp.where(floor, source_frame, 0))
                          if self.reset_floor > 0 else (empty - 1, empty))
                return (qpos, qvel, was_up), source
            source = (jnp.where(floor, source_clip, -1), source_frame, source_yaw)
            return (qpos, qvel, was_up), source
        return qpos, qvel, was_up

    def _reset(self, s: State, done, key) -> State:
        """Reset the envs where `done`: start state (`_fresh`), command drawn anew, last action zero, push schedule restarted."""
        kf, kc, kp = jax.random.split(key, 3)
        sampled_cmd = (self.sample_command(kc, s.gait_curriculum.stage) if self.zenbot_source_gait else
                       self.sample_command(kc))
        clip = None
        if self.g1_reference:
            (qpos, qvel, was_up), (clip, start, yaw) = self._fresh(kf, s.world, return_source=True)
            s = s._replace(reference_clip=jnp.where(done, clip, s.reference_clip),
                           reference_start=jnp.where(done, start, s.reference_start),
                           reference_yaw=jnp.where(done, yaw, s.reference_yaw))
        elif self.wm_floor_amp and self.reference_reward:
            (qpos, qvel, was_up), (clip, _) = self._fresh(kf, s.world, return_source=True)
            s = s._replace(reference_clip=jnp.where(done, clip, s.reference_clip))
        elif self.track_getup or self.zenbot_wm_reference_training:
            if self.zenbot_scratch_resets or self.zenbot_wm_reference_training:
                (qpos, qvel, was_up), (clip, start) = self._fresh(kf, s.world, return_source=True, command=sampled_cmd.vel)
            else:
                (qpos, qvel, was_up), (clip, start) = self._fresh(kf, s.world, return_source=True)
            s = s._replace(reference_clip=jnp.where(done, clip, s.reference_clip),
                           reference_start=jnp.where(done, start, s.reference_start))
        else:
            if self.reference is not None:
                clip = jax.random.randint(jax.random.split(kf, 6)[0], (self.N,), 0, len(self.reference.files))
                s = s._replace(reference_clip=jnp.where(done, clip, s.reference_clip))
            first_command = sampled_cmd.vel if self.wm_floor_locomotion else None
            qpos, qvel, was_up = self._fresh(kf, s.world, clip, command=first_command)
        if self.physical_gait_starts is not None:
            window, qpos, qvel, sampled_cmd, window_delays = self.physical_gait_starts.reset(
                s, jax.random.fold_in(key, 759), done, clip < 0, qpos, qvel, sampled_cmd)
            s = s._replace(gait_window=window)
        if s.episode_limit is not None:
            limit = jnp.where(clip < 0, source_gait.GAIT_STEPS, self.episode_steps) if self.zenbot_source_gait else self.episode_steps
            s = s._replace(episode_limit=jnp.where(done, limit, s.episode_limit))
        if self.zenbot_scratch_curriculum:
            s = s._replace(first_supported_stand=jnp.where(done, -1, s.first_supported_stand))
        on = done[:, None]
        d = s.data.replace(qpos=jnp.where(on, qpos, s.data.qpos), qvel=jnp.where(on, qvel, s.data.qvel), qacc_warmstart=jnp.where(on, 0.0, s.data.qacc_warmstart))
        down = jnp.where(done, clip >= 0 if self.g1_phase_amp else ~was_up, s.down)
        if self.floor_stop_command_share:
            # Practice standing after a rise as well as rising under a live movement command.
            stop = ~was_up & jax.random.bernoulli(jax.random.fold_in(kc, 22), self.floor_stop_command_share, (self.N,))
            sampled_cmd = sampled_cmd._replace(
                vel=jnp.where(stop[:, None], 0., sampled_cmd.vel),
                is_standing=sampled_cmd.is_standing | stop,
                time_left=jnp.where(stop, (self.episode_steps + 1) * env.DT, sampled_cmd.time_left),
            )
        if self.recovery_amp_steps:
            s = s._replace(recovery_left=jnp.where(done, jnp.where(was_up, 0, self.recovery_amp_steps), s.recovery_left))
        if s.supported_hold is not None:
            s = s._replace(supported_hold=s.supported_hold.reset(done))
        if s.step_history is not None:
            s = s._replace(step_history=s.step_history.reset(done))
        if self.zenbot_command_gait:
            s = s._replace(gait_ready=jnp.where(done, was_up, s.gait_ready))
        if self.zenbot_scratch_resets:
            if self.zenbot_source_gait:
                hold = jax.random.uniform(jax.random.fold_in(kc, 4), (self.N,), minval=2., maxval=4.)
                sampled_cmd = sampled_cmd._replace(time_left=jnp.where(was_up, sampled_cmd.time_left, hold))
            sampled_cmd = sampled_cmd._replace(vel=jnp.where(was_up[:, None], sampled_cmd.vel, 0.),
                                               is_standing=sampled_cmd.is_standing | ~was_up)
        cmd = self.recovery_command(env.where_cmd(done, sampled_cmd, s.cmd), down)
        if self.robot is zenbot:
            delays = jax.random.randint(jax.random.fold_in(key, 9), s.delay_steps.shape, 3, 8)
            if self.physical_gait_starts is not None:
                delays = jnp.where(window_delays >= 0, window_delays, delays)
            if self.zero_motor_delay_probability:
                immediate = jax.random.bernoulli(
                    jax.random.fold_in(key, 760), self.zero_motor_delay_probability, (self.N, 1)
                )
                delays = jnp.where(immediate, 0, delays)
            s = s._replace(delay_steps=jnp.where(on, delays, s.delay_steps))
        return s._replace(data=d, a_last=jnp.where(on, 0.0, s.a_last), a_ode=jnp.where(on, 0.0, s.a_ode), cmd=cmd, t=jnp.where(done, 0, s.t),
                          was_up=jnp.where(done, was_up, s.was_up), down=down, push_t=jnp.where(done, 0, s.push_t),
                          push_at=jnp.where(done, self._push_interval(kp), s.push_at), ep_sums=jnp.where(on, 0.0, s.ep_sums))  # fmt: skip

    def init(self, key) -> State:
        N, J = self.N, self.J
        k_world, k_reset, k_t, k_obs, key = jax.random.split(key, 5)
        world = self.world(k_world)
        data = jax.vmap(lambda q: self.dx.replace(qpos=q))(jnp.zeros((N, self.m.nq), jnp.float32))
        z, zi, zb = jnp.zeros((N,), jnp.float32), jnp.zeros((N,), jnp.int32), jnp.zeros((N,), bool)
        cmd = env.Cmd(jnp.zeros((N, 3)), z, zb, zb, z)
        s = State(data, world, jnp.zeros((N, self.obs_dim)), jnp.zeros((N, J)), jnp.zeros((N, J)), cmd, jnp.zeros((N, 3)), zi, jnp.ones((N,), bool),
                  jnp.zeros((N,), bool), zi, zi, jnp.zeros((N, len(self.terms))), key, jnp.full((N,), self.assist_newtons, jnp.float32), jnp.zeros((N, 5 if self.robot is zenbot else 0), jnp.int32))  # fmt: skip
        if self.reference is not None or self.zenbot_scratch_resets or self.zenbot_wm_reference_training:
            s = s._replace(reference_clip=zi)
            if self.g1_reference:
                s = s._replace(reference_start=zi, reference_yaw=z)
            elif self.track_getup or self.zenbot_wm_reference_training:
                s = s._replace(reference_start=zi)
        if self.recovery_amp_steps:
            s = s._replace(recovery_left=zi)
        if self.zenbot_support_weight or self.wm_floor_amp:
            s = s._replace(supported_hold=zenbot.SupportedStand.empty(N, env.DT))
        if self.zenbot_command_gait:
            s = s._replace(gait_ready=zb)
        if self.zenbot_scratch_curriculum:
            s = s._replace(first_supported_stand=jnp.full_like(zi, -1))
        if self.zenbot_source_gait:
            s = s._replace(gait_curriculum=source_gait.Curriculum.empty(N), episode_limit=zi)
        if self.physical_gait_starts is not None:
            s = s._replace(gait_window=jnp.full((N, 2), -1, jnp.int32))
        if self.wm_floor_amp:
            s = s._replace(episode_limit=zi)
        if self.wm_gait_rewards:
            s = s._replace(step_history=gait_rewards.StepHistory.empty(N))
        s = self._reset(s, jnp.ones((N,), bool), k_reset)
        if self.random_start and self.wm_floor_amp:
            # Every robot starts at t=0. Spread the first timeouts across control ticks.
            s = s._replace(episode_limit=jax.random.randint(k_t, (N,), self.episode_steps // 2, self.episode_steps + 1))
        elif self.random_start and (self.track_getup or self.zenbot_getup_stand or self.wm_stance_start_share):
            # Spread first resets across the collection buffer while t continues to measure
            # the actual time since the sampled motion pose. Later episodes run to full length.
            horizon = s.episode_limit if self.zenbot_source_gait else self.episode_steps
            limit = jax.random.randint(k_t, (N,), 1, horizon + 1)
            if self.zenbot_scratch_curriculum:
                limit = jnp.where(s.reference_clip >= 0, self.episode_steps, limit)
            s = s._replace(episode_limit=limit)
        elif self.random_start and self.robot is g1_getup:
            # The source PPO runner randomizes the initial episode counter once.
            s = s._replace(t=jax.random.randint(k_t, (N,), 0, self.episode_steps))
        elif self.random_start and not self.recovery:  # episodes then end at different times (a get-up start keeps its full time to rise)
            s = s._replace(t=jnp.where(s.down, 0, jax.random.randint(k_t, (N,), 0, self.episode_steps)))
        S = unflatten(s.data.qpos, s.data.qvel)
        c_obs = self.wm_velocity_command(s.cmd, S[1], s.was_up, s.recovery_left)
        frame = self.obs_frame(S, s.a_last, c_obs, k_obs)
        critic_obs = self.push_critic(None, S, s.a_last) if self.robot is g1_getup else None
        if self.zenbot_motion_train_obs:
            s = s._replace(motion_train_sensors=self.motion_train_sensors(s, S[1], c_obs, jnp.ones_like(zb), k_obs))
        s = s._replace(c_obs=c_obs, obs=self.push_actor(None, frame), critic_obs=critic_obs)
        if self.physical_floor_starts is not None:
            s = self.physical_floor_starts.restore(self, s, jnp.ones_like(zb), k_obs)
        if self.physical_gait_starts is not None:
            s = self.physical_gait_starts.restore(self, s, jnp.ones_like(zb), k_obs)
        return s

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
        if self.recovery or self.wm_floor_amp:
            support = jnp.asarray(self.support_link)
            link = jnp.maximum(support[g0], support[g1])
            loads = jnp.zeros((N, len(self.robot.SUPPORT_BODIES))).at[w, jnp.maximum(link, 0)].add(
                jnp.where(valid & floor & (link >= 0), normal, 0))
        return found, hit, touch, loads

    def physics(self, world: World, data, target, assist, fresh, *, gait_forces=False):
        """4 substeps with the actuator's control recomputed before each; the feet's contact and the termination contact after the last
        substep, and the number of substeps with a self-collision."""
        mx = world.model.tree_replace({k: getattr(world, k) for k in self.world_fields})
        step = jax.vmap(lambda m_, d, c: mjx.step(m_, d.replace(ctrl=c)), in_axes=(self.axes, 0, 0))

        def sub(carry, i):
            d, hits = carry
            tgt = target.reshape(self.N, env.DECIMATION, self.J)[:, i] if self.robot is zenbot else target
            if self.robot is g1_getup:
                # LeggedLab's reset event writes a persistent world-Z force on the torso.
                d = d.replace(xfrc_applied=d.xfrc_applied.at[:, self.torso, 2].set(assist))
            elif self.zenbot_scratch_curriculum or (self.recovery and not self.zenbot_getup_stand):
                loads = self.contact_stats(d)[3]
                force = zenbot.supported_lift(assist, d.qpos[:, 2], loads, fresh, i)
                d = d.replace(xfrc_applied=d.xfrc_applied.at[:, self.torso, 2].set(force))
            qd0 = d.qvel[:, self.va]
            d = step(mx, d, self.robot.actuator_ctrl(d.qpos[:, self.qa], d.qvel[:, self.va], tgt))
            found, hit, touch, loads = self.contact_stats(d)
            acc = (d.qvel[:, self.va] - qd0) / env.SIM_DT if self.robot is g1_getup or self.legged_support or gait_forces else None
            body_contacts = None
            if gait_forces:
                body_contacts = gait_rewards.measured_contacts(d, jnp.asarray(self.m.geom_bodyid),
                    self.gait_bodies[:2], self.gait_bodies[2:], self.foot_dofs, world.body_mass.sum(-1) * abs(self.m.opt.gravity[2]))
            return (d, hits + hit), (found, touch, loads, acc, body_contacts)

        (data, hits), (found, touch, loads, acc, body_contacts) = jax.lax.scan(sub, (data, jnp.zeros((self.N,))), jnp.arange(env.DECIMATION))
        illegal = touch.any(axis=0) if self.zenbot_source_gait else touch[-1]
        result = data, found[-1], hits, illegal, loads[-1] if loads is not None else None, acc[-1] if acc is not None else None
        if gait_forces:
            feet, other, foot_slip = body_contacts
            return (*result, feet[-1], other[-3:].any(0).sum(-1), foot_slip.mean(0))
        return result

    # ------------------------------------------------------------------ one policy step
    def step(self, s: State, action, clean=None, *, auto_reset=True, reference_rate_weight=1.):
        N = self.N
        key, k_reset, k_cmd, k_obs, k_push, k_push_at = jax.random.split(s.key, 6)
        a = action.astype(jnp.float32)
        clean_action = a if clean is None else clean.astype(jnp.float32)
        target = self.control_targets(a, s.world.bias, s.obs, self.context(s),
                                      elapsed=s.t if self.wm_floor_amp else None)
        assist = s.assist
        if self.zenbot_scratch_curriculum:
            assist = jnp.where((s.reference_clip >= 0) & (s.first_supported_stand < 0), assist, 0.)
        sensors = {"gait_forces": True} if self.wm_gait_rewards else {}
        data, contact, hits, touch, loads, joint_acc, *body_contacts = self.physics(s.world, s.data, target, assist, s.t == 0, **sensors)
        S0, S1 = unflatten(s.data.qpos, s.data.qvel), unflatten(data.qpos, data.qvel)
        t = s.t + 1
        v = self.view_state(S1)
        diverged = ~(jnp.isfinite(data.qpos).all(-1) & jnp.isfinite(data.qvel).all(-1)) | (jnp.abs(data.qvel) > DIVERGED).any(-1)
        up1 = self.is_up(v.height, S1[1])
        terminated = jnp.where(s.was_up, touch, (t >= env.GETUP_STEPS) & ~up1) | diverged
        was_up1 = self.next_upright(s.was_up, v.height, S1[1])
        if self.recovery or self.mixed:
            terminated = diverged
        limit = self.episode_steps if s.episode_limit is None else s.episode_limit
        timeout = (t >= limit) | self.recovery_timeout(s.reference_clip, s.reference_start, t)
        if self.zenbot_wm_reference_training:
            timeout |= self.wm_gait_worlds & s.was_up & ~was_up1
        if self.wm_floor_locomotion:
            terminated |= self.wm_gait_worlds & s.was_up & ~was_up1
        fall = (s.was_up & ~was_up1) | terminated if self.mixed else terminated
        remaining, active, deadline = None, None, False
        if self.recovery_amp_steps:
            remaining = self.next_recovery_interval(s.recovery_left, s.was_up, was_up1)
            active, deadline = remaining > 0, s.recovery_left == 1
        hold, gait_ready = s.supported_hold, s.gait_ready
        hold_score = None
        if self.wm_floor_amp:
            weight = s.world.body_mass.sum(-1) * abs(self.m.opt.gravity[2])
            posture = zenbot.standing_posture(v.height, S1[1][..., 2, 2])
            hold, hold_score = self.advance_floor_stand(
                hold, posture & ~diverged, loads[:, :2].sum(-1) >= .5 * weight, s.reference_clip, t)
            if self.wm_floor_locomotion:
                remaining = self.floor_gait_interval(s.recovery_left, was_up1, hold_score)
                active = remaining > 0
        terms = self.reward_terms(S0, S1, v.kin, a, s.a_last, s.c_obs, ~was_up1, fall.astype(jnp.float32), contact, hits, self.q_soft,
                                  torque=data.actuator_force if self.robot is g1_getup or self.legged_support else data.ctrl,
                                  joint_acc=joint_acc, clip=s.reference_clip, step=t, reference_start=s.reference_start, reference_yaw=s.reference_yaw,
                                  recovery_active=active, recovery_deadline=deadline, stand_blend=hold_score,
                                  reference_rate_weight=reference_rate_weight)
        if self.wm_min_rise_score:
            terminated |= self.wm_reference.failed_rise(terms["motion_reference"], s.reference_clip, s.reference_start,
                                                        t, active, s.was_up & ~was_up1, self.wm_min_rise_score)
        step_history = s.step_history
        if self.wm_gait_rewards:
            foot_acc_z = gait_rewards.measured_foot_acc_z(data, self.gait_bodies[:2], v.feet.com_velocity_z)
            step_history, added = gait_rewards.reward_terms(S1, v.feet, s.c_obs, data.actuator_force, joint_acc,
                self.q_soft, *body_contacts, foot_acc_z, step_history, (s.recovery_left == 0) | ~active)
            terms.update(added)
        if self.zenbot_getup_stand:
            weight = s.world.body_mass.sum(-1) * abs(self.m.opt.gravity[2])
            foot_force = loads[:, :2].sum(-1)
            posture = zenbot.standing_posture(v.height, S1[1][..., 2, 2])
            loaded = (foot_force >= .5 * weight) & ~diverged
            if self.zenbot_command_gait:
                nonfoot_load_bw = loads[:, 2:].sum(-1) / weight
                posture &= nonfoot_load_bw < .10
            standing = posture & loaded
            if not self.zenbot_standing_amp:
                _, stand_quality = zenbot.quiet_stand(S1, foot_force, weight)
                terms["stand_pose"] = stand_quality
            if self.zenbot_support_weight:
                hold, hold_score = hold.advance(posture & ~diverged, loaded)
                terms["supported_hold"] = hold_score
                if self.zenbot_command_gait:
                    gait_ready = zenbot.ready_for_gait(gait_ready, hold_score, v.height, S1[1][..., 2, 2],
                                                       nonfoot_load_bw, diverged)
                    if self.zenbot_source_gait:
                        source_active = (s.reference_clip < 0) | s.gait_ready
                        gait_fell = source_gait.failed(v.height, S1[1][:, 2, 2], touch, diverged)
                        terminated |= source_active & gait_fell
                        gait_ready = jnp.where(source_active, ~gait_fell, gait_ready)
                    elif self.zenbot_scratch_resets:
                        # Ground-born episodes can retry a fall. Gait-born episodes reset on loss of support.
                        terminated |= (s.reference_clip < 0) & ~gait_ready
                    moving = jnp.any(jnp.abs(s.c_obs) > .05, axis=-1)
                    terms["supported_hold"] = jnp.where(gait_ready & moving, 0., hold_score)
                if self.zenbot_default_stand:
                    terms["stand_pose"], terms["supported_hold"] = zenbot.default_stand_terms(S1, hold_score)
        first_stand = s.first_supported_stand
        if self.zenbot_scratch_curriculum:
            first_stand, failed, timeout = zenbot.scratch_recovery_end(
                s.reference_clip, s.reference_start, s.world.floor["lengths"], t, first_stand, gait_ready,
                self.episode_steps, timeout)
            terminated |= failed
        parts = self.weighted(terms)
        parts = jnp.where(diverged[:, None], 0.0, jnp.stack([parts[k] for k in self.terms], -1))
        if self.zenbot_source_gait:
            # In locomotion the task consists of velocity tracking (trainer) and the existing threefold action-rate cost.
            rate = jnp.asarray([name == "action_rate_l2" for name in self.terms])
            parts = jnp.where(source_active[:, None] & ~rate, 0., parts)
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
        s1 = s._replace(data=data, a_last=a, a_ode=clean_action, t=t,
                        was_up=was_up1, ep_sums=ep_sums, recovery_left=remaining, supported_hold=hold, gait_ready=gait_ready,
                        first_supported_stand=first_stand, step_history=step_history)
        if self.zenbot_wm_reference_training:
            # A later fall belongs to a new recovery; its phase cannot be matched to the birth clip.
            s1 = s1._replace(reference_clip=jnp.where(s.was_up & ~was_up1, -1, s1.reference_clip))
        if self.wm_posture_and_tracking:
            moving = ~active & ~diverged & jnp.any(s.c_obs != 0., axis=-1)
            error = gait_rewards.base_velocity(S1, self.root_com) - s.c_obs
            stats["velocity_errors"] = jnp.stack((jnp.where(moving, jnp.linalg.norm(error[:, :2], axis=-1), 0.).sum(),
                                                 jnp.where(moving, jnp.abs(error[:, 2]), 0.).sum(), moving.sum()))
        if self.wm_floor_amp:
            stats["stand_blend"] = jnp.where(active, hold_score, 1.) if self.wm_floor_locomotion else hold_score
        if self.zenbot_scratch_curriculum:
            stats["failed_floor_recovery"] = failed
            stats["applied_lift_n"] = data.xfrc_applied[:, self.torso, 2].mean()
            stats["failed_floor_resets"] = failed.sum()
        if self.mixed:
            stats["gait_falls"] = (s.was_up & ~was_up1 & ~diverged).sum()
            stats["rises"] = (~s.was_up & was_up1 & ~diverged).sum()
        if self.reference_reward:
            valid = ~diverged
            if self.g1_reference:
                valid &= self.reference.active(s.reference_clip, s.reference_start, t)
            elif self.zenbot_scratch_resets:
                valid &= (s.reference_clip >= 0) & (s.reference_start + t < self.reference.lengths[jnp.maximum(s.reference_clip, 0)])
            elif not self.track_getup:
                valid &= (s.reference_clip >= 0) & (t < self.reference.lengths[jnp.maximum(s.reference_clip, 0)])
            score = terms.get("reference_pose_score", terms["motion_reference"])
            stats["reference_score_sum"] = jnp.where(valid, score, 0.).sum()
            stats["reference_frames"] = valid.sum()
            if self.track_getup:
                moving = valid & (s.reference_start + t < self.reference.lengths[s.reference_clip])
                stats["reference_moving_score_sum"] = jnp.where(moving, score, 0.).sum()
                stats["reference_moving_frames"] = moving.sum()
        if self.recovery:
            if self.zenbot_getup_stand:
                requested = self.robot.DEFAULT_Q + self.robot.ACTION_SCALE * a - s.world.bias
                stats["pd_request_outside_fraction"] = ((requested < self.joint_range[:, 0]) |
                                                        (requested > self.joint_range[:, 1])).mean()
                if self.zenbot_command_gait:
                    stats["gait_command"] = s.c_obs
                    stats["gait_ready"] = gait_ready
                    local_speed = mv3(tr3(yaw_mat(S1[1])), S1[2])
                    velocity = jnp.column_stack((local_speed[:, :2], S1[3][:, 2]))
                    stats["gait_velocity_error"] = velocity - s.c_obs
                stats["standing_now"] = standing
                stats["supported_stand"] = stats["standing_now"].mean()
                if self.zenbot_standing_amp:
                    stats["min_foot_load_bw"] = loads[:, :2].min(axis=-1) / weight
                if not self.zenbot_standing_amp:
                    stats["stand_quality"] = jnp.where(diverged, 0., stand_quality)
                if self.zenbot_support_weight:
                    stats["support_hold_score"] = hold_score.mean()
                if self.zenbot_default_stand or self.zenbot_standing_amp:
                    stats["stand_blend"] = hold_score
            else:
                height_score = ep_sums[:, self.terms.index("target_base_height")] / (self.episode_steps * env.DT)
                avg = (height_score * fin).sum() / jnp.maximum(fin.sum(), 1)
                dec = done & (avg > 0.6 * self.weights["target_base_height"])
                s1 = s1._replace(assist=jnp.maximum(0., s.assist - dec * self.robot.ASSIST_DECREMENT))
            stats.update(assist_n=s.assist.mean(), foot_load=loads[:, :2].sum(-1))
        if self.zenbot_source_gait:
            velocity_score, velocity_error = source_gait.tracking(S1, s.c_obs)
            # The source command curriculum measures ordinary starts. A physical window can
            # prescribe a run above the current training stage and assist its first steps.
            assessed = source_active if s.gait_window is None else source_active & (s.gait_window[:, 0] < 0)
            curriculum = s.gait_curriculum.advance(s.c_obs[:, 0], velocity_error, assessed, terminated, timeout)
            if s.gait_window is not None:
                first, end = s.gait_window.T
                stats["physical_gait_window_fraction"] = ((first >= 0) & (first + s.t < end)).mean()
            s1 = s1._replace(gait_curriculum=curriculum)
            stats.update(source_gait_active=source_active, source_standing=s.cmd.is_standing,
                         source_tilt_l2=jnp.square(S1[1][:, 2, :2]).sum(axis=-1),
                         source_velocity_score=jnp.where(diverged, 0., velocity_score),
                         source_speed_stage=curriculum.stage, source_completed_episodes=curriculum.completed[7],
                         source_tracking_error=curriculum.last_metrics[0], source_fall_rate=curriculum.last_metrics[3])
        reset_history = done if auto_reset else jnp.zeros_like(done)
        if not auto_reset and self.recovery:
            stats["support_load"] = loads
        if auto_reset:
            s1 = self._reset(s1, done, k_reset)
        if self.zenbot_motion_train_wm:
            cmd = self.tick_wm_command(s1.cmd, k_cmd, S1, was_up1, remaining)
        else:
            sample = (lambda key: self.sample_command(key, s1.gait_curriculum.stage)) if self.zenbot_source_gait else self.sample_command
            cmd = self.recovery_command(env.tick(s1.cmd, k_cmd, self.cmd_ranges, sample=sample), s1.down)
        if self.zenbot_motion_train_wm and auto_reset:
            # The transition belonged to the old episode. A newborn world gets its full command interval.
            cmd = env.where_cmd(reset_history, s1.cmd, cmd)
        if self.zenbot_scratch_resets:
            # Freeze a zero command and its timer until the ground-born actor has held its feet for a second.
            recovering = (s1.reference_clip >= 0) & ~s1.gait_ready
            stopped = s1.cmd._replace(vel=jnp.zeros_like(s1.cmd.vel), is_standing=jnp.ones_like(s1.cmd.is_standing))
            cmd = env.where_cmd(recovering, stopped, cmd)
        push_t = s1.push_t + 1
        push_due = (push_t == s1.push_at) & (self.push and not self.recovery)
        push = push_due
        if self.g1_phase_amp:
            push &= ~self.reference.active(s1.reference_clip, s1.reference_start, s1.t)
        qvel = s1.data.qvel
        if self.fall_push_speed:
            push &= (s1.recovery_left == 0) & s1.was_up
            angle, magnitude = jax.random.uniform(k_push, (2, N))
            angle = angle * (2*jnp.pi)
            impulse = (self.fall_push_speed * (.4 + .6*magnitude))[:, None] * jnp.stack((jnp.cos(angle), jnp.sin(angle)), -1)
            next_xy = qvel[:, :2] + impulse
        else:
            next_xy = env.PUSH_VEL * jax.random.uniform(k_push, (N, 2), minval=-1.0, maxval=1.0)
        qvel = qvel.at[:, :2].set(jnp.where(push[:, None], next_xy, qvel[:, :2]))
        s1 = s1._replace(data=s1.data.replace(qvel=qvel), push_t=jnp.where(push_due, 0, push_t), push_at=jnp.where(push_due, self._push_interval(k_push_at), s1.push_at))
        stats["pushes"] = push.sum().astype(jnp.float32)
        S2 = unflatten(s1.data.qpos, s1.data.qvel)
        c_obs = self.wm_velocity_command(cmd, S2[1], s1.was_up, s1.recovery_left)
        obs = self.push_actor(s.obs, self.obs_frame(S2, s1.a_last, c_obs, k_obs), reset_history)
        critic_obs = self.push_critic(s.critic_obs, S2, s1.a_last, reset_history) if self.robot is g1_getup else None
        if self.zenbot_motion_train_obs:
            s1 = s1._replace(motion_train_sensors=self.motion_train_sensors(s1, S2[1], c_obs, reset_history, k_obs))
        s2 = s1._replace(obs=obs, critic_obs=critic_obs, cmd=cmd, c_obs=c_obs, key=key)
        if self.physical_floor_starts is not None and auto_reset:
            s2 = self.physical_floor_starts.restore(self, s2, reset_history, k_reset)
        if self.physical_gait_starts is not None and auto_reset:
            s2 = self.physical_gait_starts.restore(self, s2, reset_history, k_reset)
        return s2, Transition(s.data.qpos, s.data.qvel, target, data.qpos, data.qvel, reward, terminated, timeout & ~terminated, diverged, parts,
                              contact, v.clearance, stats)  # fmt: skip
