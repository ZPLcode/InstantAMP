"""Retarget the active InstantAMP 23-DoF reference banks with FRAX differential IK.

From the InstantAMP checkout, using a Python environment with frax[ik] and MuJoCo:
    python -m tools.retarget_zenbot26 /path/to/motion_train
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import zenbot
from instantamp.zenbot_models import ZENBOT23, ZENBOT26
from tools.convert_motions import convert_recovery_clip

ROOT = Path(__file__).resolve().parents[1]
KEY_BODIES = zenbot.KEY_BODIES
EXTRA_JOINTS = tuple(
    f"{name}_joint" for name in ZENBOT26.dof if name not in ZENBOT23.dof
)
BANKS = ("walk_run", "recovery")


def digest(path):
    """Return the SHA-256 digest of a model asset or reference clip."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampling_weights(directory):
    """Read the positive clip weights and check that every NPZ is sampled."""
    weights = json.loads((directory / "sampling.json").read_text())["weights"]
    if not weights or set(weights) != {
        path.stem for path in directory.glob("*.npz")
    }:
        raise ValueError(
            f"Sampling weights differ from the clips in {directory}"
        )
    values = np.array(list(weights.values()))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Invalid weights in {directory}")
    return weights


def verify(source, dest):
    """Check both target banks against the model and current source clips."""
    manifest = json.loads((dest / "manifest.json").read_text())
    if (
        manifest["target_mjcf_sha256"] != digest(ZENBOT26.xml_path)
        or manifest["source_mjcf_sha256"] != digest(ZENBOT23.xml_path)
        or manifest["joint_names"] != [f"{name}_joint" for name in ZENBOT26.dof]
    ):
        raise ValueError(f"Zenbot model differs from the retarget in {dest}")
    clips = set()
    for bank in BANKS:
        weights = sampling_weights(source / bank)
        if sampling_weights(dest / bank) != weights:
            raise ValueError(
                f"Source and target sampling weights differ for {bank}"
            )
        for name in weights:
            relative = Path(bank) / f"{name}.npz"
            info = manifest["clips"].get(str(relative), {})
            if info.get("source_sha256") != digest(
                source / relative
            ) or info.get("output_sha256") != digest(dest / relative):
                raise ValueError(
                    f"Source or target checksum mismatch for {relative}"
                )
            clips.add(str(relative))
    if clips != set(manifest["clips"]):
        raise ValueError(f"Unexpected clips in {dest / 'manifest.json'}")
    print(
        f"Verified {len(clips)} retargeted clips and their sampling weights in {dest}"
    )


class Kinematics:
    """MuJoCo world-frame body poses and per-link clearance above the ground plane."""

    def __init__(self, path, head):
        self.model = m = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(m)
        self.names = [m.joint(i).name for i in range(1, m.njnt)]
        self.bodies = [
            m.body(name).id for name in (*KEY_BODIES, "torso_Link", head)
        ]
        self.geoms = np.flatnonzero(
            (m.geom_bodyid != 0) & ((m.geom_contype | m.geom_conaffinity) != 0)
        )
        self.key_geoms = [
            list(m.geom_bodyid[self.geoms]).index(body)
            for body in self.bodies[:6]
        ]
        self.trunk_geoms = [
            list(m.geom_bodyid[self.geoms]).index(body)
            for body in self.bodies[6:]
        ]
        types = m.geom_type[self.geoms]
        self.box = types == mujoco.mjtGeom.mjGEOM_BOX
        self.cylinder = types == mujoco.mjtGeom.mjGEOM_CYLINDER
        if not (
            self.box | self.cylinder | (types == mujoco.mjtGeom.mjGEOM_CAPSULE)
        ).all():
            raise ValueError("Unsupported ground collision primitive")
        self.sizes = m.geom_size[self.geoms]

    def evaluate(self, poses):
        d = self.data
        xyz = np.empty((len(poses), len(self.bodies), 3))
        rotations = np.empty((*xyz.shape, 3))
        heights = np.empty((len(poses), len(self.geoms)))
        for i, pose in enumerate(poses):
            d.qpos[:] = pose
            mujoco.mj_kinematics(self.model, d)
            xyz[i] = d.xpos[self.bodies]
            rotations[i] = d.xmat[self.bodies].reshape(-1, 3, 3)
            z = d.geom_xmat[self.geoms].reshape(-1, 3, 3)[:, 2]
            radius = (
                np.where(self.cylinder, np.linalg.norm(z[:, :2], axis=1), 1.0)
                * self.sizes[:, 0]
            )
            depth = np.where(
                self.box,
                (abs(z) * self.sizes).sum(axis=1),
                abs(z[:, 2]) * self.sizes[:, 1] + radius,
            )
            heights[i] = d.geom_xpos[self.geoms, 2] - depth
        return xyz, rotations, heights


class Retarget:
    """Fit source poses to the target while matching limb motion and clearance."""

    def __init__(self, source, target, urdf):
        # Only reference generation needs the optional FRAX IK dependencies.
        from frax import Robot
        from frax.ik import DiffIKSolver, FrameTask, JointLimits, JointTask

        self.source, self.target = source, target
        if len(source.names) != 23 or set(target.names) != set(
            source.names
        ).union(EXTRA_JOINTS):
            raise ValueError("Expected the 23- and 26-DoF Zenbot articulations")
        self.common = [target.names.index(name) for name in source.names]
        # Work in pelvis coordinates. Preserve measured root translation and orientation
        # independently of the joint solver, then match floor clearance in world space.
        robot = Robot(urdf, joint_ordering=target.names, floating_base=None)
        self.robot = robot
        self.bounds = target.model.jnt_range[1:].T
        np.testing.assert_allclose(
            self.bounds,
            [
                robot.actuated_joint_lower_limits,
                robot.actuated_joint_upper_limits,
            ],
        )
        body_joints = target.model.body_jntadr[target.bodies]
        frames = [
            robot.joint_name_to_index[target.model.joint(i).name]
            for i in body_joints
        ]
        task = FrameTask(
            robot,
            frames,
            weight_pos=[16, 16, 8, 8, 16, 16, 2, 16],
            weight_rot=[3, 3, 1, 1, 1, 1, 1, 1],
        )
        posture = JointTask(robot, np.arange(robot.nv), "posture", weight=0.2)
        limits = JointLimits(robot, gain=1.0, v_max=None)
        solver = DiffIKSolver(
            robot,
            [task, posture],
            [limits],
            regularization=1e-5,
            solver_tol=1e-9,
        )

        @jax.jit
        def fit(seed, positions, rotations):
            ref = {
                "frame_pos": positions,
                "frame_rmat": rotations,
                "posture": seed,
            }
            state, _ = solver.converge(
                solver.init(seed), ref, max_iters=25, dt_max=1.0, tol=1e-7
            )
            return state.q

        self.fit = fit

    def clip(self, path, out):
        with np.load(path) as source:
            fps = float(source["fps"])
            q = source["dof_pos"].astype(float)
            if fps != 50 or q.shape != (len(q), 23) or len(q) < 2:
                raise ValueError(
                    f"Invalid source clip dimensions or fps: {path}"
                )
            root = np.column_stack(
                (source["root_pos"], source["root_quat"][:, [3, 0, 1, 2]])
            ).astype(float)
        if not np.isfinite(q).all() or not np.isfinite(root).all():
            raise ValueError(f"Nonfinite source motion: {path}")
        np.testing.assert_allclose(
            np.linalg.norm(root[:, 3:], axis=1), 1.0, atol=1e-6
        )
        root[:, 3:] /= np.linalg.norm(root[:, 3:], axis=1, keepdims=True)
        xyz, rot, heights = self.source.evaluate(np.column_stack((root, q)))
        seed = np.zeros((len(q), 26))
        seed[:, self.common] = q
        poses = np.column_stack((root, seed))
        nominal, _, target_heights = self.target.evaluate(poses)
        seed = np.clip(seed, *self.bounds)

        # Match the lower surfaces of supporting links across collision shapes.
        # Fade each offset as the source link leaves the floor.
        desired = xyz.copy()
        # Torso and head link origins moved in the 26-DoF model.
        desired[:, 6:] = nominal[:, 6:]
        lower = heights[:, self.source.key_geoms + self.source.trunk_geoms]
        near_floor = np.clip(1.0 - (lower - 0.015) / 0.045, 0.0, 1.0)
        weight = near_floor**2 * (3.0 - 2.0 * near_floor)
        desired[:, :, 2] += weight * (
            lower
            - target_heights[:, self.target.key_geoms + self.target.trunk_geoms]
        )

        # Share the head/trunk clearance correction between the pelvis and IK
        # when lying down. Keep the measured world targets for the other limbs.
        trunk = heights[:, self.source.trunk_geoms].min(axis=1)
        gap = np.maximum(trunk - heights.min(axis=1), trunk - 0.02)
        contact = np.clip(1.0 - gap / 0.04, 0.0, 1.0)
        trunk_dz = trunk - target_heights[:, self.target.trunk_geoms].min(
            axis=1
        )
        poses[:, 2] += 0.5 * contact**2 * (3.0 - 2.0 * contact) * trunk_dz
        pelvis = Rotation.from_quat(root[:, [4, 5, 6, 3]]).as_matrix()
        positions = (desired - poses[:, None, :3]) @ pelvis
        rotations = pelvis.transpose(0, 2, 1)[:, None] @ rot
        fitted = np.stack(
            [
                np.asarray(self.fit(s, p, r))
                for s, p, r in zip(seed, positions, rotations)
            ]
        )
        low, high = self.bounds
        violation = max(
            0.0, float(np.maximum(low - fitted, fitted - high).max())
        )
        if not np.isfinite(fitted).all() or violation > 1e-6:
            raise ValueError(
                f"Invalid IK solution for {path}: limit violation {violation}"
            )
        poses[:, 7:] = np.clip(fitted, low, high)
        speed = np.max(abs(np.diff(poses[:, 7:], axis=0)) * fps, axis=0)
        if np.any(speed > self.robot.actuated_joint_max_velocities + 1e-4):
            raise ValueError(f"Target joint speed limit exceeded in {path}")

        points, _, target_heights = self.target.evaluate(poses)
        clearance = np.maximum(0.0, heights.min(axis=1))
        dz = clearance - target_heights.min(axis=1)
        poses[:, 2] += dz
        points[:, :, 2] += dz[:, None]
        points = points[:, :6]
        convert_recovery_clip(
            out,
            poses[:, :3],
            poses[:, 3:7],
            poses[:, 7:],
            np.arange(26),
            fps,
            points,
            keep_native=True,
        )
        error = np.linalg.norm(points - xyz[:, :6], axis=2)
        return {
            "frames": len(q),
            "source_sha256": digest(path),
            "output_sha256": digest(out),
            "max_root_height_change_m": float(
                abs(poses[:, 2] - root[:, 2]).max()
            ),
            "lowest_collision_height_m": float(clearance.min()),
            "key_body_rms_error_m": float(np.sqrt(np.mean(error**2))),
            "key_body_max_error_m": float(error.max()),
            "max_joint_speed_rad_s": float(speed.max()),
        }


def prepare(motion_train, source_dir, dest):
    """Retarget both reference banks and publish their manifest together."""
    source_dir, dest = source_dir.resolve(), dest.resolve()
    if dest.exists():
        raise FileExistsError(f"Choose a new output directory: {dest}")
    assets = motion_train.resolve() / "robolab/data/robots/roboparty"
    model = "zenbot_26dof_43kg"
    source_xml = assets / "zenbot/mjcf/zenbot.xml"
    target_xml = assets / model / "mjcf" / f"{model}.xml"
    target_urdf = assets / model / "urdf" / f"{model}.urdf"
    if digest(source_xml) != digest(ZENBOT23.xml_path) or digest(
        target_xml
    ) != digest(ZENBOT26.xml_path):
        raise ValueError(
            "Motion Train MJCF differs from the bundled Zenbot articulation"
        )
    source = Kinematics(source_xml, ZENBOT23.head_body)
    target = Kinematics(target_xml, ZENBOT26.head_body)
    fit = Retarget(source, target, target_urdf)
    manifest = {
        "engine": "frax.ik.DiffIKSolver",
        "frax_version": version("frax"),
        "mujoco_version": mujoco.__version__,
        "source_directory": source_dir.name,
        "source_mjcf_sha256": digest(source_xml),
        "target_mjcf": str(target_xml.relative_to(motion_train.resolve())),
        "target_mjcf_sha256": digest(target_xml),
        "target_urdf": str(target_urdf.relative_to(motion_train.resolve())),
        "target_urdf_sha256": digest(target_urdf),
        "joint_names": target.names,
        "key_body_names": KEY_BODIES,
        "fps": 50,
        "root_quaternion_order": "xyzw",
        "root_velocity_frame": "world",
        "key_body_position_frame": "world",
        "clips": {},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".frax26-") as tmp:
        stage = Path(tmp) / dest.name
        for bank in BANKS:
            directory = source_dir / bank
            weights = sampling_weights(directory)
            for name in sorted(weights):
                relative = Path(bank) / f"{name}.npz"
                info = fit.clip(source_dir / relative, stage / relative)
                manifest["clips"][str(relative)] = info
                print(
                    f"{relative}: {info['frames']} frames, landmark RMS {1000 * info['key_body_rms_error_m']:.2f} mm",
                    flush=True,
                )
            (stage / bank / "sampling.json").write_text(
                json.dumps({"weights": weights}, indent=2) + "\n"
            )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        stage.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_train", type=Path, nargs="?")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data" / ZENBOT23.mixed_task
    )
    parser.add_argument(
        "--dest", type=Path, default=ROOT / "data" / ZENBOT26.mixed_task
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check an existing retarget without running FRAX",
    )
    args = parser.parse_args()
    if args.verify:
        verify(args.source, args.dest)
    elif args.motion_train is None:
        parser.error("motion_train is required to generate a new retarget")
    else:
        jax.config.update("jax_enable_x64", True)
        prepare(args.motion_train, args.source, args.dest)
