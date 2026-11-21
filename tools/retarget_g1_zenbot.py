"""Retarget the ten 50 Hz LeggedLab G1 recoveries to the physical 43 kg, 23-DoF Zenbot.

Bounded differential inverse kinematics tracks limb directions and root/trunk orientation,
matches support clearances in both articulations and separates colliding Zenbot links.
All exported landmarks and velocities are recomputed after smoothing the target motion.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import mujoco as mj
import numpy as np
from scipy.linalg import solve
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp import g1_getup, zenbot
from instantamp.sim import build_scene
from tools.convert_motions import velocities
from tools.prepare_g1_getup import ACTIVE_CLIPS


def body(model, name):
    return model.body("robot_" + name).id


def rotation(data, index):
    return Rotation.from_matrix(data.xmat[index].reshape(3, 3))


class Skeleton:
    def __init__(self, source):
        self.model = build_scene(robot_name="g1_getup" if source else "zenbot23_getup")
        self.data = mj.MjData(self.model)
        self.data.qpos[:3] = 0.
        mj.mj_forward(self.model, self.data)
        self.root = 1
        self.trunk = body(self.model, "torso_link" if source else zenbot.TORSO)
        self.joints = g1_getup.DOF if source else zenbot.DOF
        # Uppercase/lowercase here follow the actual imported Zenbot MJCF names.
        self.chains = []
        for side in ("left", "right"):
            hip = side + ("_hip_pitch_link" if source or side == "right" else "_hip_pitch_Link")
            suffix = "_link" if source else "_Link"
            leg = [hip, side + "_knee" + suffix, side + "_ankle_roll" + suffix]
            hand = side + ("_wrist_yaw_link" if source else "_wrist_roll_rubber_hand")
            arm = [side + "_shoulder_pitch" + suffix, side + "_elbow" + suffix, hand]
            self.chains.extend([[body(self.model, s) for s in chain] for chain in (leg, arm)])
        self.support = [self.chains[0][2], self.chains[2][2], self.chains[0][1],
                        self.chains[2][1], self.chains[1][2], self.chains[3][2]]
        self.neutral = self.data.xpos.copy()
        self.neutral_rot = self.data.xmat.reshape(-1, 3, 3).copy()
        self.colliders = [Collider(self.model, i) for i in range(1, self.model.ngeom)
                          if self.model.geom_contype[i] or self.model.geom_conaffinity[i]]
        self.by_body = {}
        for collider in self.colliders:
            self.by_body.setdefault(collider.link, []).append(collider)

    def set_pose(self, p, quat_xyzw, joints):
        self.data.qpos[:] = np.r_[p, quat_xyzw[[3, 0, 1, 2]], joints]
        mj.mj_forward(self.model, self.data)

    def lowest(self):
        return min(c.point(self.data)[2] for c in self.colliders)

    def contact_points(self):
        return np.stack([min((c.point(self.data) for c in self.by_body[b]), key=lambda a: a[2]) for b in self.support])


class Collider:
    def __init__(self, model, geom):
        self.id, self.link = geom, int(model.geom_bodyid[geom])
        self.kind = int(model.geom_type[geom])
        self.size = model.geom_size[geom].copy()
        if self.kind == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            index, length = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self.vertices = model.mesh_vert[index:index + length].copy()

    def point(self, data):
        mat = data.geom_xmat[self.id].reshape(3, 3)
        pos = data.geom_xpos[self.id]
        axis = mat[2]
        if self.kind == mj.mjtGeom.mjGEOM_MESH:
            return pos + mat @ self.vertices[np.argmin(self.vertices @ axis)]
        if self.kind == mj.mjtGeom.mjGEOM_BOX:
            return pos - mat @ (np.sign(axis) * self.size)
        if self.kind == mj.mjtGeom.mjGEOM_SPHERE:
            return pos - np.array([0, 0, self.size[0]])
        if self.kind in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            end = pos - mat[:, 2] * self.size[1] * np.sign(axis[2])
            if self.kind == mj.mjtGeom.mjGEOM_CAPSULE:
                return end - np.array([0, 0, self.size[0]])
            normal = axis[:2]
            return end - mat[:, :2] @ normal * self.size[0] / max(np.linalg.norm(normal), 1e-9)
        raise ValueError(f"Unsupported collider geometry {self.kind}")


class Retargeter:
    def __init__(self):
        self.src, self.dst = Skeleton(True), Skeleton(False)
        s, d = self.src, self.dst
        self.order = [s.joints.index(name) for name in d.joints]
        self.ranges = d.model.jnt_range[1:].copy()
        assert [d.model.joint(i + 1).name for i in range(d.model.nu)] == ["robot_" + n + "_joint" for n in d.joints]
        self.height_scale = abs(d.neutral[d.support[:2], 2].mean() / s.neutral[s.support[:2], 2].mean())
        self.identity = np.eye(d.model.nv)

    def targets(self, origin):
        s, d = self.src, self.dst
        src_root = rotation(s.data, s.root)
        trunk = rotation(s.data, s.trunk) * Rotation.from_matrix(s.neutral_rot[s.trunk].T @ d.neutral_rot[d.trunk])
        goals = []
        for i, (source_chain, dest_chain) in enumerate(zip(s.chains, d.chains)):
            sa, da = source_chain[0], dest_chain[0]
            # The shoulder is anchored on the taller Zenbot trunk. Hips are anchored on its pelvis.
            frame = trunk if i % 2 else src_root
            start = origin + frame.apply(d.neutral[da])
            for sb, db in zip(source_chain[1:], dest_chain[1:]):
                sn = s.neutral[sb] - s.neutral[sa]
                dn = d.neutral[db] - d.neutral[da]
                ratio = np.linalg.norm(dn) / np.linalg.norm(sn)
                limb = ratio * (s.data.xpos[sb] - s.data.xpos[sa])
                parent_motion = s.data.xmat[sa].reshape(3, 3) @ s.neutral_rot[sa].T
                start = start + limb + parent_motion @ (dn - ratio * sn)
                weights = (1.15, 1.6) if i % 2 == 0 else (1.1, 1.8)
                goals.append((db, start.copy(), weights[int(db == dest_chain[-1])]))
                sa, da = sb, db
        return src_root, trunk, goals

    def fit_frame(self, root, quat, q, previous, previous_seed, previous_rot, source_gap):
        s, d = self.src, self.dst
        m, data = d.model, d.data
        source_root, trunk, goals = self.targets(root)
        seed = np.clip(q[self.order], *self.ranges.T)
        if previous is None:
            estimate, reference = seed, seed
            initial = source_root
            limits = self.ranges
        else:
            reference = previous + seed - previous_seed
            limits = np.stack((np.maximum(self.ranges[:, 0], previous - zenbot.SPEED / 50),
                               np.minimum(self.ranges[:, 1], previous + zenbot.SPEED / 50)), axis=1)
            estimate = np.clip(reference, *limits.T)
            initial = Rotation.from_quat(quat) * previous_rot
        d.set_pose(root, initial.as_quat(), estimate)
        foot_rot = [rotation(s.data, a) * Rotation.from_matrix(s.neutral_rot[a].T @ d.neutral_rot[b])
                    for a, b in zip(s.support[:2], d.support[:2])]
        jp = np.empty((3, m.nv))
        jr = np.empty_like(jp)
        for _ in range(10):
            a, b = [], []

            def task(jac, error, weight):
                a.append(weight * jac)
                b.append(weight * np.atleast_1d(error))

            task(self.identity[:3], root - data.qpos[:3], 1.9)
            mj.mj_jacBody(m, data, None, jr, d.root)
            task(jr, (source_root * rotation(data, d.root).inv()).as_rotvec(), .50)
            mj.mj_jacBody(m, data, None, jr, d.trunk)
            task(jr, (trunk * rotation(data, d.trunk).inv()).as_rotvec(), .45)
            for link, goal, weight in goals:
                mj.mj_jacBody(m, data, jp, None, link)
                task(jp, goal - data.xpos[link], weight)
            for link, dest in zip(d.support[:2], foot_rot):
                mj.mj_jacBody(m, data, None, jr, link)
                task(jr, (dest * rotation(data, link).inv()).as_rotvec(), .19)
            # Match the source support order using the actual extremal point of each collision shape.
            for link, point, gap in zip(d.support, d.contact_points(), source_gap):
                if gap < .095:
                    mj.mj_jac(m, data, jp, None, point, link)
                    task(jp[2:3], gap - point[2], 4.5 * (1. - gap / .15))
            for geom in d.colliders:
                point = geom.point(data)
                if point[2] < -.001:
                    mj.mj_jac(m, data, jp, None, point, geom.link)
                    task(jp[2:3], -point[2], 4.)
            for c in data.contact:
                if 0 in (c.geom1, c.geom2) or c.dist >= -.002:
                    continue
                mj.mj_jac(m, data, jp, None, c.pos, m.geom_bodyid[c.geom1])
                mj.mj_jac(m, data, jr, None, c.pos, m.geom_bodyid[c.geom2])
                task(c.frame[:3][None] @ (jr - jp), -.001 - c.dist, 3.)
            task(self.identity[6:], seed - data.qpos[7:], .085)
            if previous is not None:
                task(self.identity[6:], reference - data.qpos[7:], .15)
            A = np.vstack(a)
            delta = solve(A.T @ A + .004 * self.identity, A.T @ np.concatenate(b), assume_a="pos")
            mj.mj_integratePos(m, data.qpos, np.clip(delta, -.22, .22), .82)
            data.qpos[7:] = np.clip(data.qpos[7:], *limits.T)
            mj.mj_forward(m, data)
            if np.max(np.abs(delta)) < .0005:
                break
        rot = rotation(data, d.root)
        return data.qpos[:3].copy(), rot.as_quat(), data.qpos[7:].copy(), seed, source_root.inv() * rot

    def refine_support(self, pos, quat, joints, gaps, ground):
        """After smoothing, align relative foot, knee and hand heights against Zenbot's own floor."""
        d = self.dst
        jac, lowest_jac = (np.empty((3, d.model.nv)) for _ in range(2))
        eye = np.eye(d.model.nu)
        correction = np.zeros(d.model.nu)
        for i, (desired, floor) in enumerate(zip(gaps, ground)):
            original = joints[i].copy()
            q = np.clip(original + .45 * correction, *self.ranges.T)
            for _ in range(14):
                d.set_pose(pos[i], quat[i], q)
                points = [(collider.link, collider.point(d.data)) for collider in d.colliders]
                lowest_link, lowest = min(points, key=lambda item: item[1][2])
                mj.mj_jac(d.model, d.data, lowest_jac, None, lowest, lowest_link)
                rows, errors = [], []
                for link, point, target in zip(d.support, d.contact_points(), desired):
                    height = point[2] - lowest[2] + floor
                    if min(height, target) > .12:
                        continue
                    mj.mj_jac(d.model, d.data, jac, None, point, link)
                    # When a sole is flat, several vertices tie. Tikhonov regularization
                    # makes the ambiguous normal derivative benign near that transition.
                    rows.append(jac[2, 6:] - lowest_jac[2, 6:])
                    errors.append(np.clip(target - height, -.06, .06))
                if not rows:
                    break
                a = np.vstack(rows)
                delta = solve(a.T @ a + .002 * eye, a.T @ errors + .001 * (original - q), assume_a="pos")
                q = np.clip(q + np.clip(.65 * delta, -.09, .09), *self.ranges.T)
                q = np.clip(q, original - .32, original + .32)
            joints[i] = q
            correction = q - original

    def separate_links(self, p, quat, joints):
        """Project overlapping collision hulls apart, carrying local corrections across adjacent frames."""
        d = self.dst
        jac_a, jac_b = (np.empty((3, d.model.nv)) for _ in range(2))
        for direction in (range(len(p)), range(len(p) - 1, -1, -1)):
            previous_correction = np.zeros(d.model.nu)
            for t in direction:
                original = joints[t].copy()
                q = np.clip(original + .72 * previous_correction, *self.ranges.T)
                for _ in range(24):
                    d.set_pose(p[t], quat[t], q)
                    rows, distances = [], []
                    for contact in d.data.contact:
                        if 0 in (contact.geom1, contact.geom2) or contact.dist >= -.0002:
                            continue
                        mj.mj_jac(d.model, d.data, jac_a, None, contact.pos, d.model.geom_bodyid[contact.geom1])
                        mj.mj_jac(d.model, d.data, jac_b, None, contact.pos, d.model.geom_bodyid[contact.geom2])
                        rows.append(contact.frame[:3] @ (jac_b - jac_a)[:, 6:])
                        distances.append(.001 - contact.dist)
                    if not rows:
                        break
                    j = np.stack(rows)
                    delta = j.T @ solve(j @ j.T + .00025 * np.eye(len(j)), np.array(distances), assume_a="pos")
                    q = np.clip(q + np.clip(delta, -.18, .18), *self.ranges.T)
                joints[t] = q
                previous_correction = q - original

    def clip(self, clip):
        pos, orient, angles = clip["root_pos"], clip["root_quat"], clip["dof_pos"]
        if float(clip["fps"]) != 50 or angles.shape != (len(pos), 29):
            raise ValueError("Expected 50 Hz G1 joint positions in the source bank")
        first = pos[0].copy()
        first[2] = 0.
        roots = (pos - first) * self.height_scale + first
        outputs, raw_gaps, source_floor = [], [], []
        prev = prev_seed = prev_rot = None
        for p, quat, q, goal in zip(pos, orient, angles, roots):
            self.src.set_pose(p, quat, q)
            floor = self.src.lowest()
            gaps = self.height_scale * np.maximum(0., self.src.contact_points()[:, 2] - min(0., floor))
            ret = self.fit_frame(goal, quat, q, prev, prev_seed, prev_rot, gaps)
            outputs.append(ret[:3])
            prev, prev_seed, prev_rot = ret[2:]
            raw_gaps.append(gaps)
            source_floor.append(floor)
        p, quat, q = map(np.stack, zip(*outputs))
        # Continuous quaternion branch and a zero-lag local polynomial remove framewise contact jitter.
        quat *= np.r_[1, np.cumprod(np.where((quat[1:] * quat[:-1]).sum(1) < 0, -1, 1))][:, None]
        smooth = lambda a: savgol_filter(a, 5, 2, axis=0)
        p, quat, q = smooth(p), smooth(quat), np.clip(smooth(q), *self.ranges.T)
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        self.refine_support(p, quat, q, raw_gaps, self.height_scale * np.maximum(source_floor, 0.))
        q = np.clip(smooth(q), *self.ranges.T)
        self.separate_links(p, quat, q)
        points, clearance, target_gaps, self_penetration = [], [], [], []
        for i in range(len(q)):
            self.dst.set_pose(p[i], quat[i], q[i])
            # Ground to the source's positive clearance. Retain airborne phases and avoid penetrating the plane.
            p[i, 2] += self.height_scale * max(source_floor[i], 0.) - self.dst.lowest()
            self.dst.set_pose(p[i], quat[i], q[i])
            points.append(self.dst.data.xpos[self.dst.support].copy())
            clearance.append(self.dst.lowest())
            target_gaps.append(self.dst.contact_points()[:, 2])
            self_penetration.append(min([0.] + [c.dist for c in self.dst.data.contact if c.geom1 and c.geom2]))
        v, w, qd = velocities(p, quat, q, 50.)
        A, B = np.array(raw_gaps), np.array(target_gaps)
        src, dst = A < .04, B < .04
        summary = dict(frames=len(q), start_root_m=float(p[0, 2]), end_root_m=float(p[-1, 2]),
                       min_floor_m=float(min(clearance)), worst_self_overlap_m=float(-min(self_penetration)),
                       joint_limit_excess_rad=float(max(0., (q - self.ranges[:, 1]).max(), (self.ranges[:, 0] - q).max())),
                       max_joint_speed_ratio=float(np.max(np.abs(qd) / zenbot.SPEED)),
                       support_f1=float(2 * (src & dst).sum() / max(1, src.sum() + dst.sum())),
                       source_support_frames=int(src.any(1).sum()), target_support_frames=int(dst.any(1).sum()))
        fields = dict(root_pos=p, root_quat=quat, dof_pos=q, root_lin_vel=v, root_ang_vel=w, dof_vel=qd,
                      key_body_pos=np.stack(points))
        return dict(fps=np.int32(50), **{k: v.astype(np.float32) for k, v in fields.items()}), summary


def prepare(source, destination):
    retarget = Retargeter()
    expected = {name + ".npz" for name in ACTIVE_CLIPS}
    if {p.name for p in source.glob("*.npz")} != expected:
        raise ValueError("The source directory must contain exactly the ten named G1 get-up references")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(robot="zenbot23_getup", mass_kg=float(retarget.dst.model.body_mass.sum()),
                  method="bounded differential IK from the ten G1 LeggedLab get-up recordings",
                  joint_names=zenbot.DOF, key_body_names=zenbot.KEY_BODIES, clips={})
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "recovery"
        stage.mkdir()
        for name in sorted(expected):
            path = source / name
            with np.load(path, allow_pickle=False) as clip:
                fields, details = retarget.clip(clip)
            np.savez_compressed(stage / name, **fields)
            details["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["clips"][name] = details
            print(name, json.dumps(details), flush=True)
        (stage / "retarget.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            if sorted(p.name for p in destination.iterdir()) != sorted(p.name for p in stage.iterdir()) or any(
                    (destination / p.name).read_bytes() != p.read_bytes() for p in stage.iterdir()):
                raise FileExistsError(f"Different data already exists in {destination}")
        else:
            stage.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/g1_getup/recovery"))
    parser.add_argument("--dest", type=Path, default=Path("data/zenbot23_getup/recovery"))
    args = parser.parse_args()
    prepare(args.source, args.dest)
