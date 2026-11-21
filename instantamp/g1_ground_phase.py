"""Contact geometry and phase timing for G1 recovery demonstrations and physical rollouts."""

import mujoco
import numpy as np
from . import env
from .stability import first_sustained


class HandContacts:
    """Palm and distal forearm contact with the floor, per side, in body weights."""

    def __init__(self, model):
        self.model = model
        self.floor = model.geom("floor").id
        self.weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
        self.colliders = np.flatnonzero(
            (model.geom_contype != 0) & (np.arange(model.ngeom) != self.floor)
        )
        self.types = {}
        for side, name in enumerate(("left", "right")):
            for kind, joint in (
                (0, "wrist_yaw_collision"),
                (0, "wrist_yaw_2_collision"),
                (1, "wrist_roll_collision"),
            ):
                self.types[model.geom("robot_" + name + "_" + joint).id] = side, kind

    def forces(self, data):
        load = np.zeros((2, 2))
        wrench = np.empty(6)
        for i, c in enumerate(data.contact):
            a, b = c.geom
            other = b if a == self.floor else a if b == self.floor else -1
            if other in self.types and c.efc_address >= 0:
                mujoco.mj_contactForce(self.model, data, i, wrench)
                load[self.types[other]] += (
                    abs(c.frame.reshape(3, 3)[:, 2] @ wrench[:3]) / self.weight
                )
        return load

    def reference_clearances(self, poses):
        """Mocap gives poses. Infer proximity relative to the lowest penetrating collision shape."""
        data = mujoco.MjData(self.model)
        result = np.full((len(poses), 2, 2), np.inf)
        distances = np.zeros(self.model.ngeom)
        for tick, pose in enumerate(poses):
            data.qpos[:] = pose
            mujoco.mj_kinematics(self.model, data)
            for geom in self.colliders:
                distances[geom] = mujoco.mj_geomDistance(
                    self.model, data, self.floor, int(geom), 2.0, None
                )
            floor = min(0.0, distances[self.colliders].min())
            for geom, (side, kind) in self.types.items():
                result[tick, side, kind] = min(
                    result[tick, side, kind], distances[geom] - floor
                )
        return result


def knee_extension_frame(poses, upright):
    """First sustained 20-degree straightening after maximum pre-stand bilateral knee flexion."""
    knees = poses[:, [10, 16]].mean(-1)
    end = len(knees) if upright is None else upright + 1
    peak = int(np.argmax(knees[:end]))
    return first_sustained(
        (np.arange(len(knees)) >= peak) & (knees < knees[peak] - np.deg2rad(20)), 4
    )


def sustained_palm(mask, end):
    if end is None:
        end = len(mask)
    return first_sustained(np.asarray(mask)[:end], round(0.12 / env.DT))
