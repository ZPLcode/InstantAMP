"""Forward kinematics of the robot in the compiled scene: a free root (the pelvis) and at most one hinge per body, anchored at
the body origin (checked at construction). Other joints are ignored: the bodies below them are never asked for, which is checked too.

The task reads body poses of the final substep's qpos; MuJoCo's xpos after a step is one substep older. This is that refresh, used
by the simulator and by imagination alike. Only the bodies on the paths from the root to the requested ones are evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import mujoco
import numpy as np

from .rot import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    """Forward kinematics for a fixed list of bodies: `forward` takes the root pose and the joint angles and returns the world
    poses of exactly those bodies."""

    def __init__(self, m: mujoco.MjModel, bodies: Sequence[str], qadr: Sequence[int] | None = None):
        """`qadr`: the qpos address of each column of the joint-angle vector q that `forward` takes; by default the hinges in
        joint order, qpos[7:]."""
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[0] == 1, "joint 0 must be the root's free joint"
        self.parent = [int(b) for b in m.body_parentid]
        self.pos = np.asarray(m.body_pos, np.float32)
        self.rot = np.asarray(quat_to_mat(np.asarray(m.body_quat, np.float32)), np.float32)
        self.axis = [None] * m.nbody
        self.col = [-1] * m.nbody  # the hinge's column in q = qpos[7:]
        col_of = {int(a): c for c, a in enumerate(qadr)} if qadr is not None else None
        other = set()  # bodies that carry a joint other than a hinge
        for j in range(1, m.njnt):
            b = int(m.jnt_bodyid[j])
            adr = int(m.jnt_qposadr[j])
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or (col_of is not None and adr not in col_of):
                # a ball joint, or a hinge outside the joint-angle vector: the body must not be on a requested path
                other.add(b)
                continue
            assert self.col[b] < 0 and b not in other, "one hinge per body"
            assert np.allclose(m.jnt_pos[j], 0), "hinges must be anchored at the body origin"
            self.axis[b], self.col[b] = np.asarray(m.jnt_axis[j], np.float32), adr - 7 if col_of is None else col_of[adr]
        self.out = [m.body(n).id for n in bodies]
        need = set()
        for b in self.out:
            while b > 1:
                need.add(b)
                b = self.parent[b]
        self.order = sorted(need)  # MuJoCo numbers parents before their children
        assert not (need & other), f"a requested body hangs below a joint that is not in q: {[m.body(b).name for b in need & other]}"

    def forward(self, p, R, q):
        """World positions (..., n, 3) and rotations (..., n, 3, 3) of the requested bodies, given the root pose (p, R)."""
        P, Rs = {1: p}, {1: R}  # by body id: body 1 is the pelvis (the free root), body 0 the world
        for b in self.order:
            par = self.parent[b]
            P[b] = P[par] + mv3(Rs[par], self.pos[b])
            Rb = mm3(Rs[par], self.rot[b])
            Rs[b] = Rb if self.col[b] < 0 else mm3(Rb, axis_angle_to_mat(self.axis[b], q[..., self.col[b]]))
        return jnp.stack([P[b] for b in self.out], -2), jnp.stack([Rs[b] for b in self.out], -3)
