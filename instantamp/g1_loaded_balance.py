"""Observe settled foot contacts for slow load transfer and post-disturbance balance."""

import mujoco
import numpy as np


class LoadedBalance:
    def __init__(self, model, feet, sole_offset):
        self.model = model
        self.feet = feet
        self.sole_offset = sole_offset
        self.floor = model.geom("floor").id
        self.foot = {geom: i for i, body in enumerate(feet) for geom in np.flatnonzero(model.geom_bodyid == body)}
        self.weight = model.body_mass.sum() * 9.81
        self.wrench = np.empty(6)
        self.reset()

    def reset(self):
        self.settling = 0
        self.settled_tilt = None
        self.last_tilt = None
        self.next_tilt_check = 0
        self.force = np.zeros(2)
        self.load_count = 0
        self.unloaded = 0
        self.slow_load = False
        self.shift = None
        self.replant = False
        self.still_frames = 0
        self.previous_feet = None
        self.impact_settling = None

    def after_impulse(self):
        """The force braking a known impulse is not a persistent external load."""
        if not self.slow_load:
            self.impact_settling = 0
            self.force[:] = 0.
            self.load_count = 0
            self.shift = None

    def _new_footprint(self, data, loads, speed):
        if not self.replant:
            return None
        soles = np.array([(data.xpos[f] + data.xmat[f].reshape(3, 3) @ self.sole_offset)[:2]
                          for f in self.feet])
        quiet_feet = (self.previous_feet is not None and
                      np.linalg.norm(soles - self.previous_feet, axis=1).max() < .055 * .02)
        self.previous_feet = soles
        planted = (quiet_feet and speed < .10 and np.all(loads[:, 2] > .1 * self.weight)
                   and loads[:, 2].sum() > .5 * self.weight and data.xmat[1, 8] > .866)
        self.still_frames = self.still_frames + 1 if planted else 0
        if self.still_frames >= 6:
            self.replant = False
            self.still_frames = 0
            return soles
        return None

    def update(self, data):
        """Return an observed new footprint and whether the outward tilt needs a forecast."""
        tilt = np.arccos(np.clip(data.xmat[1, 8], -1., 1.))
        speed = np.linalg.norm(data.qvel[:2])
        if self.settled_tilt is None:
            quiet = (speed < .035 and np.linalg.norm(data.qvel[3:6]) < .075 and
                     .68 < data.qpos[2] < .83 and data.xmat[1, 8] > .92)
            self.settling = self.settling + 1 if quiet else 0
            if self.settling < 50:
                return None, False
            self.settled_tilt = self.last_tilt = tilt
        loads = np.zeros((2, 3))
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = contact.geom
            other = b if a == self.floor else a if b == self.floor else -1
            if other in self.foot and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.model, data, i, self.wrench)
                force = self.wrench[:3] @ contact.frame.reshape(3, 3)
                loads[self.foot[other]] += np.sign(force[2]) * force

        footprint = self._new_footprint(data, loads, speed)
        if self.impact_settling is not None:
            settled = (speed < .035 and np.linalg.norm(data.qvel[3:6]) < .075
                       and .68 < data.qpos[2] < .83 and data.xmat[1, 8] > .92
                       and np.all(loads[:, 2] > .1 * self.weight)
                       and loads[:, 2].sum() > .5 * self.weight)
            self.impact_settling = self.impact_settling + 1 if settled else 0
            if self.impact_settling < 50:
                return footprint, False
            self.impact_settling = None
            self.settled_tilt = self.last_tilt = tilt
            self.next_tilt_check = 0

        # At low speed the horizontal ground reaction opposes the applied load.
        # Shift the COM target back over the support polygon gradually.
        estimate = np.clip(-loads[:, :2].sum(axis=0), -.17 * self.weight, .17 * self.weight)
        self.force += .12 * (estimate - self.force)
        magnitude = np.linalg.norm(self.force)
        self.load_count = self.load_count + 1 if magnitude > .012 * self.weight and speed < .06 else 0
        self.unloaded = self.unloaded + 1 if magnitude < .008 * self.weight else 0
        if self.load_count >= 6:
            self.slow_load = True
        if self.unloaded >= 20:
            self.slow_load = False
        self.shift = None
        if self.slow_load:
            blend = np.clip((magnitude / self.weight - .01) / .012, 0., 1.)
            distance = min(.12, .6 * .82 * magnitude / self.weight)
            self.shift = blend * distance * self.force / max(magnitude, .001)

        test_lean = False
        if self.next_tilt_check == 0:
            test_lean = tilt - self.settled_tilt > np.deg2rad(6.) and tilt - self.last_tilt > np.deg2rad(.2)
            self.last_tilt = tilt
        self.next_tilt_check = (self.next_tilt_check + 1) % 35

        return footprint, test_lean
