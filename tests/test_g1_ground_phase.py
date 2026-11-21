import unittest
import numpy as np
from instantamp.g1_ground_phase import knee_extension_frame, sustained_palm
from tools.score_g1_ground_phase import profile


class GroundPhase(unittest.TestCase):
    def test_reacquire_only_after_bilateral_contact_is_quiet(self):
        import mujoco
        from instantamp.g1_loaded_balance import LoadedBalance

        model = mujoco.MjModel.from_xml_string("""
          <mujoco><worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body name="pelvis" pos="0 0 .75"><freejoint/>
              <geom type="sphere" size=".05" mass="20" contype="0" conaffinity="0"/>
              <body name="left" pos="0 .10 -.72"><geom type="box" size=".08 .04 .03"/></body>
              <body name="right" pos="0 -.10 -.72"><geom type="box" size=".08 .04 .03"/></body>
            </body>
          </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        feet = [model.body(n).id for n in ("left", "right")]
        observer = LoadedBalance(model, feet, np.zeros(3))
        observer.replant = True
        poses = data.qpos.copy()
        for _ in range(55):
            footprint, _ = observer.update(data)
            self.assertIsNone(footprint)
        footprint, _ = observer.update(data)
        np.testing.assert_allclose(footprint, data.xpos[feet, :2])
        np.testing.assert_array_equal(data.qpos, poses)
        self.assertIsNone(observer.update(data)[0])
        self.assertIsNone(observer.shift)
        observer.after_impulse()
        for _ in range(49):
            observer.update(data)
        self.assertEqual(observer.impact_settling, 49)
        data.qvel[0] = .1
        observer.update(data)
        self.assertEqual(observer.impact_settling, 0)
        data.qvel[0] = 0.
        for _ in range(50):
            observer.update(data)
        self.assertIsNone(observer.impact_settling)

    def test_standing_ranking_checks_every_second_after_the_first_rise(self):
        from instantamp.g1_floating_base import supported_after_rise

        samples = np.ones((100, 2), dtype=bool)
        samples[48:53, 1] = False
        self.assertTrue(supported_after_rise(samples))
        samples[53, 1] = False
        self.assertFalse(supported_after_rise(samples))
        samples[:, 1] = True
        samples[12, 0] = False
        self.assertFalse(supported_after_rise(samples))
        self.assertFalse(supported_after_rise(samples[15:60]))

    def test_bilateral_transfer_follows_the_last_palm_load(self):
        from tools.score_g1_weight_transfer import transfer_time

        palms = np.zeros((70, 2))
        feet = np.full((70, 2), 0.4)
        palms[2:9, 0] = 0.3
        palms[22:32, 1] = 0.12
        feet[32:39, 1] = 0.01
        self.assertAlmostEqual(transfer_time(palms, feet, 0.02, 1.0), 0.8)
        feet[39:, 0] = 0.01
        self.assertIsNone(transfer_time(palms, feet, 0.02, 1.0))

    def test_sustained_palm_excludes_late_contact_and_transient_impacts(self):
        q = np.zeros((80, 36))
        q[:, 3] = 1.0
        q[:45, 2] = 0.36
        q[45:, 2] = 0.73
        q[:, [10, 16]] = 1.6
        q[35:, [10, 16]] = 1.2
        v = np.zeros((80, 35))
        onset = knee_extension_frame(q, 45)
        self.assertEqual(onset, 35)
        hit = np.zeros(80, bool)
        hit[5:8] = True
        hit[36:45] = True
        self.assertIsNone(sustained_palm(hit, onset))
        hit[14:20] = True
        measured = profile(q, hit, 0.02, v)
        self.assertTrue(measured["sustained_palm_before_extension"])
        self.assertAlmostEqual(measured["palm_before_extension_s"], 0.12)
        self.assertAlmostEqual(measured["extension_to_stand_s"], 0.2)

    def test_slow_amp_targets_keep_reference_and_scale_velocity(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from tools.train_g1_tracking_residual import reference_windows

        with TemporaryDirectory() as d:
            n = 25
            np.savez(
                Path(d) / "motion.npz",
                root_pos=np.zeros((n, 3)),
                root_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
                root_ang_vel=np.tile([1.0, 0.0, 0.0], (n, 1)),
                dof_pos=np.tile(np.arange(n)[:, None] * 0.1, (1, 29)),
                dof_vel=np.full((n, 29), 5.0),
                key_body_pos=np.zeros((n, 6, 3)),
            )
            fast = reference_windows(Path(d))[0]
            slow = reference_windows(Path(d), 1.5)[0]
            self.assertEqual(len(slow), 37 - 9)
            np.testing.assert_allclose(
                slow[-1, -1, 9:38], fast[-1, -1, 9:38], atol=1.0e-6
            )
            np.testing.assert_allclose(slow[:, :, 38:67], 5.0 / 1.5)
            np.testing.assert_allclose(slow[:, :, 6], 1.0 / 1.5)
