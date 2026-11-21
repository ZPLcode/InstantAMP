from types import SimpleNamespace

import numpy as np

from instantamp.native_gait import RunGains, ground_contacts
from tools.evaluate_g1_walk_recovery import RUN_PROFILE, Phase, gait_command, summarize


def test_ground_entry_forecasts_leave_live_physics_actor_and_solver_untouched():
    import mujoco
    from instantamp.g1_floating_base import FloatingBaseTracking
    from instantamp.g1_ground_phase import HandContacts
    from instantamp.sim import build_scene
    from tools.evaluate_g1_walk_recovery import select_ground_entry

    model = build_scene(robot_name="g1_getup")
    tables = SimpleNamespace(starts=np.array([0]), end=np.array([1, 1]), q=np.zeros((2, model.nq)))
    tracker = FloatingBaseTracking(model, tables, [1], 9, np.zeros((2, 59)), np.zeros((2, 7)))
    tracker.tick = 73
    tracker.probe_history.fill(-2)
    tracker.hand_wrench.fill(3)
    tracker.arm_jac.fill(4)

    class Player:
        def frame(self):
            return np.zeros((1, 114), dtype=np.float32)

        def step(self):
            mujoco.mj_step(self.model, self.data)
            # Exercise the same mutable workspaces used by a matched tracking trajectory.
            self.measured.solver.data.time += 1.0
            self.measured.probe.qpos[0] += 0.123
            self.measured.probe_history.fill(0)
            self.measured.hand_wrench.fill(0)
            self.measured.arm_jac.fill(0)
            self.previous.fill(0)

    player = Player()
    player.model, player.measured, player.policy = model, tracker, None
    player.data = mujoco.MjData(model)
    player.previous = np.full(29, 3.0, dtype=np.float32)
    player.history, player.memory = np.full((1, 570), 2.0), np.array([4.0])
    player.data.qvel[0] = 0.21
    mujoco.mj_forward(model, player.data)
    q, v = player.data.qpos.copy(), player.data.qvel.copy()
    x_probe = tracker.probe.qpos[0]
    select_ground_entry(player, HandContacts(model), 2)
    np.testing.assert_array_equal(player.data.qpos, q)
    np.testing.assert_array_equal(player.data.qvel, v)
    assert player.data.time == 0 and tracker.solver.data.time == 0
    assert tracker.tick == 73 and tracker.probe.qpos[0] == x_probe
    assert (tracker.probe_history == -2).all() and (tracker.hand_wrench == 3).all() and (tracker.arm_jac == 4).all()
    assert (player.previous == 3).all() and (player.history == 2).all() and player.memory[0] == 4


def test_flight_excludes_other_body_contacts():
    model = SimpleNamespace(geom_bodyid=np.array([0, 3, 4, 5]))
    feet = [3, 4]
    def contact(a, b, dist=0., address=0):
        return SimpleNamespace(geom=(a, b), efc_address=address, dist=dist)
    data = SimpleNamespace(contact=[contact(0, 1), contact(2, 0), contact(3, 0), contact(2, 3)])
    assert ground_contacts(model, data, 0, feet) == (3, True)
    data.contact = [contact(0, 1, .002), contact(0, 2, address=-1), contact(3, 0)]
    assert ground_contacts(model, data, 0, feet) == (0, True)
    data.contact = []
    assert ground_contacts(model, data, 0, feet) == (0, False)


def test_running_resumes_with_a_turn_and_walk_uses_the_export():
    command = gait_command(2.15, 50, True, 4)
    np.testing.assert_allclose(command, [.70, 0., .70])
    np.testing.assert_allclose(gait_command(2.15, 260, True, 4), [2.15, 0., 0.])
    np.testing.assert_allclose(gait_command(.55, 99, False, 0), [.55, 0., 0.])
    gains = RunGains.read(RUN_PROFILE)
    assert .15 < gains.push < .16


def test_running_requires_airborne_motion_before_and_after_the_fall():
    n = 2000
    trace = {"qpos": np.tile([0., 0., .75], (n, 1)), "phase": np.full(n, Phase.RESUMED),
             "body_velocity": np.tile([2., 0., 0.], (n, 1)), "airborne_substeps": np.ones(n, int),
             "floor_contact_before": np.zeros((n, 2), int)}
    evaluate = lambda: summarize(trace, 2., 0., True, [], SimpleNamespace(force=410.), RunGains.read(RUN_PROFILE))
    assert evaluate()["success"]
    trace["airborne_substeps"][-150:] = 0
    report = evaluate()
    assert report["resumed"] and not report["success"]


def test_slow_running_calibration_preserves_anchor_and_uses_planted_soles():
    from instantamp.native_gait import JogSpeedCalibration

    assert JogSpeedCalibration.read(RUN_PROFILE, .55) is None
    assert JogSpeedCalibration.read(RUN_PROFILE, 2.15) is None
    assert JogSpeedCalibration.read(RUN_PROFILE, 2.8) is None
    feet = [2, 3]
    contact = SimpleNamespace(geom=(0, 1), efc_address=0, dist=0.)
    model = SimpleNamespace(geom_bodyid=np.array([0, 2]))
    gait = SimpleNamespace(feet=feet, floor=0)
    estimates = []
    for origin in (0., 17.):
        cfg = JogSpeedCalibration.read(RUN_PROFILE, 1.85)
        assert cfg.neural == 2.025 and cfg.swing == .925
        data = SimpleNamespace(qpos=np.array([origin, 0., .75]), qvel=np.zeros(35), xpos=np.zeros((4, 3)),
                               xmat=np.repeat(np.eye(3).reshape(1, 9), 4, axis=0), contact=[contact])
        data.xpos[feet, 0] = origin
        cfg.command(gait, model, data, [1.85, 0., 0.])
        data.qpos[0] += .04
        actual = cfg.command(gait, model, data, [1.85, 0., 0.])
        np.testing.assert_allclose(cfg.forward_speed, .18)
        data.contact = []
        data.qpos[0] += .12
        np.testing.assert_allclose(cfg.command(gait, model, data, [1.85, 0., 0.]), actual)
        estimates.append(cfg.forward_speed)
    np.testing.assert_allclose(estimates[0], estimates[1])
