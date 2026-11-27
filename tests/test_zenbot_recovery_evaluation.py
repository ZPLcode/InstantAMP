"""Recovery quality must distinguish a clean rise from a successful retry."""

from types import SimpleNamespace

import numpy as np
import pytest

from instantamp import zenbot, zenbot26
from tools import evaluate_zenbot_joint
from tools.evaluate_zenbot_joint import recovery_outcome


def score(heights, profile=zenbot, **kwargs):
    heights = np.asarray(heights)
    posture = (heights > .58) & (heights <= .78)
    return recovery_outcome(heights, posture, posture.astype(float), dt=.1, profile=profile, **kwargs)


def test_26dof_requires_both_a_fresh_rise_and_the_attempt_height():
    stand = [.68] * 15
    # Start high, fall, then rise only 6 cm above the trough. No peak has been armed.
    assert score([.46, .35, .27, .33, .23] + stand, zenbot26)["first_attempt_success"]
    # A rise over 8 cm that never reaches 30 cm is preparatory.
    assert score([.18, .299, .20] + stand, zenbot26)["lift_setbacks"] == 0
    # Both 8 cm thresholds are inclusive. A new rise re-arms after each setback.
    h = [.30, .22, .30, .219] + stand
    assert score(h, zenbot26, initial_height=.22)["lift_setbacks"] == 2
    assert score(h, zenbot)["lift_setbacks"] == 0


def test_final_success_does_not_hide_an_aborted_rise():
    clean = score([.2, .3, .45, .5, .6] + [.68] * 15)
    retry = score([.2, .3, .48, .35, .3, .45, .5, .6] + [.68] * 15)
    assert clean["success"] and clean["first_attempt_success"]
    assert clean["lift_setbacks"] == 0
    assert clean["first_supported_stand_s"] == .5
    assert retry["success"] and not retry["first_attempt_success"]
    assert retry["lift_setbacks"] == 1


def test_shoulder_and_pelvis_setbacks_share_one_attempt():
    stand = [.68] * 15
    # A shoulder-only attempt before the pelvis reaches 30 cm, then a separate simultaneous setback.
    result = score([.21, .22, .21, .20, .37, .26] + stand, zenbot26, initial_shoulder_height=.15,
                   shoulder_heights=[.28, .27, .15, .14, .45, .27] + [1.] * 15)
    assert result["success"] and not result["first_attempt_success"]
    assert result["lift_setbacks"] == 2


def test_setbacks_ignore_small_bobs_but_count_separate_failed_rises():
    bobs = score([.2, .46, .44, .48, .43, .5, .6] + [.68] * 15)
    retries = score([.2, .5, .35, .33, .34, .5, .35, .5, .6] + [.68] * 15)
    assert bobs["first_attempt_success"]
    assert retries["lift_setbacks"] == 2


def test_transient_height_and_unsupported_stance_are_not_success():
    transient = score([.2] * 10 + [.68] * 5 + [.2] * 10)
    unsupported = recovery_outcome(np.full(20, .68), np.ones(20, bool), np.zeros(20), dt=.1)
    for result in (transient, unsupported):
        assert not result["success"] and not result["first_attempt_success"]
        assert result["first_supported_stand_s"] is None


def test_lost_balance_counts_even_without_a_large_height_drop():
    postures = np.r_[np.ones(10, bool), np.zeros(3, bool), np.ones(15, bool)]
    result = recovery_outcome(np.full(len(postures), .65), postures, np.ones(len(postures)), dt=.1)
    assert result["success"] and not result["first_attempt_success"]
    assert result["lift_setbacks"] == 0 and result["refalls"] == 1
    assert result["sustained_supported_stand_s"] > result["first_supported_stand_s"]


@pytest.mark.parametrize("fall", [False, True])
def test_push_trials_release_force_and_keep_fall_momentum_and_history(monkeypatch, fall):
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6),
                           xmat=np.tile(np.eye(3).ravel(), (2, 1)), xfrc_applied=np.zeros((2, 6)))
    robot = SimpleNamespace(data=data, profile=SimpleNamespace(TORSO="torso_Link"),
                            model=SimpleNamespace(body=lambda name: SimpleNamespace(id=1)), history_steps=0)
    starts, scored = [], []

    def stand():
        data.qpos[2] = .68
        data.qvel.fill(0.)
        robot.history_steps = 0
        starts.append(True)

    def step():
        robot.history_steps += 1
        if np.any(data.xfrc_applied):
            data.qvel[:3] += data.xfrc_applied[1, :3] * .001
            if fall:
                data.qpos[2] = .30

    def recovery(native, steps):
        assert native is robot and steps == 500
        assert np.linalg.norm(data.qvel[:3]) > 0.
        assert robot.history_steps == 120  # Two seconds standing plus the force pulse.
        assert not np.any(data.xfrc_applied)
        scored.append(True)
        return dict(success=True, first_attempt_success=True)

    robot.step = step
    monkeypatch.setattr(evaluate_zenbot_joint, "_recovery_trial", recovery)
    result = evaluate_zenbot_joint._push_recovery_trials(SimpleNamespace(native=robot, stand=stand), 500, forces=(300.,))
    assert len(starts) == len(result) == 4
    assert len(scored) == (4 if fall else 0)
    assert all(trial["fall_observed"] == fall for trial in result)
