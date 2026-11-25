import copy

import jax
import mujoco
import numpy as np
import pytest

from instantamp import zenbot
from instantamp.sim import build_scene
from instantamp.zenbot_domain_randomization import DomainRandomization, MODEL_FIELDS


@pytest.fixture(scope="module")
def dynamics():
    model = build_scene(robot_name=zenbot.MIXED_TASK)
    return model, DomainRandomization(model, zenbot)


def test_startup_ranges_and_cross_body_relationships(dynamics):
    model, randomization = dynamics
    values = jax.jit(lambda key: randomization.sample(key, 2048))(jax.random.key(519))
    values = jax.tree.map(np.asarray, values)
    for key, base in (("kp", zenbot.KP), ("kd", zenbot.KD)):
        ratio = values[key] / base
        assert np.all((ratio >= .8) & (ratio <= 1.2))
        assert ratio.min() < .81 and ratio.max() > 1.19
    ratio = values["dof_armature"][:, 6:] / model.dof_armature[6:]
    assert np.all((ratio >= .8 - 1e-7) & (ratio <= 1.2 + 1e-7))
    friction = values["dof_frictionloss"][:, randomization.pair_dofs]
    means = friction.mean(-1)
    reference_means = np.array([.3975, .375, .2975, .4125, .55, .35])
    assert np.all((means / reference_means >= .5 - 1e-6) & (means / reference_means <= 1.5 + 1e-6))
    relative_difference = (friction[..., 0] - friction[..., 1]) / means
    assert abs(relative_difference).max() <= .100001
    assert relative_difference.min() < -.095 and relative_difference.max() > .095
    np.testing.assert_allclose(means.mean(0), reference_means, atol=.01)

    pelvis = randomization.pelvis
    assert np.max(abs(values["body_mass"][:, pelvis] - model.body_mass[pelvis])) <= 3
    limbs = randomization.limb_bodies
    ratio = values["body_mass"][:, limbs] / model.body_mass[limbs]
    assert np.all((ratio >= .8 - 1e-7) & (ratio <= 1.2 + 1e-7))
    np.testing.assert_allclose(values["body_inertia"][:, limbs] / model.body_inertia[limbs],
                               np.broadcast_to(ratio[..., None], (2048, len(limbs), 3)), rtol=2e-6)
    offset = values["body_ipos"][:, randomization.com_bodies] - model.body_ipos[randomization.com_bodies]
    assert abs(offset).max() <= .030001
    np.testing.assert_allclose(offset[:, 0], offset[:, 1], atol=3e-8)
    untouched = [i for i in range(model.nbody) if i not in randomization.com_bodies]
    np.testing.assert_allclose(values["body_ipos"][:, untouched],
                               np.broadcast_to(model.body_ipos[untouched], (2048, len(untouched), 3)), atol=1e-8)

    mu = values["geom_friction"][:, randomization.robot_geoms, 0]
    assert mu.min() >= .3 and mu.max() <= 1.6
    assert np.unique(mu).size > 64  # Newton backend samples continuously.
    np.testing.assert_array_equal(values["geom_friction"][:, model.geom_bodyid == 0, 0], 1.)
    assert np.all(model.geom_priority == 0)


def test_reference_constants_match_mujoco_set_const(dynamics):
    model, randomization = dynamics
    values = randomization.sample(jax.random.key(125), 5)
    # Independent engine oracle: changed COM, armature and body masses must
    # reach contact/friction inverse weights, not just the forward dynamics.
    for row in range(5):
        native = copy.copy(model)
        for name in ("body_ipos", "body_mass", "body_inertia", "dof_armature"):
            getattr(native, name)[:] = np.asarray(values[name][row])
        data = mujoco.MjData(native)
        mujoco.mj_setConst(native, data)
        for name in ("body_invweight0", "dof_invweight0", "dof_M0", "actuator_acc0", "body_subtreemass"):
            np.testing.assert_allclose(np.asarray(values[name][row]), getattr(native, name),
                                       rtol=3e-4, atol=2e-5, err_msg=name)


def test_nominal_disabled_sampling_preserves_motion_train_friction(dynamics):
    model, randomization = dynamics
    first = randomization.sample(jax.random.key(0), 3, enabled=False)
    other = randomization.sample(jax.random.key(1), 3, enabled=False)
    assert set(first) == {*MODEL_FIELDS, "kp", "kd"}
    for name in first:
        np.testing.assert_array_equal(first[name], other[name])
    np.testing.assert_allclose(first["dof_frictionloss"][0, randomization.pair_dofs],
                               [[.405, .390], [.425, .325], [.250, .345], [.4125, .4125], [.55, .55], [.35, .35]])
    for name in ("body_ipos", "body_mass", "body_inertia", "body_subtreemass", "dof_armature"):
        np.testing.assert_allclose(first[name][0], getattr(model, name), rtol=1e-6)
    # Reusing a startup key is deterministic; a different key changes dynamics.
    a = randomization.sample(jax.random.key(11), 2)
    b = randomization.sample(jax.random.key(11), 2)
    c = randomization.sample(jax.random.key(12), 2)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.array_equal(a["body_mass"], c["body_mass"])


@pytest.mark.parametrize("robot_mu", [.4, 1.5])
def test_ground_contact_uses_motion_train_maximum_rule(dynamics, robot_mu):
    model, randomization = dynamics
    native = copy.copy(model)
    native.geom_friction[randomization.robot_geoms, 0] = robot_mu
    data = mujoco.MjData(native)
    data.qpos[2] = .45
    mujoco.mj_forward(native, data)
    ground_contacts = [contact for contact in data.contact
                       if 0 in native.geom_bodyid[contact.geom]]
    assert ground_contacts
    for contact in ground_contacts:
        np.testing.assert_allclose(contact.friction[:2], max(1., robot_mu))
