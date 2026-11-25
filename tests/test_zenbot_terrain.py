"""Physical terrain and its reward geometry must describe the same surface."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from instantamp.zenbot_terrain import ATLAS_BORDER, HORIZONTAL_SCALE, TerrainAtlas, _patch


@pytest.fixture(scope="module")
def atlas():
    return TerrainAtlas.create()


def test_source_distribution_and_startup_origin_semantics(atlas):
    assert atlas.kinds.shape == (10, 20)
    # Counts from MT's seed-0, 200-patch generator.  Noise sampling must not
    # consume the independent layout RNG and change which terrains are trained.
    np.testing.assert_array_equal(np.bincount(atlas.kinds.ravel(), minlength=9),
                                  [66, 24, 25, 23, 8, 15, 16, 12, 11])
    origins = np.asarray(jax.jit(lambda key: atlas.sample_origins(key, 4096))(jax.random.key(74)))
    # Every terrain column is populated evenly, and all ten rows are reachable.
    _, columns = np.unique(origins[:, 1], return_counts=True)
    assert columns.min() == 204 and columns.max() == 205
    assert len(np.unique(origins[:, 0])) == 10
    # Fixed get-up and gait world blocks must both encounter all columns.
    assert len(np.unique(origins[:2048, 1])) == len(np.unique(origins[2048:, 1])) == 20
    np.testing.assert_array_equal(atlas.height_at(jnp.array([[0., 0.], [4., 4.], [-4., -4.]])), 0.)
    np.testing.assert_array_equal(atlas.contains(jnp.array([[1., 1.], [-1., 1.], [101., 10.]])),
                                  [True, False, False])
    border = round(ATLAS_BORDER / HORIZONTAL_SCALE)
    np.testing.assert_array_equal(atlas.heights[:border], 0.)
    np.testing.assert_array_equal(atlas.heights[-border:], 0.)
    for row, col in zip(*np.nonzero(atlas.kinds == 0)):
        np.testing.assert_array_equal(atlas.origins[row, col, 2], 0.)


def test_roughness_has_real_negative_heights_and_slopes_have_platforms():
    for kind in range(1, 7):
        patch = _patch(kind, .5, np.random.default_rng(9))
        assert patch.min() < 0 < patch.max()
        np.testing.assert_array_equal(patch[:11], 0.)
        np.testing.assert_allclose(patch / .0025, np.rint(patch / .0025), atol=2e-6)
    for kind, sign in ((7, 1), (8, -1)):
        patch = _patch(kind, .5, np.random.default_rng(9))
        assert sign * patch[80, 80] > .05
        np.testing.assert_array_equal(patch[60:100, 60:100], patch[80, 80])
        assert abs(patch[80, 80]) > abs(patch[20, 80])


def test_height_queries_match_mujoco_rays_and_have_correct_derivatives():
    atlas = TerrainAtlas.create(seed=3, rows=2, cols=3)
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .01])
    atlas.add_to_spec(spec, floor)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.ngeom == 10 and model.nbody == 1
    assert model.nhfield == np.count_nonzero(atlas.kinds)
    assert np.all(model.hfield_size[:, :2] == 4.)
    rng = np.random.default_rng(14)
    xy = atlas.origins[..., :2].reshape(-1, 2).repeat(10, axis=0) + rng.uniform(-3., 3., (60, 2))
    query = np.asarray(jax.jit(atlas.height_at)(jnp.asarray(xy, jnp.float32)))
    physical = []
    for point in xy:
        geom = np.zeros(1, np.int32)
        distance = mujoco.mj_ray(model, data, np.r_[point, 1.], np.array([0., 0., -1.]), None, 1, -1, geom)
        assert geom[0] >= 0 and distance > 0
        physical.append(1 - distance)
    np.testing.assert_allclose(query, physical, atol=3e-6)
    # Use a point strictly inside a non-flat triangle, not at a cell boundary.
    changes = np.abs(np.diff(atlas.heights, axis=1))
    row, col = np.unravel_index(changes.argmax(), changes.shape)
    point = jnp.array([col + .7, row + .2]) * HORIZONTAL_SCALE
    gradient = np.asarray(jax.grad(atlas.height_at)(point))
    eps = 1e-4
    numerical = np.array([(float(atlas.height_at(point + jnp.eye(2)[i] * eps)) -
                          float(atlas.height_at(point - jnp.eye(2)[i] * eps))) / (2 * eps)
                         for i in range(2)])
    np.testing.assert_allclose(gradient, numerical, atol=2e-3)
    assert np.linalg.norm(gradient) > .01


def test_heightfield_physically_supports_a_falling_body():
    atlas = TerrainAtlas.create(seed=3, rows=1, cols=2)
    spec = mujoco.MjSpec()
    spec.option.timestep = .005
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .01])
    atlas.add_to_spec(spec, floor)
    body = spec.worldbody.add_body(name="test_body")
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.04, 0., 0.], mass=1.)
    model = spec.compile()
    data = mujoco.MjData(model)
    row, col = np.argwhere(atlas.kinds != 0)[0]
    origin = atlas.origins[row, col]
    data.qpos[:3] = origin + [0., 0., .2]
    for _ in range(160):
        mujoco.mj_step(model, data)
    assert data.ncon > 0
    assert abs(data.qpos[2] - float(atlas.height_at(data.qpos[:2])) - .04) < .01


def test_robot_cannot_overflow_warp_heightfield_pair_search():
    from mujoco_warp._src.types import MJ_MAXCONPAIR

    from instantamp import zenbot
    from instantamp.sim import build_scene

    atlas = TerrainAtlas.create(seed=3, rows=1, cols=2)
    model = build_scene(robot_name=zenbot.MIXED_TASK, terrain=atlas)
    diameter = 2 * model.geom_rbound[model.geom_bodyid != 0].max()
    pitch_x = 2 * model.hfield_size[:, 0] / (model.hfield_ncol - 1)
    pitch_y = 2 * model.hfield_size[:, 1] / (model.hfield_nrow - 1)
    # At any orientation the AABB width is bounded by the bounding sphere
    # diameter. floor(min)/ceil(max) can include one extra cell on each axis.
    cells_x = np.ceil(diameter / pitch_x).astype(int) + 1
    cells_y = np.ceil(diameter / pitch_y).astype(int) + 1
    assert np.all(2 * cells_x * cells_y <= MJ_MAXCONPAIR)
