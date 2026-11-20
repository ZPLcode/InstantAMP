import jax
import jax.numpy as jnp
import numpy as np

from instantamp import env, nets
from instantamp.inference import load, policy_output


def test_gated_export_normalizes_each_actor_and_preserves_upright_actions(tmp_path):
    arms = np.array([joint.startswith(("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                       "left_wrist", "right_wrist")) for joint in env.DOF])
    rng = np.random.default_rng(42)
    data = {}
    for name in ("base", "style"):
        arrays = {"avg/0/w": rng.normal(size=(env.OBS_DIM, env.J)).astype(np.float32),
                  "avg/0/b": rng.normal(size=env.J).astype(np.float32),
                  "obs_norm/mean": rng.normal(size=env.OBS_DIM).astype(np.float32),
                  "obs_norm/var": rng.uniform(.05, 2., size=env.OBS_DIM).astype(np.float32),
                  "ode/dmax": .4, "ode/amax": 2.}
        np.savez(tmp_path / f"{name}.npz", **arrays)
        data.update({f"{name}/{key}": value for key, value in arrays.items()})
    np.savez(tmp_path / "combined.npz", **data, blend=.2 * arms.astype(np.float32), gate=True)
    frame = np.zeros((3, env.FRAME_DIM), np.float32)
    frame[:, 5] = [-1., 0., -1.]
    for knee in ("left_knee", "right_knee"):
        i = env.DOF.index(knee)
        frame[2, 9 + i] = 2. - env.DEFAULT_Q[i]
    obs = jnp.tile(frame, (1, env.HIST))
    pi, norm, ode = load(tmp_path / "combined.npz")
    raw = jax.jit(policy_output)(pi, norm, obs)
    base, base_norm, _ = load(tmp_path / "base.npz")
    style, style_norm, _ = load(tmp_path / "style.npz")
    expected_base = nets.pi_apply(base, jnp.clip(nets.rn_apply(base_norm, obs), -10., 10.))
    expected_style = nets.pi_apply(style, jnp.clip(nets.rn_apply(style_norm, obs), -10., 10.))
    np.testing.assert_allclose(raw[0], expected_base[0], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(raw[:, ~arms], expected_base[:, ~arms], rtol=1e-6, atol=1e-5)
    target = expected_base + .2 * (expected_style - expected_base)
    np.testing.assert_allclose(raw[1:, arms], target[1:, arms], rtol=1e-6, atol=1e-5)
    assert ode == (.4, 2.)
