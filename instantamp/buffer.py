"""Device replay buffer of env transitions laid out (time, env), as XMimic's: fill level and write head on the device.

It keeps the latest steps of all envs in a ring. Collection adds one step at a time (`add`), world-model learning samples windows
of consecutive steps of one env (`sample_windows`) and policy learning samples starting states for imagination
(`sample_starts`).

Leaves are the flattened transition record (see train.InstantAMP.record). The dynamics context xi and the observation history are stored in float16; states,
actions and the task context in float32 / int32. A transition j ends a world-model window when its env was done, or when the
next transition does not start where it ended (a push overwrote the root velocity in between).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

HALF = ("xi", "obs")  # stored in float16 (obs: the 384-wide actor observation history)


def flatten(rec):
    """Turn one level of nested dicts into flat keys: {"a": {"b": x}} becomes {"a/b": x}."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update({f"{k}/{k2}": v2 for k2, v2 in v.items()})
        else:
            out[k] = v
    return out


def nest(flat):
    """The inverse of `flatten`."""
    out = {}
    for k, v in flat.items():
        if "/" in k:
            a, b = k.split("/", 1)
            out.setdefault(a, {})[b] = v
        else:
            out[k] = v
    return out


def init(T: int, rec_flat):
    """An empty buffer of T steps for records shaped like `rec_flat`, with the write head `ptr` and the fill level `size`."""
    buf = {k: jnp.zeros((T,) + v.shape, jnp.float16 if (v.dtype == jnp.float32 and k.startswith(HALF)) else v.dtype) for k, v in rec_flat.items()}
    buf["ptr"], buf["size"] = jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)
    return buf


def add(buf, rec_flat):
    """Write one step of all envs at the write head; once the buffer is full, this overwrites its oldest step."""
    ptr = buf["ptr"]
    out = {k: jax.lax.dynamic_update_index_in_dim(buf[k], rec_flat[k].astype(buf[k].dtype), ptr, 0) for k in rec_flat}
    return {**out, "ptr": (ptr + 1) % _T(buf), "size": jnp.minimum(buf["size"] + 1, _T(buf))}


def _T(buf):
    return buf["s_qpos"].shape[0]


def _rows(buf, age):
    """The row of the step that is `age` steps newer than the oldest one."""
    return (buf["ptr"] - buf["size"] + age) % _T(buf)


def _uniform_int(key, n, B):
    """B integers uniform in [0, n), where n may be a traced value."""
    return jnp.minimum((jax.random.uniform(key, (B,)) * n).astype(jnp.int32), n - 1)


def gather(buf, rows, envs):
    """The transitions at (rows, envs), with the float16 leaves converted back to float32."""
    return {k: (v[rows, envs].astype(jnp.float32) if v.dtype == jnp.float16 else v[rows, envs]) for k, v in buf.items() if k not in ("ptr", "size")}


def sample_windows(buf, key, B: int, k: int, recent: int = 0, envs=None, recovery_lengths=None):
    """B windows of k consecutive transitions, each from one env: (B, k, ...), and their validity mask (B, k). Transition j counts
    while transitions 0..j are finite and no earlier transition in the window ended its env's run of consecutive states (the
    env was done, or the next transition starts elsewhere, for example after a push). The transition after the window is read
    too, so the sampled ages leave room for it. recent > 0: only windows that start among the newest `recent` steps; envs: only these
    envs (a static array of indices)."""
    kt, ke = jax.random.split(key)
    N = buf["s_qpos"].shape[1]
    n = buf["size"] - k
    t0 = _uniform_int(kt, n, B) if recent <= 0 else jnp.maximum(n - recent, 0) + _uniform_int(kt, jnp.minimum(n, recent), B)
    env = jax.random.randint(ke, (B,), 0, N) if envs is None else jnp.asarray(envs)[jax.random.randint(ke, (B,), 0, len(envs))]
    if recovery_lengths is not None:
        # Sample the end of a complete recovery window, within the most recent filled rows.
        available_envs = jnp.arange(N) if envs is None else jnp.asarray(envs)
        age = (jnp.arange(_T(buf)) - (buf["ptr"] - buf["size"])) % _T(buf)
        clip = buf["reference_clip"][:, available_envs]
        end = buf["t"][:, available_envs]
        source = buf["reference_start"][:, available_envs] + end
        oldest = jnp.maximum(k - 1, buf["size"] - recent - 1) if recent else k - 1
        eligible = ((age >= oldest) & (age < buf["size"] - 1))[:, None]
        eligible &= (clip >= 0) & (end >= k-1) & (source < recovery_lengths[jnp.maximum(0, clip)])
        prefix = jnp.cumsum(eligible.reshape(-1), dtype=jnp.int32)
        selected = jnp.searchsorted(prefix, jax.random.randint(jax.random.fold_in(key, 1), (B,), 0,
                                                               jnp.maximum(prefix[-1], 1)), side="right")
        row, col = jnp.divmod(jnp.minimum(selected, prefix.size-1), len(available_envs))
        t0 = jnp.where(prefix[-1] > 0, age[row] - (k-1), t0)
        env = jnp.where(prefix[-1] > 0, available_envs[col], env)
    rows = _rows(buf, t0[:, None] + jnp.arange(k + 1))
    w = gather(buf, rows, env[:, None])
    nxt_q, nxt_v = w["s_qpos"][:, 1:], w["s_qvel"][:, 1:]
    jump = (jnp.abs(nxt_q - w["s1_qpos"][:, :-1]).max(-1) > 1e-5) | (jnp.abs(nxt_v - w["s1_qvel"][:, :-1]).max(-1) > 1e-5)
    cut = (w["done"][:, :-1] > 0) | jump
    w = {key_: v[:, :-1] for key_, v in w.items()}
    ok = jnp.cumprod(w["ok"], -1)
    w["mask"] = ok * jnp.cumprod(jnp.concatenate((jnp.ones((B, 1)), 1 - cut[:, :-1].astype(jnp.float32)), -1), -1)
    return w


def sample_starts(buf, key, B: int, hist: int = 0, down_frac: float = 0.0):
    """B starting states for imagination, uniform over the finite transitions: 2B random transitions are drawn and the first B
    finite ones kept (a way to skip non-finite ones with fixed array shapes). hist > 0: also the `hist` states before each start, oldest
    first, in the same env ("h_qpos", "h_qvel"), for a motion prior that looks at windows of consecutive frames. A state before a reset
    (the start is the first of its episode, or the buffer's oldest step is reached) is replaced by the earliest state that follows the
    reset, so the window never mixes two episodes.
    down_frac > 0: that share of the starts are drawn uniformly among the stored states of robots that have not been up yet
    (was_up == 0; by the inverse CDF of their mask over the whole buffer, so even a rare state is found); with none stored, uniform."""
    ka, ke, kd = jax.random.split(key, 3)
    N = buf["s_qpos"].shape[1]
    age = _uniform_int(ka, buf["size"], 2 * B)
    env = jax.random.randint(ke, (2 * B,), 0, N)
    n_d = int(round(down_frac * B))
    if n_d > 0:
        T = _T(buf)
        row_age = (jnp.arange(T) - (buf["ptr"] - buf["size"])) % T  # the age of each row; rows of age >= size hold no data yet
        down = ((buf["was_up"] <= 0) & (row_age < buf["size"])[:, None]).reshape(-1).astype(jnp.float32)
        cum = jnp.cumsum(down)
        u = jax.random.uniform(kd, (n_d,)) * cum[-1]
        flat = jnp.minimum(jnp.searchsorted(cum, u, side="right"), T * N - 1)
        has = cum[-1] > 0
        age = age.at[:n_d].set(jnp.where(has, row_age[flat // N], age[:n_d]))
        env = env.at[:n_d].set(jnp.where(has, flat % N, env[:n_d]))
    s = gather(buf, _rows(buf, age), env)
    if hist:
        back = jnp.arange(hist, 0, -1)  # steps back, oldest first
        a = age[:, None] - back  # (2B, hist) ages of the history states
        rows = _rows(buf, jnp.maximum(a, 0))
        h = gather(buf, rows, env[:, None])
        # the history state at age a is usable if no transition from a to the start's predecessor ended its env's run: the cumulative
        # product of "not done" from the newest history state backwards
        nd = (1 - h["done"]) * (a >= 0)
        valid = jnp.flip(jnp.cumprod(jnp.flip(nd, 1), 1), 1) > 0
        first = jnp.argmax(valid, 1)  # the oldest valid history state (hist if none)
        keep = jnp.arange(hist)[None, :] >= first[:, None]
        # replace the invalid (older) states by the oldest valid one, or by the start state itself if there is none
        src_q = jnp.concatenate((h["s_qpos"], s["s_qpos"][:, None]), 1)
        src_v = jnp.concatenate((h["s_qvel"], s["s_qvel"][:, None]), 1)
        pick = jnp.where(keep, jnp.arange(hist)[None, :], first[:, None])  # (2B, hist)
        pick = jnp.where(valid.any(1)[:, None], pick, hist)
        rows_i = jnp.arange(2 * B)[:, None]
        s = {**s, "h_qpos": src_q[rows_i, pick], "h_qvel": src_v[rows_i, pick]}
    idx = jnp.argsort(1 - s["ok"], stable=True)[:B]
    return jax.tree.map(lambda x: x[idx], s)


def sanitize(rec_flat):
    """Zero every non-finite float of a transition record and add ok = 1 if its states and context were finite."""
    fin = [jnp.isfinite(v).reshape(v.shape[0], -1).all(-1) for k, v in rec_flat.items() if jnp.issubdtype(v.dtype, jnp.floating)]
    ok = jnp.stack(fin, -1).all(-1)
    out = {k: (jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) if jnp.issubdtype(v.dtype, jnp.floating) else v) for k, v in rec_flat.items()}
    return {**out, "ok": ok.astype(jnp.float32)}
