"""Native standing-to-run-to-stop comparison from paired physical initial states."""
from pathlib import Path
import hashlib
import json
import sys

import jax.numpy as jnp
import numpy as np
from instantamp import zenbot
from instantamp.native_zenbot import ZenbotNative
from tools.evaluate_zenbot_directions import heading_kinematics
from tools.evaluate_zenbot_fall_recovery import standing_pose

root = Path('data/zenbot23_walk_run_recovery')
run = Path(sys.argv[1])
policy = run / 'ckpt' / (sys.argv[2] + '.npz')
dest = Path(sys.argv[3])
seeds = [305711, 934837, 8217539, 1654679, 5969183]
standing = standing_pose(root / 'walk_run/stand_4s_120f.npz')
player = ZenbotNative(root / 'recovery', policy)


def rollout(speed, steps):
    player.command = jnp.array([[speed, 0., 0.]], jnp.float32)
    rows = [player.data.qpos.copy()]
    for _ in range(steps):
        player.step()
        rows.append(player.data.qpos.copy())
    return np.array(rows)


def stats(q):
    v = heading_kinematics(q)[0]
    tilted = 1. - 2. * np.square(q[:, 4:6]).sum(1)
    fallen = (q[:, 2] < .42) | (tilted < .55)
    return dict(fell=bool(fallen.any()), minimum_height=float(q[:, 2].min()),
                first_fall_s=float(.02 * np.flatnonzero(fallen)[0]) if fallen.any() else None,
                last_second_velocity=np.mean(v[-50:], axis=0).tolist(),
                ending_xy_residual_speed=float(np.linalg.norm(v[-50:,:2], axis=1).mean()),
                ending_upright=bool(np.all((q[-50:, 2] > zenbot.UP_HEIGHT) & (tilted[-50:] > .8))))


result = []
for seed in seeds:
    delays = np.random.default_rng(seed).integers(3, 8, 5)
    branchpoint = None
    for ramp in (False, True):
        player.reset(standing, np.zeros(player.model.nv), delays, command=(0., 0., 0.))
        rollout(0., 50)
        walk, run_q = rollout(1., 200), rollout(2.5, 300)
        if branchpoint is None:
            branchpoint = run_q[-1]
        else:
            np.testing.assert_array_equal(run_q[-1], branchpoint)
        intermediate = rollout(1., 50) if ramp else None
        stop = rollout(0., 250)
        switch = stop if intermediate is None else np.vstack((intermediate[:-1], stop))
        state, ending = stats(run_q), stats(switch)
        eligible = (not stats(walk)['fell'] and not state['fell'] and
                    state['ending_upright'] and abs(state['last_second_velocity'][0] - 2.5) <= .65)
        result.append(dict(seed=seed, switch='one_second_walk_then_stop' if ramp else 'direct_stop',
            eligible_from_running=eligible, pre_run=state, switch_motion=ending,
            stopped_without_falling=bool(eligible and not ending['fell'] and ending['ending_upright'] and
                                          ending['ending_xy_residual_speed'] < .15)))
rows = {}
for variant in ('direct_stop', 'one_second_walk_then_stop'):
    trials = [r for r in result if r['switch'] == variant]
    eligible = [r for r in trials if r['eligible_from_running']]
    rows[variant] = dict(running=len(eligible), successful=sum(r['stopped_without_falling'] for r in trials),
                         fell_after_switch=sum(r['switch_motion']['fell'] for r in eligible))
dest.parent.mkdir(exist_ok=True, parents=True)
dest.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
    method='Paired starting from the same neutral stance and motor delays. After 1 s stand, 4 s walk and 6 s run, request either an immediate stop for 5 s or 1 s walk followed by a 5 s stop. The original policy weights and deployment inputs are identical. Score the switch only if it was actually running.', summary=rows, trials=result), indent=2, allow_nan=False) + '\n')
print(rows, flush=True)
