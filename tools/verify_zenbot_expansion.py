"""Prove an appended planner bank cannot change any previous successful tracked recovery."""

import argparse
import json
from pathlib import Path

import numpy as np

from instantamp import zenbot
from tools.play_zenbot_getup import ZenbotPlayer


TICK_FIELDS = {'q', 'v', 'u', 'K', 'desired_palm_load'}
MOTION_FIELDS = {'delay', 'look_gain', 'quality', 'support_style', 'reference_style', 'offset'}


def verify(base, expanded, directories):
    with np.load(base) as before, np.load(expanded) as after:
        new_fields = set(after.files) - set(before.files)
        if new_fields != {'zenbot_tracking/expansion_start'} or not set(before.files).issubset(after.files):
            raise ValueError(f'Unexpected additional controller fields {new_fields}')
        boundary = len(before['zenbot_tracking/offset']) - 1
        if int(after['zenbot_tracking/expansion_start']) != boundary:
            raise ValueError('The added bank is not separated from the original fallback')
        for key in before:
            a, b = before[key], after[key]
            short = key.removeprefix('zenbot_tracking/')
            prefix = short in TICK_FIELDS | MOTION_FIELDS and key.startswith('zenbot_tracking/')
            if a.dtype != b.dtype or not np.array_equal(a, b[:len(a)] if prefix else b):
                raise ValueError(f'Original control parameters changed at {key}')
        keys = ('offset', 'q', 'v', 'primary_count', 'quality', 'support_style', 'delay')
        old_select = {k: before['zenbot_tracking/' + k] for k in keys}
    player = ZenbotPlayer(expanded)
    sim, tracker = player.native, player.native.native_tracking
    new_db = tracker.db
    sim.native_tracking = None
    groups = {}
    for directory in directories:
        checks = []
        for path in sorted(directory.glob('[0-9]*_[0-9]*.json')):
            row = json.loads(path.read_text())
            with np.load(path.with_suffix('.npz')) as z:
                measured = z['qpos'][:26]
                velocity = z['qvel'][0]
            lags = np.asarray(row['lags'])
            sim.reset(measured[0], velocity, lags)
            tracker.lags = np.ascontiguousarray(lags[zenbot.GROUP], dtype='i')
            selections = []
            for recovered in (False, True):
                tracker.db = old_select
                old = tracker.select(sim.data, recovered=recovered)
                tracker.db = new_db
                new = tracker.select(sim.data, recovered=recovered)
                if len(old) != len(new) or any(i != j or not np.array_equal(q, q1) or
                        not np.array_equal(v, v1) for (i, q, v), (j, q1, v1) in zip(old, new)):
                    raise ValueError(f'Ordered original planner candidates differ at {path}')
                selections.append([int(i) for i, q, v in old])
            # The old player calls its generic actor from reset only if prepare()
            # failed. A divergence of the measured physical states establishes
            # that its original planning branch succeeded, before expansion is reachable.
            difference = 0.
            for tick in range(1, len(measured)):
                sim.step()
                difference = max(difference, float(np.max(np.abs(sim.data.qpos - measured[tick]))))
            success = row['held'] and not row['refall'] and row['pre_rise_attempts']['retries'] == 0
            if success and difference < 1e-8:
                raise ValueError(f'Successful fallback actor rollout needs explicit retesting at {path}')
            checks.append(dict(trial=row['trial'], seed=row['seed'], old_success=success,
                               generic_actor_max_qpos_difference=difference, same_primary_and_backup_choices=selections))
        groups[str(directory)] = checks
    return dict(original_motion_count=boundary, original_controller_fields_unchanged=True, groups=groups)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--expanded', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('directories', nargs='+', type=Path)
    args = p.parse_args()
    report = verify(args.base, args.expanded, args.directories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for directory, rows in report['groups'].items():
        print(directory, len(rows), sum(r['old_success'] for r in rows))
