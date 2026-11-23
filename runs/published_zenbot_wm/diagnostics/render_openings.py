"""Render measured native benchmark states and the original AMP clip at their recorded speeds."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from tools.evaluate_zenbot_getup import evaluate
from tools.play_zenbot_getup import ZenbotPlayer, record
from instantamp import env

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint',type=Path)
parser.add_argument('out',type=Path)
parser.add_argument('--seconds',type=float,default=6.)
parser.add_argument('--seed',type=int,default=8217539)
selection=parser.add_mutually_exclusive_group()
selection.add_argument('--trial',type=int)
selection.add_argument('--trials',type=int,nargs='+')
parser.add_argument('--cold-start',action='store_true')
parser.add_argument('--frame-motion',action='store_true')
parser.add_argument('--expected-summary',type=Path)
parser.add_argument('--traces',type=Path)
args=parser.parse_args()
expected=json.loads(args.expected_summary.read_text()) if args.expected_summary else None
digest=sha256(args.checkpoint.read_bytes()).hexdigest()
if expected:
    assert digest==expected['checkpoint_sha256'] and expected['cold_start']==args.cold_start
ref=Path('data/zenbot23_walk_run_recovery/recovery')
player=ZenbotPlayer(args.checkpoint,ref,seed=args.seed,use_plan_cache=False)
steps=round(args.seconds/env.DT)
metadata=[]
with TemporaryDirectory(prefix='zenbot_style_video_') as scratch:
    folder=Path(scratch)
    clips=[]
    selected=args.trials if args.trials else ([args.trial] if args.trial is not None else range(len(player.names)))
    for trial in selected:
        name,offset=player.names[trial]
        if offset!=0.:
            continue
        if args.traces:
            with np.load(args.traces/f'{args.seed}_{trial:02d}.npz',allow_pickle=False) as f: poses=f['qpos']
            row=json.loads((args.traces/f'{args.seed}_{trial:02d}.json').read_text())
        else:
            row,physical=evaluate(player,trial,args.seed,max(600,steps),-1,cold_start=args.cold_start)
            poses=physical['qpos']
        if expected:
            case=next(c for c in expected['cases'] if c['seed']==args.seed and c['trial']==trial)
            assert row['rise']==case['rise_s'] and row['held']==case['held']
        assert len(poses)>=steps+1
        sequence=iter(poses[:steps+1])
        data=SimpleNamespace(qpos=next(sequence))
        def advance():
            data.qpos=next(sequence)
        view=SimpleNamespace(reference=player.reference, model=player.model, names=player.names,
                             trial=trial, data=data, step=advance)
        path=folder/f'opening_{trial:02d}.mp4'
        framing = {}
        if args.frame_motion:
            with np.load(player.reference / name) as reference:
                expert_xy = reference['root_pos'][:steps+1, :2].copy()
            expert_xy += poses[0,:2] - expert_xy[0]
            xy = np.vstack((poses[:steps+1, :2], expert_xy))
            center = (xy.min(0) + xy.max(0)) / 2
            framing = dict(camera_center=[*center,.42], camera_distance=2.8 + .65 * np.linalg.norm(np.ptp(xy, axis=0)))
        record(view,path,args.seconds,**framing)
        clips.append(path)
        metadata.append(dict(trial=trial,source=name,first_rise_s=row['rise'],reference_rise_s=row['reference_rise_s'],
                             hand_load_s=row['real_hand_load_s'],transfer_s=row['transfer_s'],
                             refall=row['refall'],quiet_final_2s=row['held']))
    assert clips
    manifest=folder/'chapters.txt'
    manifest.write_text(''.join(f"file '{p.name}'\n" for p in clips))
    args.out.parent.mkdir(exist_ok=True,parents=True)
    temp=args.out.with_name(args.out.stem+'.tmp.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','1','-i',str(manifest),'-codec','copy',
                    '-movflags','+faststart',str(temp)],check=True)
    temp.replace(args.out)
sidecar=args.out.with_suffix('.json')
sidecar.write_text(json.dumps(dict(native_policy_sha256=digest, seed=args.seed, cold_start=args.cold_start,
                                  video_sha256=sha256(args.out.read_bytes()).hexdigest(), seconds_per_opening=args.seconds,
                                  cases=metadata),indent=2)+'\n')
print(len(metadata),'openings',metadata[0],flush=True)
