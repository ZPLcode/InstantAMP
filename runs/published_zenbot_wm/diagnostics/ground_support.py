import json,sys
from pathlib import Path
from statistics import median
import numpy as np

directory = Path(sys.argv[1])
report = json.loads((directory/'summary.json').read_text())
cases = report['cases']
rows=[]
for c in cases:
    r=json.loads((directory/f"{c['seed']}_{c['trial']:02d}.json").read_text())
    rows.append({**c, 'hip':r['hip_velocity_p90_rad_s'], 'source_hip':r['reference_hip_velocity_p90_rad_s'],
                 'v_z':r['vertical_velocity_p90_m_s'], 'source_v_z':r['reference_vertical_velocity_p90_m_s'],
                 'hand_loaded':r['real_hand_load_s'] is not None, 'hand_then_foot':r['transfer_s'] is not None,
                 'extension':r['extension_s'],'source_extension':r['reference_extension_s'],
                 'pelvis_height_mae':c['style'].get('source_time_pelvis_rmse_m')})
by_clip=[]
for name in dict.fromkeys(c['clip'] for c in rows):
    group=[r for r in rows if r['clip']==name]
    up=[r for r in group if r['rose']]
    item={'clip':name, 'trials':len(group),'first_held':sum(r['first_attempt_held'] for r in group),
          'hand_loaded':sum(r['hand_loaded'] for r in group),'hand_then_foot':sum(r['hand_then_foot'] for r in group)}
    if up:
        item.update(rise_ratio=median(r['rise_s']/r['reference_rise_s'] for r in up),
                    hip_ratio=median(r['hip']/r['source_hip'] for r in up),
                    vertical_ratio=median(r['v_z']/r['source_v_z'] for r in up),
                    drift_cm=median(r['post_rise_drift_cm'] for r in up))
    by_clip.append(item)
pairs=[r for r in rows if r['rose']]
summary=dict(trials=len(rows),first_held=sum(r['first_attempt_held'] for r in rows),hand_loaded=sum(r['hand_loaded'] for r in rows),
             hand_then_foot=sum(r['hand_then_foot'] for r in rows))
if pairs:
    x=lambda name,key:float(np.percentile([r[name]/r[key] for r in pairs],90))
    summary.update(hip_ratio_p90_across_trials=x('hip','source_hip'),vertical_ratio_p90_across_trials=x('v_z','source_v_z'),
                   post_rise_drift_p90_cm=float(np.percentile([r['post_rise_drift_cm'] for r in pairs],90)))
data={'checkpoint_sha256':report['checkpoint_sha256'],'test':summary,'by_clip':by_clip}
if len(sys.argv)>2:
    Path(sys.argv[2]).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
print(summary)
