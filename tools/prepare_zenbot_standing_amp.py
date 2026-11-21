"""Record independent 50 Hz standing AMP clips from a free 43 kg Zenbot in MuJoCo.

The motion_train stand_4s_120f clip repeats DEFAULT_Q at a single root pose.
Its soles penetrate this task's floor by 3.29 cm. Here floating-base equilibrium
and offline ankle feedback let MuJoCo supply the foot loads and body motion.
Only measured motion is written. No controller inputs are part of the AMP bank.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

from instantamp import env, zenbot
from instantamp.sim import build_scene
from tools.convert_motions import convert_recovery_clip


BANK = Path('data/zenbot23_walk_run_recovery')
SEEDS = (11, 29, 47, 73, 101, 131)


def sole_bottom(model, data, feet):
    # The support geoms are boxes. Project the half extents on world z.
    z = data.geom_xpos[feet, 2]
    extents = np.sum(abs(data.geom_xmat[feet, 6:9]) * model.geom_size[feet], axis=1)
    return z - extents


def equilibrium(model, feet):
    data = mj.MjData(model)
    data.qpos[7:] = zenbot.DEFAULT_Q
    mj.mj_forward(model, data)
    floor_height = data.qpos[2] - sole_bottom(model, data, feet).min()

    def floating_base_wrench(pose):
        data.qpos[2] = floor_height + pose[0]
        mj.mju_euler2Quat(data.qpos[3:7], np.r_[pose[1:], 0.], 'xyz')
        data.qacc[:] = 0.
        mj.mj_inverse(model, data)
        return data.qfrc_inverse[:6].copy()

    # Resolve the small sole compression under gravity and the center of pressure.
    # The six unactuated base coordinates must require zero applied force/torque.
    solution = least_squares(floating_base_wrench, [-.0005, 0., 0.],
                             ftol=1e-13, xtol=1e-13, gtol=1e-13)
    residual = floating_base_wrench(solution.x)
    torque = data.qfrc_inverse[6:].copy()
    if np.linalg.norm(residual) > 1e-6 or np.any(abs(torque) >= zenbot.EFFORT):
        raise RuntimeError('The default stance has no verified static contact/torque solution')
    return data.qpos.copy(), torque


def record(model, feet, pose, feedforward, seed, frames):
    data = mj.MjData(model)
    data.qpos[:] = pose
    rng = np.random.default_rng(seed)
    arms = [i + 7 for i, name in enumerate(zenbot.DOF) if 'shoulder' in name or 'elbow' in name]
    data.qpos[arms] += np.deg2rad(rng.uniform(-3., 3., len(arms)))
    data.qvel[3:] = rng.normal(0., .035, model.nv - 3)
    mj.mj_forward(model, data)
    keys = [model.body('robot_' + name).id for name in zenbot.KEY_BODIES]
    torso = model.body('robot_' + zenbot.TORSO).id
    ankle_pitch = [zenbot.DOF.index(side + '_ankle_pitch') for side in ('left', 'right')]
    ankle_roll = [zenbot.DOF.index(side + '_ankle_roll') for side in ('left', 'right')]
    limits = model.jnt_range[model.actuator_trnid[:, 0]].T
    floor = model.geom('floor').id
    weight = model.body_mass.sum() * -model.opt.gravity[2]
    error = np.zeros(model.nv)
    wrench = np.zeros(6)
    states, landmarks, quality = [], [], []
    for _ in range(frames):
        mj.mj_differentiatePos(model, error, 1., pose, data.qpos)
        effort = feedforward.copy()
        # Positive ankle torque opposes the corresponding floating-base rotation.
        for joints, axis in ((ankle_pitch, 4), (ankle_roll, 3)):
            effort[joints] += 100. * error[axis] + 20. * data.qvel[axis]
        target = np.clip(zenbot.DEFAULT_Q + effort / zenbot.KP, *limits)
        foot_loads = np.zeros(2)
        for _ in range(env.DECIMATION):
            qdot = data.qvel[6:]
            torque = np.clip(zenbot.KP * (target - data.qpos[7:]) - zenbot.KD * qdot,
                             -zenbot.EFFORT, zenbot.EFFORT)
            data.ctrl[:] = np.where((abs(qdot) >= zenbot.SPEED) & (torque * qdot > 0.), 0., torque)
            mj.mj_step(model, data)
            for k, contact in enumerate(data.contact):
                if floor not in contact.geom:
                    continue
                geom = contact.geom2 if contact.geom1 == floor else contact.geom1
                if geom not in feet:
                    raise RuntimeError('Unexpected non-foot floor contact in standing recording')
                mj.mj_contactForce(model, data, k, wrench)
                foot_loads[feet.index(geom)] += wrench[0] / (env.DECIMATION * weight)
        mj.mj_kinematics(model, data)  # refresh FK at the integrated frame, not the preceding substep
        states.append(data.qpos.copy())
        landmarks.append(data.xpos[keys].copy())
        tilt = np.rad2deg(np.arccos(np.clip(data.xmat[torso, 8], -1., 1.)))
        mae = np.rad2deg(abs(data.qpos[7:] - zenbot.DEFAULT_Q).mean())
        quality.append([tilt, mae, *foot_loads])
    quality = np.asarray(quality)
    loads = quality[:, 2:]
    stats = dict(frames=frames, max_chest_tilt_deg=float(quality[:, 0].max()),
                 max_joint_mae_deg=float(quality[:, 1].max()), min_each_foot_bw=loads.min(axis=0).tolist(),
                 total_bw_range=[float(loads.sum(1).min()), float(loads.sum(1).max())])
    if (not np.isfinite(quality).all() or stats['max_chest_tilt_deg'] > 3. or stats['max_joint_mae_deg'] > 2.
            or loads.min() < .30 or np.abs(loads.sum(1) - 1.).max() > .15):
        raise RuntimeError(f'Standing clip failed physical checks for seed {seed}: {stats}')
    return np.asarray(states), np.asarray(landmarks), stats


def inspect_template(model, feet, path):
    with np.load(path) as clip:
        states = np.column_stack((clip['root_pos'], clip['root_quat'][:, [3, 0, 1, 2]], clip['dof_pos']))
    unique = np.unique(states, axis=0)
    data = mj.MjData(model)
    depths = []
    for state in unique:
        data.qpos[:] = state
        mj.mj_kinematics(model, data)
        depths.extend(sole_bottom(model, data, feet))
    return dict(clip=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(states),
                unique_poses=len(unique), sole_height_m_range=[float(min(depths)), float(max(depths))])


def prepare(dest, template=None):
    model = build_scene(robot_name='zenbot23_getup')
    feet = [model.geom('robot_' + name + '_collision').id for name in zenbot.FEET]
    pose, torque = equilibrium(model, feet)
    files = {}
    records = []
    for seed in SEEDS:
        filename = f'stand_default_{seed:03d}.npz'
        states, keys, files[filename] = record(model, feet, pose, torque, seed, frames=200)
        records.append((filename, states, keys))
    extra = {p.name for p in dest.glob('*.npz')} - files.keys()
    if extra:
        raise ValueError(f'Unexpected motion files in {dest}: {sorted(extra)}')
    for filename, states, keys in records:
        convert_recovery_clip(dest / filename, states[:, :3], states[:, 3:7], states[:, 7:],
                              np.arange(model.nu), 50., keys, keep_native=True)
    manifest = dict(source='Native free-base MuJoCo, DEFAULT_Q static contact equilibrium and offline ankle feedback',
                    model='assets/zenbot_23dof_43kg.xml', mujoco=mj.__version__, fps=50, duration_s=4.,
                    root_quaternion_order='xyzw', root_velocity_frame='world', key_body_names=zenbot.KEY_BODIES,
                    equilibrium_pelvis_z_m=float(pose[2]), max_equilibrium_torque_nm=float(abs(torque).max()),
                    files=files)
    if template is not None and template.is_file():
        manifest['reference_check'] = inspect_template(model, feet, template)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=BANK / 'standing')
    parser.add_argument('--template', type=Path, default=BANK / 'walk_run' / 'stand_4s_120f.npz',
                        help='If present, also audit the source standing template against this scene')
    args = parser.parse_args()
    prepare(args.dest, args.template)
