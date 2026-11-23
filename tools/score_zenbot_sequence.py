"""Measure ground preparation, real hand loading and hip extension during a native Zenbot recovery."""

import numpy as np
from scipy.spatial.transform import Rotation


def first_run(mask, samples):
    if len(mask) < samples:
        return None
    starts = np.flatnonzero(
        np.convolve(mask.astype("i"), np.ones(samples), "valid") >= samples
    )
    return int(starts[0]) if len(starts) else None


def ground_timing(q, v, rise):
    """Time from reset to significant joint motion, pelvis lift, and upright at 50 Hz."""
    stop = len(q) if rise is None else rise + 1
    q, v = q[:stop], v[:stop]
    joint_change = np.sqrt(np.square(q[:, 7:] - q[0, 7:]).mean(axis=1))
    first_move = first_run(joint_change >= np.deg2rad(8.), 5)
    height = q[:, 2]
    lift = first_run(height - np.minimum.accumulate(height) >= .08, 10)
    # Track true stillness separately from time spent moving the limbs to establish support.
    still = ((height <= .46) & (np.linalg.norm(v[:, :3], axis=1) < .08)
             & (np.linalg.norm(v[:, 3:6], axis=1) < .30)
             & (np.sqrt(np.square(v[:, 6:]).mean(axis=1)) < .30))
    boundaries = np.flatnonzero(np.diff(np.r_[False, still[:-1], False]))
    durations = boundaries[1::2] - boundaries[::2]
    return dict(first_joint_motion_s=None if first_move is None else .02 * first_move,
                first_pelvis_lift_s=None if lift is None else .02 * lift,
                longest_quiet_ground_s=.02 * int(durations.max(initial=0)))


def support_diagnostics(q, v, load, rise, name, offset, directory):
    n = round(offset * 50)
    with np.load(directory / name) as clip:
        rp = clip["root_pos"][n:]
        rr = Rotation.from_quat(clip["root_quat"][n:]).as_matrix()
        rhip = clip["dof_vel"][n:][:, [0, 6]]
        rhand = clip["key_body_pos"][n:, -2:, 2]
        refup = first_run(
            (rp[:, 2] > 0.58) & (rp[:, 2] <= 0.78) & (rr[:, 2, 2] > 0.8660254), 10
        )
    end = len(q) - 1 if rise is None else max(1, rise)
    refend = len(rp) - 1 if refup is None else max(1, refup)
    # The kinematic reference contains no forces. A low palm marks its geometric opportunity to support the body.
    hand_low = first_run(np.min(rhand[:refend], axis=-1) < 0.10, 5)
    arm = load[:end, :][:, 2:4].sum(-1)
    foot = load[: end + 20, :2].sum(-1)
    hand = first_run(arm >= 0.08, 4)
    transferred = None
    if hand is not None:
        at = first_run(
            (foot[hand:] >= 0.65)
            & (load[hand : hand + len(foot) - hand, 2:4].sum(-1) < 0.06),
            5,
        )
        if at is not None:
            transferred = at + hand
    hip = np.abs(v[:end, [6, 12]])
    rh = np.abs(rhip[:refend])

    def extend(root, stop):
        if stop is None:
            return None
        begin = np.flatnonzero(root[: stop + 1, 2] <= 0.46)
        return 0.02 * (stop - begin[-1]) if len(begin) else None

    rootvel = np.abs(v[:end, 2])
    rv = np.abs(np.diff(rp[: refend + 1, 2]) / 0.02)
    return dict(
        reference_low_hand_s=None if hand_low is None else 0.02 * hand_low,
        real_hand_load_s=None if hand is None else 0.02 * hand,
        transfer_s=None if transferred is None else 0.02 * transferred,
        real_hand_peak_bw=float(arm.max()),
        real_hand_impulse_bw_s=float(0.02 * arm.sum()),
        real_forearm_peak_bw=float(load[:end, 4:6].sum(-1).max()),
        hip_velocity_p90_rad_s=float(np.percentile(hip, 90)),
        reference_hip_velocity_p90_rad_s=float(np.percentile(rh, 90)),
        vertical_velocity_p90_m_s=float(np.percentile(rootvel, 90)),
        reference_vertical_velocity_p90_m_s=float(np.percentile(rv, 90)),
        extension_s=extend(q, rise),
        reference_extension_s=extend(rp, refup),
        reference_rise_s=None if refup is None else 0.02 * refup,
    )
