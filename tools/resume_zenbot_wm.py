"""Continue a Zenbot WM run with saved hyperparameters and the current reward objective.

The learner resets an older joint-task critic when migrating to the continuous reward blend.
Actor, world model and motion discriminators remain available for continuation.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path


def continuation_args(
    checkpoint,
    output,
    until,
    *,
    with_gait_rewards=False,
    with_posture_and_tracking=False,
    range_commands=False,
    num_envs=None,
    imag_gait_mirror_weight=None,
    recovery_control_scale=None,
    getup_amp_scale=None,
    gait_ls_scale=None,
    stand_pose_weight=None,
):
    checkpoint = checkpoint.absolute()
    ckpt_dir = checkpoint.parent.parent
    if (
        checkpoint.parent.name != "full"
        or ckpt_dir.name != "ckpt"
        or not checkpoint.is_file()
    ):
        raise ValueError(
            "Choose a full learner saved as RUN/ckpt/full/NAME.npz"
        )
    companion = ckpt_dir / checkpoint.name
    if not companion.is_file():
        raise FileNotFoundError(f"Missing companion policy {companion}")
    saved = json.loads(companion.with_suffix(".json").read_text())
    config = json.loads((ckpt_dir.parent / "config.json").read_text())
    from instantamp import zenbot_models
    from instantamp.train import ZenbotMotionTrainWMConfig

    if config.get("robot") not in zenbot_models.MIXED_TASKS or not config.get(
        "motion_train_wm"
    ):
        raise ValueError("Expected a combined Zenbot world-model training run")
    if config.get("wm_getup_only"):
        raise ValueError("The checkpoint was trained only on floor recovery")
    for retired in ("gait_velocity_time_constant", "imu_replay_path", "imu_replay_share", "imu_replay_scale"):
        config.pop(retired, None)
    # Older configs with implicit widths used the original three 512-unit layers.
    config.setdefault("wm_hidden", "512,512,512")
    # Keep optional optimizer settings; reward-version migration is handled by the learner.
    config.setdefault("imag_gait_mirror_weight", 0.)
    # Archived world models/critics lack the randomized-dynamics context.
    # Keep their original physical recipe when doing an exact continuation.
    config.setdefault("motion_train_randomization", 0)
    config.setdefault("external_pushes", 1)
    inactive = {
        "g1_style": 0,
        "init_g1_full": "",
        "legged_prior_file": "",
        "legged_real_ppo": 0,
        "recovery_base_actor": "",
        "gait_command_weight": 0,
        "wm_getup_only": 0,
        "wm_stance_start_share": 0,
        "wm_motion_replay": 0,
        "wm_min_rise_score": 0,
        "wm_reference_training": 0,
        "wm_rate_ramp": 0,
        "reference_weight": 0,
    }
    for name, default in inactive.items():
        if config.pop(name, default) != default:
            raise ValueError(
                f"Cannot continue a run using retired setting {name}"
            )
    config.pop(
        "recovery_actor_share", None
    )  # Only used with recovery_base_actor.
    if config.get("wm_global_gait"):
        config.setdefault("wm_gait_frames", 3)
    if "wm_initial_gait_stage" not in config:
        # Archived runs predate the shared command curriculum and recipe.
        from instantamp.train import ZenbotJointConfig

        old_joint = config["robot"] == zenbot_models.ZENBOT26.mixed_task
        if old_joint:
            config.setdefault("wm_tracking_scale", 2.)
            config.setdefault("gait_ls_scale", 1.5)
            config.setdefault("action_rate_l2_weight", -.03)
        defaults = ZenbotJointConfig if old_joint else ZenbotMotionTrainWMConfig
        for field in dataclasses.fields(defaults):
            if (
                field.name not in inactive
                and field.name != "recovery_actor_share"
            ):
                config.setdefault(field.name, field.default)
        config["wm_initial_gait_stage"] = (
            2 if old_joint and config.get("wm_global_gait") else 0
        )
    if until <= saved["it"]:
        raise ValueError(
            f"--until must exceed the saved iteration {saved['it']}"
        )
    config.update(
        init_full=str(checkpoint),
        init_policy="",
        log_dir=str(output),
        minutes=0,
        max_iterations=until,
        jax_cache=os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", "~/.cache/instantamp/jax"
        ),
    )
    if with_gait_rewards:
        config["wm_gait_rewards"] = 1
    if with_posture_and_tracking:
        config["wm_posture_and_tracking"] = 1
    if range_commands:
        config["wm_range_commands"] = 1
    if imag_gait_mirror_weight is not None:
        config["imag_gait_mirror_weight"] = imag_gait_mirror_weight
    if recovery_control_scale is not None:
        config["wm_recovery_control_scale"] = recovery_control_scale
    if getup_amp_scale is not None:
        config["getup_amp_scale"] = getup_amp_scale
    if gait_ls_scale is not None:
        config["gait_ls_scale"] = gait_ls_scale
    if stand_pose_weight is not None:
        config["wm_stand_pose_weight"] = stand_pose_weight
    if num_envs is not None:
        if num_envs < 1:
            raise ValueError("--num-envs must be positive")
        config["num_envs"] = num_envs
    return [f"--{name}={value}" for name, value in config.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="RUN/ckpt/full/NAME.npz")
    parser.add_argument(
        "output", type=Path, help="New training output directory"
    )
    parser.add_argument(
        "--until",
        type=int,
        required=True,
        help="Absolute final learner iteration",
    )
    parser.add_argument(
        "--with-gait-rewards",
        action="store_true",
        help="Enable Motion Train foot and joint rewards in the joint task",
    )
    parser.add_argument(
        "--with-posture-and-tracking",
        action="store_true",
        help="Enable both posture rewards and instantaneous velocity-error diagnostics",
    )
    parser.add_argument(
        "--range-commands",
        action="store_true",
        help="Use the continuous Motion Train walk/run command ranges",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of parallel physics environments for the continuation",
    )
    parser.add_argument(
        "--imag-gait-mirror-weight", type=float,
        help="Override the saved mirror loss weight for imagined gait frames",
    )
    parser.add_argument(
        "--recovery-control-scale", type=float,
        help="Opt into soft recovery speed/effort/setback costs (start with 1); resets the critic when changed",
    )
    parser.add_argument(
        "--getup-amp-scale", type=float,
        help="Override the saved get-up AMP multiplier (0 uses the shared scale)",
    )
    parser.add_argument(
        "--gait-ls-scale", type=float,
        help="Override the least-squares gait AMP multiplier",
    )
    parser.add_argument(
        "--stand-pose-weight", type=float,
        help="Weight for the default joint pose at a quiet stand (negative enables the cost)",
    )
    args = parser.parse_args()
    try:
        flags = continuation_args(
            args.checkpoint,
            args.output,
            args.until,
            with_gait_rewards=args.with_gait_rewards,
            with_posture_and_tracking=args.with_posture_and_tracking,
            range_commands=args.range_commands,
            num_envs=args.num_envs,
            imag_gait_mirror_weight=args.imag_gait_mirror_weight,
            recovery_control_scale=args.recovery_control_scale,
            getup_amp_scale=args.getup_amp_scale,
            gait_ls_scale=args.gait_ls_scale,
            stand_pose_weight=args.stand_pose_weight,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from instantamp.train import main as train

    train(flags)


if __name__ == "__main__":
    main()
