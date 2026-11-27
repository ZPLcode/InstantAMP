"""Continue a Zenbot WM run with saved hyperparameters and the current reward objective.

The learner resets an older joint-task critic when migrating to the continuous reward blend.
Actor, world model and motion discriminators remain available for continuation.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path


def continuation_config(checkpoint, output, until, **overrides):
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
    from instantamp.train_config import (
        Zenbot23JointConfig,
        ZenbotJointConfig,
        ZenbotMotionTrainWMConfig,
    )

    if config.get("robot") not in zenbot_models.MIXED_TASKS or config.get("motion_train_wm") != 1:
        raise ValueError("Expected a combined Zenbot world-model training run")
    if config.get("wm_getup_only"):
        raise ValueError("The checkpoint was trained only on floor recovery")
    for retired in ("gait_velocity_time_constant", "imu_replay_path", "imu_replay_share", "imu_replay_scale",
                    "wm_recovery_control_scale", "wm_first_attempt_only"):
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
        old_joint = config["robot"] == zenbot_models.ZENBOT26.mixed_task
        if old_joint:
            config.setdefault("wm_tracking_scale", 2.)
            config.setdefault("gait_ls_scale", 1.5)
            config.setdefault("action_rate_l2_weight", -.03)
        defaults = ZenbotJointConfig if old_joint else ZenbotMotionTrainWMConfig
        for field in dataclasses.fields(defaults):
            config.setdefault(field.name, field.default)
        config["wm_initial_gait_stage"] = (
            2 if old_joint and config.get("wm_global_gait") else 0
        )
    if until <= saved["it"]:
        raise ValueError(
            f"--until must exceed the saved iteration {saved['it']}"
        )
    config.update(overrides)
    if "num_envs" in config and config["num_envs"] < 1:
        raise ValueError("--num-envs must be positive")
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
    # Records with missing fields use the defaults in effect before the new 26-DoF scratch recipe.
    recipe = Zenbot23JointConfig if config["robot"] == zenbot_models.ZENBOT23.mixed_task else ZenbotJointConfig
    return recipe(**config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("checkpoint", type=Path, help="RUN/ckpt/full/NAME.npz")
    parser.add_argument("output", type=Path, help="New training output directory")
    parser.add_argument("--until", type=int, required=True, help="Absolute final learner iteration")
    parser.add_argument("--num-envs", type=int, help="Number of parallel physics environments")

    # Published continuation commands may override individual saved recipe settings.
    for flag, field in (("--with-gait-rewards", "wm_gait_rewards"),
                        ("--with-posture-and-tracking", "wm_posture_and_tracking"),
                        ("--range-commands", "wm_range_commands")):
        parser.add_argument(flag, dest=field, action="store_const", const=1, help=argparse.SUPPRESS)
    for flag in ("--imag-gait-mirror-weight", "--getup-amp-scale", "--gait-ls-scale"):
        parser.add_argument(flag, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--stand-pose-weight", dest="wm_stand_pose_weight", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from instantamp.train import train

    try:
        cfg = continuation_config(**vars(args))
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    train(cfg)


if __name__ == "__main__":
    main()
