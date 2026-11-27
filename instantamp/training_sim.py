"""Physical environment options shared by the trainer and checkpoint evaluator."""

from pathlib import Path


def simulation_options(config):
    """Translate a saved training config into Sim arguments.

    The simulator supplies defaults for optional fields missing from older runs.
    Evaluation can override the collector count, seed and starting episode clock.
    """
    shared = (
        "num_envs", "nconmax", "njmax", "walk_dir", "floor_dir", "random_start",
        "seed", "reset_floor", "prob_shove", "push", "robot", "amp_wrists",
        "action_rate_l2_weight", "zenbot_episode_steps",
        "recovery_amp_seconds", "fall_push_speed", "support_rsi",
        "wm_floor_amp",
        "wm_lift_gate", "wm_lift_weight", "wm_fixed_floor_envs", "wm_tracking_scale",
        "wm_recovery_control_scale", "wm_stand_pose_weight",
        "floor_stop_command_share", "gait_source_commands", "wm_global_gait",
        "wm_side_share", "wm_initial_gait_stage", "wm_gait_rewards",
        "wm_posture_and_tracking", "wm_range_commands",
        "motion_train_randomization",
    )
    options = {name: config[name] for name in shared if name in config}
    motion_train = bool(config.get("motion_train_wm"))
    options["zenbot_motion_train_wm"] = motion_train
    if motion_train:
        # Archived configs selected their push mechanism through the dynamics recipe.
        options["external_pushes"] = config.get("external_pushes", 1)
    if motion_train and config.get("amp_w", 0) > 0:
        options["wm_stand_reference"] = Path(config["amp_dir"]) / "stand_4s_120f.npz"
    return options
