"""Continue a Zenbot WM run with the hyperparameters saved alongside its full checkpoint."""

import argparse
import json
import os
from pathlib import Path


def continuation_args(checkpoint, output, until, *, with_gait_rewards=False, with_posture_and_tracking=False,
                      range_commands=False, num_envs=None):
    checkpoint = checkpoint.absolute()
    ckpt_dir = checkpoint.parent.parent
    if checkpoint.parent.name != "full" or ckpt_dir.name != "ckpt" or not checkpoint.is_file():
        raise ValueError("Choose a full learner saved as RUN/ckpt/full/NAME.npz")
    companion = ckpt_dir / checkpoint.name
    if not companion.is_file():
        raise FileNotFoundError(f"Missing companion policy {companion}")
    saved = json.loads(companion.with_suffix(".json").read_text())
    config = json.loads((ckpt_dir.parent / "config.json").read_text())
    # Old joint-task runs saved the time constant of the former pelvis-speed reward.
    config.pop("gait_velocity_time_constant", None)
    if config.get("wm_global_gait"):
        config.setdefault("wm_gait_frames", 3)  # original global gait checkpoints used three frames
    if config.get("robot") != "zenbot23_walk_run_recovery" or not config.get("motion_train_wm"):
        raise ValueError("Expected the 234-input Zenbot world-model run configuration")
    if until <= saved["it"]:
        raise ValueError(f"--until must exceed the saved iteration {saved['it']}")
    config.update(init_full=str(checkpoint), init_policy="", log_dir=str(output), minutes=0,
                  max_iterations=until, jax_cache=os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/instantamp/jax"))
    if with_gait_rewards:
        config["wm_gait_rewards"] = 1
    if with_posture_and_tracking:
        config["wm_posture_and_tracking"] = 1
    if range_commands:
        config["wm_range_commands"] = 1
    if num_envs is not None:
        if num_envs < 1:
            raise ValueError("--num-envs must be positive")
        config["num_envs"] = num_envs
    return [f"--{name}={value}" for name, value in config.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="RUN/ckpt/full/NAME.npz")
    parser.add_argument("output", type=Path, help="New training output directory")
    parser.add_argument("--until", type=int, required=True, help="Absolute final learner iteration")
    parser.add_argument("--with-gait-rewards", action="store_true", help="Enable Motion Train foot and joint rewards in the joint task")
    parser.add_argument("--with-posture-and-tracking", action="store_true", help="Enable both posture rewards and instantaneous velocity-error diagnostics")
    parser.add_argument("--range-commands", action="store_true", help="Use the continuous Motion Train walk/run command ranges")
    parser.add_argument("--num-envs", type=int, help="Number of parallel physics environments for the continuation")
    args = parser.parse_args()
    try:
        flags = continuation_args(args.checkpoint, args.output, args.until, with_gait_rewards=args.with_gait_rewards,
                                  with_posture_and_tracking=args.with_posture_and_tracking,
                                  range_commands=args.range_commands, num_envs=args.num_envs)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from instantamp.train import main as train
    train(flags)


if __name__ == "__main__":
    main()
