"""Continue a Zenbot WM run with the hyperparameters saved alongside its full checkpoint."""

import argparse
import json
import os
from pathlib import Path


def continuation_args(checkpoint, output, until):
    checkpoint = checkpoint.absolute()
    ckpt_dir = checkpoint.parent.parent
    if checkpoint.parent.name != "full" or ckpt_dir.name != "ckpt" or not checkpoint.is_file():
        raise ValueError("Choose a full learner saved as RUN/ckpt/full/NAME.npz")
    companion = ckpt_dir / checkpoint.name
    if not companion.is_file():
        raise FileNotFoundError(f"Missing companion policy {companion}")
    saved = json.loads(companion.with_suffix(".json").read_text())
    config = json.loads((ckpt_dir.parent / "config.json").read_text())
    if config.get("robot") != "zenbot23_walk_run_recovery" or not config.get("motion_train_wm"):
        raise ValueError("Expected the 234-input Zenbot world-model run configuration")
    if until <= saved["it"]:
        raise ValueError(f"--until must exceed the saved iteration {saved['it']}")
    config.update(init_full=str(checkpoint), init_policy="", log_dir=str(output), minutes=0,
                  max_iterations=until, jax_cache=os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/instantamp/jax"))
    return [f"--{name}={value}" for name, value in config.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="RUN/ckpt/full/NAME.npz")
    parser.add_argument("output", type=Path, help="New training output directory")
    parser.add_argument("--until", type=int, required=True, help="Absolute final learner iteration")
    args = parser.parse_args()
    try:
        flags = continuation_args(args.checkpoint, args.output, args.until)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from instantamp.train import main as train
    train(flags)


if __name__ == "__main__":
    main()
