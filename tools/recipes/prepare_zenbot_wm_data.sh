#!/usr/bin/env bash
# Rebuild the published run's 50 Hz gait bank from the original Motion Train commit.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [MOTION_TRAIN_CHECKOUT]" >&2
    exit 2
fi
manifest=assets/zenbot_wm_reference.sha256
if sha256sum --check --status "$manifest" 2>/dev/null; then
    echo "All 29 Zenbot reference files match the training run."
    exit 0
fi
source_dir=${1:-}
if [[ -z "$source_dir" ]]; then
    revision=aafe0101ade37656f5390a7320bbd026f1f3c171
    source_dir=$(mktemp -d)
    trap 'rm -rf -- "$source_dir"' EXIT
    git -C "$source_dir" init --quiet
    git -C "$source_dir" remote add origin "${MOTION_TRAIN_URL:-git@github.com:ZPLcode/h1_train.git}"
    git -C "$source_dir" sparse-checkout set --no-cone '/robolab/data/motions/zenbot_amp_frax/' \
        '/robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py'
    git -C "$source_dir" fetch --quiet --depth 1 --filter=blob:none origin "$revision"
    git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
fi
"${SIM_PYTHON:-.venv/bin/python}" -m tools.prepare_zenbot "$source_dir" --locomotion-only
sha256sum --check "$manifest"
