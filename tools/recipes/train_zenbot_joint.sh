#!/usr/bin/env bash
# Train and evaluate one policy for walking, running and getting up.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

usage() {
    echo "Usage: $0 RUN_DIRECTORY [--robot {zenbot23_walk_run_recovery,zenbot26_walk_run_recovery}]" >&2
    echo "       [--init_policy FILE] [--max_iterations N] [--ckpt_iterations N] [--num_envs N] [--seed N]" >&2
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 || $1 == --* ]]; then
    usage
    exit 2
fi
run=$1
shift
robot=zenbot23_walk_run_recovery
training_args=()
while [[ $# -gt 0 ]]; do
    if [[ $# -lt 2 ]]; then
        usage
        exit 2
    fi
    case $1 in
        --robot) robot=$2 ;;
        --init_policy|--max_iterations|--ckpt_iterations|--num_envs|--seed)
            training_args+=("$1" "$2") ;;
        *) usage; exit 2 ;;
    esac
    shift 2
done

bash tools/recipes/prepare_zenbot_wm_data.sh --robot "$robot"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
sim_python=${SIM_PYTHON:-.venv/bin/python}
"$sim_python" -m instantamp.train --robot "$robot" --log_dir "$run" "${training_args[@]}"
JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_joint "$run" \
    --reference "data/$robot/recovery" --out "$run/evaluation/final.json"
