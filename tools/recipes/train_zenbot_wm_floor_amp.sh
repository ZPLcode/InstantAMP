#!/usr/bin/env bash
# Start one WM + AMP + PPO learner from random weights.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run=${1:?Usage: train_zenbot_wm_floor_amp.sh FRESH_RUN_DIRECTORY [--locomotion-i|--locomotion]}
mode=${2:-}
if [[ $# -gt 2 || ( -n "$mode" && "$mode" != --locomotion && "$mode" != --locomotion-i ) ]]; then
    echo "Usage: $0 FRESH_RUN_DIRECTORY [--locomotion-i|--locomotion]" >&2
    exit 2
fi
iterations=${FLOOR_MAX_ITERATIONS:-10000}
if [[ ! $iterations =~ ^[1-9][0-9]*$ ]]; then
    echo "FLOOR_MAX_ITERATIONS must be a positive integer" >&2
    exit 2
fi
task_args=(--wm_getup_only 1)
practice=0
ground_scale=6
num_envs=1024
action_rate=-.015
horizon=16
if [[ -n "$mode" ]]; then
    ground_scale=10
    num_envs=4096
    action_rate=-.03
    mirror_weight=.2
    if [[ "$mode" == --locomotion-i ]]; then
        mirror_weight=0
    fi
    task_args=(
        --wm_getup_only 0 --wm_range_commands 1 --wm_gait_rewards 1 --wm_posture_and_tracking 1
        --wm_fixed_floor_envs 1 --reset_floor .4 --wm_tracking_scale 2
        --prob_shove 0 --push 0 --gait_ls_scale "$ground_scale" --gait_amp_stride 3
        --floor_stop_command_share .5 --gait_source_commands 1 --gait_mirror_weight "$mirror_weight" --cmd_fresh 0
    )
    practice=.35
    horizon=28
fi
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
sim_python=${SIM_PYTHON:-.venv/bin/python}
"$sim_python" -m instantamp.train \
    --robot zenbot23_walk_run_recovery --motion_train_wm 1 "${task_args[@]}" \
    --wm_floor_amp 1 --reference_weight 0 \
    --wm_reset_imag_share "${FLOOR_RESET_IMAG_SHARE:-.2}" \
    --wm_tempo_negatives "${FLOOR_TEMPO_NEGATIVES:-.25}" --wm_lift_gate "${FLOOR_LIFT_GATE:-.44}" --support_rsi "${FLOOR_SUPPORT_RSI:-$practice}" --log_dir "$run" \
    --seed "${FLOOR_SEED:-42}" --max_iterations "$iterations" --minutes 0 --ckpt_iterations 500 --log_every 50 \
    --num_envs "${FLOOR_NUM_ENVS:-$num_envs}" --env_steps 16 --ppo_len 16 --buffer_steps 100 --warmup_steps 64 \
    --pi_batch 384 --pi_updates 3 --pi_lr .0003 --wm_batch 512 --amp_batch 512 \
    --wm_only_iters 150 --wm_window 8 --horizon "$horizon" --amp_getup 1 --amp_w 2 \
    --amp_delay 0 --amp_ramp 0 --recovery_amp_scale "${FLOOR_AMP_SCALE:-$ground_scale}" \
    --ppo_frac .1 --ppo_w 1 --ode 0 --action_noise .8 --action_rate_l2_weight "${FLOOR_ACTION_RATE_L2_WEIGHT:-$action_rate}" \
    --jax_cache "${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/instantamp/jax}"

# Check the final numbered iteration rather than selecting the best policy on the evaluation seeds.
policy=$(JAX_PLATFORMS=cpu "$sim_python" - "$run" "$iterations" <<'PYTHON'
import sys
from pathlib import Path
from tools.monitor_zenbot_scratch import wm_checkpoint
iteration = int(sys.argv[2])
result = wm_checkpoint(Path(sys.argv[1]), iteration)
if result is None:
    raise SystemExit(f"Missing the {iteration}-iteration checkpoint")
print(result)
PYTHON
)
JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_getup --checkpoint "$policy" --out "$run/style" --summary --openings-only --seconds 12 \
    --reference data/zenbot23_walk_run_recovery/recovery --seeds 932277 1654679 2718869 3950477 5969183
JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_fall_recovery "$policy" "$run/fall_recovery.json"
JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_fall_recovery "$policy" "$run/fall_after_getup.json" --from-floor
if [[ -n "$mode" ]]; then
    video=()
    if [[ -d data/zenbot23_getup/meshes ]] && command -v ffmpeg >/dev/null; then
        video=(--video)
    fi
    JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_command_style "$policy" "$run/command_style" "${video[@]}"
    JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_command_style "$policy" "$run/command_style_recorded" --recorded-speeds "${video[@]}"
    JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_getup --checkpoint "$policy" --out "$run/cold" --summary --cold-start --openings-only --seconds 12 \
        --reference data/zenbot23_walk_run_recovery/recovery --seeds 932277 1654679 2718869 3950477 5969183
    JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_four_skills "$policy" "$run/walk_run_stop" --cold-start \
        --stand-reference data/zenbot23_walk_run_recovery/walk_run/stand_4s_120f.npz --seeds 932277 1654679 2718869
    JAX_PLATFORMS=cpu "$sim_python" -m tools.evaluate_zenbot_directions "$policy" --out "$run/directions.json" --cold-start --all-pairs
    JAX_PLATFORMS=cpu "$sim_python" -m tools.demo_zenbot_sequence "$policy" "$run/sequence" --cold-start --seed 8217539 --trial 16
fi
