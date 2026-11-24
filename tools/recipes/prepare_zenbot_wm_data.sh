#!/usr/bin/env bash
# Install matching 50 Hz gait and recovery references for the joint Zenbot task.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

usage() {
    echo "Usage: $0 [--robot {zenbot23_walk_run_recovery,zenbot26_walk_run_recovery}] [MOTION_TRAIN_CHECKOUT]" >&2
    echo "Set LEGGED_LAB_CHECKOUT to the local checkout containing the licensed get_up_frax bank." >&2
    echo "Set FRAX_PYTHON to the Python interpreter with FRAX IK when generating the 26-DoF bank." >&2
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
robot=zenbot23_walk_run_recovery
if [[ ${1:-} == --robot && $# -ge 2 ]]; then
    robot=$2
    shift 2
fi
if [[ $# -gt 1 || ${1:-} == -* || ( $robot != zenbot23_walk_run_recovery && $robot != zenbot26_walk_run_recovery ) ]]; then
    usage
    exit 2
fi
source_dir=${1:-${MOTION_TRAIN_CHECKOUT:-}}
legged_lab=${LEGGED_LAB_CHECKOUT:-}
sim_python=${SIM_PYTHON:-.venv/bin/python}
manifest=$PWD/assets/zenbot_wm_reference.sha256
data=data/zenbot23_walk_run_recovery

# The 26-DoF retarget consumes the verified 23-DoF clips, so both use this lock.
exec 9<"$data"
flock 9
shopt -s nullglob
needs=()
for bank in walk_run recovery; do
    clips=("$data/$bank"/*.npz)
    expected=$(awk -v dir="$data/$bank/" 'index($2, dir) == 1 && /\.npz$/ {n++} END {print n+0}' "$manifest")
    if [[ ${#clips[@]} != "$expected" ]] || ! sha256sum --check --status \
            <(awk -v dir="$data/$bank/" 'index($2, dir) == 1' "$manifest") 2>/dev/null; then
        needs+=("$bank")
        if [[ $bank == recovery && -z $legged_lab ]]; then
            echo "Set LEGGED_LAB_CHECKOUT to a checkout containing the licensed Zenbot get_up_frax source clips." >&2
            exit 1
        fi
    fi
done

if [[ ${#needs[@]} == 0 ]]; then
    echo "Verified the 23-DoF gait and recovery references."
    if [[ $robot == zenbot23_walk_run_recovery ]]; then
        exit 0
    elif [[ -e data/$robot ]]; then
        exec "$sim_python" -m tools.retarget_zenbot26 --verify --source "$data" --dest "data/$robot"
    fi
fi

stage=$(mktemp -d "$data/.prepare.XXXXXX")
backup=
destination=
cleanup() {
    if [[ -n $backup && -d $backup/previous && ! -e $destination ]]; then
        mv -T -- "$backup/previous" "$destination"
    fi
    rm -rf -- "$stage"
}
trap cleanup EXIT

if [[ -z $source_dir ]]; then
    revision=4733fc9649dc3f2b99b2c3ba21fb2ec5e259e6ff
    source_dir=$stage/source
    git init --quiet "$source_dir"
    git -C "$source_dir" remote add origin "${MOTION_TRAIN_URL:-https://github.com/ZPLcode/h1_train.git}"
    paths=('/robolab/data/motions/zenbot_amp_frax_walk_sole8/'
           '/robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py'
           '/robolab/data/robots/roboparty/zenbot/mjcf/zenbot.xml')
    if [[ $robot == zenbot26_walk_run_recovery ]]; then
        paths+=('/robolab/data/robots/roboparty/zenbot/' '/robolab/data/robots/roboparty/zenbot_26dof_43kg/')
    fi
    git -C "$source_dir" sparse-checkout set --no-cone "${paths[@]}"
    git -C "$source_dir" fetch --quiet --depth 1 --filter=blob:none origin "$revision"
    git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
fi

for bank in "${needs[@]}"; do
    if [[ $bank == walk_run ]]; then
        "$sim_python" -m tools.prepare_zenbot "$source_dir" --locomotion-only --dest "$stage/$data"
    else
        "$sim_python" -m tools.prepare_zenbot "$source_dir" --legged-lab "$legged_lab" --dest "$stage/$data"
        cp -- "$data/recovery/sampling.json" "$stage/$data/recovery/"
    fi
    (cd "$stage" && sha256sum --check --quiet <(awk -v dir="$data/$bank/" 'index($2, dir) == 1' "$manifest"))
done

for bank in "${needs[@]}"; do
    destination=$data/$bank
    backup=
    if [[ -e $destination ]]; then
        backup=$(mktemp -d "$data/${bank}_backup.XXXXXX")
        mv -T -- "$destination" "$backup/previous"
        echo "Saved previous $bank bank in $backup/previous"
    fi
    mv -T -- "$stage/$destination" "$destination"
done

if [[ $robot == zenbot26_walk_run_recovery ]]; then
    if [[ ! -e data/$robot ]]; then
        "${FRAX_PYTHON:-$sim_python}" -m tools.retarget_zenbot26 "$source_dir" --source "$data" --dest "data/$robot"
    fi
    "$sim_python" -m tools.retarget_zenbot26 --verify --source "$data" --dest "data/$robot"
fi
