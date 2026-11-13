#!/bin/bash
# Download and convert the motion clips InstantAMP trains with (they are not part of the repository; their own licenses apply):
#
#   data/amp/walk_run      30 walking / running / side-stepping clips of the G1 (156 s), the demos of the locomotion prior and the
#                          upright reset frames. Source: legged_lab (github.com/zitongbai/legged_lab), GMR retargets of AMASS (ACCAD).
#   data/recovery/lafan1   the 6 LAFAN1 fallAndGetUp takes retargeted to the G1 by Unitree (HuggingFace
#                          lvhaidong/LAFAN1_Retargeting_Dataset), the demos of the get-up prior and the floor reset frames.
#
#   bash tools/download_data.sh            (needs git, git-lfs, curl and Python with NumPy)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RAW=data/raw
mkdir -p "$RAW"

# 1. legged_lab's walk-and-run clips (Git LFS pickles), sparse checkout of the clips and the joint-order config only
LL=$RAW/legged_lab
if [ ! -d "$LL/.git" ]; then
  git clone --filter=blob:none --sparse --no-checkout https://github.com/zitongbai/legged_lab "$LL"
fi
(
  cd "$LL"
  git sparse-checkout set source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run scripts/tools/retarget
  git checkout
  git lfs install --local
  git lfs pull --include "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/*"
)
$PY tools/convert_motions.py amp --src "$LL" --dst data/amp/walk_run

# 2. Unitree's LAFAN1 retarget: the fallAndGetUp takes (30 fps CSV)
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1
mkdir -p "$RAW/lafan1_g1"
for take in fallAndGetUp1_subject1 fallAndGetUp1_subject4 fallAndGetUp1_subject5 fallAndGetUp2_subject2 fallAndGetUp2_subject3 fallAndGetUp3_subject1; do
  [ -f "$RAW/lafan1_g1/$take.csv" ] || curl -fL "$HF/$take.csv" -o "$RAW/lafan1_g1/$take.csv"
done
$PY tools/convert_motions.py csv --src "$RAW/lafan1_g1" --dst data/recovery/lafan1 --pattern 'fallAndGetUp*.csv'

echo "clips: $(ls data/amp/walk_run/*.npz | wc -l) walking / running, $(ls data/recovery/lafan1/*.npz | wc -l) get-up"
