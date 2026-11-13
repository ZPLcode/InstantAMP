#!/bin/bash
# Download the G1 visual meshes that assets/g1_mjlab.xml references (Unitree G1 description, as redistributed in Holosoma's repository
# at a fixed commit) and check them against tools/mesh_sha256.txt. Only rendering needs them (tools/render.py); training and
# evaluation use the collision geometry, which is all primitives.
#   bash tools/download_meshes.sh
set -euo pipefail
COMMIT=79ba7142398e6e565e559e75620a889581d5966f
URL=https://raw.githubusercontent.com/amazon-far/holosoma/$COMMIT/src/holosoma/holosoma/data/robots/g1/meshes
cd "$(dirname "$0")/.."
mkdir -p assets/meshes
while read -r sha name; do
  [ -f "assets/meshes/$name" ] || curl -fsSL "$URL/$name" -o "assets/meshes/$name"
done < tools/mesh_sha256.txt
SHA="sha256sum"; command -v sha256sum >/dev/null || SHA="shasum -a 256"
(cd assets/meshes && $SHA -c ../../tools/mesh_sha256.txt --quiet) && echo "G1 meshes in assets/meshes ($(ls assets/meshes | wc -l | tr -d ' ') files, checked)"
