#!/usr/bin/bash
set -euo pipefail

# where to put everything
DATA_DIR=LibriSpeech

# base HTTP URL for the US mirror
BASE_URL="http://www.openslr.org/resources/12"

# list of files to download
FILES=(
  dev-clean.tar.gz
  test-clean.tar.gz
  train-clean-100.tar.gz
)

mkdir -p "${DATA_DIR}"

for f in "${FILES[@]}"; do
  echo "Downloading ${f}..."
  # -L follow redirects, -C - resume, --retry for robustness
  curl -L --retry 3 -C - "${BASE_URL}/${f}" -o "${DATA_DIR}/${f}"
done

echo "Downloads complete. Files are in ${DATA_DIR}/"