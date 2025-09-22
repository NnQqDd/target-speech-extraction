#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIGURE THIS ────────────────────────────────────────────────────────────
ARCHDIR="LibriSpeech"   # <-- where your .tar.gz files live, and where you want the sets
# ────────────────────────────────────────────────────────────────────────────────

# Go into that directory
cd "$ARCHDIR"

# List of archive filenames
archives=(
  dev-clean.tar.gz
  test-clean.tar.gz
  train-clean-100.tar.gz
)

for archive in "${archives[@]}"; do
  echo "Extracting $archive → ./"
  # Strip the outer "LibriSpeech/" folder so you get just dev-clean/, etc.
  tar xzf "$archive" --strip-components=1
done

echo "Done! Contents of each set are now inside '$ARCHDIR/'"

rm -rf dev-clean.tar.gz
rm -rf test-clean.tar.gz
rm -rf train-clean-100.tar.gz

echo "Compressed files removed"

