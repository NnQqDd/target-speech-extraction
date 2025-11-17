"""
Scan a VoxCeleb2 'dev' directory and produce a CSV with columns:
    file_path, url, speaker_id

Expected directory structure (example):
    /path/to/dev/
        id00019/
            3WoWJ3cOHzM/
                00121.mp4
                ...
            ...
        ...
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple, List
import random
random.seed(42)


DEFAULT_EXTS = {".mp4", ".m4a", ".wav", ".flac", ".aac", ".mp3"}


def find_media_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """Yield media files under root matching extensions (case-insensitive)."""
    exts_lower = {e.lower() for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            yield p


def infer_fields(file_path: Path, root: Path) -> Tuple[str, str, str]:
    """
    Given a file_path and the root directory, infer:
      - file_path (string)
      - url: the video-id folder (second-level folder under speaker)
      - speaker_id: top-level folder name (first-level under root)
    If structure is unexpected, fallback to empty strings for missing fields.
    """
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        # file_path is not under root (shouldn't normally happen) -> just return path and empty metadata
        return str(file_path.resolve()), "", ""

    parts = rel.parts  # tuple of path components relative to root
    speaker_id = parts[0] if len(parts) >= 1 else ""
    url = parts[1] if len(parts) >= 2 else ""
    return (str(file_path.resolve()), url, speaker_id)


def build_manifest(root: Path, exts: Iterable[str], write_relative: bool = False) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    files = list(find_media_files(root, exts))
    files.sort()  # deterministic order
    for p in files:
        file_abs, url, speaker_id = infer_fields(p, root)
        if write_relative:
            try:
                file_field = str(Path(file_abs).relative_to(Path.cwd()))
            except Exception:
                file_field = file_abs
        else:
            file_field = file_abs
        rows.append((file_field, url, speaker_id))
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare VoxCeleb2 dev -> CSV manifest (file_path, url, speaker_id)")
    ap.add_argument("--input", "-r", type=str, required=True,
                    help="Path to dev directory (e.g. datasets/dev/mp4)")
    ap.add_argument("--output", "-o", type=str, required=True,
                    help="Output CSV path (e.g. vox2_dev.csv)")
    ap.add_argument("--relative", action="store_true",
                    help="Write file_path as relative to current working directory when possible")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.input).expanduser().resolve()
    out_csv = Path(args.output).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ERROR: root {root} does not exist or is not a directory")

    exts = [x.strip() if x.startswith(".") else "." + x.strip() for x in DEFAULT_EXTS if x.strip()]
    rows = build_manifest(root, exts, write_relative=args.relative)
    
    all_speakers = list(set(row[2] for row in rows))
    random.shuffle(all_speakers)

    split_idx = int(len(all_speakers) * 0.8)
    train_speakers = set(all_speakers[:split_idx])
    valid_speakers = set(all_speakers[split_idx:])

    partition_map = {}
    for speaker_id in train_speakers:
        partition_map[speaker_id] = 'train'
    for speaker_id in valid_speakers:
        partition_map[speaker_id] = 'valid'

    final_rows = []
    for file_path, url, speaker_id in rows:
        partition = partition_map[speaker_id] # Default to 'train' if speaker not found (shouldn't happen)
        final_rows.append([file_path, url, speaker_id, partition])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'url', 'speaker_id', 'partition']) 
        writer.writerows(final_rows)

    print(f"Wrote CSV with {len(rows)} rows to {out_csv}")