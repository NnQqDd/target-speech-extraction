from __future__ import annotations
import os
import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


EPOCH_RE = re.compile(r"^epoch_(\d+)\.pth$")

def find_immediate_subdirs(root: Path) -> Iterable[Path]:
    """Yield immediate subdirectories (depth=1) of root."""
    if not root.exists():
        return
    for entry in root.iterdir():
        if entry.is_dir():
            yield entry


def find_latest_epoch_file(directory: Path) -> Optional[Tuple[int, Path]]:
    """Search directory for files matching epoch_<n>.pth and return (n, path)
    for the largest n. Return None if no match.
    """
    best_n: Optional[int] = None
    best_path: Optional[Path] = None
    try:
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            m = EPOCH_RE.match(entry.name)
            if not m:
                continue
            n = int(m.group(1))
            if best_n is None or n > best_n:
                best_n = n
                best_path = entry
    except PermissionError:
        # skip directories we cannot read
        return None

    if best_n is None:
        return None
    return best_n, best_path


def list_latest_checkpoints(root: Path) -> Iterable[Path]:
    """Yield absolute Paths to the selected checkpoint files (one per immediate subdir).
    """
    for subdir in find_immediate_subdirs(root):
        res = find_latest_epoch_file(subdir)
        if res is not None:
            _, path = res
            # yield absolute resolved path for clarity
            yield path.resolve()


if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    fill_path = lambda x: os.path.join(BASE_PATH, x)
    root = Path(fill_path('weights'))
    if root.exists():
        any_printed = False
        for chk in list_latest_checkpoints(root):
            print(str(chk))
            any_printed = True

        if not any_printed:
            print(f"No epoch_*.pth files found in immediate subdirectories of {root}.")
    
