"""File actions: move without overwriting, move to the recycle bin, delete."""

from __future__ import annotations

import shutil
from pathlib import Path


def free_name(path: Path) -> Path:
    """path itself if free, else name_1.ext, name_2.ext, ... (never overwrites)."""
    if not path.exists():
        return path
    number = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def destination(path: Path, root: Path, subfolder: str, target: Path | None, flat: bool) -> Path:
    """Where move puts a file: the subfolder next to it, or target (flat or keeping the structure below root)."""
    if target is None:
        return path.parent / subfolder / path.name
    if flat:
        return target / path.name
    return target / path.relative_to(root)


def move(path: Path, dest: Path) -> Path:
    """Move path to dest (or a free name next to it); a file already at dest stays where it is."""
    if dest.exists() and dest.samefile(path):
        return path
    dest.parent.mkdir(parents=True, exist_ok=True)
    final = free_name(dest)
    shutil.move(str(path), str(final))
    return final


def trash(path: Path):
    from send2trash import send2trash

    send2trash(str(path))


def delete(path: Path):
    path.unlink()
