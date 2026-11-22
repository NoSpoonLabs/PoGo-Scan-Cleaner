"""Finding and opening images, and reading the screenshot marker from their metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from PIL import Image

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
USER_COMMENT = 0x9286
EXIF_IFD = 0x8769


def find_images(folders: list[Path], skip_dirs: set[Path], skip_names: set[str]) -> Iterator[tuple[Path, Path]]:
    """(root, file) for every image below the given folders, recursively and in a stable order.

    Folders in skip_dirs (absolute) and folders named in skip_names are not entered.
    """
    skip = {os.path.normcase(str(p.resolve())) for p in skip_dirs}
    seen: set[str] = set()
    for root in folders:
        root = root.resolve()
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in skip_names
                             and os.path.normcase(os.path.join(current, d)) not in skip)
            for name in sorted(files):
                path = Path(current, name)
                key = os.path.normcase(str(path))
                if path.suffix.lower() in EXTENSIONS and key not in seen:
                    seen.add(key)
                    yield root, path


def open_image(path: Path) -> tuple[Image.Image, bool]:
    """(RGB image, has screenshot marker)."""
    with Image.open(path) as image:
        return to_rgb(image), has_screenshot_marker(image)


def to_rgb(image: Image.Image) -> Image.Image:
    image.draft("RGB", (1024, 1024))  # faster JPEG decoding at a reduced size
    return image.convert("RGB")


def has_screenshot_marker(image: Image.Image) -> bool:
    """The word "Screenshot" in the EXIF UserComment or in the XMP packet.

    iOS writes it into every screenshot, also after cropping.
    """
    try:
        comment = image.getexif().get_ifd(EXIF_IFD).get(USER_COMMENT, b"")
    except Exception:  # broken EXIF: treat as no marker
        comment = b""
    if isinstance(comment, bytes):
        comment = comment.decode("latin-1", "ignore")
    xmp = image.info.get("XML:com.adobe.xmp") or image.info.get("xmp") or b""
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")
    return "Screenshot" in str(comment) or "Screenshot" in str(xmp)
