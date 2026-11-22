from pathlib import Path

import pytest
from PIL import Image

from scan_cleaner.images import EXIF_IFD, USER_COMMENT


def make_image(path: Path, color=(0, 0, 0), marker=True, size=(60, 130)) -> Path:
    """A small synthetic image; marker=True writes "Screenshot" into the EXIF UserComment like iOS does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    exif = Image.Exif()
    if marker:
        exif.get_ifd(EXIF_IFD)[USER_COMMENT] = b"ASCII\x00\x00\x00Screenshot"
    image.save(path, exif=exif.tobytes())
    return path


@pytest.fixture
def image_factory(tmp_path):
    def factory(rel, color=(0, 0, 0), marker=True):
        return make_image(tmp_path / rel, color, marker)
    return factory
