from pathlib import Path

import numpy as np
import pytest

from scan_cleaner import actions, settings
from scan_cleaner.cli import parse
from scan_cleaner.images import find_images, open_image
from scan_cleaner.model import Head, crop_view, fit_logistic, letterbox


def test_free_name_never_overwrites(tmp_path):
    (tmp_path / "a.png").write_bytes(b"1")
    (tmp_path / "a_1.png").write_bytes(b"2")
    assert actions.free_name(tmp_path / "a.png") == tmp_path / "a_2.png"
    assert actions.free_name(tmp_path / "b.png") == tmp_path / "b.png"


def test_destination_variants(tmp_path):
    root, file = tmp_path / "pics", tmp_path / "pics" / "2024" / "x.png"
    assert actions.destination(file, root, "_sub", None, False) == root / "2024" / "_sub" / "x.png"
    assert actions.destination(file, root, "_sub", tmp_path / "t", False) == tmp_path / "t" / "2024" / "x.png"
    assert actions.destination(file, root, "_sub", tmp_path / "t", True) == tmp_path / "t" / "x.png"


def test_move_resolves_conflict(tmp_path):
    source = tmp_path / "x.png"
    source.write_bytes(b"new")
    existing = tmp_path / "out" / "x.png"
    existing.parent.mkdir()
    existing.write_bytes(b"old")
    final = actions.move(source, existing)
    assert final == tmp_path / "out" / "x_1.png"
    assert existing.read_bytes() == b"old" and final.read_bytes() == b"new" and not source.exists()


def test_find_images_skips_subfolder_and_other_files(tmp_path, image_factory):
    image_factory("a/1.png")
    image_factory("a/_PoGo-Scans/2.png")
    image_factory("a/b/3.jpg")
    (tmp_path / "a" / "notes.txt").write_text("x")
    found = [p.relative_to(tmp_path).as_posix() for _, p in find_images([tmp_path / "a"], set(), {"_PoGo-Scans"})]
    assert found == ["a/1.png", "a/b/3.jpg"]


def test_screenshot_marker(image_factory):
    assert open_image(image_factory("s.png", marker=True))[1] is True
    assert open_image(image_factory("p.jpg", marker=False))[1] is False


def test_letterbox_keeps_whole_image():
    from PIL import Image

    image = Image.new("RGB", (100, 200), (255, 255, 255))
    box = letterbox(image, 50)
    assert box.size == (50, 50)
    assert box.getpixel((25, 0)) == (255, 255, 255) and box.getpixel((0, 25)) == (0, 0, 0)
    assert crop_view(image, "top").size == (100, 100)


def test_head_thresholds_and_roundtrip(tmp_path):
    head = Head(np.array([1.0, -1.0], dtype=np.float32), 0.0, ("full",), {"file": "x.onnx"}, 0.9, 0.1)
    assert head.decide(0.95) == "positive"
    assert head.decide(0.5) == "uncertain"
    assert head.decide(0.05) == "negative"
    head.save(tmp_path / "h.json")
    loaded = Head.load(tmp_path / "h.json")
    assert loaded.positive_at == 0.9 and loaded.views == ("full",)
    assert loaded.probability(np.array([5.0, 0.0])) > 0.99


def test_fit_logistic_separates():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(1, 0.3, (40, 3)), rng.normal(-1, 0.3, (40, 3))])
    y = np.array([1] * 40 + [0] * 40, dtype=float)
    w, b = fit_logistic(x, y, 1e-3)
    assert (((x @ w + b) > 0) == (y == 1)).all()


def test_parse_defaults_to_move():
    assert parse(["C:/pics"]).command == "move"
    assert parse(["dry-run", "x"]).command == "dry-run"


def test_settings_load(tmp_path):
    good = tmp_path / "c.toml"
    good.write_text('folders = ["a"]\nflat = true\n', encoding="utf-8")
    loaded = settings.load(good)
    assert loaded.folders == ["a"] and loaded.flat is True and loaded.subfolder == "_PoGo-Scans"
    bad = tmp_path / "bad.toml"
    bad.write_text("colour = 1\n", encoding="utf-8")
    with pytest.raises(settings.ConfigError):
        settings.load(bad)
    with pytest.raises(settings.ConfigError):
        settings.load(tmp_path / "missing.toml")


def test_template_is_valid(tmp_path):
    path = tmp_path / "t.toml"
    path.write_text(settings.TEMPLATE, encoding="utf-8")
    assert settings.load(path) == settings.Settings()


def test_damaged_model_file_is_refused(tmp_path):
    from scan_cleaner.model import ModelError, ensure_backbone, sha256

    (tmp_path / "m.onnx").write_bytes(b"truncated")
    meta = {"file": "m.onnx", "sha256": "0" * 64}
    with pytest.raises(ModelError):
        ensure_backbone(meta, tmp_path)
    meta["sha256"] = sha256(tmp_path / "m.onnx")
    assert ensure_backbone(meta, tmp_path) == tmp_path / "m.onnx"
