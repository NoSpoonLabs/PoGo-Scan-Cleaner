"""End-to-end runs of the scan modes with the real classifier and a stub image model.

The stub backbone scores red images as matches and green ones as uncertain; everything else is no match.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scan_cleaner import cli
from scan_cleaner.classify import Classifier
from scan_cleaner.model import Head

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


class StubBackbone:
    def embed(self, image, views):
        r, g, _ = image.getpixel((30, 60))  # JPEG shifts colors slightly
        return np.array([10.0 if r > 200 and g < 60 else 0.0 if g > 200 and r < 60 else -10.0])


def stub_classifier(marker=True):
    return Classifier(Head(np.array([1.0]), 0.0, ("full",), {}), StubBackbone(), marker)


@pytest.fixture
def pics(tmp_path, image_factory, monkeypatch):
    monkeypatch.setattr(Classifier, "load", classmethod(lambda cls, head, model_dir, marker=True:
                                                         stub_classifier(marker)))
    image_factory("pics/hit.png", RED)
    image_factory("pics/sub/hit.png", RED)
    image_factory("pics/sub/unsure.png", GREEN)
    image_factory("pics/keep.png", BLUE)
    image_factory("pics/photo.jpg", RED, marker=False)
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    return tmp_path


def options(pics):
    return ["--log-dir", str(pics / "logs"), "--config", str(pics / "config.toml"), "--quiet"]


def run(pics, *args):
    code = cli.main([*args, "--yes", *options(pics)])
    log = next((pics / "logs").glob("*.jsonl"))
    return code, [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def files(records):
    """File records by name; files in the folder "sub" as "sub/<name>"."""
    names = {}
    for record in records:
        if record["type"] == "file":
            path = Path(record["path"])
            names[("sub/" if path.parent.name == "sub" else "") + path.name] = record
    return names


def test_dry_run_changes_nothing(pics, capsys):
    code, records = run(pics, "dry-run", str(pics / "pics"))
    assert code == 0
    assert (pics / "pics" / "hit.png").exists()
    out = capsys.readouterr().out
    assert "Mode:    DRY-RUN" in out and "nothing was changed" in out and "unsure.png" in out
    assert records[0]["type"] == "run" and records[-1]["type"] == "summary"
    assert records[-1]["positive"] == 2 and records[-1]["uncertain"] == 1
    assert files(records)["photo.jpg"]["reason"] == "no screenshot marker"


def test_move_default_into_subfolder(pics):
    code, records = run(pics, str(pics / "pics"))
    assert code == 0
    assert (pics / "pics" / "_PoGo-Scans" / "hit.png").exists()
    assert (pics / "pics" / "sub" / "_PoGo-Scans" / "hit.png").exists()
    assert (pics / "pics" / "sub" / "unsure.png").exists()  # uncertain stays
    assert (pics / "pics" / "photo.jpg").exists()  # no marker: never matched
    assert files(records)["hit.png"]["action"] == "moved"


def test_move_flat_target_renames_conflicts(pics):
    code, _ = run(pics, "move", str(pics / "pics"), "--target", str(pics / "out"), "--flat")
    assert code == 0
    assert sorted(p.name for p in (pics / "out").iterdir()) == ["hit.png", "hit_1.png"]


def test_move_target_keeps_structure(pics):
    run(pics, "move", str(pics / "pics"), "--target", str(pics / "out"))
    assert (pics / "out" / "hit.png").exists() and (pics / "out" / "sub" / "hit.png").exists()


def test_no_marker_check_includes_photos(pics):
    run(pics, "move", str(pics / "pics"), "--no-marker-check")
    assert (pics / "pics" / "_PoGo-Scans" / "photo.jpg").exists()


def test_delete_uses_recycle_bin(pics, monkeypatch):
    trashed = []
    monkeypatch.setattr(cli.actions, "trash", lambda path: trashed.append(path.name))
    code, records = run(pics, "delete", str(pics / "pics"))
    assert code == 0 and sorted(trashed) == ["hit.png", "hit.png"]
    assert files(records)["hit.png"]["action"] == "trashed"


def test_permanent_delete(pics):
    code, records = run(pics, "delete", str(pics / "pics"), "--permanent")
    assert code == 0 and not (pics / "pics" / "hit.png").exists()
    assert files(records)["hit.png"]["action"] == "deleted"


def test_changing_mode_needs_yes_without_terminal(pics, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["move", str(pics / "pics"), *options(pics)]) == 2
    assert (pics / "pics" / "hit.png").exists()


def test_confirmation_no_aborts(pics, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["delete", str(pics / "pics"), *options(pics)]) == 1
    assert (pics / "pics" / "hit.png").exists()
    out = capsys.readouterr().out
    assert "PoGo-Scan-Cleaner" in out and "Mode:    DELETE" in out and "recycle bin, restorable" in out


def test_dry_run_needs_no_confirmation(pics, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("dry-run must not ask"))
    assert cli.main(["dry-run", str(pics / "pics"), *options(pics)]) == 0


def test_missing_folder_is_an_error(pics):
    assert cli.main(["dry-run", str(pics / "nope"), "--config", str(pics / "config.toml")]) == 2


def test_plan_wording():
    target = Path("out").resolve()
    subfolder_effect = cli.make_plan("move", "_sub", None, False, False).effect
    assert subfolder_effect == 'moved into a subfolder "_sub" next to each image'
    assert cli.make_plan("move", "_sub", target, True, False).effect == f"moved to {target}, all into one folder"
    assert "not restorable" in cli.make_plan("delete", "_sub", None, False, True).sentence
    assert cli.make_plan("dry-run", "_sub", None, False, False).action == "would act"


def test_frame_centers_lines():
    lines = cli.frame(["Name", "v1"]).splitlines()
    assert len(lines) == 6 and len(set(len(line) for line in lines)) == 1 and len(lines[0]) == 44
    assert lines[2].strip("# ").strip() == "Name"


def test_target_equal_to_scanned_folder_renames_nothing(pics):
    code, _ = run(pics, "move", str(pics / "pics"), "--target", str(pics / "pics"))
    assert code == 0
    assert (pics / "pics" / "hit.png").exists() and (pics / "pics" / "sub" / "hit.png").exists()
    assert not list((pics / "pics").rglob("hit_1.png"))
