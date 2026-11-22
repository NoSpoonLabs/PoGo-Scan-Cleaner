"""Classify image files: screenshot-marker rule first, then the model."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .images import has_screenshot_marker, to_rgb
from .model import Backbone, Head, ensure_backbone


def default_head_path() -> Path:
    return Path(str(resources.files("scan_cleaner") / "data" / "head.json"))


@dataclass
class Verdict:
    decision: str  # positive | negative | uncertain | error
    probability: float | None
    reason: str


class Classifier:
    def __init__(self, head: Head, backbone: Backbone, require_marker: bool = True):
        self.head, self.backbone, self.require_marker = head, backbone, require_marker

    @classmethod
    def load(cls, head_path: Path, model_dir: Path, require_marker: bool = True):
        head = Head.load(head_path)
        backbone = Backbone(ensure_backbone(head.backbone, model_dir), head.backbone)
        return cls(head, backbone, require_marker)

    def classify(self, path: Path) -> Verdict:
        try:
            with Image.open(path) as image:
                if self.require_marker and not has_screenshot_marker(image):
                    return Verdict("negative", None, "no screenshot marker")
                rgb = to_rgb(image)  # decoded only when the model needs it
            probability = self.head.probability(self.backbone.embed(rgb, self.head.views))
        except Exception as error:  # unreadable or broken file: report, never act on it
            return Verdict("error", None, f"{type(error).__name__}: {error}")
        return Verdict(self.head.decide(probability), probability, "model")

    def classify_all(self, items: Iterable[tuple[Path, Path]], workers: int = 4) -> Iterator[tuple[tuple, Verdict]]:
        """((root, path), verdict) in input order; keeps a few images per worker in flight."""
        with ThreadPoolExecutor(max(1, workers)) as pool:
            pending: deque = deque()
            for item in items:
                pending.append((item, pool.submit(self.classify, item[1])))
                if len(pending) >= 4 * workers:
                    item, future = pending.popleft()
                    yield item, future.result()
            while pending:
                item, future = pending.popleft()
                yield item, future.result()
