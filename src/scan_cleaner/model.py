"""Image classifier: an exchangeable ONNX image encoder (backbone) plus a logistic head stored as JSON.

The head file names its backbone (file, download URL, SHA-256, input size, normalization),
so a new model is a new head file and, if needed, a new backbone file; no code change.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS = ("full", "top", "bottom")


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Scale the whole image into a size x size square, padded with black (nothing is cropped)."""
    scale = size / max(image.size)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                           Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def crop_view(image: Image.Image, view: str) -> Image.Image:
    width, height = image.size
    if view == "full":
        return image
    if view == "top":
        return image.crop((0, 0, width, height // 2))
    if view == "bottom":
        return image.crop((0, height // 2, width, height))
    raise ValueError(f"unknown view {view!r}")


def sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sigmoid(z):
    """Logistic function, clipped so that large scores cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class ModelError(Exception):
    pass


def ensure_backbone(meta: dict, model_dir: Path) -> Path:
    """Path of the backbone file in model_dir; downloads and verifies it when missing."""
    path = model_dir / meta["file"]
    if path.is_file():
        if meta.get("sha256") and sha256(path) != meta["sha256"]:
            raise ModelError(f"model file is damaged or a different version: {path}; delete it to download it again")
        return path
    if not meta.get("url"):
        raise ModelError(f"model file missing: {path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading model {meta['file']} (once) to {model_dir} ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(meta["url"], timeout=60) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as error:  # network errors are not all OSError (e.g. http.client.IncompleteRead)
        partial.unlink(missing_ok=True)
        raise ModelError(f"download failed ({error}); put {meta['file']} into {model_dir} by hand") from error
    if meta.get("sha256") and sha256(partial) != meta["sha256"]:
        partial.unlink(missing_ok=True)
        raise ModelError(f"download of {meta['file']} has a wrong checksum")
    partial.replace(path)
    return path


@dataclass
class Backbone:
    """ONNX image encoder; meta holds input size, mean and std."""

    path: Path
    meta: dict

    def __post_init__(self):
        import onnxruntime

        self.size = int(self.meta["size"])
        self.mean = np.array(self.meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(self.meta["std"], dtype=np.float32).reshape(3, 1, 1)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def pixels(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(letterbox(image, self.size), dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (array - self.mean) / self.std

    def embed(self, image: Image.Image, views=VIEWS) -> np.ndarray:
        """One L2-normalized embedding per view, concatenated."""
        batch = np.stack([self.pixels(crop_view(image, view)) for view in views])
        out = self.session.run(None, {self.input: batch})[0].astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out.reshape(-1)


@dataclass
class Head:
    """Logistic regression on the embedding, with two thresholds on the probability."""

    weights: np.ndarray
    bias: float
    views: tuple
    backbone: dict
    positive_at: float = 0.9  # probability >= positive_at: positive (acted on)
    negative_below: float = 0.1  # probability < negative_below: negative; in between: uncertain
    info: dict = field(default_factory=dict)

    def probability(self, embedding: np.ndarray) -> float:
        return float(sigmoid(float(embedding @ self.weights + self.bias)))

    def decide(self, probability: float) -> str:
        if probability >= self.positive_at:
            return "positive"
        if probability < self.negative_below:
            return "negative"
        return "uncertain"

    def save(self, path: Path):
        data = {"backbone": self.backbone, "views": list(self.views), "bias": self.bias,
                "positive_at": self.positive_at, "negative_below": self.negative_below, "info": self.info,
                "weights": [round(float(w), 6) for w in self.weights]}
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Head":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(data["weights"], dtype=np.float32), float(data["bias"]), tuple(data["views"]),
                   data["backbone"], float(data["positive_at"]), float(data["negative_below"]), data.get("info", {}))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-5, steps: int = 100) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by Newton's method (small dense problems only)."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1), dtype=x.dtype)]).astype(np.float64)
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2 * n)
    reg[-1] = 0.0
    for _ in range(steps):
        p = sigmoid(xb @ w)
        grad = xb.T @ (p - y) + reg * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1].astype(np.float32), float(w[-1])
