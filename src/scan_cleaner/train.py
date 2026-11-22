"""Maintainer commands: train a head from labeled images, and measure it with repeated k-fold cross-validation.

Labels are a CSV file with the columns path (relative to --root) and label (positive or negative).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .images import open_image
from .model import VIEWS, Backbone, Head, fit_logistic, sigmoid


def add_parsers(sub):
    for name, text in (("train", "train a classifier head from labeled images"),
                       ("evaluate", "cross-validate on labeled images (error rates, uncertain share)")):
        parser = sub.add_parser(name, help=text, description=text)
        parser.add_argument("labels", type=Path, help="CSV with the columns path and label")
        parser.add_argument("--root", type=Path, help="folder the paths are relative to (default: next to labels)")
        parser.add_argument("--backbone", type=Path, required=True, help="ONNX image encoder with its .json sidecar")
        parser.add_argument("--cache", type=Path, help="embedding cache (default: next to labels)")
        parser.add_argument("--l2", type=float, default=1e-5, help="regularization (default: %(default)s)")
        parser.add_argument("--positive-at", type=float, default=Head.positive_at, help="default: %(default)s")
        parser.add_argument("--negative-below", type=float, default=Head.negative_below, help="default: %(default)s")
        parser.add_argument("--workers", type=int, default=4)
        if name == "train":
            parser.add_argument("--out", type=Path, required=True, help="head JSON to write")
            parser.add_argument("--url", default="", help="download URL of the backbone, stored in the head")
            parser.add_argument("--name", default="", help="model name stored in the head")
            parser.set_defaults(run=run_train)
        else:
            parser.add_argument("--folds", type=int, default=5)
            parser.add_argument("--repeats", type=int, default=3)
            parser.set_defaults(run=run_evaluate)


def read_labels(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in ("positive", "negative"):
                raise SystemExit(f"error: {path}: label must be positive or negative, got {row['label']!r}")
            rows.append((row["path"], int(row["label"] == "positive")))
    return rows


def backbone_meta(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    meta["file"] = path.name
    return meta


def embeddings(args, rows) -> np.ndarray:
    """Embeddings of all labeled images; reuses the cache for unchanged files of the same backbone."""
    root = args.root or args.labels.parent
    meta = backbone_meta(args.backbone)
    cache = args.cache or args.labels.with_name(f"{args.labels.stem}.{args.backbone.stem}.npz")
    keys = []
    for rel, _ in rows:
        stat = (root / rel).stat()
        keys.append(f"{rel}|{stat.st_size}|{int(stat.st_mtime)}")
    known = {}
    if cache.is_file():
        data = np.load(cache)
        if str(data["model"]) == meta["sha256"] + "|" + ",".join(VIEWS):
            known = dict(zip(data["keys"].tolist(), data["emb"]))
    todo = [i for i, key in enumerate(keys) if key not in known]
    if todo:
        backbone = Backbone(args.backbone, meta)
        print(f"embedding {len(todo)} images ...", file=sys.stderr)

        def one(i):
            return keys[i], backbone.embed(open_image(root / rows[i][0])[0], VIEWS)

        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            known.update(pool.map(one, todo))
        np.savez(cache, keys=np.array(keys), emb=np.stack([known[k] for k in keys]),
                 model=np.array(meta["sha256"] + "|" + ",".join(VIEWS)))
    return np.stack([known[k] for k in keys])


def run_train(args) -> int:
    rows = read_labels(args.labels)
    x, y = embeddings(args, rows), np.array([label for _, label in rows], dtype=float)
    weights, bias = fit_logistic(x, y, args.l2)
    meta = backbone_meta(args.backbone)
    meta["url"] = args.url
    head = Head(weights, bias, VIEWS, meta, args.positive_at, args.negative_below,
                {"name": args.name, "l2": args.l2, "samples": int(len(y)), "positives": int(y.sum())})
    head.save(args.out)
    print(f"written: {args.out} ({int(y.sum())} positive, {int(len(y) - y.sum())} negative samples)")
    return 0


def run_evaluate(args) -> int:
    rows = read_labels(args.labels)
    x, y = embeddings(args, rows), np.array([label for _, label in rows], dtype=float)
    lo, hi = args.negative_below, args.positive_at
    out = np.zeros((args.repeats, len(y)))
    for repeat in range(args.repeats):
        folds = np.random.default_rng(repeat).permutation(len(y)) % args.folds
        for fold in range(args.folds):
            test = folds == fold
            weights, bias = fit_logistic(x[~test], y[~test], args.l2)
            out[repeat, test] = sigmoid(x[test] @ weights + bias)
    p, yy = out.reshape(-1), np.tile(y, args.repeats)
    fp = int(((p >= hi) & (yy == 0)).sum())
    fn = int(((p < lo) & (yy == 1)).sum())
    unsure = (p >= lo) & (p < hi)
    print(f"{len(y)} images ({int(y.sum())} positive), {args.folds}-fold cross-validation x {args.repeats}, "
          f"thresholds {lo} / {hi}")
    print(f"false positives (would be acted on): {fp} of {int((yy == 0).sum())}")
    print(f"false negatives (missed):            {fn} of {int((yy == 1).sum())}")
    print(f"uncertain (left untouched):          {int(unsure.sum())} ({unsure.mean():.2%})")
    print(f"recall of positives:                 {((p >= hi) & (yy == 1)).sum() / (yy == 1).sum():.2%}")
    print(f"highest negative score {p[yy == 0].max():.3f}, lowest positive score {p[yy == 1].min():.3f}")
    mean = out.mean(axis=0)
    flagged = [i for i in range(len(y)) if (mean[i] >= lo) != bool(y[i]) or lo <= mean[i] < hi]
    if flagged:
        print("\nimages wrong or uncertain (mean score, label, path):")
        for i in sorted(flagged, key=lambda i: abs(mean[i] - y[i]), reverse=True):
            print(f"  {mean[i]:.3f}  {'positive' if y[i] else 'negative'}  {rows[i][0]}")
    return 1 if fp else 0
