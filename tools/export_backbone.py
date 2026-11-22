"""Export the OpenCLIP image encoder to ONNX (maintainer tool, run once per model release).

Needs: torch, open_clip_torch, onnx, onnxruntime (not runtime dependencies of the tool).
Usage: python tools/export_backbone.py <out_dir> [--quantize]
"""

import argparse
import hashlib
import json
from pathlib import Path

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--quantize", action="store_true", help="also write a dynamic INT8 copy")
    args = parser.parse_args()

    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    visual = model.visual.eval()
    size = visual.image_size[0] if isinstance(visual.image_size, tuple) else visual.image_size
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"clip-{MODEL.lower()}-{PRETRAINED}.onnx"
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(visual, dummy, str(out), input_names=["pixels"], output_names=["embedding"],
                      dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    files = [out]
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = out.with_name(out.stem + "-int8.onnx")
        quantize_dynamic(str(out), str(quantized), weight_type=QuantType.QInt8)
        files.append(quantized)
    normalize = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
    for path in files:
        meta = {"file": path.name, "sha256": sha256(path), "size": size, "model": MODEL,
                "pretrained": PRETRAINED, "mean": list(normalize.mean), "std": list(normalize.std)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(path, path.stat().st_size // 1_000_000, "MB", meta["sha256"])


if __name__ == "__main__":
    main()
