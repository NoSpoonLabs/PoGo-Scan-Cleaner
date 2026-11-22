# PoGo-Scan-Cleaner

A local command-line tool that finds the Pokémon detail screenshots you took for an IV appraisal app
and moves or deletes them, so your picture folders and backups stop filling up with them.
It runs **offline** on your computer: no cloud service, no account, no paid API.

## What it matches

The top of the detail screen of a single Pokémon, with or without the appraisal overlay.  
Everything else stays: other game screens (map, raids, battles, the Pokémon list,
a detail screen scrolled down to the moves), other apps and photos.

Two checks decide:

1. **Screenshot marker.** iOS writes the word `Screenshot` into the metadata of every screenshot.
   An image without it is never a match (turn this off with `--no-marker-check`, e.g. for Android screenshots).
2. **Image model.** A small classifier on top of a general image model gives a probability.
   At 0.9 or above the image is a match; below 0.1 it is not;
   anything in between is **uncertain: never moved or deleted, only listed**.

## Install

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) (or pipx).

```
uv tool install git+https://github.com/NoSpoonLabs/PoGo-Scan-Cleaner
```

The first run downloads the image model once (88 MB) to the data folder (`scan-cleaner config` shows where).

## Use

```
scan-cleaner dry-run D:\Pictures          # list what would be matched, change nothing
scan-cleaner D:\Pictures                  # move matches (default mode)
scan-cleaner delete D:\Pictures           # move matches to the recycle bin
```

| Mode                     | What happens to a match                                                                  |
|--------------------------|------------------------------------------------------------------------------------------|
| `move` (default)         | moved into the subfolder `_PoGo-Scans` next to it; `--subfolder <name>` changes the name |
| `move --target <folder>` | moved into one folder with the same folder structure; `--flat`: all files directly in it |
| `dry-run`                | listed only                                                                              |
| `delete`                 | moved to the recycle bin; `--permanent` deletes for good                                 |

- Before it starts, the tool shows the mode, the folders and what happens to a match.
  `move` and `delete` then ask `Start? [y/N]`; `--yes` skips the question 
  (needed when no one can answer, e.g. in a scheduled task). `dry-run` starts without asking.
- At the end a report shows the totals and, per source folder, how many matches went where,
  followed by the uncertain images.
- Folders are searched recursively. File types: PNG, JPEG, WebP.
- A file is never overwritten: a name that exists already gets `_1`, `_2`, … appended.
- `_PoGo-Scans` folders and the target folder are skipped, so running again is safe.
- Every run writes a log: one JSON object per line, with every file, its decision and score, and where it went.
  The last line of the output names it.

## Config file

`scan-cleaner config --init` writes a commented template, e.g. `%APPDATA%\scan-cleaner\config.toml` on Windows.  
There you can set default folders (then `scan-cleaner` alone is enough), the subfolder name, a target folder and more.
Command-line options win over the file.

## Accuracy

Cross-validation on 1,822 labeled images: 0 false matches, 0 missed, 0.38 % uncertain.
The images come from one German-language iPhone; other setups may give more uncertain results,
so start with `dry-run`.

## Model

- Image model: OpenCLIP ViT-B/32 trained on LAION-2B (`laion2b_s34b_b79k`), exported to ONNX and quantized to INT8.
  OpenCLIP and these weights are MIT-licensed.
- Classifier: logistic regression on the image embeddings of the whole screenshot and its top and bottom half,
  stored in `src/scan_cleaner/data/head.json`.
  The head names its image model, so both can be swapped without code changes.

## Development

```
uv venv && uv pip install -e ".[dev]"
.venv/Scripts/python -m pytest          # Linux/macOS: .venv/bin/python
```

Maintainer commands (need your own labeled images: a CSV with the columns `path` and `label`):

- `scan-cleaner evaluate labels.csv --backbone <model.onnx>`: cross-validation report.
- `scan-cleaner train labels.csv --backbone <model.onnx> --out src/scan_cleaner/data/head.json --url <model URL>`.
- `python tools/export_backbone.py <folder> --quantize`: builds the ONNX image model (needs torch and open_clip_torch).

## Disclaimer

The detection can be wrong. Use `delete` at your own risk, `--permanent` above all, and check with `dry-run` first.
The software is provided "as is", without warranty of any kind (see [LICENSE](LICENSE)):
NoSpoonLabs and the contributors are not liable for images that are moved or deleted by mistake.

Not affiliated with, endorsed or sponsored by Niantic, Nintendo, The Pokémon Company or any appraisal app.
Pokémon and Pokémon GO are trademarks of their respective owners. The code is MIT-licensed.
