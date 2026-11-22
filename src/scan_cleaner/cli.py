"""Command line: scan-cleaner [move|dry-run|delete] [FOLDER ...], plus config, train and evaluate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import __version__, actions, settings
from .classify import Classifier, default_head_path
from .images import find_images
from .model import ModelError

MODES = ("move", "dry-run", "delete")
COMMANDS = MODES + ("config", "train", "evaluate")


def add_scan_options(parser: argparse.ArgumentParser):
    parser.add_argument("folders", nargs="*", type=Path, help="folders to scan recursively (default: from config)")
    parser.add_argument("--config", type=Path, help=f"config file (default: {settings.config_path()})")
    parser.add_argument("--subfolder", help=f"move: subfolder name (default: {settings.DEFAULT_SUBFOLDER})")
    parser.add_argument("--target", type=Path, help="move: put all found images into this folder instead")
    flat = parser.add_mutually_exclusive_group()
    flat.add_argument("--flat", action="store_true", default=None, help="move with --target: no subfolders")
    flat.add_argument("--keep-structure", dest="flat", action="store_false", help="move with --target: keep folders")
    parser.add_argument("--permanent", action="store_true", default=None, help="delete: delete for good, not to bin")
    parser.add_argument("--yes", action="store_true", help="move/delete: start without asking for confirmation")
    parser.add_argument("--no-marker-check", dest="marker", action="store_false", default=None,
                        help="also classify images without the screenshot marker in their metadata")
    parser.add_argument("--model-dir", type=Path, help="folder of the model file")
    parser.add_argument("--head", type=Path, help="classifier head JSON (default: the bundled one)")
    parser.add_argument("--log-dir", type=Path, help="folder for the run log")
    parser.add_argument("--workers", type=int, default=4, help="parallel image workers (default: 4)")
    parser.add_argument("--quiet", action="store_true", help="no progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scan-cleaner", description=(
        "Find Pokemon detail screenshots (the screens appraisal apps scan) and move or delete them. "
        "Uncertain matches are never touched, only listed."))
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for mode, text in (("move", "move matches into a subfolder or a target folder (default)"),
                       ("dry-run", "only list what would be matched"),
                       ("delete", "move matches to the recycle bin (or delete them with --permanent)")):
        add_scan_options(sub.add_parser(mode, help=text, description=text))
    config = sub.add_parser("config", help="show the config file path; --init writes a template")
    config.add_argument("--init", action="store_true", help="write a commented template if none exists")
    from . import train  # dev commands

    train.add_parsers(sub)
    return parser


def parse(argv: list[str]) -> argparse.Namespace:
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv = ["move", *argv]  # move is the default mode
    return build_parser().parse_args(argv)


class RunLog:
    """One JSON object per line: a run header, one record per image, a summary."""

    def __init__(self, folder: Path, mode: str):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = folder / f"scan-cleaner-{stamp}-{mode}.jsonl"
        self.handle = open(self.path, "w", encoding="utf-8")

    def write(self, record: dict):
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


@dataclass(frozen=True)
class Plan:
    """Everything that depends on the mode, decided once: the wording and what is done to a match."""

    mode: str
    effect: str  # overview: what happens to a match, in few words
    sentence: str  # overview: one sentence before the confirmation
    verb: str  # report: row label and title
    action: str  # log: value of "action"
    title: str  # report: heading of the per-folder list


def make_plan(mode, subfolder, target, flat, permanent) -> Plan:
    if mode == "dry-run":
        return Plan(mode, "only listed, nothing is changed",
                    "Matched screenshots will only be listed, nothing changes.", "Would move", "would act",
                    "Would move " + (f"to {target.resolve()}" if target else "within the source folders"))
    if mode == "delete":
        if permanent:
            return Plan(mode, "DELETED FOR GOOD (not to the recycle bin)",
                        "Matched screenshots will be deleted for good (not restorable).",
                        "Deleted for good", "deleted", "Deleted for good from")
        return Plan(mode, "moved to the recycle bin", "Matched screenshots will be deleted (recycle bin, restorable).",
                    "Trashed", "trashed", "Trashed from")
    if target is None:
        effect, title = f'moved into a subfolder "{subfolder}" next to each image', "Moved within the source folders"
    else:
        effect = f"moved to {target.resolve()}, " + ("all into one folder" if flat else "keeping the folder structure")
        title = f"Moved to {target.resolve()}"
    return Plan(mode, effect, "Matched screenshots will be moved, nothing is deleted.", "Moved", "moved", title)


@dataclass
class RunResult:
    folders: list[Path]
    seconds: float
    counts: dict[str, int]
    groups: dict[tuple[Path, Path | None], int]  # (source folder, destination folder or None) -> matches


def frame(lines, char="#", border=2, padding=10) -> str:
    """Centered lines in a box of char, at least 40 wide, with an empty line above and below the text."""
    width = max(40, max(len(line) for line in lines) + 2 * padding)
    side = char * border
    rule, empty = side + char * width + side, side + " " * width + side
    return "\n".join([rule, empty, *(side + line.center(width) + side for line in lines), empty, rule])


def overview(plan: Plan, folders) -> str:
    """What the run will do, shown before it starts."""
    folder_lines = "\n         ".join(f"{folder.resolve()} (with subfolders)" for folder in folders)
    return "\n".join([f"Mode:    {plan.mode.upper()}", f"Folders: {folder_lines}", f"Matches: {plan.effect}",
                      "Others:  untouched (uncertain ones are listed at the end)", "", plan.sentence])


MAX_GROUPS = 20


def report(plan: Plan, result: RunResult) -> str:
    """The end-of-run summary: totals, then where the matches went, per source folder."""
    counts, groups = result.counts, result.groups
    head = f"Scan Cleaner - {plan.mode} - {result.seconds} s"
    lines = [head + ("  (nothing was changed)" if plan.mode == "dry-run" else "")]
    for number, folder in enumerate(result.folders):
        lines.append(("Source: " if number == 0 else "        ") + str(folder.resolve()))
    rows = [("Images total", sum(counts.values())), (plan.verb, counts["positive"]),
            ("Not matched", counts["negative"]), ("Uncertain", counts["uncertain"]), ("Errors", counts["error"])]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    lines.append("")
    lines += [f"  {label:<{label_width}}  {value:>{value_width}}" for label, value in rows]
    if groups:
        ranked = sorted(groups.items(), key=lambda item: (-item[1], str(item[0][0])))
        shown, rest = ranked[:MAX_GROUPS], ranked[MAX_GROUPS:]
        lines += ["", plan.title]
        src_width = max(len(str(src)) for (src, _), _ in shown)
        count_width = max(len(str(count)) for _, count in shown)
        for (src, dest), count in shown:
            row = f"  {str(src):<{src_width}}  {count:>{count_width}}"
            lines.append(row + (f"  ->  {dest}" if dest is not None else ""))
        if rest:
            lines.append(f"  ... and {len(rest)} more folders ({sum(count for _, count in rest)} images)")
        else:
            lines.append(f"  ({len(shown)} of {len(shown)} folders)")
    return "\n".join(lines)


def run_scan(args: argparse.Namespace) -> int:
    try:
        conf = settings.load(args.config)
    except settings.ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    mode = args.command
    folders = args.folders or [Path(f).expanduser() for f in conf.folders]
    if not folders:
        print(f"error: no folders given and none in {settings.config_path()}", file=sys.stderr)
        return 2
    missing = [str(f) for f in folders if not f.is_dir()]
    if missing:
        print(f"error: not a folder: {', '.join(missing)}", file=sys.stderr)
        return 2
    subfolder = args.subfolder or conf.subfolder
    target = args.target or (Path(conf.target).expanduser() if conf.target else None)
    flat = conf.flat if args.flat is None else args.flat
    permanent = conf.permanent if args.permanent is None else args.permanent
    marker = conf.require_screenshot_marker if args.marker is None else args.marker
    plan = make_plan(mode, subfolder, target, flat, permanent)
    print(frame(["PoGo-Scan-Cleaner", f"v{__version__}"]) + "\n")
    print(overview(plan, folders))
    if mode != "dry-run" and not args.yes:
        if sys.stdin is None or not sys.stdin.isatty():
            print(f"error: {mode} needs --yes when not run in a terminal", file=sys.stderr)
            return 2
        try:
            answer = input("\nStart? [y/N] ")
        except (EOFError, KeyboardInterrupt):  # no input possible (NUL counts as a terminal on Windows), or Ctrl-C
            answer = ""
            print()
        if answer.strip().lower() not in ("y", "yes", "j", "ja"):
            print("Aborted, nothing changed.")
            return 1
    print()
    try:
        classifier = Classifier.load(args.head or default_head_path(), args.model_dir or conf.model_path(), marker)
    except (ModelError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        log = RunLog(args.log_dir or conf.log_path(), mode)
    except OSError as error:
        print(f"error: cannot write the log: {error}", file=sys.stderr)
        return 2
    log.write({"type": "run", "version": __version__, "mode": mode, "folders": [str(f.resolve()) for f in folders],
               "subfolder": subfolder, "target": str(target) if target else None, "flat": flat,
               "permanent": permanent, "marker_check": marker, "model": classifier.head.info.get("name", ""),
               "positive_at": classifier.head.positive_at, "negative_below": classifier.head.negative_below,
               "started": datetime.now().isoformat(timespec="seconds")})
    skip_dirs = {target} if target else set()
    counts = {"positive": 0, "negative": 0, "uncertain": 0, "error": 0}
    listed: dict[str, list[dict]] = {"uncertain": [], "error": []}
    groups: dict[tuple[Path, Path | None], int] = {}
    started, last = time.monotonic(), 0.0
    items = find_images(folders, skip_dirs, {subfolder})
    for done, ((root, path), verdict) in enumerate(classifier.classify_all(items, args.workers), 1):
        record = {"type": "file", "path": str(path), "decision": verdict.decision,
                  "probability": None if verdict.probability is None else round(verdict.probability, 4),
                  "reason": verdict.reason}
        if verdict.decision == "positive":
            dest = None if mode == "delete" else actions.destination(path, root, subfolder, target, flat)
            try:
                if mode == "move":
                    dest = actions.move(path, dest)
                    record["dest"] = str(dest)
                elif mode == "delete":
                    (actions.delete if permanent else actions.trash)(path)
                record["action"] = plan.action
                key = (path.parent, None if dest is None else dest.parent)
                groups[key] = groups.get(key, 0) + 1
            except OSError as error:
                record.update(decision="error", reason=f"{plan.action} failed: {error}")
        counts[record["decision"]] += 1
        listed.get(record["decision"], []).append(record)
        log.write(record)
        now = time.monotonic()
        if not args.quiet and now - last > 2:
            last = now
            print(f"\r{done} images checked, {counts['positive']} matches ...", end="", file=sys.stderr, flush=True)
    seconds = round(time.monotonic() - started, 1)
    log.write({"type": "summary", **counts, "acted": counts["positive"], "seconds": seconds})
    log.close()
    if not args.quiet:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    print(report(plan, RunResult(folders, seconds, counts, groups)))
    uncertain, errors = listed["uncertain"], listed["error"]
    if uncertain:
        print("\nUncertain (left untouched, check by hand):")
        for record in uncertain:
            print(f"  {record['probability']:.2f}  {record['path']}")
    if errors:
        print("\nErrors:")
        for record in errors:
            print(f"  {record['path']}: {record['reason']}")
    print(f"\nLog: {log.path}")
    return 1 if errors else 0


def run_config(args: argparse.Namespace) -> int:
    path = settings.config_path()
    if args.init:
        if path.exists():
            print(f"exists, unchanged: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(settings.TEMPLATE, encoding="utf-8")
            print(f"written: {path}")
    else:
        print(f"{path} ({'exists' if path.exists() else 'missing; create it with: scan-cleaner config --init'})")
    print(f"model and logs: {settings.data_dir()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.command in MODES:
        return run_scan(args)
    if args.command == "config":
        return run_config(args)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
