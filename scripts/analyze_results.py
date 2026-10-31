#!/usr/bin/env python3
"""Summarize complete or partial runs, keeping errors and coverage visible."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.analysis import analyze_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, help="Recursively discover run.json below this directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-images", type=int, default=3, help="Examples per IoU extreme per run; 0 disables")
    parser.add_argument("--area-thresholds", type=float, nargs="+", default=[.01, .1],
                        help="Predeclared union-foreground area fractions; defaults 1%% and 10%%, not COCO AP bins")
    args = parser.parse_args()
    if args.case_images < 0:
        parser.error("--case-images must be nonnegative")
    paths = list(args.runs)
    if args.root:
        paths.extend(p.parent for p in args.root.rglob("run.json"))
    if not paths:
        parser.error("Provide run directories or --root containing runs")
    print(json.dumps(analyze_runs(paths, args.output, case_images=args.case_images,
                                 area_thresholds=args.area_thresholds), indent=2))


if __name__ == "__main__":
    main()
