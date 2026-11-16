#!/usr/bin/env python3
"""Audit frozen batch-2 queue outputs and report all predeclared comparisons."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.batch2_analysis import analyze_batch2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Queue root containing plan.json; may repeat for separately gated domains")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    result = analyze_batch2(args.run_dir, args.output, bootstrap_replicates=args.bootstrap_replicates)
    keys = ("status", "formal_observed_records", "formal_ok", "formal_errors", "formal_missing_records_in_bound_outputs")
    print(json.dumps({k: result[k] for k in keys} | {"output": str(args.output.resolve()),
                                                  "unbound_outputs": len(result["outputs_without_binding"])}, indent=2))


if __name__ == "__main__":
    main()
