#!/usr/bin/env python3
"""Prepare immutable second-batch inputs; performs no model or Slurm work."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.batch2_data import prepare_scope, verify_scope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scope', choices=('core', 'suim', 'isic'), required=True)
    parser.add_argument('--prior-run', action='append', type=Path)
    parser.add_argument('--queries-per-fold', type=int, default=100)
    parser.add_argument('--transfer-queries', type=int, default=1000)
    parser.add_argument('--references-per-query', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--min-free-gib', type=float, default=32)
    parser.add_argument('--trust-prepared-pickle', action='store_true')
    parser.add_argument('--allow-download', action='store_true',
                        help='Only transfer scopes: use verified source preparation for missing assets')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            result = verify_scope(args.output_dir, runtime=args.runtime,
                                  prior_runs=args.prior_run, expected_scope=args.scope)
        else:
            result = prepare_scope(args.runtime, args.output_dir, scope=args.scope,
                prior_runs=args.prior_run, queries_per_fold=args.queries_per_fold,
                transfer_queries=args.transfer_queries, references_per_query=args.references_per_query,
                seed=args.seed, min_free_gib=args.min_free_gib,
                trust_prepared_pickle=args.trust_prepared_pickle, allow_download=args.allow_download)
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'scope': args.scope,
                          'error_type': type(exc).__name__, 'error': str(exc),
                          'model_work_performed': False, 'existing_files_preserved': True},
                         ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps({k: result[k] for k in ('status', 'request_sha256', 'receipt_sha256',
                     'query_images', 'episodes', 'manifests')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
