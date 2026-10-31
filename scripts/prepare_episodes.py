#!/usr/bin/env python3
"""Prepare one fixed manifest, reusable unchanged by all model variants."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.data import build_index, create_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['coco','lvis','fss','pascal','isic','suim','isaid'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--split', default=None)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--queries', type=int, required=True, help='Total class-balanced query draws; 0=all images')
    parser.add_argument('--references-per-query', type=int, default=1)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--query-seed', type=int, default=0)
    parser.add_argument('--protocol-id')
    parser.add_argument('--group-map', type=Path, help='JSON: relative image path -> patient/video/original-image group')
    parser.add_argument('--trust-prepared-pickle', action='store_true')
    parser.add_argument('--insufficient-references', choices=['all','error'], default='all')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    sidecar = args.output.with_suffix('.protocol.json')
    if not args.overwrite and (args.output.exists() or sidecar.exists()):
        parser.error('Output exists. Reuse it, or explicitly choose --overwrite with a new protocol version.')
    groups = json.loads(args.group_map.read_text()) if args.group_map else None
    index = build_index(args.dataset, args.data_root, fold=args.fold,
        split=args.split or ('test' if args.dataset == 'fss' else 'val'),
        split_file=args.split_file, trust_prepared_pickle=args.trust_prepared_pickle, group_map=groups)
    manifest, metadata = create_manifest(index, args.data_root, queries=args.queries,
        references_per_query=args.references_per_query, shots=args.shots, seed=args.seed,
        query_seed=args.query_seed, insufficient_references=args.insufficient_references,
        protocol_id=args.protocol_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((sidecar, metadata), (args.output, manifest)):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(path)
    print(json.dumps({'manifest': str(args.output), 'protocol': str(sidecar),
                      'manifest_sha256': metadata['manifest_sha256'],
                      'episodes': len(manifest['episodes']), 'sampling': metadata['sampling']}, indent=2))


if __name__ == '__main__':
    main()
