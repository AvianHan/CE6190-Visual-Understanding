#!/usr/bin/env python3
"""Prepare one fixed manifest, reusable unchanged by all model variants."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.data import build_index, create_manifest


def _read_sidecar(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Existing protocol sidecar is unreadable or invalid: {path}; preserved unchanged') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Existing protocol sidecar must be a JSON object: {path}; preserved unchanged')
    return value


def _publish_prepared(output, manifest, metadata, *, overwrite=False):
    """Publish a pair; recover only an identical orphan audit, never replace it."""
    sidecar = output.with_suffix('.protocol.json')
    if not overwrite and output.exists():
        raise ValueError(f'Manifest already exists: {output}; preserved unchanged')
    reuse_sidecar = not overwrite and sidecar.exists()
    if reuse_sidecar and _read_sidecar(sidecar) != metadata:
        raise ValueError(f'Existing protocol sidecar does not match the recomputed manifest: {sidecar}; '
                         'preserved unchanged. Use a new output path for different data or protocol.')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        # Prepare complete bytes for BOTH files before publishing either one.
        # A unique temporary name also avoids collisions between preflight jobs.
        for path, content in ((sidecar, metadata), (output, manifest)):
            if path == sidecar and reuse_sidecar:
                continue
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    prefix='.' + path.name + '.', suffix='.tmp',
                    dir=output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                pending.append((path, temporary))
                json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        for path, temporary in pending:
            if overwrite:
                temporary.replace(path)
            else:
                try:
                    # Same-filesystem create-only publication: a racing writer
                    # cannot have its file silently overwritten after our check.
                    os.link(temporary, path)
                except FileExistsError:
                    if path != sidecar or _read_sidecar(sidecar) != metadata:
                        raise ValueError(f'Output appeared during preparation: {path}; preserved unchanged')
    finally:
        for _, temporary in pending:
            temporary.unlink(missing_ok=True)


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
    parser.add_argument('--empty-mask-policy', choices=['error','exclude'], default='error',
                        help='LVIS only: explicitly exclude successfully decoded empty masks, with an audit')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.empty_mask_policy == 'exclude' and args.dataset != 'lvis':
        parser.error('--empty-mask-policy exclude is supported only for LVIS')
    sidecar = args.output.with_suffix('.protocol.json')
    if not args.overwrite and args.output.exists():
        parser.error('Output exists. Reuse it, or explicitly choose --overwrite with a new protocol version.')
    if not args.overwrite and sidecar.exists():
        try:
            _read_sidecar(sidecar)  # Reject damaged audits before expensive recomputation.
        except ValueError as exc:
            parser.error(str(exc))
    groups = json.loads(args.group_map.read_text()) if args.group_map else None
    index = build_index(args.dataset, args.data_root, fold=args.fold,
        split=args.split or ('test' if args.dataset == 'fss' else 'val'),
        split_file=args.split_file, trust_prepared_pickle=args.trust_prepared_pickle, group_map=groups)
    manifest, metadata = create_manifest(index, args.data_root, queries=args.queries,
        references_per_query=args.references_per_query, shots=args.shots, seed=args.seed,
        query_seed=args.query_seed, insufficient_references=args.insufficient_references,
        protocol_id=args.protocol_id, empty_mask_policy=args.empty_mask_policy)
    metadata['protocol_id'] = manifest['protocol_id']
    try:
        _publish_prepared(args.output, manifest, metadata, overwrite=args.overwrite)
    except ValueError as exc:
        parser.error(str(exc))
    eligibility = metadata.get('sample_eligibility', {})
    print(json.dumps({'manifest': str(args.output), 'protocol': str(sidecar),
                      'manifest_sha256': metadata['manifest_sha256'],
                      'episodes': len(manifest['episodes']), 'sampling': metadata['sampling'],
                      'sample_eligibility': {key: eligibility.get(key) for key in (
                          'policy', 'version', 'scope', 'input_count', 'eligible_count',
                          'excluded_count', 'pool_fingerprint', 'pool_fingerprint_scope')},
                      'eligibility_audit': str(sidecar)}, indent=2))


if __name__ == '__main__':
    main()
