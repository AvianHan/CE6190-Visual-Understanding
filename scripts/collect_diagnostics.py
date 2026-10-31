#!/usr/bin/env python3
"""Bundle bounded experiment diagnostics, excluding images, data, weights and secrets."""
import argparse
import json
from pathlib import Path
import re
import tarfile
import tempfile


def redact(text):
    text = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED_HF_TOKEN]', text)
    text = re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token)\s*[\"\x27]?\s*[:=]\s*[\"\x27]?)[^\s\"\x27,]+', r'\1[REDACTED]', text)
    text = re.sub(r'(https?://[^\s?\"\x27]+)\?[^\s\"\x27]+', r'\1?[QUERY_REDACTED]', text)
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tail-bytes', type=int, default=40000)
    a = p.parse_args()
    root = a.run_dir.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    names = {'plan.json', 'submission.json', 'collection.json', 'preflight.json',
             'summary.json', 'comparison.json', 'run.json', 'status.json',
             'cluster_probe.json', 'bootstrap_status.json', 'weight_inventory.json',
             'gpu_memory.json', 'execution_summary.json', 'failure.json', 'preflight.complete.json'}
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            if any(x in rel.parts for x in ('weights', 'data', 'sources', 'envs', 'predictions', 'episodes')):
                continue
            if path.name not in names and path.suffix not in {'.log', '.csv'}:
                continue
            with path.open('rb') as f:
                if path.stat().st_size > a.tail_bytes:
                    f.seek(-a.tail_bytes, 2)
                content = f.read().decode('utf-8', errors='replace')
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(redact(content))
        with tarfile.open(a.output, 'w:gz') as archive:
            for path in staging.rglob('*'):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(staging)), recursive=False)
    print(json.dumps({'diagnostic_bundle': str(a.output.resolve()), 'images_included': False}))
