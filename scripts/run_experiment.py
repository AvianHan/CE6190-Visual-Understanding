#!/usr/bin/env python3
"""Execute one reproducible shard of an approved experiment configuration."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ics.runner import expand_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, help="Global prefix before sharding; labels run as a subset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    config = expand_config(json.loads(args.config.read_text()))
    manifest = args.manifest or config.get("manifest")
    root = args.dataset_root or config.get("dataset_root")
    output = args.output or config.get("output")
    if not all((manifest, root, output)):
        parser.error("manifest, dataset_root and output must be specified in config or CLI")
    result = run_experiment(config, manifest_path=manifest, dataset_root=root,
                            output_dir=output, shard_index=args.shard_index,
                            num_shards=args.num_shards, resume=args.resume,
                            retry_failed=args.retry_failed, max_episodes=args.max_episodes)
    print(json.dumps(result, indent=2))
    return 2 if result["errors_in_shard"] else (3 if result["pending_in_shard"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
