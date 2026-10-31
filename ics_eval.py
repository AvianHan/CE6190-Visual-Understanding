"""Small stdlib-only evaluation contract. No model implementation is included."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

TRACKS = {"semantic", "personalized", "part"}


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("File paths must be non-empty relative strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Each asset needs a 64-character SHA256")
    return value.lower()


def _asset(asset: dict, root: Path | None, digest_cache: dict | None = None) -> None:
    if set(asset) != {"image", "image_sha256", "mask", "mask_sha256"}:
        raise ValueError("Asset must contain exactly image/image_sha256/mask/mask_sha256")
    for kind in ("image", "mask"):
        relative = _relative_path(asset[kind])
        expected = _sha256(asset[kind + "_sha256"])
        if root is not None:
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Symlink escapes dataset root")
            # Reference-sensitivity manifests reuse assets many times. Hash each
            # unchanged file once within this validation call; never persist this
            # cache across runs. Conflicting expected hashes are still rejected.
            stat = path.stat()
            cache_key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            actual = None if digest_cache is None else digest_cache.get(cache_key)
            if actual is None:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual = digest.hexdigest()
                if digest_cache is not None:
                    digest_cache[cache_key] = actual
            if actual != expected:
                raise ValueError(f"SHA256 mismatch: {relative}")


def validate_manifest(manifest: dict, *, root: Path | None = None,
                      allow_example: bool = False) -> None:
    if set(manifest) != {"schema_version", "status", "protocol_id", "episodes"}:
        raise ValueError("Unexpected manifest fields")
    if manifest["schema_version"] != 1 or not manifest["protocol_id"]:
        raise ValueError("Unsupported schema or empty protocol_id")
    if manifest["status"] == "example_only":
        if not allow_example:
            raise ValueError("Synthetic example; cannot be used as an experiment")
    elif manifest["status"] != "ready":
        raise ValueError("status must be example_only or ready")
    if not manifest["episodes"]:
        raise ValueError("Manifest must contain episodes")
    ids = set()
    digest_cache = {}
    fields = {"episode_id", "dataset", "dataset_version", "split", "fold",
              "track", "concept_id", "reference_seed", "query", "references",
              "mask_semantics", "ignore_label"}
    for episode in manifest["episodes"]:
        if set(episode) != fields:
            raise ValueError("Unexpected or missing episode fields")
        eid = episode["episode_id"]
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError("Episode IDs must be unique non-empty strings")
        ids.add(eid)
        if episode["track"] not in TRACKS:
            raise ValueError("Unknown task track")
        for key in ("dataset", "dataset_version", "split", "concept_id", "mask_semantics"):
            if not isinstance(episode[key], str) or not episode[key]:
                raise ValueError(f"Missing metadata: {key}")
        if not isinstance(episode["reference_seed"], int):
            raise ValueError("reference_seed must be integer")
        _asset(episode["query"], root, digest_cache)
        if not episode["references"]:
            raise ValueError("Each episode needs at least one reference")
        ref_hashes = set()
        for ref in episode["references"]:
            _asset(ref, root, digest_cache)
            if (ref["image"] == episode["query"]["image"] or
                    ref["image_sha256"].lower() == episode["query"]["image_sha256"].lower()):
                raise ValueError("Reference and query cannot be the same image")
            if ref["image_sha256"].lower() in ref_hashes:
                raise ValueError("Duplicate reference image")
            ref_hashes.add(ref["image_sha256"].lower())


def inference_view(episode: dict) -> dict:
    """Whitelist only authorized model inputs; GT and class names stay outside.

    Use this object as the adapter argument. Asset paths are resolved by the
    trusted loader. This is an API guard, not an OS filesystem security boundary.
    """
    return {
        "query_image": episode["query"]["image"],
        "references": [{"image": ref["image"], "mask": ref["mask"]}
                       for ref in episode["references"]],
    }


def manifest_fingerprint(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def binary_counts(prediction, target, *, ignore_label=None) -> dict:
    """Flattened binary masks only; the dataset adapter must map labels first."""
    if len(prediction) != len(target):
        raise ValueError("Mask lengths differ")
    tp = fp = fn = tn = ignored = 0
    for pred, truth in zip(prediction, target):
        if truth == ignore_label and ignore_label is not None:
            ignored += 1
            continue
        if pred not in (0, 1) or truth not in (0, 1):
            raise ValueError("Expected mapped binary mask values")
        tp += int(pred == 1 and truth == 1)
        fp += int(pred == 1 and truth == 0)
        fn += int(pred == 0 and truth == 1)
        tn += int(pred == 0 and truth == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ignored": ignored}


def summarize_counts(rows: list[dict]) -> dict:
    """Aggregate one method/run/protocol/split/track only.

    macro_class_pooled_iou = mean of per-class accumulated foreground IoU.
    class_mean_episode_iou = mean of the per-class means of episode IoU.
    mean_episode_iou = mean of valid per-episode foreground IoU.
    These are deliberately reported separately. Empty-union IoU is undefined
    and excluded; the excluded count is explicit. No cross-dataset average.
    """
    if not rows:
        raise ValueError("No results supplied")
    scope_keys = ("run_id", "method", "variant", "protocol_id", "manifest_sha256",
                  "dataset", "split", "fold", "track", "nshot")
    scope = {key: rows[0][key] for key in scope_keys}
    per_class = defaultdict(lambda: [0, 0])
    per_class_episodes = defaultdict(list)
    episode_ious, ids = [], set()
    empty = 0
    for row in rows:
        if any(row[key] != scope[key] for key in scope_keys):
            raise ValueError("Do not mix methods, runs, protocols, datasets, folds, or tracks")
        if row["episode_id"] in ids:
            raise ValueError("Duplicate result episode")
        ids.add(row["episode_id"])
        counts = row["counts"]
        if any(type(counts.get(key)) is not int or counts[key] < 0
               for key in ("tp", "fp", "fn", "tn", "ignored")):
            raise ValueError("Counts must be non-negative integers")
        intersection = counts["tp"]
        union = intersection + counts["fp"] + counts["fn"]
        per_class[row["concept_id"]][0] += intersection
        per_class[row["concept_id"]][1] += union
        if union:
            episode_ious.append(intersection / union)
            per_class_episodes[row["concept_id"]].append(intersection / union)
        else:
            empty += 1
    class_ious = {key: a / b if b else None for key, (a, b) in per_class.items()}
    class_episode_means = {key: sum(per_class_episodes[key]) / len(per_class_episodes[key])
                           if per_class_episodes[key] else None for key in per_class}
    valid = [value for value in class_ious.values() if value is not None]
    valid_class_episode = [value for value in class_episode_means.values() if value is not None]
    return {"scope": scope, "episodes": len(rows), "empty_union_episodes": empty,
            "class_pooled_iou": class_ious,
            "macro_class_pooled_iou": sum(valid) / len(valid) if valid else None,
            "per_class_mean_episode_iou": class_episode_means,
            "class_mean_episode_iou": sum(valid_class_episode) / len(valid_class_episode)
            if valid_class_episode else None,
            "mean_episode_iou": sum(episode_ious) / len(episode_ious) if episode_ious else None}


def main():
    parser = argparse.ArgumentParser(description="Validate an episode manifest; never runs a model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--allow-example", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") == "ready" and args.dataset_root is None:
        parser.error("Ready manifests require --dataset-root for file hash verification")
    validate_manifest(manifest, root=args.dataset_root, allow_example=args.allow_example)
    print(json.dumps({"valid": True, "status": manifest["status"],
                      "manifest_sha256": manifest_fingerprint(manifest),
                      "episodes": len(manifest["episodes"])}, indent=2))


if __name__ == "__main__":
    main()
