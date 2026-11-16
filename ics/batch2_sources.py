"""Reuse transfer data; acquire only absent catalog assets and organize ISIC.

No model weights or existing files are replaced. The official pinned CG-ICS
class table supplies ISIC's author categories; no diagnosis is inferred here.
"""
from __future__ import annotations

import csv
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .orchestration import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CG_PIN = "8da845743f15c18ec490567fbd86dba58c8a388f"
ISIC_TABLE_SHA256 = "312046a57ed9f563f331bb3c14517bb8d4a69c9e536a1a44bf720d5b7f263870"
ISIC_CLASSES = {"nevus": "1", "melanoma": "2", "seborrheic_keratosis": "3"}
SUIM_CLASSES = ("FV", "HD", "PF", "RI", "RO", "SR", "WR")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Transfer asset must be a regular path within data root: {path}")
    return path


def _downloader():
    path = ROOT / "scripts/download_data.py"
    spec = importlib.util.spec_from_file_location("_ics_batch2_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_mapping(runtime):
    source = Path(runtime) / "sources/CG-ICS"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=20).strip()
    if commit != CG_PIN:
        raise ValueError("ISIC mapping must come from the existing pinned CG-ICS checkout")
    table = source / "datasets/isic/class_id.csv"
    if table.is_symlink() or _sha(table) != ISIC_TABLE_SHA256:
        raise ValueError("Pinned ISIC class table changed; no reorganization attempted")
    with table.open(newline="") as f:
        rows = list(csv.DictReader(f))
    mapping = {}
    for row in rows:
        name, category = row["ID"], row["Class"]
        if not name.startswith("ISIC_") or not name[5:].isdigit() or category not in ISIC_CLASSES:
            raise ValueError("Invalid pinned ISIC mapping record")
        if name in mapping:
            raise ValueError("Duplicate ISIC mapping ID")
        mapping[name] = ISIC_CLASSES[category]
    if len(mapping) != 2594:
        raise ValueError("Unexpected pinned ISIC mapping size")
    return mapping, {"source_commit": commit, "class_table_path": str(table),
                     "class_table_sha256": ISIC_TABLE_SHA256,
                     "class_names": ISIC_CLASSES, "mapped_image_count": len(mapping)}


def _isic_missing(root, mapping):
    images = root / "ISIC/ISIC2018_Task1-2_Training_Input"
    masks = root / "ISIC/ISIC2018_Task1_Training_GroundTruth"
    missing_images, missing_masks = [], []
    for name, cid in mapping.items():
        flat, classified = images / (name + ".jpg"), images / cid / (name + ".jpg")
        for p in (flat, classified, masks / (name + "_segmentation.png")):
            _inside(root, p)
        if not flat.is_file() and not classified.is_file():
            missing_images.append(name)
        if not (masks / (name + "_segmentation.png")).is_file():
            missing_masks.append(name)
    return missing_images, missing_masks


def _publish_same_bytes(source, destination, guard):
    """Keep originals; hardlink where possible, otherwise exclusive staged copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or _sha(source) != _sha(destination):
            raise ValueError(f"Existing ISIC class image differs: {destination}")
        return "existing_identical"
    try:
        os.link(source, destination)
        return "hardlink"
    except FileExistsError:
        return _publish_same_bytes(source, destination, guard)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):
            raise
    guard.check(destination.parent, source.stat().st_size)
    fd, temp = tempfile.mkstemp(prefix=".batch2-isic-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        if _sha(temp) != _sha(source):
            raise IOError("ISIC staging copy checksum mismatch")
        # Exclusive target publication: never replace an existing user image.
        try:
            os.link(temp, destination)
        except FileExistsError:
            if _sha(destination) != _sha(source):
                raise ValueError(f"Concurrent differing image: {destination}")
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):
                raise
            try:
                with destination.open("xb") as out, Path(temp).open("rb") as inp:
                    shutil.copyfileobj(inp, out)
                    out.flush()
                    os.fsync(out.fileno())
            except FileExistsError:
                if _sha(destination) != _sha(source):
                    raise ValueError(f"Concurrent differing image: {destination}")
            if _sha(destination) != _sha(source):
                raise IOError("ISIC exclusive copy checksum mismatch; incomplete target retained")
        return "copy"
    finally:
        Path(temp).unlink(missing_ok=True)


def _organize_isic(root, mapping, guard):
    images = root / "ISIC/ISIC2018_Task1-2_Training_Input"
    actions = {}
    for name, cid in sorted(mapping.items()):
        destination = _inside(root, images / cid / (name + ".jpg"))
        flat = _inside(root, images / (name + ".jpg"))
        # Validate existing organized inputs too; a wrongly labelled copy is not reused.
        for other in ISIC_CLASSES.values():
            unexpected = _inside(root, images / other / (name + ".jpg"))
            if other != cid and unexpected.exists():
                raise ValueError(f"ISIC image is in the wrong author class: {unexpected}")
        if flat.is_file():
            action = _publish_same_bytes(flat, destination, guard)
        elif destination.is_file():
            action = "existing_organized"
        else:
            raise FileNotFoundError(f"Missing ISIC source image: {name}")
        actions[action] = actions.get(action, 0) + 1
    for cid in ISIC_CLASSES.values():
        for image in (images / cid).glob("*.jpg"):
            if mapping.get(image.stem) != cid:
                raise ValueError(f"Unmapped ISIC image in author class directory: {image}")
    return actions


def _suim_layout(root):
    base = _inside(root, root / "SUIM")
    images = _inside(root, base / "images")
    counts = {}
    if not images.is_dir() or not any(p.is_file() for p in images.iterdir()):
        return False, counts
    for name in SUIM_CLASSES:
        folder = _inside(root, base / "masks" / name)
        count = sum(p.is_file() for p in folder.iterdir()) if folder.is_dir() else 0
        counts[name] = count
    return all(counts.values()), counts


def ensure_transfer_sources(runtime, dataset, *, allow_download=False, min_free_gib=32):
    """Return provenance after source availability; model validation is separate."""
    if dataset not in {"suim", "isic"}:
        raise ValueError("Batch2 source preparation supports SUIM/ISIC only")
    runtime = Path(runtime).resolve()
    root = _inside(runtime, runtime / "data")
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(min_free_gib, (int, float)) or not 0 <= min_free_gib < float("inf"):
        raise ValueError("Invalid free space guard")
    receipts = root / "batch2_source_receipts"
    receipts.mkdir(exist_ok=True)
    with (receipts / f"{dataset}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        download = _downloader()
        guard = download.CapacityGuard(int(min_free_gib * 2**30))
        guard.check(root)
        mapping = None
        report = {"dataset": dataset, "allow_download": bool(allow_download),
                  "data_root": str(root), "models_downloaded": False, "assets": []}
        if dataset == "isic":
            mapping, evidence = _official_mapping(runtime)
            report.update(evidence)
            missing_images, missing_masks = _isic_missing(root, mapping)
            missing_assets = (["isic2018_images"] if missing_images else []) + (["isic2018_masks"] if missing_masks else [])
        else:
            complete, counts = _suim_layout(root)
            report["existing_class_file_counts"] = counts
            missing_assets = [] if complete else ["suim_merged"]
        if missing_assets and not allow_download:
            raise FileNotFoundError("Missing transfer assets (no downloads authorized by this invocation): " + ", ".join(missing_assets))
        catalog = json.loads((ROOT / "configs/data_sources.json").read_text())
        assets = {a["id"]: a for a in catalog["assets"]}
        for name in missing_assets:
            asset = assets[name]
            if asset["dataset"] != dataset:
                raise ValueError("Dataset/catalog identity mismatch")
            report["assets"].append(download.acquire(asset, root, root / "download_archives", guard))
        if dataset == "isic":
            missing_images, missing_masks = _isic_missing(root, mapping)
            if missing_images or missing_masks:
                raise FileNotFoundError(f"ISIC remains incomplete: {len(missing_images)} images, {len(missing_masks)} masks")
            report["organization_actions"] = _organize_isic(root, mapping, guard)
            report["protocol"] = "author_reorganized_public_training; not challenge test or verified patient-disjoint"
        else:
            complete, counts = _suim_layout(root)
            if not complete:
                raise FileNotFoundError("SUIM extraction does not match the pinned author merged layout")
            report["class_file_counts"] = counts
            report["protocol"] = "author_merged_seven_foreground_categories; not official test-only"
        report["status"] = "SOURCE_READY_REQUIRES_MANIFEST_AND_MODEL_VALIDATION"
        report["completeness_scope"] = ("all 2594 pinned mapping image/mask paths" if dataset == "isic" else
                                          "existing author layout; manifest records actual eligible inventory")
        atomic_json(receipts / f"{dataset}.json", report)
        return report
