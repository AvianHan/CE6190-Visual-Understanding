#!/usr/bin/env python3
"""Apply the ICS interpreter-path fix to an existing clone, with verified backups.

The full orchestration module is deliberately not bundled: this updater patches
only the recognized resolver expressions in the target's current source. All
other bytes are retained. The existing bundle updater performs the final apply.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


EXECUTABLE_HELPER = '''def resolve_executable(value: str, relative_to: Path = PROJECT_ROOT) -> str:
    """Make an interpreter path absolute without dereferencing a venv symlink."""
    p = Path(expand(value))
    return os.path.abspath(p if p.is_absolute() else relative_to / p)
'''

ORIGINAL_PATH_HELPER = '''def resolve_path(value: str, relative_to: Path = PROJECT_ROOT) -> str:
    p = Path(expand(value))
    return str(p.resolve() if p.is_absolute() else (relative_to / p).resolve())
'''

OLD_PYTHON = 'p["python"] = resolve_path(p["python"])'
NEW_PYTHON = 'p["python"] = resolve_executable(p["python"])'
OLD_MLLM = 'p[field] = resolve_path(p[field])'
NEW_MLLM = 'p[field] = resolve_executable(p[field]) if field == "mllm_python" else resolve_path(p[field])'
MATCH_NAME_NODES = tuple(getattr(ast, name) for name in ("MatchAs", "MatchStar") if hasattr(ast, name))
MATCH_MAPPING_NODE = getattr(ast, "MatchMapping", ())


def _same(left, right) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _statement(text: str):
    return ast.parse(text).body[0]


def _function(tree: ast.Module, name: str):
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ValueError(f"Unrecognized orchestration source: expected one {name} function")
    return matches[0]


def _shape(text: str):
    """Recognize one complete original/fixed shape, never a partial patch."""
    tree = ast.parse(text)
    path_helper = _function(tree, "resolve_path")
    if not _same(path_helper, _statement(ORIGINAL_PATH_HELPER)):
        raise ValueError("Unrecognized resolve_path implementation; nothing was changed")
    profile = _function(tree, "resource_profile")
    if profile.decorator_list or [arg.arg for arg in profile.args.args] != ["cluster", "name"]:
        raise ValueError("Unrecognized resource_profile signature; nothing was changed")
    has_os = any(isinstance(node, ast.Import) and any(alias.name == "os" and alias.asname is None
                 for alias in node.names) for node in tree.body)
    has_path = any(isinstance(node, ast.ImportFrom) and node.module == "pathlib" and
                   any(alias.name == "Path" and alias.asname is None for alias in node.names)
                   for node in tree.body)
    if not has_os or not has_path:
        raise ValueError("Expected os and pathlib.Path imports; nothing was changed")
    helper_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "resolve_executable"]
    # Reject other bindings conservatively in every scope. In particular, a
    # resource_profile local or an assignment under a module-level if could
    # shadow the helper while leaving the two expected call sites unchanged.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == "resolve_executable" for alias in node.names):
                raise ValueError("resolve_executable name already has another binding")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.name == "resolve_executable" and node not in helper_nodes:
                raise ValueError("resolve_executable name already has another binding")
        if ((isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
             and node.id == "resolve_executable") or
            (isinstance(node, ast.arg) and node.arg == "resolve_executable") or
            (isinstance(node, ast.ExceptHandler) and node.name == "resolve_executable") or
            (isinstance(node, MATCH_NAME_NODES) and node.name == "resolve_executable") or
            (isinstance(node, MATCH_MAPPING_NODE) and node.rest == "resolve_executable")):
            raise ValueError("resolve_executable name already has another binding")
    if len(helper_nodes) > 1 or (helper_nodes and not _same(helper_nodes[0], _statement(EXECUTABLE_HELPER))):
        raise ValueError("Unrecognized resolve_executable helper; nothing was changed")
    python_targets = [node for node in ast.walk(profile) if isinstance(node, ast.Assign)
                      and any(_same(target, _statement(OLD_PYTHON).targets[0]) for target in node.targets)]
    if len(python_targets) != 1:
        raise ValueError("Expected one resource Python assignment; nothing was changed")
    field_loops = [node for node in ast.walk(profile) if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Name) and node.target.id == "field"
                   and _same(node.iter, ast.parse('(\"mllm_python\", \"mllm_model_path\")', mode="eval").body)]
    if len(field_loops) != 1:
        raise ValueError("Expected one owned-MLLM path loop; nothing was changed")
    loop = field_loops[0]
    field_targets = [node for node in ast.walk(loop) if isinstance(node, ast.Assign)
                     and any(_same(target, _statement(OLD_MLLM).targets[0]) for target in node.targets)]
    if len(field_targets) != 1 or field_targets[0] not in loop.body:
        raise ValueError("Expected one direct owned-MLLM path assignment; nothing was changed")
    python_node, field_node = python_targets[0], field_targets[0]
    old = (_same(python_node, _statement(OLD_PYTHON)) and
           _same(field_node, _statement(OLD_MLLM)) and not helper_nodes)
    fixed = (_same(python_node, _statement(NEW_PYTHON)) and
             _same(field_node, _statement(NEW_MLLM)) and len(helper_nodes) == 1)
    if not old and not fixed:
        raise ValueError("Unrecognized or partial interpreter-path patch; nothing was changed")
    return "fixed" if fixed else "original", profile, python_node, field_node


def _offset(text: str, line: int, utf8_column: int) -> int:
    """AST columns are UTF-8 byte offsets; retain non-ASCII text exactly."""
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[:line - 1])) + len(lines[line - 1].encode("utf-8")[:utf8_column].decode("utf-8"))


def transform_orchestration(original: bytes) -> bytes:
    text = original.decode("utf-8")
    state, profile, python_node, field_node = _shape(text)
    if state == "fixed":
        return original
    newline = "\r\n" if "\r\n" in text else "\n"
    edits = []
    for node, replacement in ((python_node.value, _statement(NEW_PYTHON).value),
                              (field_node.value, _statement(NEW_MLLM).value)):
        edits.append((_offset(text, node.lineno, node.col_offset),
                      _offset(text, node.end_lineno, node.end_col_offset),
                      ast.unparse(replacement)))
    insert = _offset(text, profile.lineno, 0)
    edits.append((insert, insert, EXECUTABLE_HELPER.replace("\n", newline) + newline + newline))
    patched = text
    for start, end, replacement in sorted(edits, reverse=True):
        patched = patched[:start] + replacement + patched[end:]
    if _shape(patched)[0] != "fixed":
        raise ValueError("Interpreter-path patch validation failed; nothing was changed")
    compile(patched, "ics/orchestration.py", "exec")
    return patched.encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_bundle(source: Path) -> dict:
    manifest_path = source / "UPDATE_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Invalid update manifest")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Update manifest must contain file hashes")
    if "ics/orchestration.py" in files:
        raise ValueError("This focused bundle must not replace the complete orchestration module")
    for required in ("scripts/apply_update.py", "scripts/apply_python_fix.py"):
        if required not in files:
            raise ValueError("Update manifest omits " + required)
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Malformed bundle file hash")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts or "\\" in name:
            raise ValueError("Unsafe bundle path")
        path = source / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError("Invalid source file: " + name)
        if _sha(path.read_bytes()) != expected:
            raise ValueError("Downloaded bundle file hash mismatch: " + name)
    return manifest


def apply(source: Path, target: Path, execute: bool = False) -> dict:
    source, target = source.expanduser().resolve(), target.expanduser().resolve()
    if source == target or target.is_relative_to(source):
        raise ValueError("Run this updater from the extracted download, targeting the existing clone")
    manifest = _verified_bundle(source)
    original_path = target / "ics" / "orchestration.py"
    if original_path.is_symlink() or not original_path.is_file() or not original_path.resolve().is_relative_to(target):
        raise ValueError("Target needs its existing regular ics/orchestration.py")
    original = original_path.read_bytes()
    patched = transform_orchestration(original)
    # Load only the already hash-verified bundled updater; its normal checks,
    # backup behavior and Git preservation are retained unchanged.
    updater_path = source / "scripts" / "apply_update.py"
    namespace = {"__name__": "_ics_verified_bundle_updater", "__file__": str(updater_path)}
    exec(compile(updater_path.read_bytes(), str(updater_path), "exec"), namespace)
    with tempfile.TemporaryDirectory(prefix="ics-python-path-fix-") as temporary:
        stage = Path(temporary)
        for name in manifest["files"]:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        destination = stage / "ics" / "orchestration.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_path, destination)
        destination.write_bytes(patched)
        generated = dict(manifest, files=dict(manifest["files"]))
        generated["files"]["ics/orchestration.py"] = _sha(patched)
        (stage / "UPDATE_MANIFEST.json").write_text(json.dumps(generated, indent=2) + "\n")
        if original_path.read_bytes() != original:
            raise ValueError("Target orchestration changed while preparing the fix; retry after saving edits")
        result = namespace["apply"](stage, target, execute)
    result.update(python_path_fix="ALREADY_FIXED" if original == patched else "PRESERVE_VENV_SYMLINKS",
                  orchestration_before_sha256=_sha(original), orchestration_after_sha256=_sha(patched))
    if result.get("backup"):
        (Path(result["backup"]) / "update_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Apply verified changes, backing up every replaced file")
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.target, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
        raise SystemExit(f"Update refused: {exc}")
