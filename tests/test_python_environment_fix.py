"""Focused original-source excerpts and real symlinked venv regressions.

This does not represent a run of the unavailable full repository test suite.
No pip, model, dataset, network or Slurm operation is required.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import venv


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("apply_python_fix", ROOT / "scripts" / "apply_python_fix.py")
fix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fix)

# Exact relevant functions from the original source read in this conversation.
# The surrounding imports/constants merely make that excerpt executable here.
ORIGINAL = '''"""Focused source fixture, not the entire orchestration module."""
import os
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCE_NAMES = {"insid3", "cgics_visual", "cgics_mllm"}

def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        value = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\\$\\{[^}]+\\}|\\$[A-Za-z_]", value):
            raise ValueError(f"Unresolved environment variable in configuration: {value}")
    return value

def resolve_path(value: str, relative_to: Path = PROJECT_ROOT) -> str:
    p = Path(expand(value))
    return str(p.resolve() if p.is_absolute() else (relative_to / p).resolve())


def resource_profile(cluster: dict, name: str) -> dict:
    p = dict(cluster.get("defaults", {}))
    p.update(cluster.get("resources", {}).get(name, {}))
    if "python" not in p:
        raise ValueError(f"Resource profile {name} needs an explicit python executable")
    p["python"] = resolve_path(p["python"])
    p["gpus"] = int(p.get("gpus", 0))
    if p["gpus"] < 0:
        raise ValueError("GPU allocation cannot be negative")
    if name in RESOURCE_NAMES and p["gpus"] < 1:
        raise ValueError(f"GPU experiment profile {name} must request GPUs")
    if name == "cgics_mllm":
        mode = p.get("mllm_mode", "owned")
        if mode not in {"owned", "external"}:
            raise ValueError("mllm_mode must be owned or external")
        if mode == "owned":
            if p["gpus"] < 2:
                raise ValueError("Owned SAM3 + vLLM profile requires separate allocated GPUs (>=2)")
            for field in ("mllm_python", "mllm_model_path"):
                if not p.get(field):
                    raise ValueError(f"Owned MLLM profile requires {field}")
                p[field] = resolve_path(p[field])
        else:
            endpoint = p.get("mllm_endpoint", "")
            if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
                if not p.get("allow_remote_mllm", False):
                    raise ValueError("Remote MLLM requires explicit allow_remote_mllm: true; images leave this worker")
            if not endpoint:
                raise ValueError("External MLLM requires mllm_endpoint")
    return p

# 无关用户注释必须逐字保留。
def unrelated_function(value):
    return "unchanged:" + value
'''


class TransformationTests(unittest.TestCase):
    def load_fixed(self, text=ORIGINAL):
        result = fix.transform_orchestration(text.encode())
        namespace = {"__file__": str(ROOT / "ics" / "orchestration.py")}
        exec(compile(result, "focused-orchestration-fixture", "exec"), namespace)
        return namespace, result

    def test_patch_is_idempotent_and_preserves_unrelated_source(self):
        namespace, patched = self.load_fixed()
        self.assertEqual(fix.transform_orchestration(patched), patched)
        before = ORIGINAL.split("# 无关用户注释", 1)[1]
        self.assertTrue(patched.decode().endswith("# 无关用户注释" + before))
        self.assertEqual(namespace["unrelated_function"]("ok"), "unchanged:ok")
        self.assertIn(fix.ORIGINAL_PATH_HELPER, patched.decode())

    def test_crlf_and_non_ascii_are_preserved(self):
        original = ORIGINAL.replace("\n", "\r\n").encode()
        patched = fix.transform_orchestration(original)
        self.assertNotIn(b"\n", patched.replace(b"\r\n", b""))
        self.assertIn("# 无关用户注释必须逐字保留。\r\n".encode(), patched)

    def test_unknown_and_partial_shapes_are_rejected(self):
        variants = [
            ORIGINAL.replace('p["python"] = resolve_path(p["python"])', 'p["python"] = custom_python(p["python"])'),
            ORIGINAL.replace('p["python"] = resolve_path(p["python"])', 'p["python"] = resolve_executable(p["python"])'),
            ORIGINAL.replace('p[field] = resolve_path(p[field])', 'p[field] = custom_python(p[field])'),
            ORIGINAL.replace('("mllm_python", "mllm_model_path")', '("mllm_python", "another_path")'),
            ORIGINAL.replace('p = Path(expand(value))', 'p = Path(value)'),
            ORIGINAL + '\nresolve_executable = lambda value: value\n',
        ]
        for source in variants:
            with self.subTest(source=source[-150:]), self.assertRaises(ValueError):
                fix.transform_orchestration(source.encode())

    def test_existing_foreign_helper_is_rejected(self):
        with self.assertRaises(ValueError):
            fix.transform_orchestration((ORIGINAL + '\ndef resolve_executable(value):\n    return value\n').encode())

    def test_nested_and_local_helper_shadowing_is_rejected(self):
        variants = [
            ORIGINAL + '\nif True:\n    resolve_executable = lambda value: "wrong"\n',
            ORIGINAL.replace('    p = dict(cluster.get("defaults", {}))',
                             '    resolve_executable = lambda value: "wrong"\n    p = dict(cluster.get("defaults", {}))'),
            ORIGINAL + '\ndef outer():\n    def resolve_executable(value):\n        return value\n',
            ORIGINAL + '\nimport resolve_executable.child\n',
        ]
        for source in variants:
            with self.subTest(source=source[-100:]), self.assertRaisesRegex(ValueError, "another binding"):
                fix.transform_orchestration(source.encode())

    def test_unresolved_environment_variable_stays_an_error(self):
        namespace, _ = self.load_fixed()
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "Unresolved"):
            namespace["resolve_executable"]("${ICS_MISSING_TEST_VARIABLE}/bin/python")

    def test_relative_and_environment_paths_are_absolute_without_dereferencing(self):
        namespace, _ = self.load_fixed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "actual-python"
            real.write_text("sentinel")
            link = root / "env" / "bin" / "python"
            link.parent.mkdir(parents=True)
            link.symlink_to(real)
            with mock.patch.dict(os.environ, {"ICS_TEST_RUNTIME": str(root)}):
                self.assertEqual(namespace["resolve_executable"]("${ICS_TEST_RUNTIME}/env/bin/python"), str(link))
            self.assertEqual(namespace["resolve_executable"]("env/bin/python", root), str(link))
            self.assertEqual(namespace["resolve_path"](str(link)), str(real))

    def test_real_venv_worker_and_mllm_interpreters_keep_environment_and_assets_resolve(self):
        namespace, _ = self.load_fixed()
        with tempfile.TemporaryDirectory(prefix="ics-interpreter-test-") as temporary:
            root = Path(temporary)
            environment = root / "env"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = environment / "bin" / "python"
            self.assertTrue(python.is_symlink(), "Regression requires a symlinked venv")
            purelib = Path(subprocess.check_output([str(python), "-I", "-c",
                'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
            (purelib / "ics_venv_sentinel.py").write_text('VALUE = "venv-only"\n')
            model = root / "model"
            model.mkdir()
            model_link = root / "model-link"
            model_link.symlink_to(model, target_is_directory=True)
            cluster = {"resources": {"cgics_mllm": {"python": str(python), "gpus": 2,
                       "mllm_python": str(python), "mllm_model_path": str(model_link)}}}
            profile = namespace["resource_profile"](cluster, "cgics_mllm")
            # JSON round-trip matches the immutable plan's relevant serialization.
            restored = json.loads(json.dumps({"resources": {"cgics_mllm": profile}}))["resources"]["cgics_mllm"]
            self.assertEqual(restored["mllm_model_path"], str(model.resolve()))
            for field in ("python", "mllm_python"):
                self.assertEqual(restored[field], str(python))
                data = json.loads(subprocess.check_output([restored[field], "-I", "-c",
                    'import json,sys,ics_venv_sentinel; print(json.dumps({"prefix":sys.prefix,"value":ics_venv_sentinel.VALUE}))'], text=True))
                self.assertEqual(data, {"prefix": str(environment), "value": "venv-only"})
            baseline = subprocess.run([str(python.resolve()), "-I", "-c", "import ics_venv_sentinel"], capture_output=True)
            self.assertNotEqual(baseline.returncode, 0, "Original resolved executable must fail this regression")


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ics-patch-updater-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "download"
        self.target = self.root / "clone"
        (self.source / "scripts").mkdir(parents=True)
        (self.target / "ics").mkdir(parents=True)
        for name in ("apply_update.py", "apply_python_fix.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.source / "scripts" / name)
        (self.source / "README.md").write_text("updated documentation\n")
        (self.target / "README.md").write_text("old documentation\n")
        (self.target / "unrelated.txt").write_text("user file\n")
        (self.target / "ics" / "orchestration.py").write_text(ORIGINAL)
        subprocess.run(["git", "init", "--quiet", str(self.target)], check=True)
        subprocess.run(["git", "-C", str(self.target), "remote", "add", "origin",
                        "https://github.com/AvianHan/CE6190-Visual-Understanding.git"], check=True)
        files = ["README.md", "scripts/apply_update.py", "scripts/apply_python_fix.py"]
        self.manifest = {"schema_version": 1, "files": {
            name: hashlib.sha256((self.source / name).read_bytes()).hexdigest() for name in files}}
        self.save_manifest()

    def save_manifest(self):
        (self.source / "UPDATE_MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_dry_run_then_apply_backups_git_and_idempotence(self):
        git_config = (self.target / ".git" / "config").read_bytes()
        preview = fix.apply(self.source, self.target)
        self.assertEqual(preview["status"], "PLANNED")
        self.assertEqual((self.target / "ics/orchestration.py").read_text(), ORIGINAL)
        self.assertFalse((self.root / "clone-update-backups").exists())
        result = fix.apply(self.source, self.target, True)
        self.assertEqual(result["status"], "APPLIED")
        backup = Path(result["backup"])
        self.assertEqual((backup / "ics/orchestration.py").read_text(), ORIGINAL)
        self.assertEqual((backup / "README.md").read_text(), "old documentation\n")
        self.assertEqual((self.target / "unrelated.txt").read_text(), "user file\n")
        self.assertEqual((self.target / ".git/config").read_bytes(), git_config)
        again = fix.apply(self.source, self.target, True)
        self.assertEqual(again["status"], "UP_TO_DATE")
        self.assertEqual(again["python_path_fix"], "ALREADY_FIXED")

    def test_unknown_target_prevents_all_writes(self):
        path = self.target / "ics/orchestration.py"
        original = ORIGINAL.replace('p["python"] = resolve_path(p["python"])', 'p["python"] = unknown(p["python"])')
        path.write_text(original)
        with self.assertRaises(ValueError):
            fix.apply(self.source, self.target, True)
        self.assertEqual(path.read_text(), original)
        self.assertEqual((self.target / "README.md").read_text(), "old documentation\n")
        self.assertFalse((self.target / "scripts").exists())

    def test_bundle_hash_failure_prevents_all_writes(self):
        (self.source / "README.md").write_text("altered bytes")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            fix.apply(self.source, self.target, True)
        self.assertEqual((self.target / "ics/orchestration.py").read_text(), ORIGINAL)

    def test_wrong_git_remote_is_rejected(self):
        subprocess.run(["git", "-C", str(self.target), "remote", "set-url", "origin", "https://example.com/other.git"], check=True)
        with self.assertRaisesRegex(ValueError, "origin"):
            fix.apply(self.source, self.target, True)
        self.assertEqual((self.target / "ics/orchestration.py").read_text(), ORIGINAL)

    def test_target_symlink_is_rejected(self):
        outside = self.root / "other.py"
        outside.write_text(ORIGINAL)
        path = self.target / "ics/orchestration.py"
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "regular"):
            fix.apply(self.source, self.target, True)
        self.assertEqual(outside.read_text(), ORIGINAL)

    def test_source_traversal_and_whole_module_replacement_are_rejected(self):
        for name in ("../outside.py", "ics/orchestration.py"):
            with self.subTest(name=name):
                manifest = dict(self.manifest["files"])
                self.manifest["files"][name] = "0" * 64
                self.save_manifest()
                with self.assertRaises(ValueError):
                    fix.apply(self.source, self.target, True)
                self.manifest["files"] = manifest
        self.assertEqual((self.target / "ics/orchestration.py").read_text(), ORIGINAL)


if __name__ == "__main__":
    unittest.main()
