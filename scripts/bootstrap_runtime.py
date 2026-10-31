#!/usr/bin/env python3
"""Prepare pinned sources/environments/weights outside the project, idempotently.

Dry-run unless --execute. This installs software/downloads only; no GPU jobs.
Never accepts license terms on behalf of the user. Never prints signed URLs/tokens.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd, *, cwd=None, capture=False):
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    if p.returncode:
        raise RuntimeError(f'Command failed: {Path(str(cmd[0])).name}, exit={p.returncode}; see terminal output')
    return p.stdout.strip() if capture else ''


def checkout(path, spec):
    if path.exists():
        if not (path / '.git').exists():
            raise ValueError(f'Refusing non-Git source directory: {path}')
        actual = run(['git', 'rev-parse', 'HEAD'], cwd=path, capture=True)
        if actual != spec['commit'] or run(['git', 'status', '--porcelain'], cwd=path, capture=True):
            raise ValueError(f'Existing source differs from pin or has local edits: {path}; preserving it')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(['git', 'clone', '--no-checkout', '--filter=blob:none', spec['url'], path])
    run(['git', 'checkout', '--detach', spec['commit']], cwd=path)


def environment(runtime, name, interpreter, pins, refresh=False):
    """Independent model environments prevent vLLM replacing segmentation torch."""
    env = runtime / 'envs' / name
    py = env / 'bin' / 'python'
    shared = ['numpy==1.26.4', 'Pillow==11.3.0', 'einops==0.8.1',
              'scikit-learn==1.7.2', 'scipy==1.16.3', 'pycocotools==2.0.10',
              'tqdm==4.67.1', 'termcolor==3.3.0', 'matplotlib==3.10.7',
              'gdown>=5,<6', 'huggingface_hub==0.36.0']
    specs = {
        'insid3': {'torch': ['torch==2.7.1', 'torchvision==0.22.1'], 'index': 'cu126',
                   'deps': shared + ['torchmetrics==1.8.2']},
        'cgics': {'torch': ['torch==2.9.0', 'torchvision==0.24.0'], 'index': 'cu128',
                  'deps': shared + ['torchmetrics==1.8.2', 'timm==1.0.22', 'ftfy==6.1.1',
                  'regex', 'iopath==0.1.10', 'openai==2.8.1', 'opencv-python-headless==4.11.0.86',
                  'scikit-image==0.25.2', 'hydra-core==1.3.2', 'omegaconf==2.3.0',
                  'requests', 'pandas==2.3.3', 'decord==0.6.0', 'psutil==7.1.3']},
        'mllm': {'torch': [], 'index': None, 'deps': ['vllm==0.11.2',
                  'transformers==4.57.3', 'qwen-vl-utils==0.0.14', 'huggingface_hub==0.36.0']}
    }
    spec = specs[name]
    key = hashlib.sha256(json.dumps({'spec': spec, 'python': interpreter,
                                    'sam3_pin': pins['sources']['CG-ICS']['commit']}, sort_keys=True).encode()).hexdigest()
    marker = env / 'ics-install.json'
    if marker.exists() and not refresh:
        if json.loads(marker.read_text()).get('spec_sha256') != key:
            raise ValueError(f'Environment spec changed: {env}; use --refresh-envs after reviewing')
        run([py, '-m', 'pip', 'check'])
        import_smoke(runtime, name, py)
        return
    if not py.exists():
        if env.exists() and any(env.iterdir()):
            raise ValueError(f'Incomplete non-venv directory preserved: {env}')
        run([interpreter, '-m', 'venv', env])
    run([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools<81'])
    if spec['torch']:
        run([py, '-m', 'pip', 'install', *spec['torch'], '--index-url',
             'https://download.pytorch.org/whl/' + spec['index']])
    run([py, '-m', 'pip', 'install', *spec['deps']])
    if name == 'cgics':
        # Vendored sam3 declares numpy==1.26 (1.26.0), incompatible with its other
        # pinned scipy dependencies. Explicit compatible 1.26.4 above; do not
        # install/rewrite the vendor package or its requirements silently.
        pass  # Adapter loads pinned source directly; all runtime deps are above.
    run([py, '-m', 'pip', 'check'])
    import_smoke(runtime, name, py)
    freeze = run([py, '-m', 'pip', 'freeze'], capture=True)
    (env / 'ics-pip-freeze.txt').write_text(freeze + '\n')
    marker.write_text(json.dumps({'spec_sha256': key, 'created_at': time.time(),
                                 'torch_index': spec['index'], 'status': 'installed_not_gpu_validated'}, indent=2))


def import_smoke(runtime, name, py):
    """Import source dependencies without loading weights or allocating CUDA."""
    if name == 'insid3':
        code = ('import sys;sys.path.insert(0,sys.argv[1]);'
                'import models.insid3;import pycocotools.mask;print("INSID3 imports OK; model not loaded")')
        run([py, '-c', code, runtime/'sources/INSID3'])
    elif name == 'cgics':
        code = ('import sys;sys.path.insert(0,sys.argv[1]);'
                'from ics.adapters.cgics import _load_upstream;'
                '_load_upstream(sys.argv[2]);print("CG-ICS imports OK; model not loaded")')
        run([py, '-c', code, PROJECT, runtime/'sources/CG-ICS'])
    else:
        run([py, '-c', 'import vllm;import transformers;print("vLLM",vllm.__version__,"transformers",transformers.__version__)'])


def download_url(url, target):
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.part')
    # Restart interrupted URL transfers, never append an unchecked Range response.
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'CE6190-ICS/1.0'})
        with urllib.request.urlopen(req, timeout=60) as source, tmp.open('wb') as out:
            length = source.headers.get('Content-Length')
            shutil.copyfileobj(source, out, 8 * 1024 * 1024)
        if not tmp.stat().st_size or (length is not None and tmp.stat().st_size != int(length)):
            raise RuntimeError('incomplete weight transfer')
        tmp.replace(target)
    except Exception as exc:
        # Do not include signed download URL in errors.
        raise RuntimeError(f'{target.name} download failed ({type(exc).__name__}); partial file kept') from None


def weights(runtime, pins, names):
    folder = runtime / 'weights'
    folder.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for name in names:
        spec = pins['weights'][name]
        target = folder / spec['target']
        try:
            if name in ('sam3', 'qwen'):
                from huggingface_hub import hf_hub_download, snapshot_download
                if name == 'sam3':
                    if not target.is_file():
                        src = hf_hub_download(repo_id=spec['repo'], filename=spec['filename'],
                                              revision=spec['revision'])
                        shutil.copyfile(src, target)
                else:
                    snapshot_download(repo_id=spec['repo'], revision=spec['revision'],
                                      local_dir=target,
                                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
                inventory[name] = {'status': 'available', 'path': str(target), 'revision': spec['revision']}
            else:
                if not target.is_file():
                    url = spec.get('url') or os.environ.get(spec.get('url_env', ''), '')
                    if not url:
                        raise ValueError('Approved official DINOv3 .pth is required: place weights/dinov3_L.pth or set ICS_DINOV3_URL privately')
                    download_url(url, target)
                inventory[name] = {'status': 'available', 'path': str(target)}
            if target.is_file():
                inventory[name]['sha256'] = digest(target)
                inventory[name]['bytes'] = target.stat().st_size
        except Exception as exc:
            # HF errors can include request URLs; keep actionable public details only.
            inventory[name] = {'status': 'blocked', 'error_type': type(exc).__name__,
                               'action': ('Obtain approved official PyTorch DINOv3 checkpoint and place it at '+str(target))
                               if name == 'dinov3' else 'Check authorized model access, disk space and network; rerun to resume'}
    (runtime / 'weight_inventory.json').write_text(json.dumps(inventory, indent=2))
    print(json.dumps({'weights': inventory}, indent=2))
    return all(x['status'] == 'available' for x in inventory.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, default=os.environ.get('ICS_RUNTIME'))
    p.add_argument('--stages', nargs='+', choices=['sources', 'envs', 'weights'], default=['sources', 'envs', 'weights'])
    p.add_argument('--envs', nargs='+', choices=['insid3', 'cgics', 'mllm'], default=['insid3', 'cgics', 'mllm'])
    p.add_argument('--models', nargs='+', choices=['dinov3', 'dinov2', 'sam3', 'qwen'], default=['dinov3', 'dinov2', 'sam3', 'qwen'])
    p.add_argument('--python', default=shutil.which('python3.12'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--refresh-envs', action='store_true')
    args = p.parse_args()
    if args.runtime is None:
        p.error('--runtime or ICS_RUNTIME must name persistent server storage outside the Git project')
    runtime = args.runtime.expanduser().resolve()
    if runtime.is_relative_to(PROJECT):
        p.error('Runtime must be outside the Git project to exclude weights/data from submission')
    pins = json.loads((PROJECT / 'configs/source_pins.json').read_text())
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY', 'runtime': str(runtime), 'stages': args.stages,
                          'environments': args.envs, 'weights': args.models,
                          'note': '3 separate envs; gated downloads require existing authorization; GPU not tested'}, indent=2))
        return
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(runtime / 'cache' / 'huggingface'))
    os.environ.setdefault('PIP_CACHE_DIR', str(runtime / 'cache' / 'pip'))
    errors = []
    if 'sources' in args.stages:
        for name, spec in pins['sources'].items():
            try:
                print('Source:', name, flush=True)
                checkout(runtime / 'sources' / name, spec)
            except Exception as exc:
                errors.append({'stage': 'sources', 'item': name, 'error': str(exc)})
    if 'envs' in args.stages:
        if not args.python:
            errors.append({'stage': 'envs', 'error': 'Python3.12 not found; load an available Python3.12 module and pass --python PATH'})
        else:
            version = run([args.python, '-c', 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")'], capture=True)
            if version != '3.12':
                errors.append({'stage': 'envs', 'error': 'This environment specification requires Python3.12'})
            else:
                for name in args.envs:
                    try:
                        print('Environment:', name, flush=True)
                        environment(runtime, name, args.python, pins, args.refresh_envs)
                    except Exception as exc:
                        errors.append({'stage': 'envs', 'item': name, 'error': str(exc)})
    if 'weights' in args.stages:
        # Use the prepared lightweight hub install if the caller has no hub.
        if not os.environ.get('ICS_WEIGHT_CHILD') and (runtime/'envs/insid3/bin/python').exists():
            env = os.environ.copy(); env['ICS_WEIGHT_CHILD'] = '1'
            proc = subprocess.run([str(runtime/'envs/insid3/bin/python'), __file__, '--runtime', str(runtime),
                                   '--stages', 'weights', '--models', *args.models, '--execute'], env=env)
            if proc.returncode:
                errors.append({'stage': 'weights', 'error': 'Some models unavailable; inspect weight_inventory.json'})
        elif not weights(runtime, pins, args.models):
            errors.append({'stage': 'weights', 'error': 'Some models unavailable; inspect weight_inventory.json'})
    result = {'status': 'PREPARED_NOT_GPU_VALIDATED' if not errors else 'NOT_READY',
              'runtime': str(runtime), 'errors': errors, 'date': time.time()}
    (runtime/'bootstrap_status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
