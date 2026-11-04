#!/usr/bin/env python3
"""Print bounded, redacted, read-only evidence for an existing ICS queue."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_diagnostics import redact

MAX_SECTION_CHARS = 16000
MAX_TAIL_BYTES = 65536
MAX_JSON_BYTES = 16 * 1024 * 1024


def emit(value):
    text = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else str(value)
    # Error strings inside JSON may escape their quotes. Protect those values
    # before the shared plain-text redactor sees a backslash as the value.
    text = re.sub(r'''(?i)((?:api[_-]?key|access[_-]?token|authorization)\s*(?:\\?["'])?\s*[:=]\s*(?:\\?["'])?\s*(?:bearer\s+)?)[^\s"'\\,]+''',
                  r'\1[REDACTED]', text)
    text = redact(text)
    if len(text) > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS] + '\n[输出已截断]'
    print(text, flush=True)


def safe_path(run, relative):
    """Never follow an evidence file or parent symlink outside the selected run."""
    path = run / relative
    current = run
    for part in Path(relative).parts:
        current = current / part
        if part in ('..', '/') or current.is_symlink():
            raise ValueError('拒绝路径越界或符号链接: ' + str(relative))
    if not path.resolve().is_relative_to(run):
        raise ValueError('拒绝 run 目录外路径: ' + str(relative))
    if path.exists() and not path.is_file():
        raise ValueError('拒绝非普通文件: ' + str(relative))
    return path


def read_json(run, relative):
    with safe_path(run, relative).open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(str(relative) + ' 超过诊断读取上限')
    return json.loads(raw)


def tail(run, relative, lines):
    emit('\n【文件】' + str(relative))
    try:
        with safe_path(run, relative).open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - MAX_TAIL_BYTES))
            raw = stream.read(MAX_TAIL_BYTES)
        if size > MAX_TAIL_BYTES:
            # Discard a partial first line, which could begin inside a secret.
            raw = raw.split(b'\n', 1)[1] if b'\n' in raw else b''
            emit('[仅读取文件尾部；超长行可能省略]')
        emit('\n'.join(raw.decode('utf-8', errors='replace').splitlines()[-lines:]))
    except (OSError, ValueError) as exc:
        emit('无法读取: ' + type(exc).__name__ + ': ' + str(exc))


def command(args, label):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
        for output in (result.stdout, result.stderr):
            if output:
                emit(output)
        if result.returncode:
            emit(f'{label} 返回码: {result.returncode}')
    except subprocess.TimeoutExpired as exc:
        emit(label + ' 查询超时（30 秒）；继续收集其他证据')
        for output in (exc.stdout, exc.stderr):
            if output:
                emit(output)
    except OSError as exc:
        emit(label + ' 执行失败: ' + type(exc).__name__ + ': ' + str(exc))


def diagnose(run, tail_lines):
    run = run.resolve()
    emit('【当前仓库新版只读 status；不查询 Slurm】')
    try:
        # queue_status reads state through plan.run_dir. Reject a relocated or
        # tampered plan before invoking it, and guard its other fixed inputs.
        for name in ('rolling_state.json', 'manager.json', 'manager.lock'):
            safe_path(run, name)
        plan = read_json(run, 'plan.json')
        if isinstance(plan, dict) and 'run_dir' in plan:
            if not isinstance(plan['run_dir'], str) or Path(plan['run_dir']).resolve() != run:
                raise ValueError('plan.run_dir 与选定 run 不一致，未调用 status')
        command([sys.executable, '-I', str(ROOT / 'scripts/queue_ics.py'),
                 'status', '--run-dir', str(run)], 'status')
    except (OSError, ValueError, TypeError) as exc:
        emit('无法读取 status: ' + type(exc).__name__ + ': ' + str(exc))

    tail(run, 'manager.log', tail_lines)
    emit('\n【已保存的 data／preflight 任务】')
    tasks = {}
    try:
        state = read_json(run, 'rolling_state.json')
        if not isinstance(state, dict) or not isinstance(state.get('tasks'), dict):
            raise ValueError('state/tasks 不是 JSON object')
        tasks = state['tasks']
    except (OSError, ValueError, TypeError) as exc:
        emit('读取已保存任务失败: ' + type(exc).__name__ + ': ' + str(exc))

    job_ids = []
    for key in ('data', 'preflight'):
        task = tasks.get(key)
        if isinstance(task, dict):
            emit(json.dumps({field: task.get(field) for field in
                             ('key', 'status', 'job_id', 'error')}, ensure_ascii=False))
            job_id = str(task.get('job_id') or '')
            if re.fullmatch(r'[0-9]{1,20}(?:_[0-9]{1,20})?', job_id):
                job_ids.append(job_id)
                tail(run, f'logs/{key}-{job_id}.log', tail_lines)
            elif job_id:
                emit(key + ': job ID 格式无效，未用于文件名或查询')
        else:
            emit(key + ': 没有有效任务对象')
        tail(run, f'rolling_markers/{key}.json', tail_lines)

    if job_ids:
        emit('\n【Slurm accounting：仅查询现有 data／preflight 作业】')
        command(['sacct', '-n', '-P', '-X', '-j', ','.join(dict.fromkeys(job_ids)),
                 '--format=JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed'], 'sacct')
    else:
        emit('\n没有可查询的 data／preflight job ID')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--tail-lines', type=int, default=80)
    args = parser.parse_args(argv)
    if not 1 <= args.tail_lines <= 200:
        parser.error('--tail-lines must be between 1 and 200')
    diagnose(args.run_dir, args.tail_lines)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
