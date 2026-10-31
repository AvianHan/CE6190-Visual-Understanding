"""Sample device memory of this job's owned process trees, including vLLM workers.

NVML/nvidia-smi driver allocations differ from PyTorch allocator peaks. Report
them separately. Polling gives a lower bound on transient peaks, never an exact
continuous maximum. No unrelated process names or arguments are collected.
"""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time


def process_snapshot():
    """PID -> (parent PID, kernel start ticks), without collecting process args."""
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            tail = text[text.rfind(')') + 2:].split()
            # After comm's closing parenthesis, state is field 3. Linux stat
            # starttime is field 22, hence index 19 in this remaining sequence.
            processes[int(path.parent.name)] = (int(tail[1]), int(tail[19]))
        except (OSError, ValueError, IndexError):
            continue
    return processes


def process_parents():
    return {pid: value[0] for pid, value in process_snapshot().items()}


def descendants(roots, parents):
    ownership = dict(roots)
    for _ in range(len(parents) + 1):
        additions = {pid: ownership[parent] for pid, parent in parents.items()
                     if pid not in ownership and parent in ownership}
        if not additions:
            break
        ownership.update(additions)
    return ownership


class Monitor:
    def __init__(self, output_path, interval=1.0):
        if interval <= 0:
            raise ValueError('Sampling interval must be positive')
        self.output_path = Path(output_path)
        self.interval = interval
        self.roots = {}
        self.root_starttimes = {}
        self.known = {}
        self.known_starttimes = {}
        self.peaks = {}
        self.total_peak = None
        self.samples = 0
        self.failed_samples = 0
        self.started = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.available = bool(shutil.which('nvidia-smi')) and Path('/proc').exists()

    def add_process(self, pid, role):
        if role not in {'mllm', 'runner'}:
            raise ValueError('Role must be mllm or runner')
        pid = int(pid)
        identity = process_snapshot().get(pid)
        # A process that already exited cannot safely establish ownership of a
        # later process using the same PID. Sampling may miss such short work.
        if identity is None:
            return
        with self._lock:
            self.roots[pid] = role
            self.root_starttimes[pid] = identity[1]

    def start(self):
        self.started = time.time()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _observe(self, output, ownership):
        snapshot = {}
        for row in csv.reader(io.StringIO(output)):
            if len(row) != 3:
                continue
            try:
                pid, gpu, memory = int(row[0].strip()), row[1].strip(), float(row[2].strip())
            except ValueError:
                continue
            role = ownership.get(pid)
            if role is None or memory < 0:
                continue
            key = (role, gpu)
            snapshot[key] = snapshot.get(key, 0.0) + memory
        for key, value in snapshot.items():
            self.peaks[key] = max(self.peaks.get(key, 0), value)
        # Sum only allocations observed in the SAME sample, not separate GPU peaks.
        total = sum(snapshot.values())
        self.total_peak = total if self.total_peak is None else max(self.total_peak, total)
        self.samples += 1

    def _sample(self):
        with self._lock:
            roots = dict(self.roots)
            root_starttimes = dict(self.root_starttimes)
        before = process_snapshot()
        parents = {pid: identity[0] for pid, identity in before.items()}
        valid_roots = {pid: role for pid, role in roots.items()
                       if pid in before and before[pid][1] == root_starttimes[pid]}
        # Orphaned workers remain owned only while the SAME process exists.
        # Existence alone is insufficient because Linux can reuse their PIDs.
        valid_roots.update({pid: role for pid, role in self.known.items()
                            if pid not in valid_roots and pid in before
                            and before[pid][1] == self.known_starttimes.get(pid)})
        ownership = descendants(valid_roots, parents)
        self.known = ownership
        self.known_starttimes = {pid: before[pid][1] for pid in ownership}
        try:
            p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if p.returncode:
                self.failed_samples += 1
                return
            after = process_snapshot()
            # A PID can disappear/reappear while nvidia-smi runs. Exclude it
            # unless kernel starttime agrees on both sides of the query.
            stable = {pid: role for pid, role in ownership.items()
                      if pid in after and after[pid][1] == before[pid][1]}
            self._observe(p.stdout, stable)
        except (OSError, subprocess.TimeoutExpired):
            self.failed_samples += 1

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8)
        result = {
            'status': 'measured' if self.samples else 'unavailable',
            'scope': 'owned_runner_and_mllm_process_trees_only',
            'process_identity': 'pid_and_proc_stat_starttime_checked_before_and_after_sampling',
            'measurement': 'driver_allocated_process_memory_MiB',
            'sampling_interval_seconds': self.interval,
            'samples': self.samples, 'failed_samples': self.failed_samples,
            'peak_interpretation': 'sampled_lower_bound_not_exact_peak; separate from PyTorch allocator',
            'owned_simultaneous_total_peak_mib': self.total_peak,
            'per_role_per_gpu_sampled_peak_mib': [
                {'role': role, 'gpu_uuid': gpu, 'peak_mib': value}
                for (role, gpu), value in sorted(self.peaks.items())],
            'wall_seconds': time.time() - self.started if self.started else None,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(self.output_path)
        return result
