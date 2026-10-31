import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ics.resource_monitor import Monitor, descendants


class ResourceTests(unittest.TestCase):
    def test_owned_trees_and_simultaneous_peaks(self):
        owned = descendants({10: 'mllm', 20: 'runner'}, {10: 1, 11: 10, 12: 11, 20: 1, 30: 1})
        self.assertEqual(owned[12], 'mllm')
        self.assertNotIn(30, owned)
        with tempfile.TemporaryDirectory() as temp:
            m = Monitor(Path(temp)/'gpu_memory.json')
            m._observe('12, GPU-A, 100\n20, GPU-B, 20\n30, GPU-A, 9000\n', owned)
            m._observe('12, GPU-A, 10\n20, GPU-B, 200\n', owned)
            result = m.stop()
            self.assertEqual(result['owned_simultaneous_total_peak_mib'], 210)
            self.assertEqual(sum(r['peak_mib'] for r in result['per_role_per_gpu_sampled_peak_mib']), 300)

    def test_unavailable_is_not_zero_peak(self):
        with tempfile.TemporaryDirectory() as temp:
            result = Monitor(Path(temp)/'gpu_memory.json').stop()
            self.assertIsNone(result['owned_simultaneous_total_peak_mib'])
            self.assertEqual(result['status'], 'unavailable')

    def test_orphan_retained_but_reused_worker_and_root_pids_excluded(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('ics.resource_monitor.process_snapshot') as snapshot, \
             patch('ics.resource_monitor.subprocess.run') as query:
            m = Monitor(Path(temp)/'gpu_memory.json')
            snapshot.return_value = {10: (1, 100), 11: (10, 101)}
            m.add_process(10, 'mllm')
            query.return_value = SimpleNamespace(returncode=0, stdout='11, GPU-A, 100\n')
            m._sample()
            # A genuine worker survives reparenting after its launcher exits.
            snapshot.return_value = {11: (1, 101)}
            query.return_value.stdout = '11, GPU-A, 200\n'
            m._sample()
            self.assertEqual(m.known, {11: 'mllm'})
            # Neither a reused worker PID nor children of a reused root belong
            # to this job, even when the original numeric PIDs remain visible.
            snapshot.return_value = {11: (1, 999), 10: (1, 999), 12: (10, 1000)}
            query.return_value.stdout = '11, GPU-A, 9000\n12, GPU-A, 9000\n'
            m._sample()
            self.assertEqual(m.known, {})
            self.assertEqual(m.stop()['owned_simultaneous_total_peak_mib'], 200)

    def test_pid_reuse_during_driver_query_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('ics.resource_monitor.process_snapshot') as snapshot, \
             patch('ics.resource_monitor.subprocess.run') as query:
            m = Monitor(Path(temp)/'gpu_memory.json')
            snapshot.return_value = {10: (1, 100)}
            m.add_process(10, 'runner')
            snapshot.side_effect = [{10: (1, 100)}, {10: (1, 900)}]
            query.return_value = SimpleNamespace(returncode=0, stdout='10, GPU-A, 9000\n')
            m._sample()
            self.assertEqual(m.stop()['owned_simultaneous_total_peak_mib'], 0)
