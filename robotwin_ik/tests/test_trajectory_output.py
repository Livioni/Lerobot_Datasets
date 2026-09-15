"""Successful artifacts cannot be confused with failed optimization candidates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from robotwin_ik._embodiments import load_embodiment, sha256
from robotwin_ik._io import save_trajectory_output
from robotwin_ik.visualize_ik_rerun import load_replay
from robotwin_ik._common import main


class OutputTests(unittest.TestCase):
    def fixture(self, status):
        robot = load_embodiment('piper')
        state = np.zeros((3, 14), dtype=np.float32)
        meta = {'schema_version': 3, 'format': 'robotwin_closed_tcp_trajectory_v3', 'status': status,
                'embodiment': 'piper', 'frame_count': 3, 'robot_config_snapshot': robot.config,
                'inputs': {'urdf': str(robot.urdf_path), 'urdf_sha256': sha256(robot.urdf_path)},
                'output': {'shape': [3, 14]},
                'arms': {'left': {'gripper_column': 6}, 'right': {'gripper_column': 13}}}
        return state, meta

    def test_failed_candidate_explicit_replay_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state, meta = self.fixture('success')
            save_trajectory_output(path, state, meta, {'schema_version': 3})
            self.assertTrue((path/'robot_state.npy').is_file())
            loaded, q, _ = load_replay(path)
            np.testing.assert_array_equal(q, state)
            self.assertEqual(loaded['output']['role'], 'solution')
            state, meta = self.fixture('failed')
            save_trajectory_output(path, state, meta, {'schema_version': 3})
            self.assertFalse((path/'robot_state.npy').exists())
            self.assertTrue((path/'robot_state_candidate.npy').is_file())
            with self.assertRaisesRegex(ValueError, 'show-candidate'):
                load_replay(path)
            loaded, q, _ = load_replay(path, show_candidate=True)
            self.assertEqual(loaded['output']['role'], 'failed_candidate')
            np.testing.assert_array_equal(q, state)
            meta['status'] = 'success'
            save_trajectory_output(path, state, meta, {'schema_version': 3})
            self.assertFalse((path/'robot_state_candidate.npy').exists())
            self.assertTrue((path/'robot_state.npy').exists())

    def test_v2_results_remain_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state, meta = self.fixture('success')
            save_trajectory_output(path, state, meta, {})
            meta.update(schema_version=2, format='robotwin_closed_tcp_ik_v2')
            (path/'metadata.json').write_text(json.dumps(meta))
            _, actual, _ = load_replay(path)
            np.testing.assert_array_equal(actual, state)

    def test_cli_failure_publishes_candidate_and_exit_one(self):
        episode = Path(__file__).resolve().parents[2] / '4d_datasets/beat_block_hammer/episode_0000000'
        if not (episode/'tcp_episode.json').exists():
            self.skipTest('Local episode fixture unavailable')
        def solve(runtime, config, geometry, model, targets, ee, initial, times, *args):
            state = np.tile(initial, (len(times), 1)).astype(np.float32)
            frames = [{'success': False, 'failure_reasons': ['tcp_position_tolerance'], 'joint_step_max_abs_rad': 0.} for _ in times]
            return state, frames, {'success': 0, 'failure': len(times), 'trajectory_success': False}
        from unittest.mock import MagicMock
        runtime = MagicMock()
        runtime.torch.__version__ = 'test'; runtime.torch.version.cuda = 'test'
        runtime.torch.cuda.get_device_name.return_value = 'test'
        with tempfile.TemporaryDirectory() as tmp, patch('robotwin_ik._common.CuroboRuntime', return_value=runtime), patch(
                'robotwin_ik._common.solve_arm', side_effect=solve), patch('robotwin_ik._common.package_version', return_value='test'):
            with self.assertRaises(SystemExit) as result:
                main('piper', [str(episode), '--output-dir', tmp])
            self.assertEqual(result.exception.code, 1)
            self.assertFalse((Path(tmp)/'robot_state.npy').exists())
            self.assertEqual(json.loads((Path(tmp)/'metadata.json').read_text())['status'], 'failed')

    def test_environment_error_does_not_fabricate_a_candidate(self):
        episode = Path(__file__).resolve().parents[2] / '4d_datasets/beat_block_hammer/episode_0000000'
        if not (episode/'tcp_episode.json').exists():
            self.skipTest('Local episode fixture unavailable')
        with tempfile.TemporaryDirectory() as tmp, patch('robotwin_ik._common.CuroboRuntime', side_effect=RuntimeError('CUDA unavailable')):
            with self.assertRaises(SystemExit):
                main('piper', [str(episode), '--output-dir', tmp])
            path = Path(tmp)
            self.assertFalse((path/'robot_state.npy').exists())
            self.assertFalse((path/'robot_state_candidate.npy').exists())
            self.assertEqual(json.loads((path/'solver_error.json').read_text())['status'], 'error')



if __name__ == '__main__':
    unittest.main()
