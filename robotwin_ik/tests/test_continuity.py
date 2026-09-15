"""Whole-horizon branch selection, time scaling, and serialization acceptance."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from robotwin_ik._trajectory import select_paths, motion_metrics, smoothness_cost, validate_inputs, diverse_candidates, interpolate_candidate_gaps
from robotwin_ik._solver import evaluate_trajectory
from robotwin_ik._embodiments import load_embodiment
from robotwin_ik._common import parse_args


class TrajectoryTests(unittest.TestCase):
    def test_future_target_changes_first_branch(self):
        layers = [np.array([[0.], [1.]]), np.array([[.1], [1.1]]), np.array([[1.2]])]
        paths = select_paths(layers, np.arange(3), np.array([0.]), .3)
        self.assertEqual(len(paths), 1)
        np.testing.assert_allclose(paths[0][:, 0], [1., 1.1, 1.2])
        short = select_paths(layers[:2], np.arange(2), np.array([0.]), .3)
        np.testing.assert_allclose(short[0][:, 0], [0., .1])

    def test_k_paths_agree_with_exhaustive_search(self):
        import itertools
        layers = [np.array([[0.], [.2]]), np.array([[.1], [.4]]), np.array([[.3], [.5]])]
        times = np.array([0., .2, .5])
        expected = []
        for indices in itertools.product(range(2), repeat=3):
            q = np.array([layers[i][j] for i, j in enumerate(indices)])
            if np.max(np.abs(np.diff(q, axis=0))) <= .31:
                cost = .01 * q[0, 0]**2 + np.sum(np.diff(q[:, 0])**2/np.diff(times))
                expected.append((cost, q))
        expected.sort(key=lambda item:item[0])
        actual = select_paths(layers, times, np.array([0.]), .31)
        for q, (_, reference) in zip(actual, expected[:4]):
            np.testing.assert_allclose(q, reference)
        self.assertEqual(len(actual), min(4, len(expected)))

    def test_windings_are_not_wrapped_to_hide_a_jump(self):
        layers = [np.array([[3.1]]), np.array([[-3.1]])]
        self.assertEqual(select_paths(layers, np.array([0., 1.]), np.array([3.1]), .5), [])
        relaxed = select_paths(layers, np.array([0., 1.]), np.array([3.1]), .5, relaxed=True)
        self.assertGreater(abs(relaxed[0][1, 0]-relaxed[0][0, 0]), 6)
        self.assertEqual(len(diverse_candidates(np.array([[0.], [2*np.pi]]), np.array([0.]))), 2)

    def test_empty_layer_and_short_trajectories(self):
        self.assertEqual(select_paths([np.empty((0, 1))], np.array([0.]), np.zeros(1), .5), [])
        paths = select_paths([np.array([[1.]])], np.array([0.]), np.zeros(1), .5)
        self.assertEqual(len(paths), 1)
        for q in (np.zeros((1, 2)), np.zeros((2, 2))):
            metrics = motion_metrics(q, np.arange(len(q)))
            self.assertEqual(metrics[2].shape, (0, 2))
            self.assertEqual(smoothness_cost(q, np.arange(len(q)), np.zeros(2)), 0.)

    def test_nonuniform_timestamps_and_smoothness(self):
        times = np.array([0., .1, .3, .7])
        q = (2*times)[:, None]
        _, v, a = motion_metrics(q, times)
        np.testing.assert_allclose(v, 2.)
        np.testing.assert_allclose(a, 0., atol=1e-12)
        smooth = np.array([[0.], [.1], [.2], [.3]])
        rough = np.array([[0.], [.28], [.02], [.3]])
        self.assertLess(smoothness_cost(smooth, times, np.zeros(1)), smoothness_cost(rough, times, np.zeros(1)))

    def test_missing_ik_layers_use_a_time_aligned_global_initialization(self):
        layers = [np.array([[0.], [2.]]), np.empty((0, 1)), np.array([[2.2]])]
        times = np.array([0., .25, 1.])
        guide = interpolate_candidate_gaps(layers, times, np.array([0.]), .5)
        np.testing.assert_allclose(guide[:, 0], [2., 2.05, 2.2])
        self.assertEqual(len(layers[1]), 0)  # Interpolation creates no qualified IK node.

    def test_float32_rounding_is_included_in_hard_step_validation(self):
        robot = load_embodiment('piper')
        q = np.zeros((2, 6)); q[:, 0] = [.1, .6]
        with patch('robotwin_ik._solver.pose_residuals', return_value=(np.zeros(2), np.zeros(2))), patch(
                'robotwin_ik._solver._valid_configs', return_value=np.ones(2, dtype=bool)):
            state, frames, violation = evaluate_trajectory(MagicMock(), MagicMock(), robot.geometries['left'],
                robot.model, q, np.tile(np.eye(4), (2, 1, 1)), np.array([0., .1]), .5)
        np.testing.assert_array_equal(state, q.astype(np.float32).astype(float))
        self.assertIn('joint_step_limit', frames[1]['failure_reasons'])
        self.assertGreater(violation, 0.)

    def test_cli_and_input_contract(self):
        for value in ['0', '-1', 'nan', 'inf']:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args('piper', ['episode', '--max-joint-step-rad', value])
        for value in ['0', '1', '257']:
            with self.assertRaises(SystemExit):
                parse_args('piper', ['episode', '--ik-seeds', value])
        with self.assertRaises(SystemExit):
            parse_args('piper', ['episode', '--fallback-seeds', '32'])
        args = parse_args('piper', ['episode'])
        self.assertEqual((args.ik_seeds, args.max_joint_step_rad), (64, .5))
        for times in (np.array([]), np.array([0., 0.]), np.array([1., 0.]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                validate_inputs(np.zeros(1), np.tile(np.eye(4), (len(times), 1, 1)), times, 64, .5)


if __name__ == '__main__':
    unittest.main()
