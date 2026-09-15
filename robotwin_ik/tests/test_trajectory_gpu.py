"""GPU integration across every embodiment, plus impossible-trajectory reporting."""
import os
import unittest
import numpy as np
from robotwin_ik._embodiments import NAMES, load_embodiment
from robotwin_ik._kinematics import fk_tcp_batch, fk_link
from robotwin_ik._initial_state import load_initial_state
from robotwin_ik._common import parse_args
from robotwin_ik._aloha import load_aloha


def embodiment(name):
    robot = load_aloha(parse_args(name, ['unused'])) if name == 'aloha-agilex' else load_embodiment(name)
    config = robot.runtime_configs['left'] if name == 'aloha-agilex' else robot.runtime
    initial, _ = load_initial_state(name, robot.model, {s:g.canonical_joint_names for s,g in robot.geometries.items()})
    return robot, config, initial['left']


class IndependentBatchFkTests(unittest.TestCase):
    def test_batch_matches_scalar_for_five_embodiments(self):
        for name in (*NAMES, 'aloha-agilex'):
            robot, _, home = embodiment(name)
            g = robot.geometries['left']
            q = np.tile(home, (4, 1)); q[:, 0] += np.arange(4)*.02
            actual = fk_tcp_batch(robot.model, g, q)
            expected = np.array([fk_link(robot.model, g.ee_link, dict(zip(g.canonical_joint_names, row))) @ g.link6_from_tcp for row in q])
            np.testing.assert_allclose(actual, expected, atol=1e-12)


@unittest.skipUnless(os.environ.get('ROBOTWIN_IK_GPU_TESTS') == '1', 'Set ROBOTWIN_IK_GPU_TESTS=1 for CUDA tests')
class GpuTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from robotwin_ik._solver import CuroboRuntime
        cls.runtime = CuroboRuntime('cuda:0')

    def test_all_five_whole_trajectory_roundtrips(self):
        from robotwin_ik._solver import solve_arm
        from robotwin_ik._trajectory_opt import curobo_rotation_matrix
        runtime = self.runtime
        for name in (*NAMES, 'aloha-agilex'):
            with self.subTest(name=name):
                robot, config, initial = embodiment(name)
                g = robot.geometries['left']
                q = np.tile(initial, (3, 1)); q[:, 0] += [0., .015, .025]
                targets = fk_tcp_batch(robot.model, g, q)
                ee = np.linalg.inv(g.footprint_from_base) @ targets @ np.linalg.inv(g.link6_from_tcp)
                state, frames, summary = solve_arm(runtime, config, g, robot.model, targets, ee, initial,
                                                    np.array([0., .1, .25]), ik_seeds=4)
                self.assertTrue(summary['trajectory_success'], frames)
                self.assertTrue(all(f['success'] for f in frames))
                self.assertLessEqual(np.abs(np.diff(state.astype(float), axis=0)).max(), .5)
                self.assertTrue(all(r['numerical_error'] is None for r in summary['optimization']), summary)
                for result in summary['optimization']:
                    self.assertLessEqual(result['selected_smoothness_cost'], result['initial_smoothness_cost']+1e-9)
                # Independently verify the CUDA TCP position gradient used by refinement.
                engine = runtime.create_solver(config.robot_cfg, 1)
                order = [g.canonical_joint_names.index(n) for n in engine.joint_names]
                q_grad = runtime.torch.tensor(q[1:2], device=runtime.device, dtype=runtime.torch.float32, requires_grad=True)
                fk = engine.fk(q_grad[:, order].contiguous())
                rotation = curobo_rotation_matrix(fk.ee_quaternion)
                offset = runtime.torch.tensor(g.link6_from_tcp[:3, 3], device=runtime.device, dtype=runtime.torch.float32)
                tcp = fk.ee_position + rotation @ offset
                weight = runtime.torch.tensor([.7, -.3, .5], device=runtime.device)
                rotation_weight = runtime.torch.tensor([[.2, -.3, .1], [.4, .1, -.2], [-.1, .2, .3]], device=runtime.device)
                tcp_rotation = rotation @ runtime.torch.tensor(g.link6_from_tcp[:3,:3], device=runtime.device, dtype=runtime.torch.float32)
                ((tcp*weight).sum() + (tcp_rotation*rotation_weight).sum()).backward()
                gradient = q_grad.grad.detach().cpu().numpy()[0]
                numeric = []
                for j in range(len(initial)):
                    plus, minus = q[1:2].copy(), q[1:2].copy()
                    plus[0, j] += 1e-3; minus[0, j] -= 1e-3
                    a = np.linalg.inv(g.footprint_from_base) @ fk_tcp_batch(robot.model, g, plus)
                    b = np.linalg.inv(g.footprint_from_base) @ fk_tcp_batch(robot.model, g, minus)
                    position_difference = ((a[0,:3,3]-b[0,:3,3])*[.7,-.3,.5]).sum()
                    rotation_difference = ((a[0,:3,:3]-b[0,:3,:3])*rotation_weight.cpu().numpy()).sum()
                    numeric.append((position_difference+rotation_difference)/2e-3)
                np.testing.assert_allclose(gradient, numeric, atol=1e-3, rtol=1e-2)
                del engine
                runtime.torch.cuda.empty_cache()

    def test_unreachable_target_is_failed_candidate(self):
        from robotwin_ik._solver import solve_arm
        robot, config, initial = embodiment('piper'); g = robot.geometries['left']
        targets = fk_tcp_batch(robot.model, g, np.tile(initial, (3, 1)))
        targets[1, :3, 3] = [10., 10., 10.]
        ee = np.linalg.inv(g.footprint_from_base) @ targets @ np.linalg.inv(g.link6_from_tcp)
        state, frames, summary = solve_arm(self.runtime, config, g, robot.model, targets, ee, initial,
                                            np.array([0., .1, .2]), ik_seeds=64)
        self.assertFalse(summary['trajectory_success'])
        self.assertIn('tcp_position_tolerance', frames[1]['failure_reasons'])
        self.assertTrue(np.isfinite(state).all())
        self.assertNotIn('hold_previous_after_failure', frames[1])


if __name__ == '__main__':
    unittest.main()
