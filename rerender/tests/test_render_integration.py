"""Opt-in real GPU/URDF checks: RERENDER_GPU_TESTS=1 python -m unittest ..."""
import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from rerender._composite import compose
from rerender._data import ROOT, Robot, load_episode, load_source, load_target, validate_robot
from rerender._renderer import RobotRenderer
from robotwin_ik._kinematics import fk_link


@unittest.skipUnless(os.environ.get('RERENDER_GPU_TESTS') == '1', 'requires SAPIEN GPU and local assets')
class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get('RERENDER_TEST_EPISODE', ROOT / '4d_datasets/place_dual_shoes/episode_0000092'))
        cls.episode = load_episode(path, True)
        cls.source = load_source(cls.episode)

    def assert_link_poses(self, renderer, robot, frame):
        groups = [('left', 'right')] if robot.config.get('dual_arm', False) else [('left',), ('right',)]
        for (articulation, _), sides in zip(renderer.instances, groups):
            values = {name: 0.0 for name in robot.model.joints_by_name}
            for side in sides:
                values.update(robot.locks.get(side, {}))
            for side in sides:
                values.update(robot.values(side, frame, include_locks=False))
            root = np.asarray(robot.arms[sides[0]]['world_from_root'])
            # Independent NumPy FK checks all link origins, including fingers
            # whose prismatic axes live in rotated joint frames.
            for link in articulation.get_links():
                expected = root @ fk_link(robot.model, link.name, values)
                np.testing.assert_allclose(link.entity_pose.to_transformation_matrix(), expected, atol=2e-5,
                                           err_msg=f'{robot.name}/{sides}/{link.name}')

    def test_real_source_and_target_outputs(self):
        episode = self.episode
        with RobotRenderer(self.source, episode.height, episode.width) as old_renderer:
            for slug in ['ur5_wsg', 'arx_x5']:
                target = load_target(episode.path / 'TCP_prediction_ik' / slug, episode)
                with RobotRenderer(target, episode.height, episode.width) as new_renderer:
                    for frame in [0, min(120, episode.count - 1), episode.count - 1]:
                        with self.subTest(target=slug, frame=frame):
                            rgb, depth = episode.read_frame(frame)
                            old = old_renderer.render(frame, episode.intrinsics[frame], episode.extrinsics[frame])
                            new = new_renderer.render(frame, episode.intrinsics[frame], episode.extrinsics[frame])
                            self.assert_link_poses(old_renderer, self.source, frame)
                            self.assert_link_poses(new_renderer, target, frame)
                            np.testing.assert_allclose(new_renderer.camera.get_extrinsic_matrix(),
                                                       episode.extrinsics[frame, :3], atol=1e-6)
                            visible = old.depth > 0
                            self.assertGreater(np.mean(np.abs(old.depth[visible] - depth[visible]) < .005), .5)
                            self.assertGreater(np.count_nonzero(new.depth), 100)
                            for mode, d in [('no_depth', None), ('with_depth', depth)]:
                                result = compose(rgb, old.depth, new.rgb, new.depth, d)
                                untouched = ~(result.erased | result.drawn)
                                np.testing.assert_array_equal(result.rgb[untouched], rgb[untouched])
                                np.testing.assert_array_equal(result.rgb[result.erased & ~result.drawn], 0)
                                output = episode.path / 'rerender_images/third_views' / slug / mode / episode.rgb_paths[frame].name
                                if output.is_file():
                                    with Image.open(output) as image:
                                        np.testing.assert_array_equal(np.array(image), result.rgb)

    def test_camera_updates_each_frame(self):
        episode = self.episode
        with RobotRenderer(self.source, episode.height, episode.width) as renderer:
            k = episode.intrinsics[0].copy()
            e = episode.extrinsics[0].copy()
            original = renderer.render(0, k, e)
            k[0, 0] *= 1.1
            k[0, 2] += 4
            e[0, 3] += .05
            moved = renderer.render(0, k, e)
            np.testing.assert_allclose(renderer.camera.get_extrinsic_matrix(), e[:3], atol=1e-6)
            np.testing.assert_allclose(renderer.camera.get_intrinsic_matrix(), k, atol=1e-5)
            self.assertFalse(np.array_equal(original.depth, moved.depth))

    def test_other_embodiment_geometry_at_home(self):
        # These are constructed home poses, not the episode's failed IK candidates.
        from robotwin_ik._embodiments import load_embodiment
        for name in ['franka-panda', 'piper']:
            with self.subTest(embodiment=name):
                emb = load_embodiment(name)
                state = np.concatenate([emb.homestates['left'], [0], emb.homestates['right'], [1]])[None]
                arms, offset = {}, 0
                for side in ['left', 'right']:
                    names = list(emb.geometries[side].canonical_joint_names)
                    arms[side] = dict(active_joint_names=names, joint_columns=list(range(offset, offset+len(names))),
                                      gripper_column=offset+len(names), world_from_root=emb.world_from_root[side])
                    offset += len(names)+1
                robot = validate_robot(Robot(name, emb.urdf_path, emb.config, state, arms,
                                              {side: emb.runtime.lock_joints for side in arms}, emb.model), 1, True)
                with RobotRenderer(robot, self.episode.height, self.episode.width) as renderer:
                    result = renderer.render(0, self.episode.intrinsics[0], self.episode.extrinsics[0])
                    self.assert_link_poses(renderer, robot, 0)
                    pixels = np.count_nonzero(result.depth)
                    self.assertGreater(pixels, 100)
                    self.assertLess(pixels, self.episode.height*self.episode.width*.8)


if __name__ == '__main__':
    unittest.main()
