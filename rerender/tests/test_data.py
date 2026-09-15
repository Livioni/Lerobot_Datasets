import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from rerender.__main__ import check_outputs, select_frames
from rerender._data import load_episode, load_target, sha256


URDF = '''<robot name="test">
<link name="base"/><link name="arm"/><link name="finger"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="grip" type="prismatic"><parent link="arm"/><child link="finger"/>
<axis xyz="0 1 0"/><limit lower="0" upper="0.05" effort="1" velocity="1"/></joint>
</robot>'''


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episode = self.root / 'episode'
        self.ik = self.episode / 'TCP_prediction_ik/test'
        self.assets = self.root / 'embodiments'
        for folder in ['images/third_views', 'intrinsics', 'extrinsics', 'TCP_prediction_ik/test']:
            (self.episode / folder).mkdir(parents=True, exist_ok=True)
        (self.assets / 'test').mkdir(parents=True)
        self.urdf = self.assets / 'test/robot.urdf'
        self.urdf.write_text(URDF)
        self.meta = dict(format_version='robotwin_4d_v1', embodiment='aloha_agilex', num_frames=2,
                         image_height=3, image_width=4, extrinsics={'transform': 'world_to_camera'},
                         depth={'unit': 'millimeter', 'invalid_value': 0})
        self.write_meta()
        for i in range(2):
            Image.fromarray(np.full((3, 4, 3), 70, np.uint8)).save(
                self.episode / 'images/third_views' / f'{i:06d}.png')
        np.save(self.episode / 'intrinsics/third_views.npy', np.array([[2, 0, 2], [0, 2, 1.5], [0, 0, 1.]]))
        np.save(self.episode / 'extrinsics/third_views.npy', np.eye(4)[:3])
        np.save(self.ik / 'robot_state.npy', np.array([[.1, 0, -.1, 1], [.2, 1, -.2, 0]], np.float32))
        self.target = dict(
            format='robotwin_closed_tcp_trajectory_v3', schema_version=3, status='success',
            embodiment='test', frame_count=2, camera='third_views',
            inputs=dict(urdf=str(self.urdf), urdf_sha256=sha256(self.urdf),
                        extrinsics_semantics='world_to_camera',
                        extrinsics_sha256=sha256(self.episode / 'extrinsics/third_views.npy')),
            robot_config_snapshot=dict(urdf_path='robot.urdf', dual_arm=False,
                                       gripper_scale=[0, .05],
                                       gripper_name=[{'base': 'grip', 'mimic': []}] * 2),
            output=dict(shape=[2, 4], state_file='robot_state.npy', role='solution'),
            arms={side: dict(active_joint_names=['joint'], joint_columns=[i * 2],
                             gripper_column=i * 2 + 1, world_from_root=np.eye(4).tolist())
                  for i, side in enumerate(['left', 'right'])})
        self.write_target()

    def write_meta(self):
        (self.episode / 'metadata.json').write_text(json.dumps(self.meta))

    def write_target(self):
        (self.ik / 'metadata.json').write_text(json.dumps(self.target))

    def load(self):
        return load_target(self.ik, load_episode(self.episode, False), self.assets)

    def test_success_and_gripper_mapping(self):
        robot = self.load()
        self.assertAlmostEqual(robot.values('left', 0)['joint'], .1)
        self.assertEqual(robot.values('left', 0)['grip'], 0)
        self.assertEqual(robot.values('right', 0)['grip'], .05)

    def test_v2_metadata(self):
        self.target.update(format='robotwin_closed_tcp_ik_v2', schema_version=2)
        self.target.pop('status')
        diagnostics = {'frames': [{'left': {'success': True}, 'right': {'success': True}}] * 2}
        (self.ik / 'diagnostics.json').write_text(json.dumps(diagnostics))
        self.write_target()
        self.assertEqual(self.load().state.shape, (2, 4))

    def test_no_depth_does_not_require_depth_files(self):
        episode = load_episode(self.episode, False)
        self.assertIsNone(episode.read_frame(0)[1])
        with self.assertRaisesRegex(ValueError, 'PNG frames'):
            load_episode(self.episode, True)

    def test_failed_candidate_is_rejected(self):
        self.target['status'] = 'failed'
        self.write_target()
        with self.assertRaisesRegex(ValueError, 'Failed IK'):
            self.load()

    def test_success_cannot_point_to_candidate(self):
        self.target['output']['state_file'] = 'robot_state_candidate.npy'
        self.write_target()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.load()

    def test_model_and_extrinsic_hashes(self):
        for field in ['urdf_sha256', 'extrinsics_sha256']:
            with self.subTest(field=field):
                original = self.target['inputs'][field]
                self.target['inputs'][field] = 'wrong'
                self.write_target()
                with self.assertRaisesRegex(ValueError, 'hash|extrinsics'):
                    self.load()
                self.target['inputs'][field] = original

    def test_column_overlap_and_nonfinite_state(self):
        self.target['arms']['right']['joint_columns'] = [0]
        self.write_target()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            self.load()
        self.target['arms']['right']['joint_columns'] = [2]
        self.write_target()
        state = np.load(self.ik / 'robot_state.npy')
        state[0, 0] = np.nan
        np.save(self.ik / 'robot_state.npy', state)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.load()

    def test_frame_number_mismatch(self):
        folder = self.episode / 'images/third_views'
        (folder / '000001.png').rename(folder / '000002.png')
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            load_episode(self.episode, False)

    def test_calibration_shapes(self):
        for rows, per_frame in [(3, False), (4, False), (3, True), (4, True)]:
            e = np.eye(4)[:rows]
            if per_frame:
                e = np.stack([e, e])
            np.save(self.episode / 'extrinsics/third_views.npy', e)
            self.assertEqual(load_episode(self.episode, False).extrinsics.shape, (2, 4, 4))
        k = np.repeat(np.eye(3)[None], 2, axis=0)
        np.save(self.episode / 'intrinsics/third_views.npy', k)
        self.assertEqual(load_episode(self.episode, False).intrinsics.shape, (2, 3, 3))

    def test_depth_pairing_units_and_size(self):
        folder = self.episode / 'depths/third_views'
        folder.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((3, 4), 1250, np.uint16)).save(folder / f'{i:06d}.png')
        episode = load_episode(self.episode, True)
        self.assertTrue(np.all(episode.read_frame(0)[1] == 1.25))
        Image.fromarray(np.zeros((2, 4), np.uint16)).save(folder / '000001.png')
        with self.assertRaisesRegex(ValueError, 'depth'):
            episode.read_frame(1)

    def test_output_protection_and_frame_selection(self):
        path = self.root / 'out.png'
        path.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            check_outputs([path], False)
        check_outputs([path], True)
        self.assertEqual(path.read_bytes(), b'original')
        self.assertEqual(select_frames('1,0,1', 2), [0, 1])
        with self.assertRaises(ValueError):
            select_frames('2', 2)

    def test_v2_unverified_or_failed_is_rejected(self):
        self.target.update(format='robotwin_closed_tcp_ik_v2', schema_version=2)
        self.target.pop('status')
        self.write_target()
        with self.assertRaisesRegex(ValueError, 'requires diagnostics'):
            self.load()
        diagnostics = {'frames': [{'left': {'success': True}, 'right': {'success': False}}] * 2}
        (self.ik / 'diagnostics.json').write_text(json.dumps(diagnostics))
        with self.assertRaisesRegex(ValueError, 'Failed'):
            self.load()

    def test_uint8_depth_is_not_mistaken_for_millimeters(self):
        folder = self.episode / 'depths/third_views'
        folder.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((3, 4), 100, np.uint8)).save(folder / f'{i:06d}.png')
        episode = load_episode(self.episode, True)
        with self.assertRaisesRegex(ValueError, 'uint16'):
            episode.read_frame(0)


if __name__ == '__main__':
    unittest.main()
