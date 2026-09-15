"""Validate real recordings and reject ambiguous/corrupt initial states."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import h5py
from robotwin_ik._embodiments import NAMES, load_embodiment
from robotwin_ik._initial_state import BUILTIN_SEED_SOURCE, load_initial_state, load_example_initial_state
from robotwin_ik._aloha import DEFAULT_URDF, arm_geometry, load_runtime_robot_config, DEFAULT_LEFT_CONFIG, DEFAULT_RIGHT_CONFIG
import convert_robotwin_tcp as tc


class InitialStateTests(unittest.TestCase):
    def test_all_five_builtin_states_without_recordings(self):
        for name in (*NAMES, 'aloha-agilex'):
            with self.subTest(name=name):
                if name == 'aloha-agilex':
                    model = tc.load_robot_model(str(DEFAULT_URDF))
                    names = {}
                    for side, config in [('left', DEFAULT_LEFT_CONFIG), ('right', DEFAULT_RIGHT_CONFIG)]:
                        runtime = load_runtime_robot_config(config, DEFAULT_URDF)
                        names[side] = arm_geometry(model, side, runtime.base_link, runtime.ee_link).canonical_joint_names
                else:
                    robot = load_embodiment(name); model = robot.model
                    names = {s:g.canonical_joint_names for s,g in robot.geometries.items()}
                with patch('robotwin_ik._initial_state.load_example_initial_state', side_effect=AssertionError('Must not read HDF5')):
                    arms, provenance = load_initial_state(name, model, names)
                expected = np.zeros(7 if name == 'franka-panda' else 6)
                if name == 'franka-panda':
                    expected = np.asarray([0, .19634954084936207, 0, -2.617993877991494, 0,
                                           2.941592653589793, .7853981633974483], dtype=np.float32).astype(float)
                elif name == 'ur5-wsg':
                    expected = np.asarray([-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0], dtype=np.float32).astype(float)
                for side in names:
                    np.testing.assert_array_equal(arms[side], expected)
                self.assertEqual(provenance['source'], BUILTIN_SEED_SOURCE)
                self.assertIsNone(provenance['path'])
                # Bind values by name, not by an assumed iteration order.
                reversed_names = {s:tuple(reversed(n)) for s,n in names.items()}
                reordered, _ = load_initial_state(name, model, reversed_names)
                np.testing.assert_array_equal(reordered['left'], expected[::-1])
                arms['left'][:] = 42
                fresh, _ = load_initial_state(name, model, names)
                np.testing.assert_array_equal(fresh['left'], expected)
                self.assertEqual(provenance['frame_index'], 0)
                self.assertEqual(len(provenance['state_vector']), 16 if name == 'franka-panda' else 14)

    def test_reads_only_first_state_and_validates_layout(self):
        robot = load_embodiment('piper')
        names = {s:g.canonical_joint_names for s,g in robot.geometries.items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'initial.hdf5'
            with h5py.File(path,'w') as f:
                for side in names:
                    # Future values must not be used as seeds or validated.
                    f[f'joint_action/{side}_arm'] = np.array([[0.]*6,[np.nan]*6])
                    f[f'joint_action/{side}_gripper'] = [1.,np.nan]
                f['joint_action/vector'] = np.array([[0.]*6+[1.]+[0.]*6+[1.], [np.nan]*14])
            arms, provenance = load_initial_state('piper', robot.model, names, path)
            self.assertEqual(provenance['source'], 'target_embodiment_example_frame_0')
            np.testing.assert_array_equal(arms['left'], np.zeros(6))
            with h5py.File(path,'r+') as f:
                f['joint_action/vector'][0,0] = .5
            with self.assertRaisesRegex(ValueError,'disagrees'):
                load_example_initial_state(path, robot.model, names)
            with h5py.File(path,'r+') as f:
                f['joint_action/left_arm'][0,0] = 100.
            with self.assertRaisesRegex(ValueError,'outside URDF limits'):
                load_example_initial_state(path, robot.model, names)
            with self.assertRaises(OSError):
                load_initial_state('piper', robot.model, names, Path(tmp)/'missing.hdf5')


if __name__ == '__main__':
    unittest.main()
