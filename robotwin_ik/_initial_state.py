"""Embedded recording-derived initial states, with an optional HDF5 override."""
from pathlib import Path
import numpy as np
from ._embodiments import sha256

INITIAL_SEED_SOURCE = 'target_embodiment_example_frame_0'

BUILTIN_SEED_SOURCE = 'builtin_target_example_frame_0'


def _both_arms(names, values, right_names=None):
    return {'left': dict(zip(names, values)),
            'right': dict(zip(right_names or names, values))}


# Exact float64 values read from joint_action/*_arm[0] of the five supplied
# recordings (RoboTwin stores float32 drive targets as float64 in HDF5).
# These constants do not read config homestate or require the recordings.
BUILTIN_INITIAL_JOINTS = {
    'franka-panda': _both_arms(
        tuple(f'panda_joint{i}' for i in range(1, 8)),
        (0., 0.19634954631328583, 0., -2.6179938316345215, 0.,
         2.9415926933288574, 0.7853981852531433)),
    'ARX-X5': _both_arms(tuple(f'joint{i}' for i in range(1, 7)), (0.,) * 6),
    'piper': _both_arms(tuple(f'joint{i}' for i in range(1, 7)), (0.,) * 6),
    'ur5-wsg': _both_arms(
        ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'),
        (-1.544700026512146, -1.544700026512146, -1.544700026512146,
         -1.5793999433517456, 1.5793999433517456, 0.)),
    'aloha-agilex': _both_arms(
        tuple(f'fl_joint{i}' for i in range(1, 7)), (0.,) * 6,
        tuple(f'fr_joint{i}' for i in range(1, 7))),
}


def load_initial_state(embodiment, model, joint_names, hdf5_path=None):
    if hdf5_path is not None:
        return load_example_initial_state(hdf5_path, model, joint_names)
    if embodiment not in BUILTIN_INITIAL_JOINTS:
        raise ValueError(f'No built-in initial state for {embodiment}')
    arms = {}
    for side in ('left', 'right'):
        named_values = BUILTIN_INITIAL_JOINTS[embodiment][side]
        names = joint_names[side]
        if len(names) != len(named_values) or set(names) != set(named_values):
            raise ValueError(f'{embodiment}: {side} joint names disagree with built-in state')
        q = np.array([named_values[name] for name in names], dtype=np.float64)
        _validate_joint_values(q, model, names, f'built-in {embodiment}/{side}')
        arms[side] = q
    vector = np.concatenate([arms['left'], [1.], arms['right'], [1.]])
    return arms, {
        'source': BUILTIN_SEED_SOURCE, 'path': None, 'sha256': None,
        'original_example': f'{embodiment}.hdf5', 'frame_index': 0,
        'dataset': 'joint_action/vector',
        'joint_names': {s: list(joint_names[s]) for s in arms},
        'joint_order_basis': 'built-in named joints reordered to active_joint_names',
        'state_vector': vector.tolist(), 'gripper_open': {'left': 1., 'right': 1.},
    }


def _validate_joint_values(q, model, names, source):
    if not np.isfinite(q).all():
        raise ValueError(f'{source}: nonfinite initial state')
    for name, value in zip(names, q):
        joint = model.joints_by_name[name]
        if ((joint.lower is not None and value < joint.lower - 1e-6) or
                (joint.upper is not None and value > joint.upper + 1e-6)):
            raise ValueError(f'{source}: initial {name}={value} outside URDF limits')


def load_example_initial_state(path, model, joint_names):
    """The RoboTwin schema orders each arm by config.yml arm_joints_name.

    Read arm datasets rather than guessing offsets in a 14/16 column vector;
    cross-check the recorded vector to catch inconsistent layouts.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('Reading example initial states requires h5py in the solver environment') from exc
    path = Path(path).expanduser().resolve()
    arms, grippers = {}, {}
    with h5py.File(path, 'r') as recording:
        lengths = []
        for side in ('left', 'right'):
            names = joint_names[side]
            key = f'joint_action/{side}_arm'
            if key not in recording:
                raise ValueError(f'{path}: missing {key}')
            dataset = recording[key]
            if dataset.ndim != 2 or dataset.shape[0] == 0 or dataset.shape[1] != len(names):
                raise ValueError(f'{path}: {key} must have shape [T,{len(names)}], T > 0')
            q = np.asarray(dataset[0], dtype=np.float64)
            _validate_joint_values(q, model, names, str(path))
            grip_key = f'joint_action/{side}_gripper'
            if grip_key not in recording or recording[grip_key].shape != (dataset.shape[0],):
                raise ValueError(f'{path}: invalid {grip_key}')
            grip = float(recording[grip_key][0])
            if not np.isfinite(grip) or not 0 <= grip <= 1:
                raise ValueError(f'{path}: invalid normalized initial gripper value')
            arms[side], grippers[side] = q, grip
            lengths.append(dataset.shape[0])
        vector = np.concatenate([arms['left'], [grippers['left']], arms['right'], [grippers['right']]])
        key = 'joint_action/vector'
        if (lengths[0] != lengths[1] or key not in recording or
                recording[key].shape != (lengths[0], len(vector))):
            raise ValueError(f'{path}: invalid joint_action/vector shape or inconsistent frame counts')
        if not np.allclose(recording[key][0], vector, rtol=0, atol=1e-7):
            raise ValueError(f'{path}: first state vector disagrees with left/right arm datasets')
    provenance = {
        'source': INITIAL_SEED_SOURCE, 'path': str(path), 'sha256': sha256(path),
        'frame_index': 0, 'dataset': 'joint_action/vector',
        'arm_datasets': {s: f'joint_action/{s}_arm' for s in arms},
        'joint_names': {s: list(joint_names[s]) for s in arms},
        'joint_order_basis': 'RoboTwin joint_action arm order = config.yml arm_joints_name',
        'state_vector': vector.tolist(), 'gripper_open': grippers,
    }
    return arms, provenance
