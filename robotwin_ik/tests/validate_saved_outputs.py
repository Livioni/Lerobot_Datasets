"""Independently verify saved v2 results and v3 solutions/failed candidates.

python -m robotwin_ik.tests.validate_saved_outputs <output-directory-or-episode> [...]
"""
import json
from pathlib import Path
import sys
import numpy as np
from robotwin_ik.visualize_ik_rerun import load_replay, replay_positions
from robotwin_ik._io import load_prediction, load_extrinsics
from robotwin_ik._solver import rotation_error_radians
from robotwin_ik._trajectory import motion_metrics


def validate_directory(directory):
    raw = json.loads((directory/'metadata.json').read_text())
    if raw.get('schema_version') not in (2, 3):
        return None
    v3 = raw['schema_version'] == 3
    failed = v3 and raw['status'] == 'failed'
    meta, state, model = load_replay(directory, show_candidate=failed)
    prediction = load_prediction(Path(meta['inputs']['prediction_json']))
    np.testing.assert_array_equal(meta['timestamps_seconds'], prediction.timestamps)
    extrinsics = load_extrinsics(Path(meta['inputs']['extrinsics']), len(state))
    actual = replay_positions(meta, state, model)
    diagnostic = json.loads((directory/'diagnostics.json').read_text())
    assert len(diagnostic['frames']) == len(state)
    successes, max_pos, max_rot, max_step = 0, 0., 0., 0.
    per_arm = {}
    for side, arm in meta['arms'].items():
        targets = np.linalg.inv(extrinsics) @ prediction.camera_tcp[side]
        q = state[:, arm['joint_columns']].astype(float)
        steps, velocity, acceleration = motion_metrics(q, prediction.timestamps)
        np.testing.assert_array_equal(state[:, arm['gripper_column']], prediction.gripper_open[side])
        bounds = np.ones(len(state), dtype=bool)
        for column, name in zip(arm['joint_columns'], arm['active_joint_names']):
            joint = model.joints_by_name[name]
            if joint.lower is not None: bounds &= state[:, column].astype(float) >= joint.lower
            if joint.upper is not None: bounds &= state[:, column].astype(float) <= joint.upper
        arm_successes = 0
        for i, frame in enumerate(diagnostic['frames']):
            d = frame[side]
            pos = float(np.linalg.norm(targets[i, :3, 3]-actual[side][i, :3, 3]))
            rot = float(np.degrees(rotation_error_radians(targets[i], actual[side][i])))
            assert abs(pos-d['position_residual_m']) < 2e-6, (directory, side, i, pos)
            assert abs(rot-d['rotation_residual_deg']) < 2e-4, (directory, side, i, rot)
            step = float(np.max(np.abs(steps[i-1]))) if i else None
            if v3:
                assert d['joint_limits_valid'] == bool(bounds[i])
                if i:
                    assert abs(step-d['joint_step_max_abs_rad']) < 1e-12
                    np.testing.assert_allclose(d['joint_velocity_rad_s'], velocity[i-1], atol=1e-10)
                if 0 < i < len(state)-1:
                    np.testing.assert_allclose(d['joint_acceleration_rad_s2'], acceleration[i-1], atol=1e-9)
                assert d['success'] == (len(d['failure_reasons']) == 0)
                if step is not None and step > meta['solver']['max_joint_step_rad']:
                    assert 'joint_step_limit' in d['failure_reasons']
                if pos > .003 + 1e-9: assert 'tcp_position_tolerance' in d['failure_reasons']
                if rot > 2 + 1e-6: assert 'tcp_rotation_tolerance' in d['failure_reasons']
            if d['success']:
                successes += 1; arm_successes += 1
                max_pos = max(max_pos, pos); max_rot = max(max_rot, rot)
                assert pos <= .003002 and rot <= 2.0002
                assert bounds[i]
                if v3 and i: assert step <= meta['solver']['max_joint_step_rad']
            if i: max_step = max(max_step, step)
        per_arm[side] = {'successful_frames': arm_successes, 'total_frames': len(state),
                         'max_joint_step_rad': float(np.max(np.abs(steps))) if len(steps) else 0.}
    if v3:
        assert (meta['status'] == 'success') == (successes == 2*len(state))
        stale = 'robot_state.npy' if failed else 'robot_state_candidate.npy'
        assert not (directory/stale).exists(), f'Stale conflicting output: {directory/stale}'
    return {'directory': str(directory), 'embodiment': meta['embodiment'], 'status': meta['status'],
            'shape': list(state.shape), 'successful_arm_frames': successes, 'total_arm_frames': 2*len(state),
            'max_success_position_error_mm': max_pos*1000, 'max_success_rotation_error_deg': max_rot,
            'max_joint_step_rad': max_step, 'arms': per_arm,
            'wall_time_seconds': meta['total_wall_time_seconds']}


def validate(path):
    directories = [path] if (path/'metadata.json').exists() and (path/'diagnostics.json').exists() else sorted({p.parent for p in path.rglob('metadata.json') if (p.parent/'diagnostics.json').exists()})
    return [row for d in directories if (row := validate_directory(d)) is not None]


if __name__ == '__main__':
    print(json.dumps([row for path in sys.argv[1:] for row in validate(Path(path))], indent=2))
