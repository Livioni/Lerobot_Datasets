"""Shared whole-trajectory entry point for all five RoboTwin embodiments."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import numpy as np
from ._embodiments import DEFAULT_EMBODIMENTS_ROOT, load_embodiment, sha256, slug
from ._io import (OUTPUT_FILENAMES, POSITION_TOLERANCE_METERS, ROTATION_TOLERANCE_DEGREES,
                  RANDOM_SEED, load_prediction, load_extrinsics, positive_seed_count,
                  atomic_write_json, save_trajectory_output, package_version)
from ._solver import CuroboRuntime, solve_arm
from ._initial_state import load_initial_state


def parse_args(name, argv=None):
    parser = argparse.ArgumentParser(description=f'Whole closed-TCP trajectory solving for dual {name}.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--prediction-json', type=Path)
    parser.add_argument('--initial-state-hdf5', type=Path,
                        help='Optional target recording override; default uses embedded first-frame state')
    assets = parser.add_mutually_exclusive_group()
    assets.add_argument('--embodiments-root', type=Path, default=DEFAULT_EMBODIMENTS_ROOT)
    assets.add_argument('--robotwin-root', type=Path, help='Explicit legacy asset root override')
    if name == 'aloha-agilex':
        parser.add_argument('--urdf', type=Path)
        parser.add_argument('--left-config', type=Path)
        parser.add_argument('--right-config', type=Path)
        parser.set_defaults(embodiment_distance=None, task_config=None)
    else:
        parser.add_argument('--embodiment-distance', type=float)
        parser.add_argument('--task-config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--solver', choices=('trajectory', 'differential'), default='trajectory',
                        help='Whole-trajectory search or sequential cuRobo 2 LM differential IK')
    parser.add_argument('--ik-seeds', type=positive_seed_count, default=64,
                        help='Trajectory: initial seeds (expands to 256); differential: recovery LM seeds')
    parser.add_argument('--max-joint-step-rad', type=float, default=.5,
                        help='Positive hard limit on actual per-joint changes between trajectory frames')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_joint_step_rad) or args.max_joint_step_rad <= 0:
        parser.error('--max-joint-step-rad must be finite and positive')
    return args


def build_targets(robot, prediction, extrinsics):
    """World TCP -> URDF-root TCP -> cuRobo-base EE, for every original sample."""
    world_from_camera = np.linalg.inv(extrinsics)
    world, targets = {}, {}
    for side, geometry in robot.geometries.items():
        world[side] = world_from_camera @ prediction.camera_tcp[side]
        root_tcp = np.linalg.inv(robot.world_from_root[side]) @ world[side]
        base_ee = np.linalg.inv(geometry.footprint_from_base) @ root_tcp @ np.linalg.inv(geometry.link6_from_tcp)
        targets[side] = (root_tcp, base_ee)
    return world, targets


def main(name, argv=None):
    args = parse_args(name, argv)
    started = time.perf_counter()
    episode = args.episode.expanduser().resolve()
    if not episode.is_dir():
        raise SystemExit(f'Episode not found: {episode}')
    prediction_path = (args.prediction_json or episode / 'tcp_episode.json').expanduser().resolve()
    default_output = episode / ('TCP_prediction_differential_ik'
                                if args.solver == 'differential' else 'TCP_prediction_ik')
    if name != 'aloha-agilex':
        default_output /= slug(name)
    output = (args.output_dir or default_output).expanduser().resolve()
    if output == episode:
        raise SystemExit('--output-dir must differ from the source episode directory')
    if any((output / f).exists() for f in OUTPUT_FILENAMES) and not args.overwrite:
        raise SystemExit(f'Outputs already exist: {output}; pass --overwrite')
    try:
        prediction = load_prediction(prediction_path)
        count = len(prediction.timestamps)
        extrinsics_path = episode / 'extrinsics' / f'{prediction.camera}.npy'
        extrinsics = load_extrinsics(extrinsics_path, count)
        if name == 'aloha-agilex':
            from ._aloha import load_aloha
            robot = load_aloha(args)
            configs = robot.runtime_configs
        else:
            robot = load_embodiment(name, args.embodiments_root, args.embodiment_distance, args.task_config,
                                   robotwin_root=args.robotwin_root)
            configs = {s: robot.runtime for s in robot.geometries}
        initial, provenance = load_initial_state(
            name, robot.model, {s: g.canonical_joint_names for s, g in robot.geometries.items()},
            args.initial_state_hdf5)
        _, targets = build_targets(robot, prediction, extrinsics)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    stages = {'input_preparation': time.perf_counter()-started}
    print(f'{name}: {args.solver}, {count} frames, joint step <= {args.max_joint_step_rad:g} rad', flush=True)
    outputs, diagnostics, summaries, arms = {}, {}, {}, {}
    columns, offset = [], 0
    try:
        stamp = time.perf_counter()
        if args.solver == 'differential':
            from ._differential import DifferentialRuntime, solve_arm_differential
            runtime = DifferentialRuntime(args.device)
            arm_solver = solve_arm_differential
        else:
            runtime = CuroboRuntime(args.device)
            arm_solver = solve_arm
        stages['runtime_initialization'] = time.perf_counter()-stamp
        np.random.seed(RANDOM_SEED)
        runtime.torch.manual_seed(RANDOM_SEED)
        runtime.torch.cuda.manual_seed_all(RANDOM_SEED)
        for side, geometry in robot.geometries.items():
            stamp = time.perf_counter()
            outputs[side], diagnostics[side], summaries[side] = arm_solver(
                runtime, configs[side], geometry, robot.model, *targets[side],
                initial[side], prediction.timestamps, args.ik_seeds, args.max_joint_step_rad)
            stages[f'{side}_arm'] = time.perf_counter()-stamp
            dof = len(geometry.canonical_joint_names)
            arms[side] = {'active_joint_names': geometry.canonical_joint_names,
                          'joint_columns': list(range(offset, offset+dof)), 'gripper_column': offset+dof,
                          'base_link': geometry.base_link, 'ee_link': geometry.ee_link,
                          'world_from_root': robot.world_from_root[side],
                          'root_from_base': geometry.footprint_from_base, 'ee_from_tcp': geometry.link6_from_tcp,
                          'initial_joint_state': initial[side]}
            columns.extend([f'{side}/{joint}_rad' for joint in geometry.canonical_joint_names])
            columns.append(f'{side}/gripper_open_binary')
            offset += dof+1
            runtime.torch.cuda.empty_cache()
    except Exception as exc:
        # An environment/solver exception does not produce a fabricated candidate.
        # Keep previously published results coherent and report this attempt separately.
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / 'solver_error.json', {
            'schema_version': 3, 'status': 'error', 'embodiment': name,
            'prediction_json': str(prediction_path), 'error': f'{type(exc).__name__}: {exc}',
            'stage_wall_time_seconds': stages})
        raise SystemExit(f'{type(exc).__name__}: {exc}; see {output / "solver_error.json"}') from exc
    state = np.column_stack([outputs['left'], prediction.gripper_open['left'],
                             outputs['right'], prediction.gripper_open['right']]).astype(np.float32)
    if state.shape != (count, offset) or not np.isfinite(state).all():
        raise RuntimeError('Invalid generated trajectory')
    counts = {s: {k: summaries[s][k] for k in ('success', 'failure')} for s in arms}
    failures = [{'frame_index': i, 'side': s, 'reasons': diagnostics[s][i]['failure_reasons']}
                for i in range(count) for s in arms if not diagnostics[s][i]['success']]
    status = 'success' if all(s['trajectory_success'] for s in summaries.values()) else 'failed'
    elapsed = time.perf_counter()-started
    metadata = {
        'schema_version': 3, 'format': 'robotwin_closed_tcp_trajectory_v3',
        'embodiment': name, 'status': status, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'frame_count': count, 'frame_rate_hz': prediction.frame_rate_hz,
        'timestamps_seconds': prediction.timestamps, 'camera': prediction.camera, 'initialization': provenance,
        'inputs': {'episode': str(episode), 'prediction_json': str(prediction_path), 'prediction_sha256': sha256(prediction_path),
                   'extrinsics': str(extrinsics_path), 'extrinsics_sha256': sha256(extrinsics_path),
                   'extrinsics_semantics': 'world_to_camera', 'urdf': str(robot.urdf_path), 'urdf_sha256': sha256(robot.urdf_path),
                   'embodiments_root': str(robot.asset_dir.parent),
                   'task_config': str(args.task_config.expanduser().resolve()) if args.task_config else None},
        'coordinate_chain': 'camera closed TCP -> world -> target URDF root -> cuRobo base -> EE',
        'trajectory_retargeting': 'World closed TCP position, orientation and timestamps unchanged',
        'embodiment_distance_m': robot.distance, 'robot_config_snapshot': robot.config,
        'tcp_calibration': robot.calibration, 'runtime_robot_cfg': {s: c.robot_cfg for s, c in configs.items()},
        'runtime_config_corrections': {s: getattr(c, 'corrections', []) for s, c in configs.items()},
        'arms': arms, 'output': {'shape': list(state.shape), 'dtype': 'float32', 'columns': columns},
        'solver': {'type': 'whole_trajectory_candidate_graph_and_joint_optimization',
                   'position_tolerance_m': POSITION_TOLERANCE_METERS, 'rotation_tolerance_deg': ROTATION_TOLERANCE_DEGREES,
                   'ik_seeds': args.ik_seeds, 'maximum_ik_seeds': 256, 'maximum_initial_paths': 4,
                   'random_seed': RANDOM_SEED, 'initial_seed_source': provenance['source'],
                   'initial_state_role': 'initialization_and_first_frame_soft_preference',
                   'max_joint_step_rad': args.max_joint_step_rad,
                   'velocity_and_acceleration': 'time_aware_smoothing_and_diagnostics_only',
                   'self_collision_check': True, 'world_obstacle_collision_check': False,
                   'inter_arm_collision_check': False, 'validation_samples': 'original_timestamps'},
        'ground_truth_usage': 'Recorded target frame-0 values initialize search; no source episode states or future example states are used',
        'counts': counts, 'stage_wall_time_seconds': stages, 'total_wall_time_seconds': elapsed,
        'environment': {'python': sys.version.split()[0], 'torch': runtime.torch.__version__,
                        'torch_cuda': runtime.torch.version.cuda, 'curobo': package_version('curobo', runtime.curobo),
                        'gpu': runtime.torch.cuda.get_device_name(runtime.device)},
    }
    if args.solver == 'differential':
        settings = metadata['solver']
        settings.update(type='curobo_lm_offline_tracking_with_recovery', ik_seeds=1,
                        recovery_seeds=args.ik_seeds, lm_max_iterations=128,
                        initial_state_role='sequential_tracking_initial_state',
                        velocity_and_acceleration='diagnostics_only_no_controller_dt_clamping',
                        seed_position_weight=1., seed_orientation_weight=1.,
                        seed_velocity_weight=0., seed_acceleration_weight=0.,
                        initialization='previous_frame_with_multiseed_LM_recovery',
                        dt_source='original_timestamp_intervals_for_diagnostics',
                        main_optimizer_enabled=False)
        for key in ('maximum_ik_seeds', 'maximum_initial_paths'):
            settings.pop(key)
    report = {'schema_version': 3, 'summary': {'status': status, 'frame_count': count, 'counts': counts,
              'failed_frames': failures, 'total_wall_time_seconds': elapsed}, 'arms': summaries,
              'frames': [{'frame_index': i, 'time_seconds': prediction.timestamps[i],
                          **{s: diagnostics[s][i] for s in arms}} for i in range(count)]}
    save_trajectory_output(output, state, metadata, report)
    (output / 'solver_error.json').unlink(missing_ok=True)
    print(f'Saved {output / metadata["output"]["state_file"]}: {state.shape}, {status}, {elapsed:.2f}s', flush=True)
    for side in arms:
        maximum = max((f['joint_step_max_abs_rad'] or 0 for f in diagnostics[side]), default=0.)
        print(f'  {side}: {counts[side]}, maximum joint step={maximum:.6f} rad', flush=True)
        if args.solver == 'differential':
            matched = summaries[side]['tcp_tracking_success']
            reasons = sorted({r for f in diagnostics[side] for r in f['failure_reasons']})
            print(f'    TCP within tolerance: {matched}/{count}; failure reasons: {reasons}', flush=True)
    if status != 'success':
        raise SystemExit(1)
