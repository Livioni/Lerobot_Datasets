"""Sequential cuRobo LM differential IK using the cuRobo 2 API."""
from __future__ import annotations

import copy
from dataclasses import replace
import sys
import numpy as np

from ._io import POSITION_TOLERANCE_METERS, ROTATION_TOLERANCE_RADIANS, RANDOM_SEED
from ._solver import evaluate_trajectory, _locked_mask, pose_residuals
from ._kinematics import joint_bounds, nearest_valid_branch
from ._trajectory import validate_inputs


def differential_robot_config(robot_cfg):
    """Translate the legacy RoboTwin URDF configuration without changing assets."""
    cfg = copy.deepcopy(robot_cfg)
    kin = cfg['kinematics']
    if kin.get('use_usd_kinematics'):
        raise ValueError('Differential IK requires a URDF robot configuration')
    for key in ('use_usd_kinematics', 'usd_path', 'usd_robot_root', 'isaac_usd_path',
                'usd_flip_joints', 'usd_flip_joint_limits'):
        kin.pop(key, None)
    kin['tool_frames'] = [kin.pop('ee_link')]
    # Legacy link_names selects extra FK outputs, not controlled end effectors.
    kin.pop('link_names', None)
    kin['cspace']['default_joint_position'] = kin['cspace'].pop('retract_config')
    return cfg


class DifferentialRuntime:
    def __init__(self, device):
        try:
            import torch
            import curobo
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
        except ImportError as exc:
            raise RuntimeError(
                'Differential IK requires cuRobo 2 (curobo.inverse_kinematics). '
                'Use the repository curobo source via PYTHONPATH or a separate cuRobo 2 environment. '
                f'Python: {sys.executable}; {exc}'
            ) from exc
        requested = torch.device(device)
        if requested.type != 'cuda':
            raise ValueError('--device must be a CUDA device')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f'Invalid CUDA device: {device}')
        torch.cuda.set_device(index)
        self.torch, self.curobo = torch, curobo
        self.device = torch.device(f'cuda:{index}')
        self.device_cfg = DeviceCfg(device=self.device, dtype=torch.float32)
        self.IK, self.Cfg = InverseKinematics, InverseKinematicsCfg
        self.GoalToolPose, self.JointState, self.Pose = GoalToolPose, JointState, Pose

    def tensor(self, value):
        return self.torch.as_tensor(value, device=self.device, dtype=self.torch.float32).contiguous()

    def create_solver(self, robot_cfg, num_seeds=1):
        config = self.Cfg.create(
            robot=differential_robot_config(robot_cfg), device_cfg=self.device_cfg,
            optimizer_configs=['ik/lbfgs_ik.yml'], num_seeds=num_seeds, seed_solver_num_seeds=num_seeds,
            use_cuda_graph=True, self_collision_check=True, scene_model=None,
            position_tolerance=POSITION_TOLERANCE_METERS,
            orientation_tolerance=ROTATION_TOLERANCE_RADIANS, random_seed=RANDOM_SEED,
            optimization_dt=None, success_requires_convergence=False,
            seed_position_weight=1., seed_orientation_weight=1.,
            seed_velocity_weight=0., seed_acceleration_weight=0.,
        )
        solver = self.IK(config)
        seed_solver = solver.seed_ik_solver
        solver.seed_ik_solver = type(seed_solver)(replace(
            seed_solver.config, max_iterations=128, inner_iterations=16,
            lambda_initial=.01))
        return solver


def solve_arm_differential(runtime, runtime_config, geometry, model, footprint_targets,
                           link6_targets, q_initial, timestamps, ik_seeds=64,
                           max_joint_step_rad=.5):
    """Converge at each original TCP, with explicit multi-seed LM recovery.

    These are offline numerical iterations, not physical controller ticks. Passing
    current_state.dt to cuRobo LM would activate velocity clamping, even with zero
    regularization weights, and incorrectly limit the first frame's initialization.
    """
    validate_inputs(q_initial, footprint_targets, timestamps, ik_seeds, max_joint_step_rad)
    validate_inputs(q_initial, link6_targets, timestamps, ik_seeds, max_joint_step_rad)
    names = list(geometry.canonical_joint_names)
    if len(q_initial) != len(names):
        raise ValueError('Initial state does not match active joints')
    solver = runtime.create_solver(runtime_config.robot_cfg)
    if len(solver.joint_names) != len(names) or set(solver.joint_names) != set(names):
        raise ValueError('CuRobo active joints do not match the target arm')
    order = [names.index(name) for name in solver.joint_names]
    inverse = [list(solver.joint_names).index(name) for name in names]
    lower, upper = joint_bounds(model, names)
    previous = np.asarray(q_initial, dtype=np.float32)
    recovery = None
    values, feasible, details = [], [], []

    def candidates(engine, goal, initial):
        # Bounds projection applies only to an optimization seed. Returned states
        # are never clipped, and all reported states undergo exact independent FK.
        initial = np.clip(initial, lower, upper)
        seed = engine.seed_ik_solver.solve_single(
            goal_tool_poses=goal, current_state=None,
            seed_config=runtime.tensor(initial[order][None, None]),
            return_seeds=engine.config.num_seeds)
        raw = seed.solution.detach().cpu().numpy().reshape(-1, len(names))[:, inverse]
        if not np.isfinite(raw).all():
            raise RuntimeError('Nonfinite differential IK candidate')
        raw = np.asarray([nearest_valid_branch(model, names, q, previous) for q in raw],
                         dtype=np.float32)
        result = engine.solve_pose(
            goal_tool_poses=goal, seed_config=runtime.tensor(raw[:, order][None]),
            current_state=None, return_seeds=engine.config.num_seeds, run_optimizer=False)
        q = result.solution.detach().cpu().numpy().reshape(-1, len(names))[:, inverse].copy()
        valid = result.feasible.detach().cpu().numpy().reshape(-1).astype(bool)
        valid &= _locked_mask(result, runtime_config.lock_joints).reshape(-1)
        valid &= np.all((q >= lower) & (q <= upper), axis=1)
        return q, valid

    for i, target in enumerate(link6_targets):
        goal = runtime.GoalToolPose.from_poses(
            {geometry.ee_link: runtime.Pose.from_matrix(runtime.tensor(target[None]))},
            ordered_tool_frames=solver.tool_frames, num_goalset=1)
        q, valid = candidates(solver, goal, previous)

        def assess(q, valid):
            pos, rot = pose_residuals(model, geometry, q,
                                     np.repeat(footprint_targets[i:i+1], len(q), axis=0))
            pose_ok = (pos <= POSITION_TOLERANCE_METERS) & (rot <= ROTATION_TOLERANCE_RADIANS)
            step = np.max(np.abs(q - previous), axis=1)
            accepted = pose_ok & valid & ((step <= max_joint_step_rad) | (i == 0))
            return pos, rot, pose_ok, accepted

        pos, rot, pose_ok, accepted = assess(q, valid)
        recovered = not accepted.any()
        if recovered:
            if recovery is None:
                recovery = runtime.create_solver(runtime_config.robot_cfg, num_seeds=ik_seeds)
                if list(recovery.joint_names) != list(solver.joint_names):
                    raise ValueError('Recovery solver joint order differs from local solver')
            extra, extra_valid = candidates(recovery, goal, previous)
            q, valid = np.concatenate([q, extra]), np.concatenate([valid, extra_valid])
            pos, rot, pose_ok, accepted = assess(q, valid)
        # Prefer a fully valid nearby solution. If no such solution was found,
        # retain an honest best candidate and explicitly fail its constraints.
        eligible = np.flatnonzero(accepted)
        if not len(eligible):
            eligible = np.flatnonzero(pose_ok & valid)
        if len(eligible):
            index = eligible[np.argmin(np.sum((q[eligible] - previous) ** 2, axis=1))]
        else:
            error = (pos / POSITION_TOLERANCE_METERS) ** 2 + (rot / ROTATION_TOLERANCE_RADIANS) ** 2
            error += np.sum((np.maximum(lower-q, 0) + np.maximum(q-upper, 0)) ** 2, axis=1)
            error += (~valid).astype(float)
            index = int(np.argmin(error))
        chosen = q[index].copy()
        # Evaluate the selected payload separately to give useful constraint names.
        checked = solver.solve_pose(goal_tool_poses=goal, current_state=None,
                                    seed_config=runtime.tensor(chosen[order][None, None]),
                                    return_seeds=1, run_optimizer=False)
        constraints = solver.metrics_rollout.compute_metrics_from_action(
            runtime.tensor(chosen[order][None, None])).costs_and_constraints.constraints
        constraint_values = {name: float(value.sum().item())
                             for name, value in zip(constraints.names, constraints.values)}
        values.append(chosen)
        feasible.append(bool(checked.feasible.all().item()) and
                        bool(_locked_mask(checked, runtime_config.lock_joints).all()))
        details.append({
            'lm_recovery_used': recovered, 'lm_candidate_count': len(q),
            'tcp_tracking_success': bool(pose_ok[index]),
            'model_constraint_values': constraint_values,
            'violated_model_constraints': [n for n, v in constraint_values.items() if v > 0],
            'differential_update_max_abs_rad': float(np.max(np.abs(chosen-previous))),
        })
        previous = chosen
        if i == 0 or (i+1) % 50 == 0 or i+1 == len(timestamps):
            print(f'  {geometry.side}: differential IK {i+1}/{len(timestamps)}', flush=True)
    q, frames, violation = evaluate_trajectory(
        runtime, None, geometry, model, values, footprint_targets, timestamps,
        max_joint_step_rad, constraint_feasible=np.asarray(feasible))
    for frame, detail in zip(frames, details):
        frame.update(detail)
    successes = sum(frame['success'] for frame in frames)
    return q.astype(np.float32), frames, {
        'success': successes, 'failure': len(timestamps)-successes,
        'tcp_tracking_success': sum(d['tcp_tracking_success'] for d in details),
        'trajectory_success': violation == 0, 'constraint_violation': violation,
        'method': 'curobo_lm_offline_tracking_with_recovery',
        'lm_max_iterations': 128, 'recovery_seeds': ik_seeds,
        'recovery_frames': sum(d['lm_recovery_used'] for d in details),
        'failed_candidate_policy': 'continue_from_finite_candidate_for_diagnostics',
    }
