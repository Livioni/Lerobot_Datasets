from __future__ import annotations
import copy
import logging
import sys
import time
from typing import Any
import numpy as np
import convert_robotwin_tcp as tcp_converter
from ._io import POSITION_TOLERANCE_METERS, ROTATION_TOLERANCE_RADIANS, RANDOM_SEED
from ._kinematics import ArmGeometry, fk_link, fk_tcp_batch, joint_bounds, nearest_valid_branch
from ._trajectory import validate_inputs, motion_metrics, smoothness_cost, select_paths, diverse_candidates, interpolate_candidate_gaps

def rotation_error_radians(target: np.ndarray, actual: np.ndarray) -> float:
    relative = target[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))

def independent_fk_tcp(model, geometry, q_canonical):
    return fk_link(model, geometry.ee_link,
                   dict(zip(geometry.canonical_joint_names, q_canonical))) @ geometry.link6_from_tcp


class CuroboExtensionLogFilter(logging.Filter):
    """Clarify CuRobo's unconditional JIT messages without hiding build failures."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for suffix in (
            " not found, JIT compiling...",
            " not found, jit compiling...",
            " binary not found, jit compiling...",
        ):
            if message.endswith(suffix):
                name = message[:-len(suffix)]
                if name in {
                    "kinematics_fused_cu", "geom_cu", "lbfgs_step_cu",
                    "line_search_cu", "tensor_step_cu",
                }:
                    record.msg = (
                        f"{name}: loading via PyTorch JIT cache "
                        "(builds only if needed)"
                    )
                    record.args = ()
        return True

class CuroboRuntime:
    def __init__(self, device: str) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "IK requires CUDA-enabled PyTorch and compatible CuRobo in the "
                "current Python environment; the environment need not be named RoboTwin. "
                f"Python: {sys.executable}; import error: {error}"
            ) from error

        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError(f"--device must be a CUDA device, got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is invalid for {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(index)
        logger = logging.getLogger("curobo")
        extension_filter = CuroboExtensionLogFilter()
        logger.addFilter(extension_filter)
        try:
            import curobo
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        except ImportError as error:
            raise RuntimeError(
                "IK requires compatible CuRobo (curobo.wrap.reacher.ik_solver API) "
                "and CUDA-enabled PyTorch in the current Python environment; "
                "the environment need not be named RoboTwin. "
                f"Python: {sys.executable}; import error: {error}"
            ) from error
        finally:
            logger.removeFilter(extension_filter)
        self.torch = torch
        self.curobo = curobo
        self.Pose = Pose
        self.IKSolver = IKSolver
        self.IKSolverConfig = IKSolverConfig
        self.device = torch.device(f"cuda:{index}")
        self.tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)

    def create_solver(self, robot_cfg: dict[str, Any], num_seeds: int) -> Any:
        config = self.IKSolverConfig.load_from_robot_config(
            copy.deepcopy(robot_cfg),
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            position_threshold=POSITION_TOLERANCE_METERS,
            rotation_threshold=ROTATION_TOLERANCE_RADIANS,
            use_cuda_graph=True,
            self_collision_check=True,
            self_collision_opt=True,
            sync_cuda_time=True,
            store_debug=False,
            regularization=True,
            seed=RANDOM_SEED,
        )
        return self.IKSolver(config)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def pose_from_matrix(self, matrix: np.ndarray) -> Any:
        tensor = self.torch.as_tensor(
            matrix, device=self.device, dtype=self.torch.float32
        )
        return self.Pose.from_matrix(tensor)


def pose_residuals(model, geometry, q, targets):
    actual = fk_tcp_batch(model, geometry, q)
    position = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=-1)
    relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ actual[:, :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) * .5, -1, 1))
    return position, rotation


def _tensor(runtime, values):
    return runtime.torch.as_tensor(values, device=runtime.device, dtype=runtime.torch.float32).contiguous()


def _locked_mask(result, locks):
    shape = tuple(result.success.shape)
    mask = np.ones(shape, dtype=bool)
    if not locks:
        return mask
    names = list(result.js_solution.joint_names or [])
    full = result.js_solution.position.detach().cpu().numpy()
    for name, value in locks.items():
        if name not in names or full.shape[:-1] != shape:
            return np.zeros(shape, dtype=bool)
        mask &= np.isclose(full[..., names.index(name)], value, atol=1e-5, rtol=0)
    return mask


def _valid_configs(runtime, validator, q, canonical_names):
    order = [canonical_names.index(name) for name in validator.joint_names]
    return validator.check_valid(_tensor(runtime, q[:, order])).detach().cpu().numpy().reshape(-1).astype(bool)


def evaluate_trajectory(runtime, validator, geometry, model, q, targets, timestamps, step_limit):
    """Validate the exact float32 payload; do not rely on optimizer success flags."""
    q = np.asarray(q, dtype=np.float32).astype(float)
    if not np.isfinite(q).all():
        raise ValueError('Trajectory contains nonfinite values')
    lower, upper = joint_bounds(model, geometry.canonical_joint_names)
    positions, rotations = pose_residuals(model, geometry, q, targets)
    feasible = _valid_configs(runtime, validator, q, geometry.canonical_joint_names)
    bounds = np.all((q >= lower) & (q <= upper), axis=1)
    steps, velocity, acceleration = motion_metrics(q, timestamps)
    frames = []
    violation = 0.
    for i in range(len(q)):
        step = float(np.max(np.abs(steps[i-1]))) if i else None
        reasons = []
        if positions[i] > POSITION_TOLERANCE_METERS:
            reasons.append('tcp_position_tolerance')
        if rotations[i] > ROTATION_TOLERANCE_RADIANS:
            reasons.append('tcp_rotation_tolerance')
        if not bounds[i]:
            reasons.append('joint_limits')
        if not feasible[i]:
            reasons.append('model_constraints')
        if i and step > step_limit:
            reasons.append('joint_step_limit')
        violation += max(positions[i] / POSITION_TOLERANCE_METERS - 1, 0) ** 2
        violation += max(rotations[i] / ROTATION_TOLERANCE_RADIANS - 1, 0) ** 2
        violation += float(not bounds[i]) + float(not feasible[i])
        if i:
            violation += max(step / step_limit - 1, 0) ** 2
        frames.append({
            'success': not reasons, 'failure_reasons': reasons,
            'position_residual_m': float(positions[i]),
            'rotation_residual_rad': float(rotations[i]),
            'rotation_residual_deg': float(np.degrees(rotations[i])),
            'joint_limits_valid': bool(bounds[i]), 'constraint_feasible': bool(feasible[i]),
            'joint_step_max_abs_rad': step,
            'joint_step_l2_rad': float(np.linalg.norm(steps[i-1])) if i else None,
            'joint_velocity_rad_s': velocity[i-1].tolist() if i else None,
            'joint_acceleration_rad_s2': acceleration[i-1].tolist() if 0 < i < len(q)-1 else None,
        })
    return q, frames, float(violation)


def _equivalent_candidates(model, names, values, initial):
    """Retain raw and legal equivalent windings, instead of wrapping edge distances."""
    if not len(values):
        return values
    lower, upper = joint_bounds(model, names)
    nearest = np.asarray([nearest_valid_branch(model, names, q, initial) for q in values])
    variants = [values, nearest]
    for j, name in enumerate(names):
        if model.joints_by_name[name].joint_type not in ('revolute', 'continuous'):
            continue
        for sign in (-1, 1):
            shifted = nearest.copy()
            shifted[:, j] += sign * 2 * np.pi
            variants.append(shifted)
    combined = np.concatenate(variants)
    return combined[np.all((combined >= lower) & (combined <= upper), axis=1)]


def _collect_candidates(runtime, solver, validator, geometry, model, targets, ee_targets,
                        initial, layers, approximate, seeds, propagation, locks):
    """Fixed-shape batched IK; adjacent candidate sets are read from a frozen pass."""
    count = len(targets)
    batch = min(32, count)
    names = geometry.canonical_joint_names
    order = [names.index(name) for name in solver.joint_names]
    inverse = [list(solver.joint_names).index(name) for name in names]
    snapshot = [layer.copy() for layer in layers]
    started = time.perf_counter()
    reported_success = 0
    for start in range(0, count, batch):
        indices = np.minimum(np.arange(start, start+batch), count-1)
        seed_list = []
        width = min(seeds - 1, 33) if propagation else 1
        for t in indices:
            pool = [initial[None]]
            if propagation:
                # Both directions influence the next batch pass; no previous output exists.
                for neighbor in (max(0, t-1), min(count-1, t+1), t):
                    pool.append(diverse_candidates(snapshot[neighbor], initial, maximum=max(1, (width-1)//3)))
            pool = np.concatenate(pool)[:width]
            seed_list.append(np.concatenate([pool, np.tile(initial, (width-len(pool), 1))]))
        result = solver.solve_batch(runtime.pose_from_matrix(ee_targets[indices]),
                                    seed_config=_tensor(runtime, np.asarray(seed_list)[..., order]),
                                    return_seeds=seeds, num_seeds=seeds, use_nn_seed=False)
        runtime.synchronize()
        values = result.solution.detach().cpu().numpy()[..., inverse]
        success = result.success.detach().cpu().numpy().astype(bool)
        # Lock values come from the runtime model and are verified in the full IK result.
        success &= _locked_mask(result, locks)
        for local, t in enumerate(indices[:min(batch, count-start)]):
            finite = np.isfinite(values[local]).all(axis=1)
            raw = values[local][finite]
            if not len(raw):
                continue
            pos, rot = pose_residuals(model, geometry, raw, np.repeat(targets[t][None], len(raw), axis=0))
            errors = (pos / POSITION_TOLERANCE_METERS) ** 2 + (rot / ROTATION_TOLERANCE_RADIANS) ** 2
            best = int(np.argmin(errors))
            if approximate[t] is None or errors[best] < approximate[t][0]:
                approximate[t] = (float(errors[best]), raw[best].copy())
            qualified = success[local][finite] & (pos <= POSITION_TOLERANCE_METERS) & (rot <= ROTATION_TOLERANCE_RADIANS)
            reported_success += int(qualified.sum())
            extended = _equivalent_candidates(model, names, raw[qualified], initial)
            merged = diverse_candidates(np.concatenate([layers[t], extended]), initial)
            if len(merged):
                valid = _valid_configs(runtime, validator, merged, names)
                lower, upper = joint_bounds(model, names)
                valid &= np.all((merged >= lower) & (merged <= upper), axis=1)
                # Equivalent turns can incur floating-point FK differences; recheck them.
                p, r = pose_residuals(model, geometry, merged, np.repeat(targets[t][None], len(merged), axis=0))
                layers[t] = merged[valid & (p <= POSITION_TOLERANCE_METERS) & (r <= ROTATION_TOLERANCE_RADIANS)]
        if start == 0 or start+batch >= count or (start//batch) % 4 == 0:
            print(f'  {geometry.side}: candidates {min(start+batch, count)}/{count}, seeds={seeds}, neighbors={propagation}', flush=True)
    return {'seeds': seeds, 'neighbor_propagation': propagation,
            'qualified_ik_candidates': reported_success, 'candidate_counts': [len(x) for x in layers],
            'wall_time_seconds': time.perf_counter()-started}


def solve_arm(runtime, runtime_config, geometry, model, footprint_targets, link6_targets,
              q_initial, timestamps, ik_seeds=64, max_joint_step_rad=.5):
    """Search IK branches and optimize all TCP frames jointly, returning no held frames."""
    from ._trajectory_opt import refine_trajectory
    validate_inputs(q_initial, footprint_targets, timestamps, ik_seeds, max_joint_step_rad)
    validate_inputs(q_initial, link6_targets, timestamps, ik_seeds, max_joint_step_rad)
    names = geometry.canonical_joint_names
    if len(q_initial) != len(names):
        raise ValueError('Initial state does not match active joints')
    count = len(timestamps)
    initial = np.asarray(q_initial, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    validator = runtime.create_solver(runtime_config.robot_cfg, 1)
    if set(validator.joint_names) != set(names) or len(validator.joint_names) != len(names):
        raise ValueError('CuRobo active joints do not match the target arm')
    layers = [np.empty((0, len(names))) for _ in range(count)]
    approximate = [None] * count
    search, paths = [], []
    seeds = ik_seeds
    while True:
        solver = runtime.create_solver(runtime_config.robot_cfg, seeds)
        for propagation in (False, True, True):
            search.append(_collect_candidates(runtime, solver, validator, geometry, model,
                                             footprint_targets, link6_targets, initial, layers,
                                             approximate, seeds, propagation, runtime_config.lock_joints))
            paths = select_paths(layers, timestamps, initial, max_joint_step_rad)
            if paths and propagation:
                break
        del solver
        if paths or seeds >= 256:
            break
        seeds = min(seeds*2, 256)
    graph_connected = bool(paths)
    if not paths:
        if all(item is None for item in approximate):
            raise RuntimeError('IK returned no finite candidate for the entire trajectory')
        guide = interpolate_candidate_gaps(layers, timestamps, initial, max_joint_step_rad)
        lower, upper = joint_bounds(model, names)
        relaxed_layers, penalties = [], []
        for t, layer in enumerate(layers):
            if len(layer):
                relaxed_layers.append(layer)
                penalties.append(np.zeros(len(layer)))
            else:
                values = [guide[t]]
                if approximate[t] is not None:
                    raw = nearest_valid_branch(model, names, approximate[t][1], initial)
                    # Bounds projection is only for an optimization INITIALIZATION.
                    # Final states are never clipped and always undergo independent FK.
                    values.append(np.clip(raw, lower+1e-6, upper-1e-6))
                values = diverse_candidates(values, initial)
                p, r = pose_residuals(model, geometry, values, np.repeat(footprint_targets[t][None], len(values), axis=0))
                node_cost = np.maximum(p/POSITION_TOLERANCE_METERS-1, 0)**2
                node_cost += np.maximum(r/ROTATION_TOLERANCE_RADIANS-1, 0)**2
                node_cost += ~_valid_configs(runtime, validator, values, names)
                relaxed_layers.append(values)
                penalties.append(1e4 * node_cost)
        paths = select_paths(relaxed_layers, timestamps, initial, max_joint_step_rad,
                             count=3, relaxed=True, node_penalties=penalties)
        if not any(np.array_equal(guide, path) for path in paths):
            paths.append(guide)
    if not paths:
        raise RuntimeError('No finite whole-trajectory initialization could be generated')
    def evaluate(q):
        return evaluate_trajectory(runtime, validator, geometry, model, q, footprint_targets,
                                   timestamps, max_joint_step_rad)
    best = None
    best_rank = None
    optimization = []
    for index, path in enumerate(paths):
        print(f'  {geometry.side}: joint optimization {index+1}/{len(paths)} ({count} frames)', flush=True)
        q, frames, violation, report = refine_trajectory(
            runtime, runtime_config, geometry, model, path, footprint_targets, timestamps,
            initial, max_joint_step_rad, evaluate)
        rank = (violation > 0, violation, smoothness_cost(q, timestamps, initial))
        if best_rank is None or rank < best_rank:
            best, best_rank = (q, frames, violation), rank
        optimization.append(report)
    q, frames, violation = best
    # Final validation occurs after all optimization and after output quantization.
    q, frames, violation = evaluate(q)
    for i, frame in enumerate(frames):
        frame['candidate_count'] = len(layers[i])
    successes = sum(f['success'] for f in frames)
    summary = {'success': successes, 'failure': count-successes,
               'trajectory_success': violation == 0, 'graph_connected': graph_connected,
               'search': search, 'optimization': optimization,
               'constraint_violation': violation, 'smoothness_cost': smoothness_cost(q, timestamps, initial)}
    return q.astype(np.float32), frames, summary
