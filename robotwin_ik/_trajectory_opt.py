"""Full-horizon constrained smoothing on the installed cuRobo differentiable FK."""
from __future__ import annotations

import math
import time
import numpy as np
import torch

from ._io import POSITION_TOLERANCE_METERS, ROTATION_TOLERANCE_RADIANS
from ._kinematics import joint_bounds
from ._trajectory import smoothness_cost


def quaternion_matrix(q, torch):
    """Differentiable wxyz quaternion rotation, with a sign-invariant result."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


class _CuroboQuaternionGradient(torch.autograd.Function):
    """Translate ordinary quaternion derivatives to cuRobo's angular cotangent.

    The installed fused FK backward ignores the scalar component and consumes
    [0, dL/dtheta_world], rather than dL/d(w,x,y,z). For a left perturbation,
    dq = .5 * [0, dtheta_world] * q. This adapter implements its transpose.
    """
    @staticmethod
    def forward(ctx, quaternion):
        ctx.save_for_backward(quaternion)
        return quaternion.clone()

    @staticmethod
    def backward(ctx, gradient):
        (q,) = ctx.saved_tensors
        w, v = q[..., :1], q[..., 1:]
        gw, gv = gradient[..., :1], gradient[..., 1:]
        angular = .5 * (w*gv - gw*v + torch.linalg.cross(v, gv, dim=-1))
        return torch.cat((torch.zeros_like(gw), angular), dim=-1).contiguous()


def curobo_rotation_matrix(quaternion):
    return quaternion_matrix(_CuroboQuaternionGradient.apply(quaternion), torch)


def refine_trajectory(runtime, config, geometry, model, path, targets, timestamps,
                      initial, step_limit, evaluate):
    """Augmented-Lagrangian inequalities, optimized over all [T,dof] variables.

    Feasible incumbents are retained independently of the optimization iterates.
    Actual acceptance uses independent FK and model checks, never the loss alone.
    """
    from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
    torch = runtime.torch
    started = time.perf_counter()
    # Independent buffers: IK graphs, validation and autograd must not resize each other.
    engine = runtime.create_solver(config.robot_cfg, 1)
    collision = SelfCollisionCost(SelfCollisionCostConfig(
        weight=1., tensor_args=runtime.tensor_args,
        self_collision_kin_config=engine.kinematics.get_self_collision_config()))
    to_tensor = lambda value: torch.as_tensor(value, device=runtime.device, dtype=torch.float32)
    order = [geometry.canonical_joint_names.index(name) for name in engine.joint_names]
    lower, upper = joint_bounds(model, geometry.canonical_joint_names)
    finite_low, finite_high = np.isfinite(lower), np.isfinite(upper)
    low, high = to_tensor(lower[finite_low]+1e-6), to_tensor(upper[finite_high]-1e-6)
    base_targets = np.linalg.inv(geometry.footprint_from_base) @ targets
    target_p, target_r = to_tensor(base_targets[:, :3, 3]), to_tensor(base_targets[:, :3, :3])
    offset = to_tensor(geometry.link6_from_tcp)
    q_initial = to_tensor(initial)
    dt = to_tensor(np.diff(timestamps))
    dt_middle = (dt[1:]+dt[:-1]) * .5
    time_scale = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.
    # Chordal SO(3) distance equals 4 sin(theta/2)^2, stable at theta=0 and pi.
    rotation_scale = 4 * math.sin(ROTATION_TOLERANCE_RADIANS / 2) ** 2
    q = to_tensor(path).clone().detach().requires_grad_(True)

    def terms():
        state = engine.fk(q[:, order].contiguous())
        ee_rotation = curobo_rotation_matrix(state.ee_quaternion)
        tcp_p = state.ee_position + torch.matmul(ee_rotation, offset[:3, 3])
        tcp_r = ee_rotation @ offset[:3, :3]
        position = ((tcp_p-target_p) ** 2).sum(-1) / POSITION_TOLERANCE_METERS ** 2 - .90
        rotation = ((tcp_r-target_r) ** 2).sum((-1, -2)) * .5 / rotation_scale - .90
        constraints = [position, rotation]
        if finite_low.any():
            constraints.append((low-q[:, finite_low]).flatten())
        if finite_high.any():
            constraints.append((q[:, finite_high]-high).flatten())
        spheres = state.link_spheres_tensor.unsqueeze(0)
        constraints.append(collision(spheres).flatten() / .001)
        cost = .01 * ((q[0]-q_initial) ** 2).sum()
        if len(dt):
            steps = q[1:]-q[:-1]
            velocity = steps / dt[:, None]
            constraints.append((steps.abs()/step_limit - .98).flatten())
            cost = cost + (velocity.square() * dt[:, None]).sum()
            if len(dt_middle):
                acceleration = (velocity[1:]-velocity[:-1]) / dt_middle[:, None]
                cost = cost + time_scale ** 2 * (acceleration.square()*dt_middle[:, None]).sum()
        return cost / len(path), torch.cat([c.flatten() for c in constraints])

    best_q, best_frames, best_violation = evaluate(path)
    best_rank = (best_violation > 0, best_violation, smoothness_cost(best_q, timestamps, initial))
    initial_cost = best_rank[-1]
    with torch.no_grad():
        _, first_g = terms()
    multipliers = torch.zeros_like(first_g)
    penalty = 10.
    evaluations = 0
    history = []
    error = None
    try:
        for outer in range(5):
            optimizer = torch.optim.LBFGS([q], lr=1., max_iter=20, history_size=20,
                                         tolerance_grad=1e-7, tolerance_change=1e-10,
                                         line_search_fn='strong_wolfe')
            for block in range(3):
                def closure():
                    nonlocal evaluations
                    optimizer.zero_grad(set_to_none=True)
                    smooth, g = terms()
                    shifted = torch.relu(multipliers + penalty*g)
                    loss = smooth + ((shifted.square()-multipliers.square()) / (2*penalty)).sum() / len(path)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite trajectory objective')
                    loss.backward()
                    if q.grad is None or not torch.isfinite(q.grad).all():
                        raise FloatingPointError('Nonfinite trajectory gradient')
                    evaluations += 1
                    return loss
                optimizer.step(closure)
                candidate = q.detach().cpu().numpy().copy()
                checked, frames, violation = evaluate(candidate)
                rank = (violation > 0, violation, smoothness_cost(checked, timestamps, initial))
                if rank < best_rank:
                    best_q, best_frames, best_violation, best_rank = checked, frames, violation, rank
            with torch.no_grad():
                smooth, g = terms()
                multipliers = torch.relu(multipliers + penalty*g).detach()
                maximum = float(torch.relu(g).max())
            history.append({'outer_iteration': outer+1, 'penalty': penalty,
                            'maximum_normalized_violation': maximum,
                            'smoothness_per_frame': float(smooth), 'best_feasible': best_violation == 0})
            # Constraint margins protect final float32 acceptance; stop after convergence.
            if maximum < 1e-4 and best_violation == 0:
                break
            penalty *= 10
    except (FloatingPointError, RuntimeError) as exc:
        if isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower():
            raise
        # Retain an already independently verified incumbent on numerical failures.
        error = f'{type(exc).__name__}: {exc}'
    runtime.synchronize()
    report = {'method': 'full_horizon_augmented_lagrangian_lbfgs',
              'objective_evaluations': evaluations, 'outer_iterations': history,
              'initial_smoothness_cost': initial_cost, 'selected_smoothness_cost': best_rank[-1],
              'selected_constraint_violation': best_violation, 'numerical_error': error,
              'wall_time_seconds': time.perf_counter()-started}
    return best_q, best_frames, best_violation, report
