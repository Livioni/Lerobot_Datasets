"""URDF kinematics shared by the solver, diagnostics, and replay."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import convert_robotwin_tcp as urdf


@dataclass(frozen=True)
class ArmGeometry:
    side: str
    base_link: str
    ee_link: str
    canonical_joint_names: tuple[str, ...]
    footprint_from_base: np.ndarray  # URDF root <- cuRobo base
    link6_from_tcp: np.ndarray  # Generic EE <- closed TCP; legacy field name.


def joint_matrix(joint: urdf.Joint, value: float) -> np.ndarray:
    """Exact URDF origin @ motion; never silently clamp an arm solution."""
    result = np.eye(4)
    result[:3, :3] = urdf.rotation_matrix_from_rpy(joint.origin_rpy)
    result[:3, 3] = joint.origin_xyz
    motion = np.eye(4)
    if joint.joint_type in ('revolute', 'continuous'):
        motion[:3, :3] = urdf.rotation_matrix_from_axis_angle(joint.axis, value)
    elif joint.joint_type == 'prismatic':
        axis = np.asarray(joint.axis, dtype=float)
        motion[:3, 3] = axis / np.linalg.norm(axis) * value
    elif joint.joint_type != 'fixed':
        raise ValueError(f'Unsupported joint type: {joint.joint_type}')
    return result @ motion


def fk_link(model, link: str, values: dict[str, float]) -> np.ndarray:
    pose = np.eye(4)
    for joint in urdf.joint_chain(model, link):
        if joint.joint_type != 'fixed' and joint.name not in values:
            raise ValueError(f'Missing FK value for joint {joint.name}')
        pose = pose @ joint_matrix(joint, values.get(joint.name, 0.0))
    return pose


def joint_bounds(model, names):
    """URDF positional bounds in the requested joint order."""
    joints = [model.joints_by_name[name] for name in names]
    return (np.array([j.lower if j.lower is not None else -np.inf for j in joints]),
            np.array([j.upper if j.upper is not None else np.inf for j in joints]))


def fk_tcp_batch(model, geometry, configurations):
    """Independent NumPy URDF FK for [N,dof], without clipping joint values."""
    q = np.asarray(configurations, dtype=float)
    lookup = {name: i for i, name in enumerate(geometry.canonical_joint_names)}
    result = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for joint in urdf.joint_chain(model, geometry.ee_link):
        result = result @ joint_matrix(joint, 0.)
        if joint.joint_type == 'fixed':
            continue
        value = q[:, lookup[joint.name]]
        axis = np.asarray(joint.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        if joint.joint_type in ('revolute', 'continuous'):
            x, y, z = axis
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            c, s = np.cos(value)[:, None, None], np.sin(value)[:, None, None]
            motion[:, :3, :3] = c * np.eye(3) + (1-c) * np.outer(axis, axis) + s * skew
        elif joint.joint_type == 'prismatic':
            motion[:, :3, 3] = value[:, None] * axis
        result = result @ motion
    return result @ geometry.link6_from_tcp


def nearest_valid_branch(model, names, candidate, reference):
    """Select equivalent revolute angles nearest reference, within true limits."""
    output = np.asarray(candidate, dtype=float).copy()
    for i, name in enumerate(names):
        joint = model.joints_by_name[name]
        value = output[i]
        if not np.isfinite(value) or joint.joint_type not in ('revolute', 'continuous'):
            continue
        lower = joint.lower if joint.lower is not None else -math.inf
        upper = joint.upper if joint.upper is not None else math.inf
        turn = round((reference[i] - value) / (2 * math.pi))
        first = math.ceil((lower - value) / (2 * math.pi)) if math.isfinite(lower) else -math.inf
        last = math.floor((upper - value) / (2 * math.pi)) if math.isfinite(upper) else math.inf
        if first <= last:
            output[i] = value + min(max(turn, first), last) * 2 * math.pi
    return output


def pose_matrix(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Robot pose must be finite [x,y,z,qw,qx,qy,qz]')
    norm = np.linalg.norm(values[3:])
    if norm < 1e-12:
        raise ValueError('Zero quaternion in robot pose')
    w, x, y, z = values[3:] / norm
    out = np.eye(4)
    out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    out[:3, 3] = values[:3]
    return out


def gripper_positions(model, config, side_index: int, openness: float):
    """RoboTwin drive targets and the corresponding limit-clamped display qpos."""
    low, high = config['gripper_scale']
    base_value = low + np.clip(openness, 0, 1) * (high - low)
    spec = config['gripper_name'][side_index]
    raw = {spec['base']: base_value}
    raw.update({name: base_value * scale + bias for name, scale, bias in spec['mimic']})
    return {name: float(np.clip(value,
                model.joints_by_name[name].lower if model.joints_by_name[name].lower is not None else -np.inf,
                model.joints_by_name[name].upper if model.joints_by_name[name].upper is not None else np.inf))
            for name, value in raw.items()}
