"""Read RoboTwin assets without modifying its configuration files."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import yaml
import convert_robotwin_tcp as urdf
from ._kinematics import ArmGeometry, fk_link, pose_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBODIMENTS_ROOT = REPO_ROOT / 'embodiments/RobotTwin_embodiments'
NAMES = ('franka-panda', 'ARX-X5', 'piper', 'ur5-wsg')


def slug(name):
    return name.lower().replace('-', '_')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RuntimeConfig:
    robot_cfg: dict[str, Any]
    source_document: dict[str, Any]
    lock_joints: dict[str, float]
    corrections: list[str]


@dataclass
class Embodiment:
    name: str
    asset_dir: Path
    urdf_path: Path
    config: dict
    model: Any
    runtime: RuntimeConfig
    geometries: dict[str, ArmGeometry]
    world_from_root: dict[str, np.ndarray]
    homestates: dict[str, np.ndarray]
    calibration: dict
    distance: float


def resolve_distance(name, distance, task_config):
    if task_config is not None:
        task = yaml.safe_load(Path(task_config).expanduser().read_text())
        selected = task.get('embodiment')
        if not isinstance(selected, list) or len(selected) != 3 or selected[:2] != [name, name]:
            raise ValueError(f'{task_config}: expected embodiment: [{name}, {name}, distance]')
        if distance is None:
            distance = selected[2]
    distance = 0.6 if distance is None else float(distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Embodiment distance must be positive and finite')
    return distance


def load_embodiment(name, embodiments_root=DEFAULT_EMBODIMENTS_ROOT, distance=None, task_config=None, *, robotwin_root=None):
    if name not in NAMES:
        raise ValueError(f'Unknown embodiment {name}')
    distance = resolve_distance(name, distance, task_config)
    if robotwin_root is not None:  # Compatibility for explicitly supplied old CLI paths.
        embodiments_root = Path(robotwin_root).expanduser() / 'assets/embodiments'
    asset_dir = Path(embodiments_root).expanduser().resolve() / name
    config = yaml.safe_load((asset_dir / 'config.yml').read_text())
    urdf_path = (asset_dir / config['urdf_path']).resolve()
    model = urdf.load_robot_model(str(urdf_path))
    config_path = asset_dir / 'curobo_tmp.yml'
    if not config_path.is_file():
        config_path = asset_dir / 'curobo.yml'
    document = yaml.safe_load(config_path.read_text())
    robot_cfg = copy.deepcopy(document['robot_cfg'])
    kin = robot_cfg['kinematics']
    kin['urdf_path'] = str(urdf_path)
    kin['asset_root_path'] = str(asset_dir)
    spheres = asset_dir / Path(kin['collision_spheres']).name
    if not spheres.is_file():
        raise FileNotFoundError(spheres)
    kin['collision_spheres'] = str(spheres)
    locks = {key: float(value) for key, value in (kin.get('lock_joints') or {}).items()}
    kin['lock_joints'] = locks
    corrections = []
    cs = kin['cspace']
    size = len(cs['joint_names'])
    for key in ('retract_config', 'null_space_weight', 'cspace_distance_weight'):
        values = cs[key]
        if len(values) == size + 1 and name in ('ARX-X5', 'piper') and len(set(values)) == 1:
            cs[key] = values[:size]
            corrections.append(f'{key}: removed unnamed uniform trailing entry ({size+1} -> {size})')
        elif len(values) != size:
            raise ValueError(f'{name}: {key} length does not match cspace joint_names')
    if len(set(cs['joint_names'])) != size:
        raise ValueError(f'{name}: duplicate cspace joint_names')
    calibration = yaml.safe_load((Path(__file__).parent / 'tcp_calibrations.yml').read_text())[name]
    if sha256(urdf_path) != calibration['urdf_sha256']:
        raise ValueError(f'{name}: URDF changed; recalibrate closed TCP before solving')
    # Fail explicitly if supplied assets no longer match this measured calibration.
    for relative, expected in calibration['mesh_sha256'].items():
        if sha256(asset_dir / relative) != expected:
            raise ValueError(f'{name}: calibration mesh changed: {relative}; recalibrate before solving')
    transforms, geometries, homes = {}, {}, {}
    poses = config['robot_pose']
    for index, side in enumerate(('left', 'right')):
        names = tuple(config['arm_joints_name'][index])
        if set(cs['joint_names']) - set(locks) != set(names):
            raise ValueError(f'{name}: cspace active joints differ from arm_joints_name')
        home = np.asarray(config['homestate'][index], dtype=float)
        if home.shape != (len(names),) or not np.isfinite(home).all():
            raise ValueError(f'{name}: invalid homestate')
        for joint_name, value in zip(names, home):
            joint = model.joints_by_name[joint_name]
            if (joint.lower is not None and value < joint.lower) or (joint.upper is not None and value > joint.upper):
                raise ValueError(f'{name}: homestate outside limits: {joint_name}')
        world_from_root = pose_matrix(poses[min(index, len(poses)-1)])
        world_from_root[0, 3] += (-0.5 if index == 0 else 0.5) * distance
        ee = kin['ee_link']
        if config['move_group'][index] != ee:
            raise ValueError(f'{name}: unexpected move_group / cuRobo ee_link mismatch')
        ee_from_tcp = np.eye(4)
        ee_from_tcp[:3, :3] = np.asarray(config['delta_matrix'])
        ee_from_tcp[:3, 3] = calibration['translation_m']
        # Verify the Sapien joint-axis frame cancels global_trans_matrix.
        ee_joint = model.joints_by_name[config['ee_joints'][index]]
        axis = np.asarray(ee_joint.axis, dtype=float); axis /= np.linalg.norm(axis)
        second = np.cross(axis, [0, 0, 1] if abs(axis[0]) > 0.9 else [1, 0, 0])
        second /= np.linalg.norm(second)
        axis_rotation = np.column_stack((axis, second, np.cross(axis, second)))
        if ee_joint.child != ee or not np.allclose(axis_rotation @ config['global_trans_matrix'], np.eye(3), atol=1e-6):
            raise ValueError(f'{name}: unsupported EE joint-axis convention')
        geometries[side] = ArmGeometry(side, kin['base_link'], ee, names,
                                      fk_link(model, kin['base_link'], {}), ee_from_tcp)
        homes[side] = home
        transforms[side] = world_from_root
    return Embodiment(name, asset_dir, urdf_path, config, model,
                      RuntimeConfig(robot_cfg, document, locks, corrections),
                      geometries, transforms, homes, calibration, distance)
