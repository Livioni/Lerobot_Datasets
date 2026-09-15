"""Load calibrated episodes and successful IK outputs without Rerun or cuRobo."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import yaml

import convert_robotwin_tcp as urdf
from robotwin_ik._kinematics import gripper_positions, pose_matrix

ROOT = Path(__file__).resolve().parents[1]
EMBODIMENTS = ROOT / 'embodiments/RobotTwin_embodiments'
SOURCE_CONFIG = ROOT / 'embodiments/aloha-agilex/config.yml'
CAMERA = 'third_views'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'{path}: expected an object')
    return result


def validate_transform(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{label}: expected finite 4x4 transforms')
    rotation = value[..., :3, :3]
    if (not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError(f'{label}: invalid rigid transform')
    return value


def frame_paths(directory, count):
    paths = list(directory.glob('*.png'))
    if len(paths) != count or any(not p.stem.isdecimal() for p in paths):
        raise ValueError(f'{directory}: expected {count} numerically named PNG frames')
    paths.sort(key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(count)):
        raise ValueError(f'{directory}: frame numbers must be contiguous from zero')
    return paths


@dataclass
class Episode:
    path: Path
    metadata: dict
    rgb_paths: list[Path]
    depth_paths: list[Path] | None
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    height: int
    width: int

    @property
    def count(self):
        return len(self.rgb_paths)

    def read_frame(self, index):
        with Image.open(self.rgb_paths[index]) as image:
            rgb = np.array(image.convert('RGB'))
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f'{self.rgb_paths[index]}: image size disagrees with metadata')
        depth = None
        if self.depth_paths is not None:
            with Image.open(self.depth_paths[index]) as image:
                raw = np.array(image)
            if (raw.shape != (self.height, self.width) or raw.dtype.kind not in 'iu' or raw.dtype.itemsize < 2
                    or np.any(raw < 0) or np.any(raw > 65535)):
                raise ValueError(f'{self.depth_paths[index]}: expected uint16 millimeter depth')
            depth = raw.astype(np.float64) * 0.001
        return rgb, depth


def load_episode(path, with_depth):
    path = Path(path).expanduser().resolve()
    meta = load_json(path / 'metadata.json')
    if meta.get('format_version') != 'robotwin_4d_v1':
        raise ValueError('Expected robotwin_4d_v1 episode metadata')
    if meta.get('embodiment') not in ('aloha_agilex', 'aloha-agilex'):
        raise ValueError('The source embodiment must be aloha-agilex')
    if meta.get('extrinsics', {}).get('transform') != 'world_to_camera':
        raise ValueError('Episode extrinsics must be world_to_camera in OpenCV coordinates')
    count = int(meta['num_frames'])
    height, width = int(meta['image_height']), int(meta['image_width'])
    if min(count, height, width) <= 0:
        raise ValueError('Frame count and image dimensions must be positive')
    rgb = frame_paths(path / 'images' / CAMERA, count)
    depth = None
    if with_depth:
        spec = meta.get('depth', {})
        if spec.get('unit') != 'millimeter' or spec.get('invalid_value') != 0:
            raise ValueError('Expected millimeter scene depth with zero as invalid value')
        depth = frame_paths(path / 'depths' / CAMERA, count)
        if [p.name for p in rgb] != [p.name for p in depth]:
            raise ValueError('RGB and depth filenames do not match')
    k = np.load(path / 'intrinsics' / f'{CAMERA}.npy', allow_pickle=False).astype(float)
    if k.shape == (3, 3):
        k = np.broadcast_to(k, (count, 3, 3))
    if (k.shape != (count, 3, 3) or not np.isfinite(k).all()
            or np.any(k[:, (0, 1), (0, 1)] <= 0)
            or not np.allclose(k[:, 2], [0, 0, 1]) or not np.allclose(k[:, 1, 0], 0)):
        raise ValueError('Expected finite pinhole intrinsics [3,3] or [T,3,3]')
    e = np.load(path / 'extrinsics' / f'{CAMERA}.npy', allow_pickle=False).astype(float)
    if e.shape in ((3, 4), (4, 4)):
        e = np.broadcast_to(e, (count, *e.shape))
    if e.shape == (count, 3, 4):
        bottom = np.broadcast_to([0, 0, 0, 1], (count, 1, 4))
        e = np.concatenate((e, bottom), axis=1)
    if e.shape != (count, 4, 4):
        raise ValueError('Expected extrinsics [3,4], [4,4], [T,3,4] or [T,4,4]')
    return Episode(path, meta, rgb, depth, k, validate_transform(e, 'extrinsics'), height, width)


@dataclass
class Robot:
    name: str
    urdf_path: Path
    config: dict
    state: np.ndarray
    arms: dict
    locks: dict
    model: urdf.RobotModel

    @property
    def slug(self):
        return self.name.lower().replace('-', '_')

    def values(self, side, index, include_locks=True):
        arm = self.arms[side]
        values = dict(self.locks.get(side, {})) if include_locks else {}
        values.update(zip(arm['active_joint_names'], self.state[index, arm['joint_columns']]))
        values.update(gripper_positions(self.model, self.config, ('left', 'right').index(side),
                                       self.state[index, arm['gripper_column']]))
        return values


def validate_robot(robot, count, binary_gripper):
    state = robot.state
    if state.ndim != 2 or len(state) != count or not np.isfinite(state).all():
        raise ValueError('Robot state must be a finite [T,D] array matching the episode')
    if set(robot.arms) != {'left', 'right'}:
        raise ValueError('Expected left and right arm metadata')
    used = []
    for side, arm in robot.arms.items():
        names, columns = arm['active_joint_names'], arm['joint_columns']
        if len(names) != len(columns) or len(names) != len(set(names)):
            raise ValueError(f'{side}: invalid active joint mapping')
        cols = [*columns, arm['gripper_column']]
        if any(not isinstance(c, int) or c < 0 or c >= state.shape[1] for c in cols):
            raise ValueError(f'{side}: state column out of bounds')
        used.extend(cols)
        for name in names:
            if name not in robot.model.joints_by_name:
                raise ValueError(f'{side}: unknown joint {name}')
            joint = robot.model.joints_by_name[name]
            q = state[:, columns[names.index(name)]]
            if ((joint.lower is not None and np.any(q < joint.lower - 1e-4))
                    or (joint.upper is not None and np.any(q > joint.upper + 1e-4))):
                raise ValueError(f'{side}: {name} exceeds URDF position limits')
        root = np.asarray(arm['world_from_root'])
        if root.shape != (4, 4):
            raise ValueError(f'{side}: world_from_root must be 4x4')
        validate_transform(root, f'{side} root')
        g = state[:, arm['gripper_column']]
        if np.any((g < 0) | (g > 1)) or (binary_gripper and not np.isin(g, [0, 1]).all()):
            raise ValueError(f'{side}: invalid normalized gripper states')
        values = robot.values(side, 0)
        if not set(values).issubset(robot.model.joints_by_name) or not np.isfinite(list(values.values())).all():
            raise ValueError(f'{side}: invalid locked/gripper joint mapping')
    if sorted(used) != list(range(state.shape[1])):
        raise ValueError('Arm mappings must cover every state column exactly once')
    return robot


def load_source(episode):
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    path = (SOURCE_CONFIG.parent / config['urdf_path']).resolve()
    state = np.load(episode.path / 'robot_state.npy', allow_pickle=False)
    if state.shape != (episode.count, 14):
        raise ValueError('Aloha source state must have shape [T,14]')
    arms = {}
    for index, side in enumerate(('left', 'right')):
        arms[side] = dict(active_joint_names=config['arm_joints_name'][index],
                          joint_columns=list(range(index * 7, index * 7 + 6)),
                          gripper_column=index * 7 + 6,
                          world_from_root=pose_matrix(config['robot_pose'][0]))
    return validate_robot(Robot('aloha-agilex', path, config, state, arms, {},
                                urdf.load_robot_model(str(path))), episode.count, False)


def load_target(ik_dir, episode, embodiments_root=EMBODIMENTS):
    ik_dir = Path(ik_dir).expanduser().resolve()
    meta = load_json(ik_dir / 'metadata.json')
    if meta.get('format') not in ('robotwin_closed_tcp_ik_v2', 'robotwin_closed_tcp_trajectory_v3'):
        raise ValueError('Expected v2/v3 IK output metadata')
    if meta.get('status', 'success') != 'success':
        raise ValueError('Failed IK trajectory: only successful robot_state.npy is accepted')
    if meta['format'] == 'robotwin_closed_tcp_trajectory_v3':
        if (meta.get('schema_version') != 3 or meta.get('status') != 'success'
                or meta['output'].get('state_file') != 'robot_state.npy'
                or meta['output'].get('role') != 'solution'):
            raise ValueError('IK status/state filename/role disagree')
    if meta['format'] == 'robotwin_closed_tcp_ik_v2' and 'status' not in meta:
        diagnostics_path = ik_dir / 'diagnostics.json'
        if not diagnostics_path.is_file():
            raise ValueError('v2 IK without status requires diagnostics.json to verify success')
        frames = load_json(diagnostics_path).get('frames', [])
        if (len(frames) != episode.count or any(
                frame.get(side, {}).get('success') is not True
                for frame in frames for side in ('left', 'right'))):
            raise ValueError('Failed or incomplete v2 IK diagnostics; only successful trajectories are accepted')
    if meta['frame_count'] != episode.count or meta.get('camera') != CAMERA:
        raise ValueError('IK frame count or camera does not match this episode')
    if meta['inputs'].get('extrinsics_semantics') != 'world_to_camera':
        raise ValueError('IK extrinsics must use world_to_camera')
    if sha256(episode.path / 'extrinsics' / f'{CAMERA}.npy') != meta['inputs']['extrinsics_sha256']:
        raise ValueError('Camera extrinsics differ from those used for IK')
    # Fixed cameras can have identical hashes across episodes. Verify the local
    # prediction too when available, without depending on stale absolute paths.
    prediction = episode.path / Path(meta['inputs'].get('prediction_json', 'tcp_episode.json')).name
    expected = meta['inputs'].get('prediction_sha256')
    if expected and prediction.is_file() and sha256(prediction) != expected:
        raise ValueError('Prediction hash differs: IK may belong to a different episode')
    name = meta['embodiment']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid embodiment name')
    config = meta['robot_config_snapshot']
    local = Path(embodiments_root).expanduser().resolve() / name / config['urdf_path']
    recorded = Path(meta['inputs']['urdf']).expanduser()
    candidates = [local.resolve(), recorded.resolve()]
    path = next((p for p in candidates if p.is_file() and sha256(p) == meta['inputs']['urdf_sha256']), None)
    if path is None:
        raise ValueError(f'No matching URDF for {name}; missing resource or URDF hash changed')
    state = np.load(ik_dir / 'robot_state.npy', allow_pickle=False)
    if list(state.shape) != meta['output']['shape']:
        raise ValueError('IK state shape disagrees with metadata')
    locks = {}
    runtime = meta.get('runtime_robot_cfg', {})
    for side in ('left', 'right'):
        cfg = runtime.get(side, runtime)
        locks[side] = cfg.get('kinematics', {}).get('lock_joints') or {}
    robot = Robot(name, path, config, state, meta['arms'], locks, urdf.load_robot_model(str(path)))
    return validate_robot(robot, episode.count, True)
