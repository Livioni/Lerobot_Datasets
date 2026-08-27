#!/usr/bin/env python3
"""Convert RoboTwin 4D episodes from joint states to camera-frame TCP states.

The output layout for every episode is::

    episode_xxxxxxx/
      TCP/
        left_state.npy   # [T, 7]: x, y, z, roll, pitch, yaw, gripper_open
        right_state.npy  # [T, 7]: x, y, z, roll, pitch, yaw, gripper_open
        metadata.json

Position is in metres, RPY is in radians, and ``gripper_open`` is binary
(``1 = open``, ``0 = closed``).  Poses are expressed in the selected OpenCV
camera frame (+X right, +Y down, +Z forward).  ``head_view`` is used by
default.

Examples:
    # Convert one episode.
    conda run -n rerun python convert_robotwin_tcp.py \
        4d_datasets/adjust_bottle/episode_0000000

    # Convert every task/episode below 4d_datasets using 8 workers.
    conda run -n rerun python convert_robotwin_tcp.py 4d_datasets --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


ARX5_WORLD_TO_BASE_ROTATION = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
ARX5_WORLD_TO_BASE_TRANSLATION = np.array([0.65, 0.0, 0.0], dtype=np.float64)
ARX5_FINGERTIP_CONTACT_OFFSET_METERS = np.array(
    [0.062765, 0.0, -0.000610], dtype=np.float64
)


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class RobotModel:
    root_link: str
    joints_by_name: dict[str, Joint]
    joints_by_child: dict[str, Joint]


@dataclass(frozen=True)
class ConversionConfig:
    urdf: Path
    camera: str
    gripper_threshold: float
    overwrite: bool


def _vector(
    element: ET.Element | None, attribute: str, default: str
) -> tuple[float, float, float]:
    values = (element.get(attribute, default) if element is not None else default).split()
    if len(values) != 3:
        raise ValueError(f"Expected three values in {attribute!r}, got {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def load_robot_model(urdf_path: str) -> RobotModel:
    """Parse only the URDF fields needed for forward kinematics."""
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    links = {element.get("name") for element in root.findall("link")}
    links.discard(None)
    joints_by_name: dict[str, Joint] = {}
    joints_by_child: dict[str, Joint] = {}

    for element in root.findall("joint"):
        name = element.get("name")
        joint_type = element.get("type")
        parent_element = element.find("parent")
        child_element = element.find("child")
        if not name or not joint_type or parent_element is None or child_element is None:
            raise ValueError(f"Malformed joint in {path}")
        parent = parent_element.get("link")
        child = child_element.get("link")
        if not parent or not child:
            raise ValueError(f"Joint {name} has no parent or child link")

        limit = element.find("limit")
        joint = Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=_vector(element.find("origin"), "xyz", "0 0 0"),
            origin_rpy=_vector(element.find("origin"), "rpy", "0 0 0"),
            axis=_vector(element.find("axis"), "xyz", "1 0 0"),
            lower=(
                float(limit.get("lower"))
                if limit is not None and limit.get("lower")
                else None
            ),
            upper=(
                float(limit.get("upper"))
                if limit is not None and limit.get("upper")
                else None
            ),
        )
        joints_by_name[name] = joint
        if child in joints_by_child:
            raise ValueError(f"URDF link {child} has more than one parent joint")
        joints_by_child[child] = joint

    root_links = sorted(links - set(joints_by_child))
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, found {root_links}")
    return RobotModel(root_links[0], joints_by_name, joints_by_child)


def rotation_matrix_from_rpy(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("A movable URDF joint has a zero-length axis")
    x, y, z = direction / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    complement = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def joint_transform(joint: Joint, value: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy(joint.origin_rpy)
    transform[:3, 3] = joint.origin_xyz

    if joint.lower is not None:
        value = max(value, joint.lower)
    if joint.upper is not None:
        value = min(value, joint.upper)

    motion = np.eye(4, dtype=np.float64)
    if joint.joint_type in ("revolute", "continuous"):
        motion[:3, :3] = rotation_matrix_from_axis_angle(joint.axis, value)
    elif joint.joint_type == "prismatic":
        direction = np.asarray(joint.axis, dtype=np.float64)
        motion[:3, 3] = direction / np.linalg.norm(direction) * value
    elif joint.joint_type != "fixed":
        raise ValueError(f"Unsupported URDF joint type {joint.joint_type!r}")
    return transform @ motion


def joint_chain(model: RobotModel, target_link: str) -> list[Joint]:
    chain: list[Joint] = []
    current = target_link
    visited: set[str] = set()
    while current != model.root_link:
        if current in visited:
            raise ValueError(f"Cycle in URDF chain at {current}")
        visited.add(current)
        joint = model.joints_by_child.get(current)
        if joint is None:
            raise ValueError(f"{target_link} is not connected to URDF root {model.root_link}")
        chain.append(joint)
        current = joint.parent
    return list(reversed(chain))


def link_pose_series(
    model: RobotModel, target_link: str, arm_values: np.ndarray, prefix: str
) -> np.ndarray:
    chain = joint_chain(model, target_link)
    poses = np.empty((len(arm_values), 4, 4), dtype=np.float64)
    movable_columns = {f"{prefix}_joint{index}": index - 1 for index in range(1, 7)}
    for frame_index in range(len(arm_values)):
        pose = np.eye(4, dtype=np.float64)
        for joint in chain:
            column = movable_columns.get(joint.name)
            value = 0.0 if column is None else float(arm_values[frame_index, column])
            pose = pose @ joint_transform(joint, value)
        poses[frame_index] = pose
    return poses


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], horizontal)
    if horizontal > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def homogeneous_extrinsics(path: Path, frame_count: int) -> np.ndarray:
    extrinsics = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if extrinsics.shape == (3, 4) or extrinsics.shape == (4, 4):
        extrinsics = np.broadcast_to(extrinsics, (frame_count, *extrinsics.shape)).copy()
    if extrinsics.shape == (frame_count, 3, 4):
        bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (frame_count, 1, 4))
        extrinsics = np.concatenate((extrinsics, bottom), axis=1)
    if extrinsics.shape != (frame_count, 4, 4):
        raise ValueError(
            f"{path} has shape {extrinsics.shape}; expected [T,3,4] or "
            f"[T,4,4] for T={frame_count}"
        )
    if not np.all(np.isfinite(extrinsics)):
        raise ValueError(f"{path} contains non-finite values")
    if not np.allclose(extrinsics[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{path} has invalid homogeneous bottom rows")
    return extrinsics


def tcp_pose_series(model: RobotModel, robot_state: np.ndarray, side: str) -> np.ndarray:
    state_offset, prefix = (0, "fl") if side == "left" else (7, "fr")
    arm_values = robot_state[:, state_offset : state_offset + 6]
    link6_poses = link_pose_series(model, f"{prefix}_link6", arm_values, prefix)
    finger_origins = np.asarray(
        [
            model.joints_by_name[f"{prefix}_joint{index}"].origin_xyz
            for index in (7, 8)
        ],
        dtype=np.float64,
    )
    link6_to_tcp = np.eye(4, dtype=np.float64)
    link6_to_tcp[:3, 3] = finger_origins.mean(axis=0) + ARX5_FINGERTIP_CONTACT_OFFSET_METERS
    return link6_poses @ link6_to_tcp


def world_to_base_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ARX5_WORLD_TO_BASE_ROTATION
    transform[:3, 3] = ARX5_WORLD_TO_BASE_TRANSLATION
    return transform


def pose_matrices_to_state(poses: np.ndarray, gripper_open: np.ndarray) -> np.ndarray:
    states = np.empty((len(poses), 7), dtype=np.float32)
    states[:, :3] = poses[:, :3, 3]
    states[:, 3:6] = np.asarray(
        [rotation_matrix_to_rpy(rotation) for rotation in poses[:, :3, :3]],
        dtype=np.float32,
    )
    states[:, 6] = gripper_open.astype(np.float32)
    return states


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert_episode(episode: Path, config: ConversionConfig) -> tuple[str, str]:
    output_dir = episode / "TCP"
    left_output = output_dir / "left_state.npy"
    right_output = output_dir / "right_state.npy"
    metadata_output = output_dir / "metadata.json"
    if (
        not config.overwrite
        and left_output.is_file()
        and right_output.is_file()
        and metadata_output.is_file()
    ):
        return str(episode), "skipped"

    robot_state = np.asarray(
        np.load(episode / "robot_state.npy", allow_pickle=False), dtype=np.float64
    )
    if robot_state.ndim != 2 or robot_state.shape[1] != 14:
        raise ValueError(
            f"{episode / 'robot_state.npy'} has shape {robot_state.shape}; "
            "expected [T,14]"
        )
    if len(robot_state) == 0 or not np.all(np.isfinite(robot_state)):
        raise ValueError(f"{episode / 'robot_state.npy'} is empty or contains non-finite values")

    camera_file = episode / "extrinsics" / f"{config.camera}.npy"
    if not camera_file.is_file():
        raise FileNotFoundError(f"Missing camera extrinsics: {camera_file}")
    world_to_camera = homogeneous_extrinsics(camera_file, len(robot_state))

    model = load_robot_model(str(config.urdf))
    base_to_world = np.linalg.inv(world_to_base_transform())
    base_to_world_series = np.broadcast_to(base_to_world, world_to_camera.shape)
    camera_from_base = world_to_camera @ base_to_world_series

    outputs: dict[str, np.ndarray] = {}
    for side, gripper_column in (("left", 6), ("right", 13)):
        base_from_tcp = tcp_pose_series(model, robot_state, side)
        camera_from_tcp = camera_from_base @ base_from_tcp
        gripper_open = robot_state[:, gripper_column] >= config.gripper_threshold
        outputs[side] = pose_matrices_to_state(camera_from_tcp, gripper_open)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(left_output, outputs["left"])
    atomic_save_npy(right_output, outputs["right"])
    atomic_write_json(
        metadata_output,
        {
            "format": "robotwin_tcp_v1",
            "shape": [len(robot_state), 7],
            "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
            "position_unit": "meter",
            "rotation_unit": "radian",
            "rpy_convention": (
                "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))"
            ),
            "coordinate_frame": (
                f"{config.camera} OpenCV camera (+x right, +y down, +z forward)"
            ),
            "camera": config.camera,
            "extrinsics": "world_to_camera, read per frame from extrinsics/<camera>.npy",
            "gripper": {
                "type": "binary",
                "open": 1,
                "closed": 0,
                "source_threshold": config.gripper_threshold,
            },
            "tcp_definition": "midpoint of the two inner fingertip contact faces",
            "source": "robot_state.npy",
        },
    )
    return str(episode), "converted"


def discover_episodes(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if (source / "robot_state.npy").is_file() and (source / "extrinsics").is_dir():
        return [source]
    episodes = sorted(
        path
        for path in source.rglob("episode_*")
        if path.is_dir()
        and (path / "robot_state.npy").is_file()
        and (path / "extrinsics").is_dir()
    )
    if not episodes:
        raise SystemExit(f"No RoboTwin episodes found under {source}")
    return episodes


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin joint states into per-camera TCP xyz/rpy/gripper arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="One episode, one task, or the 4d_datasets root")
    parser.add_argument(
        "--camera", default="head_view", help="Camera name under episode/extrinsics"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=(
            script_root
            / "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        ),
        help="ARX5/Aloha AgileX URDF used for forward kinematics",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Raw gripper values at or above this threshold are saved as open (1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel episode workers; use 1 for sequential processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing complete TCP outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        raise SystemExit(f"URDF not found: {urdf}")
    if args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")
    if not math.isfinite(args.gripper_threshold):
        raise SystemExit("--gripper-threshold must be finite")

    episodes = discover_episodes(args.source)
    config = ConversionConfig(urdf, args.camera, args.gripper_threshold, args.overwrite)
    print(f"Found {len(episodes)} episode(s); camera={args.camera}; workers={args.workers}")

    converted = skipped = failed = 0
    errors: list[tuple[Path, Exception]] = []
    if args.workers == 1 or len(episodes) == 1:
        for index, episode in enumerate(episodes, start=1):
            try:
                _, status = convert_episode(episode, config)
                converted += status == "converted"
                skipped += status == "skipped"
                print(f"[{index}/{len(episodes)}] {status}: {episode}")
            except Exception as error:
                failed += 1
                errors.append((episode, error))
                print(f"[{index}/{len(episodes)}] FAILED: {episode}: {error}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(convert_episode, episode, config): episode
                for episode in episodes
            }
            for index, future in enumerate(as_completed(futures), start=1):
                episode = futures[future]
                try:
                    _, status = future.result()
                    converted += status == "converted"
                    skipped += status == "skipped"
                    print(f"[{index}/{len(episodes)}] {status}: {episode}")
                except Exception as error:
                    failed += 1
                    errors.append((episode, error))
                    print(f"[{index}/{len(episodes)}] FAILED: {episode}: {error}")

    print(f"Done: converted={converted}, skipped={skipped}, failed={failed}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
