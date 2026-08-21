#!/usr/bin/env python3
"""Visualize a local LeRobot v3 dataset with Rerun.

Examples:
    conda run -n rerun python visualize_lerobot_rerun.py \
        --root lerobot_datasets_v3.0/w2_datasets \
        --dataset plug_in_socket_lerobot --episode 0
    conda run -n rerun python visualize_lerobot_rerun.py --list-datasets

When ``--dataset`` is omitted, the script automatically selects the only
compatible dataset, or shows an interactive menu if several are found.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb


TIMELINE = "episode_time"
DEFAULT_RERUN_PORT = 9876
PIPER_ROBOT_TYPE = "agilex_piper_bimanual"
PIPER_STATE_FEATURE = "observation.state"
ROBOT_ENTITY_PATH = "robot"
ROBOT_TRANSFORMS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/joint_transforms"
ROBOT_STATIC_TRANSFORMS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/tf_static"
ROBOT_EEF_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/eef"
EEF_AXIS_LENGTH_METERS = 0.12
ROBOT_FOOTPRINT_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/reference_frames/footprint"
FOOTPRINT_AXIS_LENGTH_METERS = 1
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_WIDTH = 640
CAMERA_BASE_FRAME = "footprint"
CALIBRATED_CAMERAS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/calibrated_cameras"
PIPER_GRIPPER_JOINT_NAMES = {
    *(f"fl_joint{index}" for index in (7, 8)),
    *(f"fr_joint{index}" for index in (7, 8)),
}


@dataclass(frozen=True)
class CameraCalibration:
    """One OpenCV camera calibrated relative to the robot base frame."""

    feature_key: str
    base_to_camera: np.ndarray
    intrinsic: np.ndarray
    height: int
    width: int

    @property
    def camera_name(self) -> str:
        return safe_entity_name(self.feature_key.rsplit(".", 1)[-1])

    @property
    def entity_path(self) -> str:
        return f"{CALIBRATED_CAMERAS_ENTITY_PATH}/{self.camera_name}"

    @property
    def frame_name(self) -> str:
        return f"calibrated_{self.camera_name}"


def default_aloha_urdf(script_root: Path) -> Path:
    return (
        script_root
        / "embodiments"
        / "aloha_new_description"
        / "urdf"
        / "aloha_tracer2_dabai_dark.urdf"
    )


def choose_rerun_port(preferred: int = DEFAULT_RERUN_PORT) -> int:
    """Use the preferred port when available, otherwise ask the OS for a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        try:
            candidate.bind(("127.0.0.1", preferred))
        except OSError:
            pass
        else:
            return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def discover_datasets(root: Path) -> list[Path]:
    """Return immediate child directories that look like LeRobot datasets."""
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").is_file()
    )


def choose_dataset(root: Path, requested: str | None) -> Path:
    datasets = discover_datasets(root)
    if requested:
        requested_path = Path(requested).expanduser()
        candidates = [requested_path, Path.cwd() / requested_path, root / requested_path]
        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "meta" / "info.json").is_file():
                return candidate
        names = ", ".join(path.name for path in datasets) or "none"
        raise SystemExit(f"Dataset not found: {requested!r}. Available datasets: {names}")

    if not datasets:
        raise SystemExit(f"No LeRobot dataset containing meta/info.json found under {root}")
    if len(datasets) == 1:
        print(f"Using dataset: {datasets[0].name}")
        return datasets[0]
    if not sys.stdin.isatty():
        names = ", ".join(path.name for path in datasets)
        raise SystemExit(f"Multiple datasets found; pass --dataset. Available: {names}")

    print("Select a dataset:")
    for index, path in enumerate(datasets, start=1):
        print(f"  {index}. {path.name}")
    while True:
        answer = input("Dataset number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(datasets):
            return datasets[int(answer) - 1]
        print(f"Enter a number from 1 to {len(datasets)}.")


def read_info(dataset: Path) -> dict[str, Any]:
    with (dataset / "meta" / "info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise SystemExit(
            f"{dataset.name} uses LeRobot {info.get('codebase_version', 'unknown')}; "
            "this script currently expects the v3 layout."
        )
    return info


def _read_calibration_matrices(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read the small matrix-only camera schema without requiring PyYAML."""
    cameras: dict[str, dict[str, list[list[float]]]] = {}
    in_cameras = False
    current_camera: str | None = None
    current_matrix: str | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read camera calibration {path}: {error}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        stripped = content.strip()
        indent = len(content) - len(content.lstrip(" "))

        if indent == 0:
            in_cameras = stripped == "cameras:"
            current_camera = None
            current_matrix = None
            continue
        if not in_cameras:
            continue
        if indent == 2 and stripped.endswith(":"):
            current_camera = stripped[:-1].strip().strip("'\"")
            if not current_camera:
                raise ValueError(f"Empty camera key at {path}:{line_number}")
            cameras.setdefault(current_camera, {})
            current_matrix = None
            continue
        if indent == 4 and stripped.endswith(":"):
            matrix_name = stripped[:-1]
            current_matrix = matrix_name if matrix_name in {"extrinsic", "intrinsic"} else None
            if current_camera is not None and current_matrix is not None:
                cameras[current_camera].setdefault(current_matrix, [])
            continue
        if (
            indent >= 4
            and stripped.startswith("- ")
            and current_camera is not None
            and current_matrix is not None
        ):
            try:
                row = ast.literal_eval(stripped[2:].strip())
                values = [float(value) for value in row]
            except (SyntaxError, ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid {current_matrix} row at {path}:{line_number}"
                ) from error
            cameras[current_camera][current_matrix].append(values)

    return {
        camera_key: {
            name: np.asarray(rows, dtype=np.float64)
            for name, rows in matrices.items()
        }
        for camera_key, matrices in cameras.items()
    }


def load_camera_calibration(
    path: Path,
    feature_key: str | None,
    height: int,
    width: int,
) -> CameraCalibration:
    """Load and validate a base-to-camera OpenCV calibration."""
    calibration_path = path.expanduser().resolve()
    cameras = _read_calibration_matrices(calibration_path)
    if feature_key is None:
        if len(cameras) != 1:
            choices = ", ".join(sorted(cameras)) or "none"
            raise ValueError(
                "Camera calibration must contain exactly one camera when "
                f"--camera-feature is omitted; found: {choices}"
            )
        feature_key = next(iter(cameras))
    if feature_key not in cameras:
        choices = ", ".join(sorted(cameras)) or "none"
        raise ValueError(
            f"Camera {feature_key!r} is not in {calibration_path}; available: {choices}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("Camera resolution must contain positive HEIGHT and WIDTH")

    matrices = cameras[feature_key]
    missing = [name for name in ("extrinsic", "intrinsic") if name not in matrices]
    if missing:
        raise ValueError(
            f"Camera {feature_key!r} is missing: {', '.join(missing)}"
        )
    base_to_camera = matrices["extrinsic"]
    intrinsic = matrices["intrinsic"]
    if base_to_camera.shape != (4, 4):
        raise ValueError(
            f"Camera extrinsic must be 4x4, got {base_to_camera.shape}"
        )
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Camera intrinsic must be 3x3, got {intrinsic.shape}")
    if not np.all(np.isfinite(base_to_camera)) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("Camera calibration contains non-finite values")
    if not np.allclose(base_to_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("Camera extrinsic has an invalid homogeneous bottom row")

    rotation = base_to_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Camera extrinsic rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("Camera extrinsic rotation determinant is not +1")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("Camera focal lengths must be positive")
    if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("Camera intrinsic has an invalid homogeneous bottom row")

    return CameraCalibration(
        feature_key=feature_key,
        base_to_camera=base_to_camera,
        intrinsic=intrinsic,
        height=height,
        width=width,
    )


def read_episode(dataset: Path, episode_index: int) -> dict[str, Any]:
    episode_files = sorted((dataset / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise SystemExit(f"No episode metadata parquet files found in {dataset}")

    for path in episode_files:
        table = pq.read_table(path, filters=[("episode_index", "=", episode_index)])
        if table.num_rows:
            return table.slice(0, 1).to_pylist()[0]
    raise SystemExit(f"Episode {episode_index} does not exist in {dataset.name}")


def format_dataset_path(pattern: str, **values: Any) -> Path:
    """Format both LeRobot's named placeholders and common format specs."""
    return Path(pattern.format(**values))


def load_episode_data(
    dataset: Path, info: dict[str, Any], episode: dict[str, Any]
) -> pa.Table:
    relative_path = format_dataset_path(
        info["data_path"],
        chunk_index=episode["data/chunk_index"],
        file_index=episode["data/file_index"],
    )
    data_path = dataset / relative_path
    if not data_path.is_file():
        raise SystemExit(f"Episode data file is missing: {data_path}")
    return pq.read_table(
        data_path,
        filters=[("episode_index", "=", int(episode["episode_index"]))],
    )


def safe_entity_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def flatten_component_names(value: Any) -> list[str]:
    """Flatten both v3 name lists and legacy grouped name dictionaries."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for nested_value in value.values():
            result.extend(flatten_component_names(nested_value))
        return result
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for nested_value in value:
            result.extend(flatten_component_names(nested_value))
        return result
    return []


def feature_component_names(feature: dict[str, Any]) -> list[str]:
    """Return unique, entity-safe names for every component of a vector."""
    shape = feature.get("shape", [])
    if not shape:
        return []

    names = flatten_component_names(feature.get("names"))
    if len(names) != int(shape[0]):
        return []

    unique_names: list[str] = []
    occurrences: dict[str, int] = {}
    for index, name in enumerate(names):
        base_name = safe_entity_name(name) or f"component_{index}"
        occurrences[base_name] = occurrences.get(base_name, 0) + 1
        occurrence = occurrences[base_name]
        unique_names.append(base_name if occurrence == 1 else f"{base_name}_{occurrence}")
    return unique_names


def raw_feature_component_names(feature: dict[str, Any]) -> list[str]:
    """Return the original component names when they match the vector shape."""
    shape = feature.get("shape", [])
    if not shape:
        return []
    names = flatten_component_names(feature.get("names"))
    if len(names) != int(shape[0]):
        return []
    return names


def numeric_vector_features(info: dict[str, Any], table: pa.Table) -> list[str]:
    result: list[str] = []
    for key, feature in info.get("features", {}).items():
        if key not in table.column_names or feature.get("dtype") == "video":
            continue
        if feature_component_names(feature):
            result.append(key)
    return result


def log_signals(
    info: dict[str, Any], table: pa.Table, timestamps: np.ndarray
) -> list[tuple[str, str]]:
    """Log every named numeric vector and return (display name, entity root)."""
    views: list[tuple[str, str]] = []
    time_column = rr.TimeColumn(TIMELINE, duration=timestamps)
    for feature_key in numeric_vector_features(info, table):
        feature = info["features"][feature_key]
        values = np.asarray(table[feature_key].to_pylist(), dtype=np.float64)
        root = f"signals/{safe_entity_name(feature_key)}"
        for column_index, column_name in enumerate(feature_component_names(feature)):
            rr.send_columns(
                f"{root}/{column_name}",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=values[:, column_index]),
            )
        views.append((feature_key, root))
    return views


def extract_video_clip(
    source: Path, destination: Path, start_seconds: float, duration_seconds: float
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required for video visualization but was not found in PATH")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.9f}",
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Failed to extract video clip from {source}") from error


def log_videos(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    temporary_directory: Path,
) -> list[tuple[str, str]]:
    views: list[tuple[str, str]] = []
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    for video_key in video_keys:
        metadata_prefix = f"videos/{video_key}"
        required = [
            f"{metadata_prefix}/chunk_index",
            f"{metadata_prefix}/file_index",
            f"{metadata_prefix}/from_timestamp",
            f"{metadata_prefix}/to_timestamp",
        ]
        if not all(key in episode for key in required):
            print(f"Warning: skipping {video_key}; episode video metadata is incomplete")
            continue

        source = dataset / format_dataset_path(
            info["video_path"],
            video_key=video_key,
            chunk_index=episode[f"{metadata_prefix}/chunk_index"],
            file_index=episode[f"{metadata_prefix}/file_index"],
        )
        if not source.is_file():
            print(f"Warning: skipping missing video: {source}")
            continue

        start = float(episode[f"{metadata_prefix}/from_timestamp"])
        end = float(episode[f"{metadata_prefix}/to_timestamp"])
        camera_name = safe_entity_name(video_key.rsplit(".", 1)[-1])
        clip_path = temporary_directory / f"{camera_name}.mp4"
        extract_video_clip(source, clip_path, start, end - start)

        entity_path = f"cameras/{camera_name}"
        video_asset = rr.AssetVideo(path=clip_path)
        frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
        rr.log(entity_path, video_asset, static=True)
        rr.send_columns(
            entity_path,
            indexes=[
                rr.TimeColumn(
                    TIMELINE,
                    duration=np.asarray(frame_timestamps_ns, dtype=np.float64) * 1e-9,
                )
            ],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )
        views.append((camera_name, entity_path))
    return views


def piper_state_indices(
    info: dict[str, Any], table: pa.Table
) -> tuple[dict[str, int] | None, str | None]:
    """Validate the Piper state schema and return component indexes by name."""
    if info.get("robot_type") != PIPER_ROBOT_TYPE:
        return None, f"robot_type is not {PIPER_ROBOT_TYPE}"
    if PIPER_STATE_FEATURE not in table.column_names:
        return None, f"{PIPER_STATE_FEATURE} is missing from episode data"

    feature = info.get("features", {}).get(PIPER_STATE_FEATURE)
    if not isinstance(feature, dict):
        return None, f"{PIPER_STATE_FEATURE} metadata is missing"
    names = raw_feature_component_names(feature)
    if not names:
        return None, f"{PIPER_STATE_FEATURE} component names do not match its shape"

    required = [
        *(f"left_joint_{index}" for index in range(1, 7)),
        "left_gripper",
        *(f"right_joint_{index}" for index in range(1, 7)),
        "right_gripper",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        return None, "missing state components: " + ", ".join(missing)
    return {name: names.index(name) for name in required}, None


def prepare_follower_visual_urdf(source: Path, destination: Path) -> None:
    """Create a follower-only URDF while preserving the source mesh transforms."""
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as error:
        raise RuntimeError(f"Could not parse URDF {source}: {error}") from error

    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.get("name") not in PIPER_GRIPPER_JOINT_NAMES:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        if origin is None or axis is None:
            continue

        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        direction = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ")
        if rpy.size != 3 or direction.size != 3:
            raise RuntimeError(f"Invalid origin/axis on URDF joint {joint.get('name')}")

        # Rerun 0.35 applies a prismatic axis directly in the parent frame,
        # whereas URDF defines it in the rotated joint frame. Express the axis
        # in the parent frame in this temporary copy so the fingers slide
        # sideways instead of telescoping along the tool axis.
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        parent_direction = rotation @ direction
        axis.set("xyz", " ".join(f"{value:.12g}" for value in parent_direction))

    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        link_name = link.get("name", "")
        if link_name.startswith(("bl_", "br_")):
            for visual in list(link.findall("visual")):
                link.remove(visual)
            continue

        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or not mesh.get("filename", "").lower().endswith(".dae"):
                continue
            # Rerun turns a URDF <material> into one Asset3D albedo factor,
            # which masks every material embedded in a multi-material DAE.
            # ColladaLoader-based viewers instead retain those embedded colors.
            for material in list(visual.findall("material")):
                visual.remove(material)

    # Keep the original DAE references. They contain both the Collada node
    # transforms that assemble each link and the original multi-material look.
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def prepend_ros_package_path(package_root: Path) -> None:
    """Make sibling ROS packages visible to Rerun's package URI resolver."""
    root = str(package_root.resolve())
    existing = [
        path
        for path in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep)
        if path
    ]
    if root not in existing:
        os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join([root, *existing])


def gripper_finger_positions(
    gripper_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Convert a total opening width to the two opposing URDF finger positions."""
    requested_position = np.asarray(gripper_width, dtype=np.float64) * 0.5
    positive_position = np.maximum(requested_position, 0.0)
    was_clipped = not np.allclose(positive_position, requested_position)
    return positive_position, -positive_position, was_clipped


def rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_axis_angle(
    axis: tuple[float, float, float], angle: float
) -> np.ndarray:
    """Return a rotation matrix for a URDF revolute-joint motion."""
    direction = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise RuntimeError("A revolute URDF joint has a zero-length axis")
    x, y, z = direction / norm
    cosine = np.cos(angle)
    sine = np.sin(angle)
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


def urdf_joint_transform(joint: Any, value: float = 0.0) -> np.ndarray:
    """Return one URDF parent-to-child transform at the requested joint value."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy(joint.origin_rpy)
    transform[:3, 3] = np.asarray(joint.origin_xyz, dtype=np.float64)

    motion = np.eye(4, dtype=np.float64)
    if joint.joint_type in ("revolute", "continuous"):
        motion[:3, :3] = rotation_matrix_from_axis_angle(joint.axis, value)
    elif joint.joint_type == "prismatic":
        direction = np.asarray(joint.axis, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise RuntimeError(f"URDF joint {joint.name} has a zero-length axis")
        motion[:3, 3] = direction / norm * value
    elif joint.joint_type != "fixed":
        raise RuntimeError(f"Unsupported URDF joint type {joint.joint_type!r}")
    return transform @ motion


def urdf_joint_chain(urdf_tree: Any, target_link: str) -> list[Any]:
    """Return the ordered joint chain from the URDF root to ``target_link``."""
    joints_by_child = {joint.child_link: joint for joint in urdf_tree.joints()}
    root_link = urdf_tree.root_link().name
    current_link = target_link
    reversed_chain: list[Any] = []
    visited: set[str] = set()
    while current_link != root_link:
        if current_link in visited:
            raise RuntimeError(f"Cycle found in the URDF at link {current_link}")
        visited.add(current_link)
        joint = joints_by_child.get(current_link)
        if joint is None:
            raise RuntimeError(
                f"URDF link {target_link} is not connected to root link {root_link}"
            )
        reversed_chain.append(joint)
        current_link = joint.parent_link
    return list(reversed(reversed_chain))


def clamp_urdf_joint_value(joint: Any, value: float) -> float:
    """Match the clamped joint motion used by the visible Rerun robot."""
    lower = joint.limit_lower
    upper = joint.limit_upper
    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return value


def compute_link_pose_series(
    urdf_tree: Any,
    target_link: str,
    joint_values: dict[str, np.ndarray],
    frame_count: int,
) -> np.ndarray:
    """Compute root-frame poses for a URDF link over an episode."""
    chain = urdf_joint_chain(urdf_tree, target_link)
    poses = np.empty((frame_count, 4, 4), dtype=np.float64)
    for frame_index in range(frame_count):
        pose = np.eye(4, dtype=np.float64)
        for joint in chain:
            values = joint_values.get(joint.name)
            value = 0.0 if values is None else float(values[frame_index])
            value = clamp_urdf_joint_value(joint, value)
            pose = pose @ urdf_joint_transform(joint, value)
        poses[frame_index] = pose
    return poses


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to URDF-style roll/pitch/yaw radians."""
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], horizontal)
    if horizontal > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def log_robot_footprint_frame() -> None:
    """Make the root frame used by the EEF pose labels visible in the 3D view."""
    axis_length = FOOTPRINT_AXIS_LENGTH_METERS
    rr.log(
        ROBOT_FOOTPRINT_ENTITY_PATH,
        rr.CoordinateFrame("footprint"),
        static=True,
    )
    rr.log(
        ROBOT_FOOTPRINT_ENTITY_PATH,
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * axis_length,
            radii=[0.006] * 3,
            colors=[
                [255, 60, 60],
                [60, 220, 80],
                [70, 130, 255],
            ],
            labels=["X", "Y", "Z"],
            show_labels=True,
        ),
        static=True,
    )
    rr.log(
        ROBOT_FOOTPRINT_ENTITY_PATH,
        rr.Points3D(
            [[0.0, 0.0, 0.0]],
            radii=[0.018],
            colors=[[255, 255, 255]],
            labels=["footprint\nEEF xyz/rpy reference"],
            show_labels=True,
        ),
        static=True,
    )


def log_calibrated_camera(calibration: CameraCalibration) -> None:
    """Log a calibrated OpenCV pinhole attached to the robot root frame."""
    # The YAML extrinsic maps base-frame points into OpenCV camera coordinates:
    #     p_camera = T_camera_base @ p_base
    # Rerun's ParentFromChild pose instead locates the camera in the base frame.
    camera_to_base = np.linalg.inv(calibration.base_to_camera)
    rr.log(
        calibration.entity_path,
        # Rerun 0.35 needs this explicit association when a named-frame
        # pinhole entity is used as the origin of a Spatial2DView.
        rr.CoordinateFrame(calibration.frame_name),
        rr.Transform3D(
            translation=camera_to_base[:3, 3],
            mat3x3=camera_to_base[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            parent_frame=CAMERA_BASE_FRAME,
            child_frame=calibration.frame_name,
        ),
        rr.Pinhole(
            image_from_camera=calibration.intrinsic,
            resolution=[calibration.width, calibration.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.15,
            parent_frame=CAMERA_BASE_FRAME,
            child_frame=calibration.frame_name,
        ),
        static=True,
    )

    position = camera_to_base[:3, 3]
    vertical_fov = np.rad2deg(
        2.0
        * np.arctan(
            calibration.height / (2.0 * float(calibration.intrinsic[1, 1]))
        )
    )
    print(
        f"Loaded calibrated camera: {calibration.feature_key} "
        f"({calibration.height}x{calibration.width} HxW, "
        f"position [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m, "
        f"vertical FOV {vertical_fov:.2f} deg)"
    )


def log_piper_eef_poses(
    urdf_tree: Any,
    state_values: np.ndarray,
    state_indices: dict[str, int],
    timestamps: np.ndarray,
) -> None:
    """Log animated left/right EEF axes and root-frame pose labels."""
    time_column = rr.TimeColumn(TIMELINE, duration=timestamps)
    frame_count = len(timestamps)
    for dataset_side, urdf_prefix, color in (
        ("left", "fl", [80, 200, 255]),
        ("right", "fr", [255, 170, 70]),
    ):
        joint_values = {
            f"{urdf_prefix}_joint{joint_index}": state_values[
                :, state_indices[f"{dataset_side}_joint_{joint_index}"]
            ]
            for joint_index in range(1, 7)
        }
        poses = compute_link_pose_series(
            urdf_tree,
            f"{urdf_prefix}_link6",
            joint_values,
            frame_count,
        )

        # Both finger joints originate at the gripper center. Use their common
        # origin as the EEF position while retaining the link6 orientation.
        finger_origins = np.asarray(
            [
                urdf_tree.get_joint_by_name(f"{urdf_prefix}_joint{joint_index}").origin_xyz
                for joint_index in (7, 8)
            ],
            dtype=np.float64,
        )
        if not np.allclose(finger_origins[0], finger_origins[1]):
            raise RuntimeError(
                f"{dataset_side} gripper finger origins do not share an EEF center"
            )
        link_to_eef = np.eye(4, dtype=np.float64)
        link_to_eef[:3, 3] = finger_origins[0]
        poses = poses @ link_to_eef

        translations = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        rpy_degrees = np.rad2deg(
            np.asarray([rotation_matrix_to_rpy(rotation) for rotation in rotations])
        )
        side_label = "L" if dataset_side == "left" else "R"
        labels = [
            (
                f"{side_label} EEF  xyz[m] "
                f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}\n"
                f"rpy[deg] {angles[0]:+.1f} {angles[1]:+.1f} {angles[2]:+.1f}"
            )
            for position, angles in zip(translations, rpy_degrees)
        ]

        entity_path = f"{ROBOT_EEF_ENTITY_PATH}/{dataset_side}"
        eef_frame = f"{urdf_prefix}_eef"
        rr.log(entity_path, rr.CoordinateFrame(eef_frame), static=True)
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=finger_origins[0],
                parent_frame=f"{urdf_prefix}_link6",
                child_frame=eef_frame,
            ),
            static=True,
        )
        rr.log(entity_path, rr.TransformAxes3D(EEF_AXIS_LENGTH_METERS), static=True)
        rr.log(
            entity_path,
            rr.Points3D(
                [[0.0, 0.0, 0.0]],
                radii=[0.012],
                colors=[color],
                show_labels=True,
            ),
            static=True,
        )
        rr.send_columns(
            entity_path,
            indexes=[time_column],
            columns=rr.Points3D.columns(
                positions=np.zeros((frame_count, 3), dtype=np.float32),
                labels=labels,
            ).partition(lengths=[1] * frame_count),
        )


def log_piper_robot_replay(
    table: pa.Table,
    timestamps: np.ndarray,
    state_indices: dict[str, int],
    urdf_path: Path,
    temporary_directory: Path,
    recording: rr.RecordingStream,
    package_root: Path,
) -> None:
    """Log follower geometry and animated Piper joint transforms."""
    prepend_ros_package_path(package_root)
    prepared_urdf = temporary_directory / "aloha_follower_visual.urdf"
    prepare_follower_visual_urdf(urdf_path, prepared_urdf)

    try:
        urdf_tree = rr.urdf.UrdfTree.from_file_path(
            prepared_urdf,
            entity_path_prefix=ROBOT_ENTITY_PATH,
            static_transform_entity_path=ROBOT_STATIC_TRANSFORMS_ENTITY_PATH,
        )
    except Exception as error:
        raise RuntimeError(f"Rerun could not load URDF {urdf_path}: {error}") from error

    joint_names = [
        *(f"fl_joint{index}" for index in range(1, 9)),
        *(f"fr_joint{index}" for index in range(1, 9)),
    ]
    missing_joints = [
        name for name in joint_names if urdf_tree.get_joint_by_name(name) is None
    ]
    if missing_joints:
        raise RuntimeError("URDF is missing replay joints: " + ", ".join(missing_joints))

    try:
        urdf_tree.log_urdf_to_recording(recording)
    except Exception as error:
        raise RuntimeError(f"Rerun could not log URDF {urdf_path}: {error}") from error

    log_robot_footprint_frame()

    state_values = np.asarray(table[PIPER_STATE_FEATURE].to_pylist(), dtype=np.float64)
    if state_values.ndim != 2 or state_values.shape[0] != len(timestamps):
        raise RuntimeError(
            f"Unexpected {PIPER_STATE_FEATURE} shape {state_values.shape}; "
            f"expected ({len(timestamps)}, components)"
        )
    if not np.all(np.isfinite(state_values)):
        raise RuntimeError(f"{PIPER_STATE_FEATURE} contains non-finite values")

    time_column = rr.TimeColumn(TIMELINE, duration=timestamps)
    gripper_was_clipped = False
    for dataset_side, urdf_prefix in (("left", "fl"), ("right", "fr")):
        for joint_index in range(1, 7):
            component = f"{dataset_side}_joint_{joint_index}"
            joint = urdf_tree.get_joint_by_name(f"{urdf_prefix}_joint{joint_index}")
            assert joint is not None
            rr.send_columns(
                ROBOT_TRANSFORMS_ENTITY_PATH,
                indexes=[time_column],
                columns=joint.compute_transform_columns(
                    state_values[:, state_indices[component]], clamp=True
                ),
            )

        gripper_width = state_values[:, state_indices[f"{dataset_side}_gripper"]]
        positive_position, negative_position, was_clipped = gripper_finger_positions(
            gripper_width
        )
        gripper_was_clipped = gripper_was_clipped or was_clipped
        for joint_index, values in (
            (7, positive_position),
            (8, negative_position),
        ):
            joint = urdf_tree.get_joint_by_name(f"{urdf_prefix}_joint{joint_index}")
            assert joint is not None
            rr.send_columns(
                ROBOT_TRANSFORMS_ENTITY_PATH,
                indexes=[time_column],
                # The state stores the complete finger-to-finger opening in meters.
                # Do not clamp each half to the narrower limits in this visual URDF.
                columns=joint.compute_transform_columns(values, clamp=False),
            )

    if gripper_was_clipped:
        print(
            "Warning: negative gripper widths were clipped to a closed (0 m) gripper"
        )

    log_piper_eef_poses(urdf_tree, state_values, state_indices, timestamps)


def maybe_log_robot_replay(
    args: argparse.Namespace,
    info: dict[str, Any],
    table: pa.Table,
    timestamps: np.ndarray,
    temporary_directory: Path,
    recording: rr.RecordingStream,
    script_root: Path,
) -> bool:
    """Log an automatically detected or explicitly requested robot replay."""
    if args.no_robot:
        return False

    explicit_urdf = args.urdf is not None
    state_indices, incompatibility = piper_state_indices(info, table)
    if incompatibility is not None or state_indices is None:
        message = f"robot replay unavailable: {incompatibility}"
        if explicit_urdf:
            raise SystemExit(message)
        print(f"Warning: {message}")
        return False

    urdf_path = (
        args.urdf.expanduser().resolve()
        if explicit_urdf
        else default_aloha_urdf(script_root).resolve()
    )
    if not urdf_path.is_file():
        message = f"robot replay URDF not found: {urdf_path}"
        if explicit_urdf:
            raise SystemExit(message)
        print(f"Warning: {message}")
        return False

    package_root = script_root / "embodiments"
    try:
        log_piper_robot_replay(
            table,
            timestamps,
            state_indices,
            urdf_path,
            temporary_directory,
            recording,
            package_root,
        )
    except RuntimeError as error:
        if explicit_urdf:
            raise SystemExit(str(error)) from error
        print(f"Warning: robot replay unavailable: {error}")
        return False

    print(f"Loaded robot replay: {urdf_path}")
    return True


def make_blueprint(
    video_views: list[tuple[str, str]],
    signal_views: list[tuple[str, str]],
    robot_replay: bool = False,
    calibrated_camera: CameraCalibration | None = None,
) -> rrb.Blueprint:
    camera_views: list[Any] = [
        rrb.Spatial2DView(origin=path, name=name) for name, path in video_views
    ]
    if robot_replay:
        robot_view = rrb.Spatial3DView(origin=ROBOT_ENTITY_PATH, name="Robot replay")
        robot_area: Any = robot_view
        if calibrated_camera is not None:
            calibrated_view = rrb.Spatial2DView(
                # A pinhole at the 2D view origin projects all included 3D
                # robot geometry using the logged intrinsics and extrinsics.
                origin=calibrated_camera.entity_path,
                # Do not include /cameras/** here: those video entities use
                # implicit path frames and are intentionally disconnected
                # from the robot's named ``footprint`` transform tree.
                contents=[f"/{ROBOT_ENTITY_PATH}/**"],
                name=(
                    f"Main camera replay "
                    f"({calibrated_camera.width}x{calibrated_camera.height})"
                ),
                visual_bounds=rrb.VisualBounds2D(
                    x_range=[0.0, float(calibrated_camera.width)],
                    y_range=[0.0, float(calibrated_camera.height)],
                ),
            )
            robot_area = rrb.Tabs(
                calibrated_view,
                robot_view,
                active_tab=0,
                name="Robot replay",
            )
        if camera_views:
            top_area: Any = rrb.Horizontal(
                robot_area,
                rrb.Grid(*camera_views, name="Cameras"),
                column_shares=[1, 2],
                name="Replay",
            )
        else:
            top_area = robot_area
    elif camera_views:
        top_area = rrb.Horizontal(*camera_views, name="Cameras")
    else:
        top_area = rrb.TextDocumentView(origin="episode_info", name="Episode")

    plots: list[Any] = [
        rrb.TimeSeriesView(origin=path, name=name) for name, path in signal_views
    ]
    plot_area: Any
    if plots:
        plot_area = rrb.Tabs(*plots, active_tab=0, name="Signals")
    else:
        plot_area = rrb.TextDocumentView(origin="episode_info", name="Episode")

    lower = rrb.Horizontal(
        rrb.TextDocumentView(origin="episode_info", name="Episode info"),
        plot_area,
        column_shares=[1, 4],
    )
    layout = rrb.Vertical(
        top_area,
        lower,
        row_shares=[2, 1],
    )
    return rrb.Blueprint(layout, auto_views=False, collapse_panels=True)


def log_episode_info(
    dataset: Path, info: dict[str, Any], episode: dict[str, Any], frame_count: int
) -> None:
    tasks = episode.get("tasks") or []
    task_text = ", ".join(str(task) for task in tasks) or "(unknown)"
    duration = frame_count / float(info["fps"])
    document = (
        f"# {dataset.name}\n\n"
        f"- **Episode:** {episode['episode_index']}\n"
        f"- **Task:** {task_text}\n"
        f"- **Frames:** {frame_count}\n"
        f"- **FPS:** {info['fps']}\n"
        f"- **Duration:** {duration:.2f} s\n"
        f"- **Robot:** {info.get('robot_type', 'unknown')}\n"
    )
    rr.log("episode_info", rr.TextDocument(document, media_type=rr.MediaType.MARKDOWN), static=True)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Visualize a local LeRobot v3 episode using Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="Directory containing one or more datasets",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset directory name or path; prompts when omitted and several exist",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--no-video", action="store_true", help="Only load numeric signals"
    )
    robot_group = parser.add_mutually_exclusive_group()
    robot_group.add_argument(
        "--urdf",
        type=Path,
        help=(
            "Use this URDF for an eligible Piper replay instead of the default "
            "embodiments/aloha_new_description model"
        ),
    )
    robot_group.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable automatic URDF replay for compatible Piper datasets",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help=(
            "YAML calibration whose extrinsic maps robot-base points into the "
            "OpenCV camera frame; adds a tracked camera-view robot replay"
        ),
    )
    parser.add_argument(
        "--camera-feature",
        help=(
            "Camera feature key inside --camera-calibration; inferred when the "
            "YAML contains exactly one camera"
        ),
    )
    parser.add_argument(
        "--camera-resolution",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH),
        help="Calibrated replay resolution in HxW order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write an .rrd recording instead of opening the Rerun viewer",
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List detected datasets and exit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    root = args.root.expanduser().resolve()
    if args.list_datasets:
        datasets = discover_datasets(root)
        if datasets:
            for dataset in datasets:
                print(dataset.name)
        else:
            print(f"No datasets found under {root}")
        return

    dataset = choose_dataset(root, args.dataset)
    info = read_info(dataset)
    if args.episode < 0 or args.episode >= int(info.get("total_episodes", 0)):
        raise SystemExit(
            f"Episode must be between 0 and {int(info.get('total_episodes', 0)) - 1}"
        )
    episode = read_episode(dataset, args.episode)
    table = load_episode_data(dataset, info, episode)
    if not table.num_rows:
        raise SystemExit(f"Episode {args.episode} contains no frames")

    calibrated_camera: CameraCalibration | None = None
    if args.camera_calibration is not None:
        if args.no_robot:
            raise SystemExit("--camera-calibration cannot be used with --no-robot")
        try:
            calibrated_camera = load_camera_calibration(
                args.camera_calibration,
                args.camera_feature,
                args.camera_resolution[0],
                args.camera_resolution[1],
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    timestamps -= timestamps[0]

    app_id = f"lerobot_{safe_entity_name(dataset.name)}"
    rr.init(app_id, spawn=False)
    recording = rr.get_global_data_recording()
    if recording is None:
        raise SystemExit("Rerun recording failed to initialize")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output)
    else:
        rerun_port = choose_rerun_port()
        if rerun_port != DEFAULT_RERUN_PORT:
            print(
                f"Rerun port {DEFAULT_RERUN_PORT} is occupied; "
                f"using free port {rerun_port} instead."
            )
        recording.spawn(port=rerun_port)

    print(
        f"Loading {dataset.name}, episode {args.episode} "
        f"({table.num_rows} frames at {info['fps']} FPS)..."
    )
    log_episode_info(dataset, info, episode, table.num_rows)
    signal_views = log_signals(info, table, timestamps)

    with tempfile.TemporaryDirectory(prefix="lerobot-rerun-") as temporary:
        temporary_directory = Path(temporary)
        video_views: list[tuple[str, str]] = []
        if not args.no_video:
            video_views = log_videos(dataset, info, episode, temporary_directory)
        robot_replay = maybe_log_robot_replay(
            args,
            info,
            table,
            timestamps,
            temporary_directory,
            recording,
            script_root,
        )
        if calibrated_camera is not None:
            if not robot_replay:
                raise SystemExit(
                    "calibrated camera replay requires a compatible robot replay"
                )
            log_calibrated_camera(calibrated_camera)
        rr.send_blueprint(
            make_blueprint(
                video_views,
                signal_views,
                robot_replay,
                calibrated_camera,
            )
        )
        recording.flush()

    if args.output:
        print(f"Saved Rerun recording: {args.output.expanduser().resolve()}")
    else:
        print("Episode loaded in Rerun. Use the episode_time timeline to scrub or play.")


if __name__ == "__main__":
    main()
