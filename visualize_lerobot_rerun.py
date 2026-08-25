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
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageStat


TIMELINE = "episode_time"
DEFAULT_RERUN_PORT = 9876
PIPER_ROBOT_TYPE = "agilex_piper_bimanual"
ARX5_ROBOT_TYPE = "unified_robot"
PIPER_STATE_FEATURE = "observation.state"
ROBOT_ENTITY_PATH = "robot"
ROBOT_TRANSFORMS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/joint_transforms"
ROBOT_STATIC_TRANSFORMS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/tf_static"
ROBOT_EEF_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/eef"
ROBOT_LINK6_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/link6"
EEF_AXIS_LENGTH_METERS = 0.12
ROBOT_FOOTPRINT_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/reference_frames/footprint"
FOOTPRINT_AXIS_LENGTH_METERS = 1
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_WIDTH = 640
CAMERA_BASE_FRAME = "footprint"
CALIBRATED_CAMERAS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/calibrated_cameras"
POINT_CLOUDS_ENTITY_PATH = f"{ROBOT_ENTITY_PATH}/scene_point_cloud"
DEFAULT_POINT_CLOUD_STRIDE = 2
DEPTH_QUANTIZATION_MAX = 4095
ARX5_WORLD_TO_BASE_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
ARX5_WORLD_TO_BASE_TRANSLATION = np.array(
    [0.65, 0.0, 0.0],
    dtype=np.float64,
)
PIPER_GRIPPER_JOINT_NAMES = {
    *(f"fl_joint{index}" for index in (7, 8)),
    *(f"fr_joint{index}" for index in (7, 8)),
}
# Piper gripper: joint7 gets +width/2, joint8 gets -width/2 (opposing limits).
# Arx5 gripper: both fingers get +width/2 (both limits are [0, upper]).
PIPER_GRIPPER_SIGNS = (1.0, -1.0)
ARX5_GRIPPER_SIGNS = (1.0, 1.0)
# Geometric centroid of the inner fingertip contact face in the Arx5
# link7/link8 mesh, expressed from either finger joint origin.  The two
# fingers translate symmetrically, so their contact-center midpoint keeps this
# fixed offset from the midpoint of the two joint origins.
ARX5_FINGERTIP_CONTACT_OFFSET_METERS = (0.062765, 0.0, -0.000610)

# Arx5 always uses the complete robot exterior. Camera DAEs with equivalent
# lightweight meshes are replaced to keep Rerun responsive. box2_Link keeps
# its source DAE because its Collada node hierarchy assembles the upper chassis
# and is not equivalent to the footprint-framed collision STL.
ARX5_FULL_MESH_OVERRIDES = {
    # These camera DAEs are roughly 40 MB each and have equivalent STLs.
    "camera_link1": "aloha_maniskill_sim/meshes/camera_link1.STL",
    "camera_link2": "aloha_maniskill_sim/meshes/camera_link2.STL",
}
ARX5_FORCE_SOLID_TEXTURE_LINKS = frozenset(
    {"box2_Link", "left_camera", "right_camera"}
)
ARX5_FULL_MATERIAL_OVERRIDES = {
    "camera_base_link": (0.18, 0.18, 0.20, 1.0),
    "camera_link1": (0.08, 0.08, 0.09, 1.0),
    "camera_link2": (0.08, 0.08, 0.09, 1.0),
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


@dataclass(frozen=True)
class PointCloudCamera:
    """One paired RGB-D stream with per-frame OpenCV calibration."""

    camera_name: str
    rgb_key: str
    depth_key: str
    intrinsic_key: str
    camera_pose_key: str | None
    extrinsic_key: str | None
    height: int
    width: int
    depth_min: float
    depth_max: float
    depth_shift: float
    depth_use_log: bool
    depth_unit: str
    invalid_value: float

    @property
    def entity_path(self) -> str:
        return f"{POINT_CLOUDS_ENTITY_PATH}/{safe_entity_name(self.camera_name)}"

    @property
    def frame_name(self) -> str:
        return f"point_cloud_{safe_entity_name(self.camera_name)}"


def default_aloha_urdf(script_root: Path) -> Path:
    return (
        script_root
        / "embodiments"
        / "aloha_new_description"
        / "urdf"
        / "aloha_tracer2_dabai_dark.urdf"
    )


def default_arx5_urdf(script_root: Path) -> Path:
    return (
        script_root
        / "embodiments"
        / "aloha-agilex"
        / "urdf"
        / "arx5_description_isaac.urdf"
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


def discover_point_cloud_cameras(
    info: dict[str, Any], table: pa.Table
) -> tuple[list[PointCloudCamera], list[str]]:
    """Discover paired RGB-D streams with per-frame calibration columns."""
    features = info.get("features", {})
    if not isinstance(features, dict):
        return [], ["dataset feature metadata is not a dictionary"]

    cameras: list[PointCloudCamera] = []
    issues: list[str] = []
    for depth_key, depth_feature in features.items():
        if not isinstance(depth_feature, dict) or depth_feature.get("dtype") != "video":
            continue
        depth_info = depth_feature.get("info") or {}
        if not isinstance(depth_info, dict):
            continue
        is_depth = bool(
            depth_info.get("is_depth_map", depth_info.get("video.is_depth_map", False))
        )
        if not is_depth:
            continue
        if not depth_key.endswith("_depth"):
            issues.append(
                f"skipping {depth_key}: depth feature name does not end in '_depth'"
            )
            continue

        rgb_key = depth_key[: -len("_depth")]
        rgb_feature = features.get(rgb_key)
        if not isinstance(rgb_feature, dict) or rgb_feature.get("dtype") != "video":
            issues.append(f"skipping {depth_key}: paired RGB feature {rgb_key} is missing")
            continue

        try:
            depth_shape = tuple(int(value) for value in depth_feature.get("shape", ()))
            rgb_shape = tuple(int(value) for value in rgb_feature.get("shape", ()))
        except (TypeError, ValueError):
            issues.append(f"skipping {depth_key}: RGB-D feature shapes are invalid")
            continue
        if len(depth_shape) != 3 or depth_shape[2] != 1:
            issues.append(
                f"skipping {depth_key}: expected depth shape (H, W, 1), got {depth_shape}"
            )
            continue
        if rgb_shape != (depth_shape[0], depth_shape[1], 3):
            issues.append(
                f"skipping {depth_key}: paired RGB shape {rgb_shape} does not match "
                f"{(depth_shape[0], depth_shape[1], 3)}"
            )
            continue

        camera_name = rgb_key.rsplit(".", 1)[-1]
        calibration_prefix = f"calibration.{camera_name}"
        intrinsic_key = f"{calibration_prefix}.intrinsic_matrix"
        pose_candidate = f"{calibration_prefix}.camera_pose_matrix"
        extrinsic_candidate = f"{calibration_prefix}.extrinsic_matrix"
        if intrinsic_key not in features or intrinsic_key not in table.column_names:
            issues.append(f"skipping {depth_key}: missing {intrinsic_key}")
            continue
        camera_pose_key = (
            pose_candidate
            if pose_candidate in features and pose_candidate in table.column_names
            else None
        )
        extrinsic_key = (
            extrinsic_candidate
            if extrinsic_candidate in features and extrinsic_candidate in table.column_names
            else None
        )
        if camera_pose_key is None and extrinsic_key is None:
            issues.append(
                f"skipping {depth_key}: missing both {pose_candidate} and "
                f"{extrinsic_candidate}"
            )
            continue

        def depth_parameter(name: str) -> Any:
            return depth_info.get(f"video.{name}", depth_info.get(name))

        raw_parameters = {
            name: depth_parameter(name)
            for name in ("depth_min", "depth_max", "shift", "use_log")
        }
        missing_parameters = [
            name for name, value in raw_parameters.items() if value is None
        ]
        if missing_parameters:
            issues.append(
                f"skipping {depth_key}: missing depth quantization metadata "
                + ", ".join(missing_parameters)
            )
            continue
        if depth_info.get("invalid_value") is None or depth_info.get("depth_unit") is None:
            issues.append(
                f"skipping {depth_key}: invalid_value and depth_unit metadata are required"
            )
            continue

        try:
            depth_min = float(raw_parameters["depth_min"])
            depth_max = float(raw_parameters["depth_max"])
            depth_shift = float(raw_parameters["shift"])
            invalid_value = float(depth_info["invalid_value"])
        except (TypeError, ValueError):
            issues.append(f"skipping {depth_key}: depth metadata must be numeric")
            continue
        depth_use_log = bool(raw_parameters["use_log"])
        depth_unit = str(depth_info["depth_unit"]).lower()
        if depth_unit not in {"m", "mm"}:
            issues.append(
                f"skipping {depth_key}: unsupported depth unit {depth_unit!r}"
            )
            continue
        if not np.all(
            np.isfinite([depth_min, depth_max, depth_shift, invalid_value])
        ):
            issues.append(f"skipping {depth_key}: depth metadata is not finite")
            continue
        if depth_max <= depth_min:
            issues.append(f"skipping {depth_key}: depth_max must exceed depth_min")
            continue
        if depth_use_log and depth_min + depth_shift <= 0.0:
            issues.append(
                f"skipping {depth_key}: depth_min + shift must be positive in log mode"
            )
            continue

        cameras.append(
            PointCloudCamera(
                camera_name=camera_name,
                rgb_key=rgb_key,
                depth_key=depth_key,
                intrinsic_key=intrinsic_key,
                camera_pose_key=camera_pose_key,
                extrinsic_key=extrinsic_key,
                height=depth_shape[0],
                width=depth_shape[1],
                depth_min=depth_min,
                depth_max=depth_max,
                depth_shift=depth_shift,
                depth_use_log=depth_use_log,
                depth_unit=depth_unit,
                invalid_value=invalid_value,
            )
        )
    return cameras, issues


def dequantize_depth_codes(
    quantized: np.ndarray,
    depth_min: float,
    depth_max: float,
    shift: float,
    use_log: bool,
) -> np.ndarray:
    """Convert LeRobot 12-bit depth codes into float32 metres."""
    codes = np.asarray(quantized, dtype=np.float32)
    normalized = codes / np.float32(DEPTH_QUANTIZATION_MAX)
    if use_log:
        if depth_min + shift <= 0.0:
            raise ValueError("depth_min + shift must be positive in log mode")
        log_min = np.log(float(depth_min + shift))
        log_max = np.log(float(depth_max + shift))
        depth = np.exp(normalized * (log_max - log_min) + log_min) - shift
    else:
        depth = normalized * (depth_max - depth_min) + depth_min
    return np.clip(depth, depth_min, depth_max).astype(np.float32, copy=False)


def quantized_code_for_depth_value(
    value: float,
    unit: str,
    depth_min: float,
    depth_max: float,
    shift: float,
    use_log: bool,
) -> int:
    """Map a physical invalid-depth sentinel to its stored 12-bit code."""
    value_metres = float(value) * (0.001 if unit == "mm" else 1.0)
    if use_log:
        if depth_min + shift <= 0.0:
            raise ValueError("depth_min + shift must be positive in log mode")
        if value_metres + shift <= 0.0:
            normalized = 0.0
        else:
            normalized = (
                np.log(value_metres + shift) - np.log(depth_min + shift)
            ) / (np.log(depth_max + shift) - np.log(depth_min + shift))
    else:
        normalized = (value_metres - depth_min) / (depth_max - depth_min)
    return int(
        np.rint(np.clip(normalized, 0.0, 1.0) * DEPTH_QUANTIZATION_MAX)
    )


def backproject_rgbd_to_world(
    depth_metres: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    stride: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project aligned RGB-D pixels and transform them into base/world."""
    if stride <= 0:
        raise ValueError("point-cloud stride must be positive")
    depth = np.asarray(depth_metres, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {depth.shape}")
    if colors.shape != (*depth.shape, 3):
        raise ValueError(
            f"RGB shape {colors.shape} does not match depth shape {depth.shape}"
        )
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("point-cloud calibration must contain 3x3 and 4x4 matrices")

    rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
    columns = np.arange(0, depth.shape[1], stride, dtype=np.float32)
    pixel_u, pixel_v = np.meshgrid(columns, rows)
    sampled_depth = depth[::stride, ::stride]
    sampled_colors = colors[::stride, ::stride]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(
                f"valid-mask shape {mask.shape} does not match depth shape {depth.shape}"
            )
        valid &= mask[::stride, ::stride]

    z = sampled_depth[valid]
    x = (pixel_u[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (pixel_v[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.column_stack((x, y, z))
    rotation = camera_to_world[:3, :3]
    world_points = (
        camera_points[:, 0, None] * rotation[None, :, 0]
        + camera_points[:, 1, None] * rotation[None, :, 1]
        + camera_points[:, 2, None] * rotation[None, :, 2]
        + camera_to_world[:3, 3]
    )
    return (
        np.asarray(world_points, dtype=np.float32),
        np.ascontiguousarray(sampled_colors[valid]),
    )


def point_cloud_world_to_base_transform(info: dict[str, Any]) -> np.ndarray:
    """Return the dataset-world to robot-footprint transform.

    RoboTwin's ``unified_robot`` frame has +Y where the Arx5 URDF footprint
    has +X, and its origin is 0.65 m in front of the footprint origin. This
    fixed transform is independently recovered by matching both wrist-camera
    positions to their URDF forward-kinematics positions over the episode.
    """
    transform = np.eye(4, dtype=np.float64)
    if info.get("robot_type") == ARX5_ROBOT_TYPE:
        transform[:3, :3] = ARX5_WORLD_TO_BASE_ROTATION
        transform[:3, 3] = ARX5_WORLD_TO_BASE_TRANSLATION
    return transform


def _matrix_series(
    table: pa.Table, feature_key: str, matrix_shape: tuple[int, int]
) -> np.ndarray:
    values = np.asarray(table[feature_key].to_pylist(), dtype=np.float64)
    expected_shape = (table.num_rows, *matrix_shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"{feature_key} has shape {values.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{feature_key} contains non-finite values")
    return values


def point_cloud_calibration_series(
    camera: PointCloudCamera, table: pa.Table
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate per-frame intrinsics and camera-to-world poses."""
    intrinsics = _matrix_series(table, camera.intrinsic_key, (3, 3))
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError(f"{camera.intrinsic_key} contains non-positive focal lengths")
    if not np.allclose(intrinsics[:, 2, :], [0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{camera.intrinsic_key} has invalid homogeneous rows")

    if camera.camera_pose_key is not None:
        camera_to_world = _matrix_series(table, camera.camera_pose_key, (4, 4))
    else:
        assert camera.extrinsic_key is not None
        world_to_camera = _matrix_series(table, camera.extrinsic_key, (4, 4))
        try:
            camera_to_world = np.linalg.inv(world_to_camera)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{camera.extrinsic_key} contains a singular matrix") from error

    if not np.allclose(camera_to_world[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"camera pose for {camera.camera_name} has invalid bottom rows")
    rotations = camera_to_world[:, :3, :3]
    rotation_products = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
    if not np.allclose(rotation_products, np.eye(3), atol=1e-4):
        raise ValueError(f"camera pose for {camera.camera_name} is not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
        raise ValueError(f"camera pose for {camera.camera_name} has invalid determinants")
    return intrinsics, camera_to_world


class FFmpegRawVideoReader:
    """Stream a fixed number of raw frames from one episode video segment."""

    def __init__(
        self,
        source: Path,
        start_seconds: float,
        frame_count: int,
        pixel_format: str,
        frame_shape: tuple[int, ...],
        dtype: str | np.dtype[Any],
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for point-cloud visualization")
        self.source = source
        self.frame_count = frame_count
        self.frame_shape = frame_shape
        self.dtype = np.dtype(dtype)
        self.frame_bytes = int(np.prod(frame_shape)) * self.dtype.itemsize
        command = [ffmpeg, "-v", "error", "-nostdin"]
        if start_seconds > 0.0:
            command.extend(["-ss", f"{start_seconds:.9f}"])
        command.extend(
            [
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-frames:v",
                str(frame_count),
                "-f",
                "rawvideo",
                "-pix_fmt",
                pixel_format,
                "pipe:1",
            ]
        )
        self.process: subprocess.Popen[bytes] | None = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
        )

    def __enter__(self) -> FFmpegRawVideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def read_frame(self, frame_index: int) -> np.ndarray:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError(f"FFmpeg reader for {self.source} is closed")
        frame_data = process.stdout.read(self.frame_bytes)
        if len(frame_data) != self.frame_bytes:
            if process.poll() is None:
                process.kill()
            _remaining_output, error_output = process.communicate()
            self.process = None
            detail = error_output.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"FFmpeg decoded {len(frame_data)} of {self.frame_bytes} bytes for "
                f"frame {frame_index} from {self.source}"
                + (f": {detail}" if detail else "")
            )
        return np.frombuffer(frame_data, dtype=self.dtype).reshape(self.frame_shape)

    def finish(self) -> None:
        process = self.process
        if process is None or process.stdout is None or process.stderr is None:
            return
        extra_output, error_output = process.communicate()
        return_code = process.returncode
        self.process = None
        detail = error_output.decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed while decoding {self.source}"
                + (f": {detail}" if detail else "")
            )
        if extra_output:
            raise RuntimeError(
                f"FFmpeg produced unexpected extra frame data for {self.source}"
            )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        self.process = None


def episode_video_source(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    video_key: str,
) -> tuple[Path, float]:
    metadata_prefix = f"videos/{video_key}"
    required = [
        f"{metadata_prefix}/chunk_index",
        f"{metadata_prefix}/file_index",
        f"{metadata_prefix}/from_timestamp",
    ]
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(
            f"episode video metadata for {video_key} is missing: " + ", ".join(missing)
        )
    source = dataset / format_dataset_path(
        info["video_path"],
        video_key=video_key,
        chunk_index=episode[f"{metadata_prefix}/chunk_index"],
        file_index=episode[f"{metadata_prefix}/file_index"],
    )
    if not source.is_file():
        raise ValueError(f"video file is missing: {source}")
    return source, float(episode[f"{metadata_prefix}/from_timestamp"])


def log_point_cloud_camera(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    table: pa.Table,
    timestamps: np.ndarray,
    camera: PointCloudCamera,
    stride: int,
) -> None:
    """Decode, reconstruct, and log one RGB-D camera for the whole episode."""
    dataset_fps = float(info["fps"])
    for video_key in (camera.rgb_key, camera.depth_key):
        stream_info = info["features"][video_key].get("info") or {}
        stream_fps = stream_info.get("video.fps")
        if stream_fps is not None and not np.isclose(
            float(stream_fps), dataset_fps, atol=1e-6
        ):
            raise ValueError(
                f"{video_key} is {stream_fps} FPS but the dataset is {dataset_fps} FPS"
            )

    intrinsics, camera_to_world = point_cloud_calibration_series(camera, table)
    camera_to_base = point_cloud_world_to_base_transform(info) @ camera_to_world
    rgb_source, rgb_start = episode_video_source(
        dataset, info, episode, camera.rgb_key
    )
    depth_source, depth_start = episode_video_source(
        dataset, info, episode, camera.depth_key
    )
    invalid_code = quantized_code_for_depth_value(
        camera.invalid_value,
        camera.depth_unit,
        camera.depth_min,
        camera.depth_max,
        camera.depth_shift,
        camera.depth_use_log,
    )

    print(
        f"Generating RGB point cloud: {camera.camera_name} "
        f"({camera.width}x{camera.height}, stride {stride})..."
    )
    total_points = 0
    with FFmpegRawVideoReader(
        depth_source,
        depth_start,
        table.num_rows,
        "gray12le",
        (camera.height, camera.width),
        "<u2",
    ) as depth_reader, FFmpegRawVideoReader(
        rgb_source,
        rgb_start,
        table.num_rows,
        "rgb24",
        (camera.height, camera.width, 3),
        np.uint8,
    ) as rgb_reader:
        for frame_index, timestamp in enumerate(timestamps):
            quantized = depth_reader.read_frame(frame_index)
            rgb = rgb_reader.read_frame(frame_index)
            depth_metres = dequantize_depth_codes(
                quantized,
                camera.depth_min,
                camera.depth_max,
                camera.depth_shift,
                camera.depth_use_log,
            )
            points, colors = backproject_rgbd_to_world(
                depth_metres,
                rgb,
                intrinsics[frame_index],
                camera_to_base[frame_index],
                stride,
                valid_mask=quantized != invalid_code,
            )
            rr.set_time(TIMELINE, duration=float(timestamp))
            rr.log(camera.entity_path, rr.Points3D(points, colors=colors))
            total_points += len(points)
        depth_reader.finish()
        rgb_reader.finish()

    average_points = total_points / max(table.num_rows, 1)
    print(
        f"Loaded RGB point cloud: {camera.camera_name} "
        f"({average_points:.0f} points/frame average)"
    )


def log_point_clouds(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    table: pa.Table,
    timestamps: np.ndarray,
    stride: int,
) -> list[str]:
    """Log all compatible RGB-D streams and return their entity paths."""
    cameras, issues = discover_point_cloud_cameras(info, table)
    for issue in issues:
        print(f"Warning: {issue}")
    if not cameras:
        raise SystemExit(
            "--point-cloud was requested, but no paired RGB-D stream with "
            "per-frame calibration was found"
        )

    entity_paths: list[str] = []
    for camera in cameras:
        # Points have already been transformed numerically into the base frame.
        # Rerun still needs an explicit named-frame edge for every entity that
        # carries spatial data; a CoordinateFrame on the parent entity is not
        # inherited by its children.
        rr.log(
            camera.entity_path,
            rr.CoordinateFrame(camera.frame_name),
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                mat3x3=np.eye(3),
                relation=rr.TransformRelation.ParentFromChild,
                parent_frame=CAMERA_BASE_FRAME,
                child_frame=camera.frame_name,
            ),
            static=True,
        )
        try:
            log_point_cloud_camera(
                dataset,
                info,
                episode,
                table,
                timestamps,
                camera,
                stride,
            )
        except (ValueError, RuntimeError) as error:
            raise SystemExit(
                f"Failed to generate point cloud for {camera.camera_name}: {error}"
            ) from error
        entity_paths.append(camera.entity_path)
    return entity_paths


def extract_video_clip(
    source: Path,
    destination: Path,
    start_seconds: float,
    duration_seconds: float,
    transcode: bool = False,
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
        "-an",
        "-avoid_negative_ts",
        "make_zero",
    ]
    if transcode:
        # Re-encode formats Rerun cannot decode (e.g. hevc/gray12le depth
        # videos) as h264/yuv420p so they can be loaded as AssetVideo.
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    else:
        command.extend(["-c", "copy"])
    command.append(str(destination))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Failed to extract video clip from {source}") from error


def log_videos(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    temporary_directory: Path,
) -> list[tuple[str, str, bool]]:
    """Return ``(display_name, entity_path, is_depth)`` for every video feature."""
    views: list[tuple[str, str, bool]] = []
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    for video_key in video_keys:
        feature_info = info["features"][video_key].get("info", {})
        is_depth = bool(feature_info.get("is_depth_map", False))
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
        extract_video_clip(source, clip_path, start, end - start, transcode=is_depth)

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
        views.append((camera_name, entity_path, is_depth))
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


def arx5_state_indices(
    info: dict[str, Any], table: pa.Table
) -> tuple[dict[str, int] | None, str | None]:
    """Validate the Arx5/RoboTwin state schema and return component indexes by name.

    The RoboTwin dataset stores 14 values per side pair:
    ``left_joint_0..6`` (7) + ``right_joint_0..6`` (7), where ``joint_6``
    is the gripper opening width.  These are mapped to the 1-indexed URDF
    joint names (``fl_joint1..6`` arm + ``fl_joint7/8`` gripper) that the
    replay function expects.
    """
    if info.get("robot_type") != ARX5_ROBOT_TYPE:
        return None, f"robot_type is not {ARX5_ROBOT_TYPE}"
    if PIPER_STATE_FEATURE not in table.column_names:
        return None, f"{PIPER_STATE_FEATURE} is missing from episode data"

    feature = info.get("features", {}).get(PIPER_STATE_FEATURE)
    if not isinstance(feature, dict):
        return None, f"{PIPER_STATE_FEATURE} metadata is missing"
    names = raw_feature_component_names(feature)
    if not names:
        return None, f"{PIPER_STATE_FEATURE} component names do not match its shape"

    required = [
        *(f"left_joint_{index}" for index in range(7)),
        *(f"right_joint_{index}" for index in range(7)),
    ]
    missing = [name for name in required if name not in names]
    if missing:
        return None, "missing state components: " + ", ".join(missing)

    indices: dict[str, int] = {}
    for side, offset in (("left", 0), ("right", 7)):
        for joint_index in range(1, 7):
            state_name = f"{side}_joint_{joint_index - 1}"
            indices[f"{side}_joint_{joint_index}"] = names.index(state_name)
        indices[f"{side}_gripper"] = names.index(f"{side}_joint_6")
    return indices, None


# Collada namespace used by Blender-exported DAE files.
_COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"


def patch_dae_textures(
    source_dae: Path,
    output_dir: Path,
    *,
    force_solid_textures: bool = False,
) -> Path:
    """Make externally textured DAEs self-contained using their source colors.

    Rerun's Collada loader expects ``<input semantic="TEXCOORD">`` on every
    textured triangle set. Some Arx5 Blender exports reference a tiny material
    color image without exporting UV data. Replace those invalid texture
    references with the image's representative color. DAEs that do contain
    texture coordinates are returned unchanged unless *force_solid_textures*
    is set for a mesh whose external images Rerun cannot resolve from an RRD.
    """
    try:
        tree = ET.parse(source_dae)
    except (ET.ParseError, OSError) as error:
        raise RuntimeError(f"Could not parse DAE {source_dae}: {error}") from error

    root = tree.getroot()
    ns = _COLLADA_NS
    if not root.findall(f".//{{{ns}}}texture"):
        return source_dae

    has_texture_coordinates = any(
        element.get("semantic") == "TEXCOORD"
        for element in root.findall(f".//{{{ns}}}input")
    )
    if has_texture_coordinates and not force_solid_textures:
        return source_dae

    image_references: dict[str, str] = {}
    for image in root.findall(f"{{{ns}}}library_images/{{{ns}}}image"):
        init_from = image.find(f"{{{ns}}}init_from")
        if image.get("id") and init_from is not None and init_from.text:
            image_references[image.get("id", "")] = init_from.text.strip()

    def texture_rgba(
        profile: ET.Element, texture: ET.Element
    ) -> tuple[float, float, float, float] | None:
        """Resolve a Collada sampler chain and average its source image."""
        parameters = {
            parameter.get("sid", ""): parameter
            for parameter in profile.findall(f"{{{ns}}}newparam")
            if parameter.get("sid")
        }
        sampler = parameters.get(texture.get("texture", ""))
        if sampler is None:
            return None
        surface_source = sampler.find(f"{{{ns}}}sampler2D/{{{ns}}}source")
        if surface_source is None or not surface_source.text:
            return None
        surface = parameters.get(surface_source.text.strip())
        if surface is None:
            return None
        image_source = surface.find(f"{{{ns}}}surface/{{{ns}}}init_from")
        if image_source is None or not image_source.text:
            return None
        image_reference = image_references.get(image_source.text.strip())
        if not image_reference:
            return None
        image_path = (source_dae.parent / image_reference).resolve()
        try:
            with Image.open(image_path) as image:
                means = ImageStat.Stat(image.convert("RGBA")).mean
        except (OSError, ValueError):
            return None
        if len(means) != 4:
            return None
        return tuple(float(value) / 255.0 for value in means)

    for effect in root.findall(f"{{{ns}}}library_effects/{{{ns}}}effect"):
        profile = effect.find(f"{{{ns}}}profile_COMMON")
        if profile is None:
            continue
        for technique in profile.findall(f"{{{ns}}}technique"):
            for shading in list(technique):
                for channel in ("diffuse", "emission", "ambient", "specular"):
                    channel_element = shading.find(f"{{{ns}}}{channel}")
                    if channel_element is None:
                        continue
                    texture = channel_element.find(f"{{{ns}}}texture")
                    if texture is None:
                        continue
                    rgba = texture_rgba(profile, texture)
                    if rgba is None:
                        rgba = (0.5, 0.5, 0.5, 1.0)
                    if channel != "diffuse":
                        rgba = (0.0, 0.0, 0.0, 1.0)
                    channel_element.remove(texture)
                    color = ET.SubElement(
                        channel_element,
                        f"{{{ns}}}color",
                        {"sid": channel},
                    )
                    color.text = " ".join(f"{value:.6g}" for value in rgba)
        for newparam in list(profile.findall(f"{{{ns}}}newparam")):
            if (
                newparam.find(f"{{{ns}}}surface") is not None
                or newparam.find(f"{{{ns}}}sampler2D") is not None
            ):
                profile.remove(newparam)

    for images in root.findall(f"{{{ns}}}library_images"):
        root.remove(images)

    # Collada importers generally expect the schema as the document's default
    # namespace. ElementTree otherwise serializes it as an ``ns0:`` prefix,
    # which makes Assimp (and some Rerun builds) treat a valid file as empty.
    ET.register_namespace("", ns)
    output_path = output_dir / source_dae.name
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def prepare_follower_visual_urdf(
    source: Path,
    destination: Path,
    strip_prefixes: tuple[str, ...] = ("bl_", "br_"),
    strip_link_names: frozenset[str] = frozenset(),
    patch_dae: bool = False,
    mesh_overrides: Mapping[str, str] | None = None,
    force_solid_texture_links: frozenset[str] = frozenset(),
    box_overrides: Mapping[str, tuple[float, float, float]] | None = None,
    material_overrides: Mapping[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Create a follower-only URDF while preserving the source mesh transforms.

    Visuals are stripped from links whose names start with any prefix in
    *strip_prefixes* or match an entry in *strip_link_names*.  Collision
    geometry is always removed.  When *patch_dae* is true, ``.dae`` meshes
    whose textures lack UV coordinates are copied to a temp directory with a
    representative solid-color material. *force_solid_texture_links* applies
    the same self-contained conversion even when UVs exist. Optional overrides
    replace oversized full-profile assets with lightweight meshes or primitives.
    """
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as error:
        raise RuntimeError(f"Could not parse URDF {source}: {error}") from error

    root = tree.getroot()
    mesh_overrides = mesh_overrides or {}
    box_overrides = box_overrides or {}
    material_overrides = material_overrides or {}
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
        if link_name.startswith(strip_prefixes) or link_name in strip_link_names:
            for visual in list(link.findall("visual")):
                link.remove(visual)
            continue

        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            if link_name in mesh_overrides:
                for child in list(geometry):
                    geometry.remove(child)
                override_path = Path(mesh_overrides[link_name])
                if not override_path.is_absolute():
                    override_path = source.parent / override_path
                ET.SubElement(
                    geometry,
                    "mesh",
                    {"filename": str(override_path.resolve())},
                )
            elif link_name in box_overrides:
                for child in list(geometry):
                    geometry.remove(child)
                ET.SubElement(
                    geometry,
                    "box",
                    {
                        "size": " ".join(
                            f"{value:.6g}" for value in box_overrides[link_name]
                        )
                    },
                )

            mesh = visual.find("./geometry/mesh")
            filename = ""
            if mesh is not None:
                filename = mesh.get("filename", "")
                # Convert bare relative mesh paths to absolute so Rerun can
                # resolve them from the temp URDF copy. ``package://`` URIs
                # are left untouched for ROS_PACKAGE_PATH resolution.
                if filename and not filename.startswith(("package://", "/")):
                    mesh.set("filename", str((source.parent / filename).resolve()))
                    filename = mesh.get("filename", "")
                # Make invalid or explicitly selected external textures
                # self-contained before the temporary URDF is logged.
                if patch_dae and filename.lower().endswith(".dae"):
                    abs_path = Path(filename)
                    if abs_path.is_file():
                        patched = patch_dae_textures(
                            abs_path,
                            destination.parent,
                            force_solid_textures=(
                                link_name in force_solid_texture_links
                            ),
                        )
                        mesh.set("filename", str(patched.resolve()))
                        filename = mesh.get("filename", "")
                if filename.lower().endswith(".dae"):
                    # A URDF-wide albedo masks embedded multi-material colors.
                    for material in list(visual.findall("material")):
                        visual.remove(material)

            if link_name in material_overrides:
                for material in list(visual.findall("material")):
                    visual.remove(material)
                material = ET.SubElement(
                    visual,
                    "material",
                    {"name": f"{link_name}_rerun_material"},
                )
                ET.SubElement(
                    material,
                    "color",
                    {
                        "rgba": " ".join(
                            f"{value:.6g}"
                            for value in material_overrides[link_name]
                        )
                    },
                )

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
    tcp_offset_from_finger_origins: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Log animated left/right EEF (link6) and TCP poses."""
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
        link6_poses = compute_link_pose_series(
            urdf_tree,
            f"{urdf_prefix}_link6",
            joint_values,
            frame_count,
        )

        side_label = "L" if dataset_side == "left" else "R"

        # --- Log actual EEF (link6 position) ---
        link6_translations = link6_poses[:, :3, 3]
        link6_rotations = link6_poses[:, :3, :3]
        link6_rpy_degrees = np.rad2deg(
            np.asarray([rotation_matrix_to_rpy(rotation) for rotation in link6_rotations])
        )
        link6_labels = [
            (
                f"{side_label} EEF (link6)  xyz[m] "
                f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}\n"
                f"rpy[deg] {angles[0]:+.1f} {angles[1]:+.1f} {angles[2]:+.1f}"
            )
            for position, angles in zip(link6_translations, link6_rpy_degrees)
        ]

        link6_entity_path = f"{ROBOT_LINK6_ENTITY_PATH}/{dataset_side}"
        link6_frame = f"{urdf_prefix}_link6_eef"
        rr.log(link6_entity_path, rr.CoordinateFrame(link6_frame), static=True)
        # Connect link6_eef frame to link6 with identity transform (same position)
        rr.log(
            link6_entity_path,
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                parent_frame=f"{urdf_prefix}_link6",
                child_frame=link6_frame,
            ),
            static=True,
        )
        rr.log(link6_entity_path, rr.TransformAxes3D(EEF_AXIS_LENGTH_METERS * 0.8), static=True)
        rr.log(
            link6_entity_path,
            rr.Points3D(
                [[0.0, 0.0, 0.0]],
                radii=[0.012],
                colors=[color],
                show_labels=True,
            ),
            static=True,
        )
        rr.send_columns(
            link6_entity_path,
            indexes=[time_column],
            columns=rr.Points3D.columns(
                positions=np.zeros((frame_count, 3), dtype=np.float32),
                labels=link6_labels,
            ).partition(lengths=[1] * frame_count),
        )

        # --- Log TCP (fingertip contact-center position) ---
        # Both fingers translate symmetrically, so the midpoint between their
        # contact faces is fixed relative to link6.  The profile-specific
        # offset moves the old finger-joint midpoint to the contact surface.
        finger_origins = np.asarray(
            [
                urdf_tree.get_joint_by_name(f"{urdf_prefix}_joint{joint_index}").origin_xyz
                for joint_index in (7, 8)
            ],
            dtype=np.float64,
        )
        tcp_translation = finger_origins.mean(axis=0) + np.asarray(
            tcp_offset_from_finger_origins,
            dtype=np.float64,
        )
        link_to_tcp = np.eye(4, dtype=np.float64)
        link_to_tcp[:3, 3] = tcp_translation
        tcp_poses = link6_poses @ link_to_tcp

        tcp_translations = tcp_poses[:, :3, 3]
        tcp_rotations = tcp_poses[:, :3, :3]
        tcp_rpy_degrees = np.rad2deg(
            np.asarray([rotation_matrix_to_rpy(rotation) for rotation in tcp_rotations])
        )
        tcp_labels = [
            (
                f"{side_label} TCP (fingertip contact center)  xyz[m] "
                f"{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}\n"
                f"rpy[deg] {angles[0]:+.1f} {angles[1]:+.1f} {angles[2]:+.1f}"
            )
            for position, angles in zip(tcp_translations, tcp_rpy_degrees)
        ]

        tcp_entity_path = f"{ROBOT_EEF_ENTITY_PATH}/{dataset_side}"
        tcp_frame = f"{urdf_prefix}_tcp"
        rr.log(tcp_entity_path, rr.CoordinateFrame(tcp_frame), static=True)
        rr.log(
            tcp_entity_path,
            rr.Transform3D(
                translation=tcp_translation,
                parent_frame=f"{urdf_prefix}_link6",
                child_frame=tcp_frame,
            ),
            static=True,
        )
        rr.log(tcp_entity_path, rr.TransformAxes3D(EEF_AXIS_LENGTH_METERS), static=True)
        rr.log(
            tcp_entity_path,
            rr.Points3D(
                [[0.0, 0.0, 0.0]],
                radii=[0.012],
                colors=[color],
                show_labels=True,
            ),
            static=True,
        )
        rr.send_columns(
            tcp_entity_path,
            indexes=[time_column],
            columns=rr.Points3D.columns(
                positions=np.zeros((frame_count, 3), dtype=np.float32),
                labels=tcp_labels,
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
    gripper_signs: tuple[float, float] = PIPER_GRIPPER_SIGNS,
    strip_prefixes: tuple[str, ...] = ("bl_", "br_"),
    strip_link_names: frozenset[str] = frozenset(),
    patch_dae: bool = False,
    mesh_overrides: Mapping[str, str] | None = None,
    force_solid_texture_links: frozenset[str] = frozenset(),
    box_overrides: Mapping[str, tuple[float, float, float]] | None = None,
    material_overrides: Mapping[str, tuple[float, float, float, float]] | None = None,
    gripper_mode: str = "width",
    tcp_offset_from_finger_origins: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Log follower geometry and animated bimanual joint transforms.

    *gripper_mode* selects how the gripper state value is interpreted:

    - ``"width"`` (Piper): value is the total finger-to-finger opening in
      metres; each finger moves by ``value * 0.5``.
    - ``"normalized"`` (Arx5/RoboTwin): value is a ``[0, 1]`` scalar where
      ``1`` means fully open (URDF joint upper limit) and ``0`` means
      closed; each finger moves by ``value * joint_upper_limit``.
    """
    prepend_ros_package_path(package_root)
    prepared_urdf = temporary_directory / "aloha_follower_visual.urdf"
    prepare_follower_visual_urdf(
        urdf_path,
        prepared_urdf,
        strip_prefixes=strip_prefixes,
        strip_link_names=strip_link_names,
        patch_dae=patch_dae,
        mesh_overrides=mesh_overrides,
        force_solid_texture_links=force_solid_texture_links,
        box_overrides=box_overrides,
        material_overrides=material_overrides,
    )

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
        if gripper_mode == "normalized":
            # Binary [0,1] scalar: 1 = fully open (joint upper limit).
            joint7 = urdf_tree.get_joint_by_name(f"{urdf_prefix}_joint7")
            assert joint7 is not None
            upper = joint7.limit_upper if joint7.limit_upper is not None else 0.04
            positive_position = np.clip(gripper_width, 0.0, 1.0) * float(upper)
            negative_position = -positive_position
            was_clipped = False
        else:
            positive_position, negative_position, was_clipped = (
                gripper_finger_positions(gripper_width)
            )
        gripper_was_clipped = gripper_was_clipped or was_clipped
        joint7_value = positive_position * gripper_signs[0]
        joint8_value = positive_position * gripper_signs[1]
        for joint_index, values in (
            (7, joint7_value),
            (8, joint8_value),
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

    log_piper_eef_poses(
        urdf_tree,
        state_values,
        state_indices,
        timestamps,
        tcp_offset_from_finger_origins=tcp_offset_from_finger_origins,
    )


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

    # Try Piper schema first, then Arx5/RoboTwin.
    state_indices, incompatibility = piper_state_indices(info, table)
    profile = "piper"
    if state_indices is None:
        state_indices, arx5_incompat = arx5_state_indices(info, table)
        if state_indices is None:
            message = (
                f"piper replay unavailable: {incompatibility}; "
                f"arx5 replay unavailable: {arx5_incompat}"
            )
            if explicit_urdf:
                raise SystemExit(message)
            print(f"Warning: {message}")
            return False
        profile = "arx5"
        incompatibility = arx5_incompat

    if explicit_urdf:
        urdf_path = args.urdf.expanduser().resolve()
    elif profile == "arx5":
        urdf_path = default_arx5_urdf(script_root).resolve()
    else:
        urdf_path = default_aloha_urdf(script_root).resolve()

    if not urdf_path.is_file():
        message = f"robot replay URDF not found: {urdf_path}"
        if explicit_urdf:
            raise SystemExit(message)
        print(f"Warning: {message}")
        return False

    if profile == "arx5":
        package_root = urdf_path.parent
        gripper_signs = ARX5_GRIPPER_SIGNS
        strip_prefixes = ()
        strip_link_names = frozenset()
        mesh_overrides = ARX5_FULL_MESH_OVERRIDES
        force_solid_texture_links = ARX5_FORCE_SOLID_TEXTURE_LINKS
        box_overrides = {}
        material_overrides = ARX5_FULL_MATERIAL_OVERRIDES
        patch_dae = True
        gripper_mode = "normalized"
        tcp_offset_from_finger_origins = ARX5_FINGERTIP_CONTACT_OFFSET_METERS
    else:
        package_root = script_root / "embodiments"
        gripper_signs = PIPER_GRIPPER_SIGNS
        strip_prefixes = ("bl_", "br_")
        strip_link_names = frozenset()
        patch_dae = False
        mesh_overrides = {}
        force_solid_texture_links = frozenset()
        box_overrides = {}
        material_overrides = {}
        gripper_mode = "width"
        tcp_offset_from_finger_origins = (0.0, 0.0, 0.0)

    try:
        log_piper_robot_replay(
            table,
            timestamps,
            state_indices,
            urdf_path,
            temporary_directory,
            recording,
            package_root,
            gripper_signs=gripper_signs,
            strip_prefixes=strip_prefixes,
            strip_link_names=strip_link_names,
            patch_dae=patch_dae,
            mesh_overrides=mesh_overrides,
            force_solid_texture_links=force_solid_texture_links,
            box_overrides=box_overrides,
            material_overrides=material_overrides,
            gripper_mode=gripper_mode,
            tcp_offset_from_finger_origins=tcp_offset_from_finger_origins,
        )
    except RuntimeError as error:
        if explicit_urdf:
            raise SystemExit(str(error)) from error
        print(f"Warning: robot replay unavailable: {error}")
        return False

    print(f"Loaded robot replay ({profile}): {urdf_path}")
    return True


def make_blueprint(
    video_views: list[tuple[str, str, bool]],
    signal_views: list[tuple[str, str]],
    robot_replay: bool = False,
    calibrated_camera: CameraCalibration | None = None,
    point_cloud_paths: list[str] | None = None,
) -> rrb.Blueprint:
    point_cloud_paths = point_cloud_paths or []
    rgb_views: list[Any] = [
        rrb.Spatial2DView(origin=path, name=name)
        for name, path, is_depth in video_views
        if not is_depth
    ]
    depth_views: list[Any] = [
        rrb.Spatial2DView(origin=path, name=name)
        for name, path, is_depth in video_views
        if is_depth
    ]
    all_camera_views = rgb_views + depth_views

    if robot_replay or point_cloud_paths:
        spatial_view_name = "Robot replay" if robot_replay else "Scene point cloud"
        robot_view = rrb.Spatial3DView(
            origin=ROBOT_ENTITY_PATH,
            name=spatial_view_name,
        )
        robot_area: Any = robot_view
        if calibrated_camera is not None:
            calibrated_view = rrb.Spatial2DView(
                origin=calibrated_camera.entity_path,
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

        if rgb_views or depth_views:
            camera_rows: list[Any] = []
            if rgb_views:
                camera_rows.append(rrb.Grid(*rgb_views, name="RGB"))
            if depth_views:
                camera_rows.append(rrb.Grid(*depth_views, name="Depth"))
            camera_area: Any = (
                rrb.Vertical(*camera_rows, name="Cameras")
                if len(camera_rows) > 1
                else camera_rows[0]
            )
            top_area: Any = rrb.Horizontal(
                robot_area,
                camera_area,
                column_shares=[1, 2],
                name="Replay",
            )
        else:
            top_area = robot_area
    elif all_camera_views:
        top_area = rrb.Horizontal(*all_camera_views, name="Cameras")
    else:
        top_area = rrb.TextDocumentView(origin="episode_info", name="Episode")

    plots: list[Any] = [
        rrb.TimeSeriesView(origin=path, name=name) for name, path in signal_views
    ]
    plot_area: Any
    if plots:
        # Show all signal plots simultaneously in a grid instead of
        # hiding them behind switchable tabs.
        plot_area = rrb.Grid(*plots, name="Signals")
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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
        "--no-video",
        action="store_true",
        help="Do not add 2D video views (point-cloud decoding remains available)",
    )
    parser.add_argument(
        "--point-cloud",
        action="store_true",
        help="Reconstruct paired RGB-D streams in the dataset world/base frame",
    )
    parser.add_argument(
        "--point-cloud-stride",
        type=positive_int,
        default=DEFAULT_POINT_CLOUD_STRIDE,
        metavar="N",
        help="Sample every Nth depth/RGB pixel along each image axis",
    )
    robot_group = parser.add_mutually_exclusive_group()
    robot_group.add_argument(
        "--urdf",
        type=Path,
        help=(
            "Use this URDF for an eligible replay instead of the auto-detected "
            "default (aloha_new_description for Piper, aloha-agilex for Arx5)"
        ),
    )
    robot_group.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable automatic URDF replay for compatible bimanual datasets",
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
        video_views: list[tuple[str, str, bool]] = []
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
        point_cloud_paths: list[str] = []
        if args.point_cloud:
            if not robot_replay:
                log_robot_footprint_frame()
            point_cloud_paths = log_point_clouds(
                dataset,
                info,
                episode,
                table,
                timestamps,
                args.point_cloud_stride,
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
                point_cloud_paths,
            )
        )
        recording.flush()

    if args.output:
        print(f"Saved Rerun recording: {args.output.expanduser().resolve()}")
    else:
        print("Episode loaded in Rerun. Use the episode_time timeline to scrub or play.")


if __name__ == "__main__":
    main()
