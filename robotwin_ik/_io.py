from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import importlib.metadata
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
import convert_robotwin_tcp as tcp_converter

POSITION_TOLERANCE_METERS = 0.003
ROTATION_TOLERANCE_DEGREES = 2.0
ROTATION_TOLERANCE_RADIANS = math.radians(ROTATION_TOLERANCE_DEGREES)
RANDOM_SEED = 1531
OUTPUT_FILENAMES = ("robot_state.npy", "robot_state_candidate.npy", "metadata.json", "diagnostics.json")

@dataclass(frozen=True)
class PredictionData:
    camera_tcp: dict[str, np.ndarray]
    gripper_open: dict[str, np.ndarray]
    timestamps: np.ndarray
    frame_rate_hz: float
    camera: str
    metadata: dict[str, Any]

def positive_seed_count(value: str) -> int:
    parsed = int(value)
    if not 2 <= parsed <= 256:
        raise argparse.ArgumentTypeError("IK seed count must be between 2 and 256")
    return parsed

def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON/YAML object")
    return value

def _finite_vector(value: object, shape: tuple[int, ...], description: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must be finite with shape {shape}, got {array}")
    return array

def pose_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = tcp_converter.rotation_matrix_from_rpy(rpy)
    pose[:3, 3] = xyz
    return pose

def load_prediction(path: Path) -> PredictionData:
    try:
        document = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read prediction JSON {path}: {error}") from error

    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty frames array")
    frame_count = len(frames)
    declared_count = document.get("num_frames")
    if declared_count is not None and int(declared_count) != frame_count:
        raise ValueError(
            f"{path} declares num_frames={declared_count}, but has {frame_count} frames"
        )

    frame_rate_hz = float(document.get("frame_rate_hz", 0.0))
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"{path} has invalid frame_rate_hz={frame_rate_hz!r}")
    camera = str(document.get("view", ""))
    if not camera:
        raise ValueError(f"{path} is missing the camera view")
    if str(document.get("position_unit", "meter")) != "meter":
        raise ValueError(f"{path} position_unit must be 'meter'")
    if str(document.get("rotation_unit", "radian")) != "radian":
        raise ValueError(f"{path} rotation_unit must be 'radian'")

    gripper_metadata = _mapping(document.get("gripper", {}), f"{path}.gripper")
    threshold = float(gripper_metadata.get("binary_threshold", 0.5))
    if not math.isfinite(threshold):
        raise ValueError(f"{path} gripper.binary_threshold must be finite")

    camera_tcp = {
        side: np.empty((frame_count, 4, 4), dtype=np.float64)
        for side in ("left", "right")
    }
    gripper_open = {
        side: np.empty(frame_count, dtype=np.float32) for side in ("left", "right")
    }
    timestamps = np.empty(frame_count, dtype=np.float64)

    for expected_index, raw_frame in enumerate(frames):
        frame = _mapping(raw_frame, f"frames[{expected_index}]")
        frame_index = int(frame.get("frame_index", expected_index))
        if frame_index != expected_index:
            raise ValueError(
                f"frames[{expected_index}].frame_index={frame_index}; expected contiguous order"
            )
        timestamp = float(frame.get("time_seconds", expected_index / frame_rate_hz))
        if not math.isfinite(timestamp):
            raise ValueError(f"frames[{expected_index}].time_seconds is not finite")
        timestamps[expected_index] = timestamp

        for side in ("left", "right"):
            item = _mapping(frame.get(side), f"frames[{expected_index}].{side}")
            xyz = _finite_vector(
                item.get("xyz_m"), (3,), f"frames[{expected_index}].{side}.xyz_m"
            )
            rpy = _finite_vector(
                item.get("rpy_rad"), (3,), f"frames[{expected_index}].{side}.rpy_rad"
            )
            camera_tcp[side][expected_index] = pose_from_xyz_rpy(xyz, rpy)

            if "gripper_probability" in item:
                probability = float(item["gripper_probability"])
                if not math.isfinite(probability):
                    raise ValueError(
                        f"frames[{expected_index}].{side}.gripper_probability is not finite"
                    )
                is_open = probability >= threshold
                if "gripper_open" in item and bool(item["gripper_open"]) != is_open:
                    raise ValueError(
                        f"frames[{expected_index}].{side} gripper_open disagrees with "
                        f"probability threshold {threshold:g}"
                    )
            elif "gripper_open" in item and isinstance(item["gripper_open"], bool):
                is_open = bool(item["gripper_open"])
            else:
                raise ValueError(
                    f"frames[{expected_index}].{side} has no valid gripper prediction"
                )
            gripper_open[side][expected_index] = float(is_open)

    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path} timestamps must be strictly increasing")
    return PredictionData(
        camera_tcp=camera_tcp,
        gripper_open=gripper_open,
        timestamps=timestamps,
        frame_rate_hz=frame_rate_hz,
        camera=camera,
        metadata=document,
    )

def load_extrinsics(path: Path, frame_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing world_to_camera extrinsics: {path}")
    return tcp_converter.homogeneous_extrinsics(path, frame_count)

def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

def package_version(name: str, module: object) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(sanitize_json(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def save_trajectory_output(output, state, metadata, diagnostics):
    """Publish only the accepted state filename; candidates remain explicitly separate."""
    success = metadata['status'] == 'success'
    filename = 'robot_state.npy' if success else 'robot_state_candidate.npy'
    stale = 'robot_state_candidate.npy' if success else 'robot_state.npy'
    metadata['output']['state_file'] = filename
    metadata['output']['role'] = 'solution' if success else 'failed_candidate'
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(output / filename, state)
    atomic_write_json(output / 'diagnostics.json', diagnostics)
    (output / stale).unlink(missing_ok=True)
    # Metadata is the last commit marker and names exactly the state just written.
    atomic_write_json(output / 'metadata.json', metadata)
