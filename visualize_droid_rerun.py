#!/usr/bin/env python3
"""Visualize one extracted DROID episode with Rerun.

The viewer synchronizes the two fixed third-person RGB-D cameras, colored
point clouds in the Franka base frame, the measured Panda/Robotiq URDF replay,
camera-matched robot replay views, camera-space TCP trajectories/poses, and
observation/action signals.

Example:
    conda run --no-capture-output -n rerun python visualize_droid_rerun.py \
        '4d_datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023'

On Linux without X11/Wayland, the script automatically starts the Rerun Web
viewer and prints the local URL (plus SSH forwarding instructions when useful).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb

from rerun_viewer import (
    add_viewer_arguments,
    configure_rerun_output,
    wait_for_web_viewer,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODE = (
    SCRIPT_ROOT
    / "4d_datasets"
    / "droid_episodes"
    / "AUTOLab__Fri_Aug_18_11:40:54_2023"
)
DEFAULT_URDF = (
    SCRIPT_ROOT
    / "embodiments"
    / "franka-panda-robotiq-2f85"
    / "panda_robotiq_2f85.urdf"
)
TIME_TIMELINE = "episode_time"
FRAME_TIMELINE = "frame"
WORLD_FRAME = "world"
ROBOT_FRAME_PREFIX = "robot/"
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
STATE_COLOR = np.array([55, 205, 255], dtype=np.uint8)
ACTION_COLOR = np.array([255, 155, 55], dtype=np.uint8)


@dataclass(frozen=True)
class Camera:
    serial: str
    label: str
    intrinsic: np.ndarray  # [T,3,3]
    base_to_camera: np.ndarray  # [T,4,4]
    rgb_files: tuple[Path, ...]
    depth_files: tuple[Path, ...]
    timestamps_ms: np.ndarray
    tcp_state: np.ndarray  # [T,7]: camera XYZ, fixed-axis XYZ RPY, gripper_open
    tcp_world_poses: np.ndarray  # [T,4,4]
    width: int
    height: int

    @property
    def frame_name(self) -> str:
        return f"camera/{self.serial}"

    @property
    def entity_path(self) -> str:
        return f"cameras/{self.serial}"


@dataclass(frozen=True)
class Episode:
    path: Path
    metadata: dict[str, Any]
    cameras: tuple[Camera, Camera]
    timestamps_s: np.ndarray
    observation_joints: np.ndarray
    action_joints: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.timestamps_s)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def finite_nonnegative(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def finite_positive(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        default=DEFAULT_EPISODE,
        help="Extracted DROID episode directory",
    )
    parser.add_argument(
        "--cameras",
        nargs=2,
        metavar=("SERIAL_1", "SERIAL_2"),
        help="The two third-person camera serials (auto-discovered by default)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Franka Panda + Robotiq 2F-85 replay URDF",
    )
    parser.add_argument(
        "--point-cloud-stride",
        type=positive_int,
        default=2,
        help="Back-project every Nth RGB-D pixel along each image axis",
    )
    parser.add_argument(
        "--min-depth-m",
        type=finite_nonnegative,
        default=0.10,
        help="Reject depth at or below this distance",
    )
    parser.add_argument(
        "--max-depth-m",
        type=finite_positive,
        default=3.0,
        help="Reject depth beyond this distance",
    )
    parser.add_argument(
        "--history",
        type=positive_int,
        default=60,
        help="Number of recent TCP positions to show",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Only load the first N frames (useful for a quick check)",
    )
    parser.add_argument(
        "--no-point-cloud",
        action="store_true",
        help="Show RGB/depth and replay without back-projecting point clouds",
    )
    parser.add_argument(
        "--no-depth-images",
        action="store_true",
        help="Hide the two raw depth image views (point clouds remain enabled)",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable the URDF robot mesh replay",
    )
    add_viewer_arguments(parser)
    args = parser.parse_args(argv)
    if args.min_depth_m >= args.max_depth_m:
        parser.error("--min-depth-m must be smaller than --max-depth-m")
    if args.output is not None and args.output.suffix.lower() != ".rrd":
        parser.error("--output must end in .rrd")
    return args


def load_array(path: Path, shape: tuple[int, ...], frame_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    expected = (frame_count, *shape)
    if values.shape != expected:
        raise ValueError(f"{path} has shape {values.shape}; expected {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values")
    return values


def load_calibration(
    path: Path, frame_count: int, matrix_shape: tuple[int, int]
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.shape == matrix_shape:
        values = np.broadcast_to(values, (frame_count, *matrix_shape)).copy()
    if values.ndim == 3 and values.shape[1:] == matrix_shape and len(values) >= frame_count:
        values = values[:frame_count]
    expected = (frame_count, *matrix_shape)
    if values.shape != expected:
        raise ValueError(f"{path} has shape {values.shape}; expected {matrix_shape} or {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values")
    return values


def discover_camera_serials(episode: Path) -> list[str]:
    locations = (
        episode / "images",
        episode / "depths",
        episode / "intrinsic",
        episode / "extrinsic",
    )
    for location in locations:
        if not location.is_dir():
            raise FileNotFoundError(f"Missing directory {location}")
    image_serials = {path.name for path in locations[0].iterdir() if path.is_dir()}
    depth_serials = {path.name for path in locations[1].iterdir() if path.is_dir()}
    intrinsic_serials = {path.stem for path in locations[2].glob("*.npy")}
    extrinsic_serials = {path.stem for path in locations[3].glob("*.npy")}
    return sorted(image_serials & depth_serials & intrinsic_serials & extrinsic_serials)


def load_camera(episode: Path, serial: str, label: str, frame_count: int) -> Camera:
    rgb_directory = episode / "images" / serial
    depth_directory = episode / "depths" / serial
    rgb_files = tuple(sorted(rgb_directory.glob("*.png")))
    depth_files = tuple(sorted(depth_directory.glob("*.png")))
    if len(rgb_files) < frame_count or len(depth_files) < frame_count:
        raise ValueError(
            f"Camera {serial} has {len(rgb_files)} RGB and {len(depth_files)} depth "
            f"frames; expected at least {frame_count}"
        )
    rgb_files = rgb_files[:frame_count]
    depth_files = depth_files[:frame_count]
    expected_names = tuple(f"{index:06d}.png" for index in range(frame_count))
    if tuple(path.name for path in rgb_files) != expected_names:
        raise ValueError(f"Camera {serial} RGB filenames are not a contiguous zero-based sequence")
    if tuple(path.name for path in depth_files) != expected_names:
        raise ValueError(f"Camera {serial} depth filenames are not a contiguous zero-based sequence")

    with Image.open(rgb_files[0]) as image:
        width, height = image.size
    with Image.open(depth_files[0]) as image:
        if image.size != (width, height):
            raise ValueError(f"Camera {serial} RGB/depth resolutions differ")
        if np.asarray(image).dtype != np.uint16:
            raise ValueError(f"Camera {serial} depth PNGs must contain uint16 millimeters")

    intrinsic = load_calibration(
        episode / "intrinsic" / f"{serial}.npy", frame_count, (3, 3)
    )
    if np.any(intrinsic[:, 0, 0] <= 0.0) or np.any(intrinsic[:, 1, 1] <= 0.0):
        raise ValueError(f"Camera {serial} has a non-positive focal length")
    base_to_camera = load_calibration(
        episode / "extrinsic" / f"{serial}.npy", frame_count, (4, 4)
    )
    if not np.allclose(base_to_camera[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Camera {serial} extrinsics are not homogeneous transforms")
    rotations = base_to_camera[:, :3, :3]
    if not np.allclose(
        rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=2e-4
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4):
        raise ValueError(f"Camera {serial} extrinsic rotations are invalid")

    timestamp_path = depth_directory / "timestamps.npy"
    if not timestamp_path.is_file():
        raise FileNotFoundError(f"Missing {timestamp_path}")
    timestamps_ms = np.asarray(np.load(timestamp_path, allow_pickle=False), dtype=np.float64)
    if timestamps_ms.ndim != 1 or len(timestamps_ms) < frame_count:
        raise ValueError(
            f"{timestamp_path} has shape {timestamps_ms.shape}; expected at least [{frame_count}]"
        )
    timestamps_ms = timestamps_ms[:frame_count]
    if not np.all(np.isfinite(timestamps_ms)) or np.any(np.diff(timestamps_ms) <= 0.0):
        raise ValueError(f"{timestamp_path} must contain finite, increasing timestamps")

    tcp_directory = episode / "TCP" / serial
    tcp_metadata = json.loads((tcp_directory / "metadata.json").read_text(encoding="utf-8"))
    expected_columns = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"]
    if (tcp_metadata.get("columns") != expected_columns
            or tcp_metadata.get("position_unit") != "meter"
            or tcp_metadata.get("rotation_unit") != "radian"
            or tcp_metadata.get("coordinate_frame") != "OpenCV camera (+x right, +y down, +z forward)"
            or tcp_metadata.get("rpy_convention") != "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))"):
        raise ValueError(f"Unsupported TCP format in {tcp_directory / 'metadata.json'}")
    source_count = int(json.loads((episode / "metadata.json").read_text())["frame_count"])
    tcp_state = load_array(tcp_directory / "state.npy", (7,), source_count)[:frame_count]
    if np.any((tcp_state[:, 6] < 0.0) | (tcp_state[:, 6] > 1.0)):
        raise ValueError(f"TCP gripper_open must be in [0, 1]: {tcp_directory}")
    # The saved extrinsic maps base -> camera; invert it for the world replay.
    camera_poses = np.stack([pose_from_cartesian(row) for row in tcp_state])
    tcp_world_poses = np.linalg.inv(base_to_camera) @ camera_poses

    return Camera(
        serial=serial,
        label=label,
        intrinsic=intrinsic,
        base_to_camera=base_to_camera,
        rgb_files=rgb_files,
        depth_files=depth_files,
        timestamps_ms=timestamps_ms,
        tcp_state=tcp_state,
        tcp_world_poses=tcp_world_poses,
        width=width,
        height=height,
    )


def load_episode(
    episode_path: Path,
    requested_cameras: list[str] | None,
    max_frames: int | None,
) -> Episode:
    episode = episode_path.expanduser().resolve()
    if not episode.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode}")
    metadata_path = episode / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_frame_count = int(metadata.get("frame_count", 0))
    if source_frame_count <= 0:
        raise ValueError(f"Invalid frame_count in {metadata_path}")
    frame_count = min(source_frame_count, max_frames or source_frame_count)

    available = discover_camera_serials(episode)
    selected = requested_cameras or available
    if len(selected) != 2:
        raise ValueError(
            "Exactly two RGB-D cameras are required; found "
            f"{available}. Pass --cameras SERIAL_1 SERIAL_2 to select them."
        )
    if len(set(selected)) != 2 or any(serial not in available for serial in selected):
        raise ValueError(f"Requested cameras {selected} are not two distinct members of {available}")
    cameras = tuple(
        load_camera(episode, serial, f"third_person_{index}", frame_count)
        for index, serial in enumerate(selected, start=1)
    )

    if not np.allclose(cameras[0].tcp_state[:, 6], cameras[1].tcp_state[:, 6], atol=1e-6):
        raise ValueError("TCP camera streams disagree on gripper opening")

    timestamp_stack = np.stack([camera.timestamps_ms for camera in cameras])
    timestamps_s = np.median(timestamp_stack, axis=0)
    timestamps_s = (timestamps_s - timestamps_s[0]) * 0.001
    if np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("The camera-median episode timeline is not strictly increasing")

    depth_metadata_path = episode / "depths" / "metadata.json"
    if depth_metadata_path.is_file():
        depth_metadata = json.loads(depth_metadata_path.read_text(encoding="utf-8"))
        units = str(depth_metadata.get("units", "millimeters")).lower()
        if units not in ("millimeter", "millimeters", "mm"):
            raise ValueError(f"Unsupported depth unit {units!r} in {depth_metadata_path}")

    return Episode(
        path=episode,
        metadata=metadata,
        cameras=cameras,  # type: ignore[arg-type]
        timestamps_s=timestamps_s,
        observation_joints=load_array(
            episode / "observations" / "joint_position.npy", (7,), source_frame_count
        )[:frame_count],
        action_joints=load_array(
            episode / "action" / "joint_position.npy", (7,), source_frame_count
        )[:frame_count],
    )


ASSIMP_Z_UP_ROOT_MATRIX = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float64,
)


def remove_assimp_z_up_root_rotation(glb_path: Path) -> bool:
    """Remove Assimp's DAE Z-up conversion from a temporary GLB.

    URDF mesh coordinates already live in the link frame. Assimp represents a
    COLLADA ``Z_UP`` asset as a Y-up GLB by adding a 90-degree rotation to the
    scene root. Applying that asset transform inside a URDF rotates the mesh
    away from the link's joints. Keep the original vertices and remove only
    the exact, known Assimp axis-conversion matrix.
    """
    data = glb_path.read_bytes()
    if len(data) < 20:
        raise RuntimeError(f"Assimp produced a truncated GLB: {glb_path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise RuntimeError(f"Assimp produced an invalid GLB: {glb_path}")

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    modified = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError(f"Invalid GLB chunk header in {glb_path}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise RuntimeError(f"Invalid GLB chunk length in {glb_path}")
        payload = data[chunk_start:chunk_end]
        if chunk_type == 0x4E4F534A:  # JSON
            document = json.loads(payload.rstrip(b" \t\r\n\0"))
            nodes = document.get("nodes", [])
            root_indices = {
                node_index
                for scene in document.get("scenes", [])
                for node_index in scene.get("nodes", [])
            }
            json_modified = False
            for node_index in root_indices:
                if not 0 <= node_index < len(nodes):
                    continue
                matrix = nodes[node_index].get("matrix")
                if isinstance(matrix, list) and len(matrix) == 16 and np.allclose(
                    np.asarray(matrix, dtype=np.float64),
                    ASSIMP_Z_UP_ROOT_MATRIX,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    del nodes[node_index]["matrix"]
                    json_modified = True
            if json_modified:
                payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
                payload += b" " * (-len(payload) % 4)
                modified = True
        chunks.append((chunk_type, payload))
        offset = chunk_end
    if offset != len(data):
        raise RuntimeError(f"Invalid trailing data in {glb_path}")

    if modified:
        body = b"".join(
            struct.pack("<II", len(payload), chunk_type) + payload
            for chunk_type, payload in chunks
        )
        glb_path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return modified


def prepare_visual_urdf(source: Path, destination: Path) -> None:
    """Create a Rerun-safe visual URDF without touching the source assets.

    Rerun 0.35's COLLADA importer panics on the legacy Robotiq DAEs (missing
    asset metadata and duplicate IDs). Convert every referenced DAE to a
    temporary GLB with Assimp; the URDF's original mesh scale and link-frame
    mesh orientation are retained.
    """
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as error:
        raise RuntimeError(f"Could not parse URDF {source}: {error}") from error
    assimp = shutil.which("assimp")
    converted: dict[Path, Path] = {}
    for link in tree.getroot().findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for mesh in link.findall("./visual/geometry/mesh"):
            filename = mesh.get("filename")
            if filename and not filename.startswith(("/", "package://")):
                mesh_path = (source.parent / filename).resolve()
            elif filename and filename.startswith("/"):
                mesh_path = Path(filename)
            else:
                continue
            if not mesh_path.is_file():
                raise RuntimeError(f"URDF mesh does not exist: {mesh_path}")
            if mesh_path.suffix.lower() == ".dae":
                if assimp is None:
                    raise RuntimeError(
                        "The Franka/Robotiq replay contains legacy DAE meshes that "
                        "crash Rerun 0.35. Install Assimp (`brew install assimp`) "
                        "or run with --no-robot."
                    )
                converted_path = converted.get(mesh_path)
                if converted_path is None:
                    converted_path = (
                        destination.parent / f"robotiq_{len(converted):02d}_{mesh_path.stem}.glb"
                    )
                    try:
                        result = subprocess.run(
                            [
                                assimp,
                                "export",
                                str(mesh_path),
                                str(converted_path),
                                "-f",
                                "glb2",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError(
                            f"Assimp timed out converting {mesh_path.name} to GLB"
                        ) from error
                    if result.returncode != 0 or not converted_path.is_file():
                        detail = (result.stderr or result.stdout).strip()
                        raise RuntimeError(
                            f"Assimp could not convert {mesh_path.name} to GLB: {detail}"
                        )
                    remove_assimp_z_up_root_rotation(converted_path)
                    converted[mesh_path] = converted_path
                mesh_path = converted_path
            mesh.set("filename", str(mesh_path))
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Return the fixed-axis XYZ (roll/pitch/yaw) rotation used by DROID."""
    roll, pitch, yaw = (float(value) for value in rpy)
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


def pose_from_cartesian(values: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_from_rpy(values[3:6])
    pose[:3, 3] = values[:3]
    return pose


def backproject_rgbd(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    base_to_camera: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    sampled_depth = np.asarray(depth_mm[::stride, ::stride], dtype=np.float32) * 0.001
    sampled_rgb = np.asarray(rgb[::stride, ::stride, :3], dtype=np.uint8)
    rows = np.arange(0, depth_mm.shape[0], stride, dtype=np.float32)
    columns = np.arange(0, depth_mm.shape[1], stride, dtype=np.float32)
    pixel_u, pixel_v = np.meshgrid(columns, rows)
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > min_depth_m)
        & (sampled_depth <= max_depth_m)
    )
    z = sampled_depth[valid]
    x = (pixel_u[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (pixel_v[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.column_stack((x, y, z))
    camera_to_base = np.linalg.inv(base_to_camera)
    # ``einsum`` avoids spurious floating-point warnings emitted by some
    # NumPy/Accelerate matmul builds for otherwise finite Nx3 inputs.
    base_points = (
        np.einsum("ni,ji->nj", camera_points, camera_to_base[:3, :3])
        + camera_to_base[:3, 3]
    )
    colors = np.ascontiguousarray(sampled_rgb[valid])
    return base_points.astype(np.float32), colors, int(np.count_nonzero(valid))


def log_world_axes() -> None:
    rr.log(
        "world",
        rr.CoordinateFrame(WORLD_FRAME),
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )
    rr.log(
        "world/origin",
        rr.CoordinateFrame(WORLD_FRAME),
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * 0.20,
            colors=[[255, 55, 55], [55, 220, 75], [65, 125, 255]],
            labels=["base +X", "base +Y", "base +Z"],
            radii=0.003,
        ),
        static=True,
    )


def log_cameras(cameras: tuple[Camera, Camera]) -> None:
    for camera in cameras:
        # Stored DROID extrinsics map base/world points into OpenCV camera
        # coordinates. Invert them to place the physical camera in the base.
        camera_to_base = np.linalg.inv(camera.base_to_camera[0])
        rr.log(
            camera.entity_path,
            rr.CoordinateFrame(camera.frame_name),
            rr.Transform3D(
                translation=camera_to_base[:3, 3],
                mat3x3=camera_to_base[:3, :3],
                relation=rr.TransformRelation.ParentFromChild,
                parent_frame=WORLD_FRAME,
                child_frame=camera.frame_name,
            ),
            rr.Pinhole(
                image_from_camera=camera.intrinsic[0],
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.12,
                color=[80, 210, 255] if camera.label.endswith("1") else [255, 175, 65],
                parent_frame=WORLD_FRAME,
                child_frame=camera.frame_name,
            ),
            static=True,
        )
        rr.log(f"{camera.entity_path}/rgb", rr.CoordinateFrame(camera.frame_name), static=True)
        rr.log(f"{camera.entity_path}/depth", rr.CoordinateFrame(camera.frame_name), static=True)


def log_robot_replay(
    recording: Any,
    episode: Episode,
    urdf_path: Path,
    temporary_directory: Path,
) -> None:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Robot replay URDF does not exist: {urdf_path}")
    prepared = temporary_directory / "panda_robotiq_visual.urdf"
    prepare_visual_urdf(urdf_path, prepared)
    try:
        tree = rr.urdf.UrdfTree.from_file_path(
            prepared,
            # Rerun 0.35 treats slashes in this particular string as escaped
            # characters, so use one path segment for the URDF asset subtree.
            entity_path_prefix="robot_model",
            frame_prefix=ROBOT_FRAME_PREFIX,
            static_transform_entity_path="robot/static_transforms",
        )
        recording.send_chunks(tree.stream(include_joint_transforms=False))
    except Exception as error:
        raise RuntimeError(f"Rerun could not load {urdf_path}: {error}") from error

    rr.log(
        "robot/root_transform",
        rr.Transform3D(
            translation=[0.0, 0.0, 0.0],
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
            parent_frame=WORLD_FRAME,
            child_frame=f"{ROBOT_FRAME_PREFIX}{tree.root_link().name}",
        ),
        static=True,
    )
    time_column = rr.TimeColumn(TIME_TIMELINE, duration=episode.timestamps_s)
    arm_lookup = {name: index for index, name in enumerate(ARM_JOINT_NAMES)}
    gripper_values = (1.0 - episode.cameras[0].tcp_state[:, 6]) * 0.8
    for joint in tree.joints():
        if joint.name in arm_lookup:
            values = episode.observation_joints[:, arm_lookup[joint.name]]
        elif joint.name == "finger_joint":
            values = gripper_values
        elif joint.mimic is not None:
            if joint.mimic.joint != "finger_joint":
                raise ValueError(f"Unsupported URDF mimic source {joint.mimic.joint!r}")
            values = gripper_values * joint.mimic.multiplier + joint.mimic.offset
        else:
            values = np.zeros(episode.frame_count, dtype=np.float64)
        rr.send_columns(
            f"robot/joint_transforms/{joint.name}",
            indexes=[time_column],
            columns=joint.compute_transform_columns(values, clamp=False),
        )


def log_scalar_series(episode: Episode) -> None:
    time_column = rr.TimeColumn(TIME_TIMELINE, duration=episode.timestamps_s)
    for camera in episode.cameras:
        for index, name in enumerate(("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")):
            rr.send_columns(
                f"signals/tcp/{camera.serial}/{name}",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=camera.tcp_state[:, index]),
            )
        rr.send_columns(
            f"signals/gripper/{camera.serial}/open",
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=camera.tcp_state[:, 6]),
        )
    for index, name in enumerate(ARM_JOINT_NAMES):
        for role, values in (
            ("observation", episode.observation_joints),
            ("action", episode.action_joints),
        ):
            rr.send_columns(
                f"signals/joints/{name}/{role}",
                indexes=[time_column],
                columns=rr.Scalars.columns(scalars=values[:, index]),
            )


def language_lines(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        value = metadata.get(key)
        values = value if isinstance(value, list) else [value]
        lines.extend(str(item) for item in values if item)
    return lines


def log_episode_info(episode: Episode, args: argparse.Namespace) -> None:
    instruction_lines = language_lines(episode.metadata)
    instructions = "\n".join(f"  {index}. {text}" for index, text in enumerate(instruction_lines, 1))
    camera_lines = []
    for camera in episode.cameras:
        camera_to_base = np.linalg.inv(camera.base_to_camera[0])
        position = camera_to_base[:3, 3]
        camera_lines.append(
            f"  - `{camera.label}` / `{camera.serial}`: "
            f"{camera.width}×{camera.height}, base XYZ "
            f"[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m"
        )
    cadence = np.diff(episode.timestamps_s)
    fps = 1.0 / float(np.median(cadence)) if len(cadence) else 0.0
    status = str(episode.metadata.get("status", "unknown"))
    document = "\n".join(
        [
            f"# DROID: {episode.path.name}",
            "",
            f"- **Stage/status:** `{status}`",
            f"- **Frames:** {episode.frame_count}",
            f"- **Duration:** {episode.timestamps_s[-1]:.3f} s",
            f"- **Median rate:** {fps:.2f} Hz",
            "- **Arm state:** `observations/joint_position.npy` (measured)",
            "- **TCP pose / gripper opening:** `TCP/<camera>/state.npy` (1=open, 0=closed)",
            "- **Joint action:** `action/joint_position.npy` (commanded absolute target)",
            "- **World frame:** Franka base; right-handed, +Z up",
            "- **Camera frame:** OpenCV RDF; stored extrinsic is base → camera",
            (
                f"- **Valid depth:** {args.min_depth_m:g} < z ≤ {args.max_depth_m:g} m, "
                f"pixel stride {args.point_cloud_stride}"
            ),
            "",
            "## Instructions",
            instructions or "  (none)",
            "",
            "## Third-person cameras",
            *camera_lines,
            "",
            "The source contains an episode outcome/status, but no per-frame semantic phase labels. "
            "The current panel therefore reports trajectory progress without inventing task stages.",
        ]
    )
    rr.log(
        "dashboard/episode",
        rr.TextDocument(document, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def log_pose_visuals(episode: Episode, index: int, history: int) -> None:
    start = max(0, index - history + 1)
    for camera, color in zip(episode.cameras, (STATE_COLOR, ACTION_COLOR)):
        root = f"world/tcp/{camera.serial}"
        if index == 0:
            for name in ("current", "axes", "history"):
                rr.log(f"{root}/{name}", rr.CoordinateFrame(WORLD_FRAME), static=True)
        pose = camera.tcp_world_poses[index]
        position = pose[:3, 3]
        opening = camera.tcp_state[index, 6]
        rr.log(
            f"{root}/current",
            rr.Points3D([position], colors=[color], radii=0.009,
                        labels=[f"TCP (open={opening:.2f})"]),
        )
        rr.log(
            f"{root}/axes",
            rr.Arrows3D(
                origins=np.repeat(position[None], 3, axis=0),
                vectors=pose[:3, :3].T * 0.055,
                colors=[[255, 55, 55], [55, 220, 75], [65, 125, 255]],
                radii=0.0015,
            ),
        )
        rr.log(
            f"{root}/history",
            rr.LineStrips3D([camera.tcp_world_poses[start:index + 1, :3, 3]],
                            colors=[color], radii=0.002),
        )


def log_current_document(
    episode: Episode,
    index: int,
    point_counts: dict[str, int],
) -> None:
    progress = 100.0 * index / max(episode.frame_count - 1, 1)
    points = ", ".join(
        f"{camera.label}: {point_counts.get(camera.serial, 0):,}"
        for camera in episode.cameras
    )
    document = "\n".join(
        [
            "# Current frame",
            "",
            f"- **Progress:** {index + 1}/{episode.frame_count} ({progress:.1f}%)",
            f"- **Time:** {episode.timestamps_s[index]:.3f} s",
            f"- **Valid sampled points:** {points}",
            *[
                f"- **TCP {camera.serial} (camera frame):** "
                f"XYZ `{camera.tcp_state[index, :3].round(4).tolist()}` m; "
                f"RPY `{camera.tcp_state[index, 3:6].round(4).tolist()}` rad; "
                f"open **{camera.tcp_state[index, 6]:.4f}** (1=open, 0=closed)"
                for camera in episode.cameras
            ],
        ]
    )
    rr.log(
        "dashboard/current",
        rr.TextDocument(document, media_type=rr.MediaType.MARKDOWN),
    )


def camera_eye_controls(camera: Camera) -> rrb.EyeControls3D:
    """Create a 3D eye aligned with one calibrated OpenCV camera."""
    camera_to_base = np.linalg.inv(camera.base_to_camera[0])
    position = camera_to_base[:3, 3]
    rotation = camera_to_base[:3, :3]
    # OpenCV camera coordinates are right/down/forward. Rerun expects a world
    # eye position, a point in front of it, and an upward direction.
    look_target = position + 0.55 * rotation[:, 2]
    eye_up = -rotation[:, 1]
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.FirstPerson,
        position=position,
        look_target=look_target,
        eye_up=eye_up,
        speed=0.25,
        # A tracked Pinhole takes over the exact calibrated camera pose in the
        # Rerun viewer. The explicit vectors above also make the initial view
        # deterministic if tracking is unavailable in another Rerun version.
        tracking_entity=camera.entity_path,
    )


def make_blueprint(episode: Episode, show_depth: bool) -> rrb.Blueprint:
    spatial = rrb.Spatial3DView(
        origin="world",
        # Include only the pinhole entities, not their raw depth children: the
        # 3D scene must use the explicitly filtered colored point clouds.
        contents=[
            "world/**",
            "robot_model/**",
            "robot/**",
            *(camera.entity_path for camera in episode.cameras),
        ],
        name="Two-view RGB-D point cloud + Franka replay",
        eye_controls=rrb.EyeControls3D(
            position=[1.35, 1.35, 1.10],
            look_target=[0.48, 0.0, 0.15],
            eye_up=[0.0, 0.0, 1.0],
        ),
    )
    comparison_views: list[Any] = []
    for camera in episode.cameras:
        comparison_views.extend(
            [
                rrb.Spatial2DView(
                    origin=f"{camera.entity_path}/rgb",
                    name=f"{camera.label} original RGB ({camera.serial})",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0.0, float(camera.width)],
                        y_range=[0.0, float(camera.height)],
                    ),
                ),
                rrb.Spatial3DView(
                    origin="world",
                    contents=["robot_model/**", "robot/**", f"world/tcp/{camera.serial}/**"],
                    name=f"{camera.label} robot replay ({camera.serial})",
                    eye_controls=camera_eye_controls(camera),
                    line_grid=False,
                ),
            ]
        )
    comparison = rrb.Grid(
        *comparison_views,
        grid_columns=2,
        name="Original RGB vs camera-matched robot replay",
    )
    if show_depth:
        depth_views = [
            rrb.Spatial2DView(
                origin=f"{camera.entity_path}/depth",
                name=f"{camera.label} depth ({camera.serial})",
                visual_bounds=rrb.VisualBounds2D(
                    x_range=[0.0, float(camera.width)],
                    y_range=[0.0, float(camera.height)],
                ),
            )
            for camera in episode.cameras
        ]
        cameras: Any = rrb.Tabs(
            comparison,
            rrb.Grid(*depth_views, grid_columns=2, name="Depth images"),
            active_tab=0,
            name="Third-person comparison",
        )
    else:
        cameras = comparison
    top = rrb.Horizontal(spatial, cameras, column_shares=[2, 3], name="Replay")

    info = rrb.Tabs(
        rrb.TextDocumentView(origin="dashboard/current", name="Current frame / stage"),
        rrb.TextDocumentView(origin="dashboard/episode", name="Episode metadata"),
        active_tab=0,
        name="DROID episode",
    )
    signals = rrb.Tabs(
        rrb.TimeSeriesView(origin="signals/tcp", name="TCP camera-frame XYZ / RPY"),
        rrb.TimeSeriesView(origin="signals/joints", name="Joint state vs action"),
        rrb.TimeSeriesView(origin="signals/gripper", name="TCP gripper opening (1=open)"),
        active_tab=0,
        name="State and action",
    )
    lower = rrb.Horizontal(info, signals, column_shares=[2, 5])
    return rrb.Blueprint(
        rrb.Vertical(top, lower, row_shares=[3, 1]),
        rrb.TimePanel(timeline=TIME_TIMELINE, expanded=True),
        auto_views=False,
        collapse_panels=True,
    )


def read_rgb_depth(camera: Camera, index: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(camera.rgb_files[index]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(camera.depth_files[index]) as image:
        depth_mm = np.asarray(image)
    expected_rgb_shape = (camera.height, camera.width, 3)
    expected_depth_shape = (camera.height, camera.width)
    if (
        rgb.shape != expected_rgb_shape
        or depth_mm.shape != expected_depth_shape
        or depth_mm.dtype != np.uint16
    ):
        raise ValueError(
            f"Camera {camera.serial} frame {index} has RGB/depth shapes "
            f"{rgb.shape}/{depth_mm.shape} and depth dtype {depth_mm.dtype}; expected "
            f"{expected_rgb_shape}/{expected_depth_shape} and uint16"
        )
    return rgb, depth_mm


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        episode = load_episode(args.episode, args.cameras, args.max_frames)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid DROID episode: {error}") from error

    output: Path | None = None
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("droid_two_camera_robot_replay", spawn=False)
    recording = rr.get_global_data_recording()
    if recording is None:
        raise SystemExit("Rerun recording failed to initialize")
    use_web = configure_rerun_output(recording, output, web=args.web)

    print(
        f"Loading {episode.frame_count} DROID frames from {episode.path.name}...",
        flush=True,
    )

    log_world_axes()
    log_cameras(episode.cameras)
    log_episode_info(episode, args)
    log_scalar_series(episode)
    rr.send_blueprint(make_blueprint(episode, not args.no_depth_images))

    point_totals = {camera.serial: 0 for camera in episode.cameras}
    with tempfile.TemporaryDirectory(prefix="droid_rerun_") as temporary:
        if not args.no_robot:
            try:
                log_robot_replay(
                    recording,
                    episode,
                    args.urdf.expanduser().resolve(),
                    Path(temporary),
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                raise SystemExit(f"Robot replay failed: {error}") from error

        for index, seconds in enumerate(episode.timestamps_s):
            rr.set_time(FRAME_TIMELINE, sequence=index)
            rr.set_time(TIME_TIMELINE, duration=float(seconds))
            point_counts: dict[str, int] = {}
            for camera in episode.cameras:
                rgb, depth_mm = read_rgb_depth(camera, index)
                rr.log(f"{camera.entity_path}/rgb", rr.Image(rgb))
                if not args.no_depth_images:
                    rr.log(
                        f"{camera.entity_path}/depth",
                        rr.DepthImage(
                            depth_mm,
                            meter=1000.0,
                            depth_range=[
                                args.min_depth_m * 1000.0,
                                args.max_depth_m * 1000.0,
                            ],
                            colormap="Turbo",
                        ),
                    )
                if not args.no_point_cloud:
                    points, colors, count = backproject_rgbd(
                        depth_mm,
                        rgb,
                        camera.intrinsic[index],
                        camera.base_to_camera[index],
                        args.point_cloud_stride,
                        args.min_depth_m,
                        args.max_depth_m,
                    )
                    point_path = f"world/point_cloud/{camera.serial}"
                    if index == 0:
                        rr.log(point_path, rr.CoordinateFrame(WORLD_FRAME), static=True)
                    rr.log(point_path, rr.Points3D(points, colors=colors, radii=0.003))
                    point_counts[camera.serial] = count
                    point_totals[camera.serial] += count
                else:
                    point_counts[camera.serial] = 0
            log_pose_visuals(episode, index, args.history)
            log_current_document(episode, index, point_counts)

        recording.flush()

    camera_summary = ", ".join(
        f"{camera.label}/{camera.serial}: "
        f"{point_totals[camera.serial] / episode.frame_count:.0f} valid sampled points/frame"
        for camera in episode.cameras
    )
    print(
        f"Visualized {episode.frame_count} DROID frames over "
        f"{episode.timestamps_s[-1]:.3f} s ({camera_summary})."
    )
    if output is not None:
        print(f"Saved Rerun recording: {output}")
    elif use_web:
        wait_for_web_viewer(recording)


if __name__ == "__main__":
    main()
