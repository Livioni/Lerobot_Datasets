#!/usr/bin/env python3
"""Visualize converted RoboTwin TCP trajectories in Rerun.

For each frame this script replays the RoboTwin ARX5 URDF and shows the current
left/right TCP, its coordinate axes, gripper state, and the most recent N TCP
samples.  The history length is 10 by default.  RGB-D and camera-frame TCP
poses are transformed into the fixed robot-footprint frame so the robot,
trajectory, and colored point cloud remain aligned during playback.

Example:
    conda run -n rerun python visualize_robotwin_tcp_rerun.py \
        4d_datasets/adjust_bottle/episode_0000000 --history 10
"""

from __future__ import annotations

import argparse
import json
import socket
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

import convert_robotwin_tcp as tcp_converter
import visualize_lerobot_rerun as lerobot_viz


TIMELINE = "episode_time"
DEFAULT_RERUN_PORT = 9876
SIDE_COLORS = {
    "left": np.array([80, 200, 255], dtype=np.uint8),
    "right": np.array([255, 170, 70], dtype=np.uint8),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def choose_rerun_port(preferred: int = DEFAULT_RERUN_PORT) -> int:
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


def rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
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


def states_to_pose_matrices(states: np.ndarray) -> np.ndarray:
    poses = np.broadcast_to(np.eye(4, dtype=np.float64), (len(states), 4, 4)).copy()
    poses[:, :3, 3] = states[:, :3]
    poses[:, :3, :3] = np.asarray(
        [rotation_matrix_from_rpy(rpy) for rpy in states[:, 3:6]], dtype=np.float64
    )
    return poses


def load_homogeneous_extrinsics(path: Path, frame_count: int) -> np.ndarray:
    extrinsics = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if extrinsics.shape in ((3, 4), (4, 4)):
        extrinsics = np.broadcast_to(extrinsics, (frame_count, *extrinsics.shape)).copy()
    if extrinsics.shape == (frame_count, 3, 4):
        bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (frame_count, 1, 4))
        extrinsics = np.concatenate((extrinsics, bottom), axis=1)
    if extrinsics.shape != (frame_count, 4, 4):
        raise ValueError(f"{path} has shape {extrinsics.shape}; expected [T,3,4]")
    return extrinsics


def load_tcp_states(episode: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    tcp_dir = episode / "TCP"
    metadata_path = tcp_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; run convert_robotwin_tcp.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    states = {
        side: np.asarray(
            np.load(tcp_dir / f"{side}_state.npy", allow_pickle=False),
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    shapes = {side: value.shape for side, value in states.items()}
    if any(value.ndim != 2 or value.shape[1] != 7 for value in states.values()):
        raise ValueError(f"TCP states must have shape [T,7], got {shapes}")
    if len(states["left"]) != len(states["right"]):
        raise ValueError(f"Left/right frame counts differ: {shapes}")
    if not all(np.all(np.isfinite(value)) for value in states.values()):
        raise ValueError("TCP states contain non-finite values")
    return states, metadata


def backproject_rgbd(
    depth_mm: np.ndarray, rgb: np.ndarray, intrinsic: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    sampled_depth = (
        np.asarray(depth_mm[::stride, ::stride], dtype=np.float32) * 0.001
    )
    sampled_rgb = np.asarray(rgb[::stride, ::stride, :3], dtype=np.uint8)
    rows = np.arange(0, depth_mm.shape[0], stride, dtype=np.float32)
    columns = np.arange(0, depth_mm.shape[1], stride, dtype=np.float32)
    pixel_u, pixel_v = np.meshgrid(columns, rows)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    z = sampled_depth[valid]
    x = (pixel_u[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (pixel_v[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.column_stack((x, y, z)), np.ascontiguousarray(sampled_rgb[valid])


def transform_points(points: np.ndarray, parent_from_child: np.ndarray) -> np.ndarray:
    """Transform row-vector 3D points with a homogeneous pose matrix."""
    rotation = parent_from_child[:3, :3]
    return (
        points[:, 0, None] * rotation[None, :, 0]
        + points[:, 1, None] * rotation[None, :, 1]
        + points[:, 2, None] * rotation[None, :, 2]
        + parent_from_child[None, :3, 3]
    )


def load_robot_state(episode: Path, frame_count: int) -> np.ndarray:
    path = episode / "robot_state.npy"
    state = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if state.shape != (frame_count, 14):
        raise ValueError(
            f"{path} has shape {state.shape}; expected ({frame_count}, 14)"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{path} contains non-finite values")
    return state


def log_robot_replay(
    recording: rr.RecordingStream,
    robot_state: np.ndarray,
    timestamps: np.ndarray,
    urdf_path: Path,
    temporary_directory: Path,
) -> None:
    """Load the RoboTwin ARX5 URDF and animate both follower arms."""
    lerobot_viz.prepend_ros_package_path(urdf_path.parent)
    prepared_urdf = temporary_directory / "robotwin_arx5_visual.urdf"
    lerobot_viz.prepare_follower_visual_urdf(
        urdf_path,
        prepared_urdf,
        patch_dae=True,
        mesh_overrides=lerobot_viz.ARX5_FULL_MESH_OVERRIDES,
        force_solid_texture_links=lerobot_viz.ARX5_FORCE_SOLID_TEXTURE_LINKS,
        box_overrides={},
        material_overrides=lerobot_viz.ARX5_FULL_MATERIAL_OVERRIDES,
    )
    try:
        urdf_tree = rr.urdf.UrdfTree.from_file_path(
            prepared_urdf,
            entity_path_prefix=lerobot_viz.ROBOT_ENTITY_PATH,
            static_transform_entity_path=(
                lerobot_viz.ROBOT_STATIC_TRANSFORMS_ENTITY_PATH
            ),
        )
        urdf_tree.log_urdf_to_recording(recording)
    except Exception as error:
        raise RuntimeError(f"Could not load RoboTwin URDF {urdf_path}: {error}") from error

    time_column = rr.TimeColumn(TIMELINE, duration=timestamps)
    for side, prefix, offset in (("left", "fl", 0), ("right", "fr", 7)):
        for joint_index in range(1, 7):
            joint = urdf_tree.get_joint_by_name(f"{prefix}_joint{joint_index}")
            if joint is None:
                raise RuntimeError(
                    f"URDF is missing joint {prefix}_joint{joint_index}"
                )
            rr.send_columns(
                lerobot_viz.ROBOT_TRANSFORMS_ENTITY_PATH,
                indexes=[time_column],
                columns=joint.compute_transform_columns(
                    robot_state[:, offset + joint_index - 1], clamp=True
                ),
            )

        gripper = np.clip(robot_state[:, offset + 6], 0.0, 1.0)
        for joint_index in (7, 8):
            joint = urdf_tree.get_joint_by_name(f"{prefix}_joint{joint_index}")
            if joint is None:
                raise RuntimeError(f"URDF is missing joint {prefix}_joint{joint_index}")
            upper = float(joint.limit_upper if joint.limit_upper is not None else 0.04765)
            rr.send_columns(
                lerobot_viz.ROBOT_TRANSFORMS_ENTITY_PATH,
                indexes=[time_column],
                columns=joint.compute_transform_columns(gripper * upper, clamp=False),
            )

    print(f"Loaded RoboTwin robot replay: {urdf_path}")


def bind_spatial_entities_to_footprint() -> None:
    """Attach every numerically base-framed spatial leaf to the URDF frame graph.

    Rerun named coordinate frames are not inherited through entity paths.  Each
    leaf that logs a spatial archetype therefore needs its own identity edge to
    the URDF's ``footprint`` frame.
    """
    entity_paths = [
        "robot/scene",
        *(
            f"robot/tcp/{side}/{leaf}"
            for side in ("left", "right")
            for leaf in ("current", "history", "history_points", "axes")
        ),
    ]
    for index, entity_path in enumerate(entity_paths):
        frame_name = f"tcp_visual_base_{index}"
        rr.log(
            entity_path,
            rr.CoordinateFrame(frame_name),
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                mat3x3=np.eye(3),
                relation=rr.TransformRelation.ParentFromChild,
                parent_frame=lerobot_viz.CAMERA_BASE_FRAME,
                child_frame=frame_name,
            ),
            static=True,
        )


def log_static_scene(
    episode: Path,
    camera: str,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    history: int,
) -> None:
    rr.log("robot", rr.ViewCoordinates.FLU, static=True)
    lerobot_viz.log_robot_footprint_frame()
    bind_spatial_entities_to_footprint()
    rr.log(
        "camera",
        rr.Pinhole(
            image_from_camera=intrinsic,
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
        static=True,
    )
    rr.log(
        "episode_info",
        rr.TextDocument(
            "\n".join(
                [
                    f"# {episode.parent.name}/{episode.name}",
                    "",
                    "- 3D replay frame: robot footprint (FLU)",
                    f"- Stored TCP frame: `{camera}` camera (OpenCV RDF)",
                    f"- Frames: {frame_count}",
                    f"- FPS: {fps:g}",
                    f"- TCP history: last {history} samples including current",
                    "- TCP state: `[x, y, z, roll, pitch, yaw, gripper_open]`",
                ]
            ),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )


def make_blueprint(show_rgb: bool, show_robot: bool) -> rrb.Blueprint:
    spatial_name = (
        "Robot replay + TCP trajectory + RGB-D scene"
        if show_robot
        else "TCP trajectory + RGB-D scene"
    )
    spatial = rrb.Spatial3DView(origin="robot", name=spatial_name)
    signal_view = rrb.TimeSeriesView(origin="signals", name="Gripper open/closed")
    info_view = rrb.TextDocumentView(origin="episode_info", name="Episode")
    if show_rgb:
        rgb_view = rrb.Spatial2DView(origin="camera/rgb", name="Camera RGB")
        upper: object = rrb.Horizontal(spatial, rgb_view, column_shares=[2, 1])
    else:
        upper = spatial
    layout = rrb.Vertical(
            upper,
            rrb.Horizontal(info_view, signal_view, column_shares=[1, 2]),
            row_shares=[3, 1],
        )
    return rrb.Blueprint(
        layout,
        rrb.TimePanel(timeline=TIMELINE, expanded=True),
        auto_views=False,
        collapse_panels=True,
    )


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Visualize RoboTwin URDF replay and TCP trajectories with Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("episode", type=Path, help="Converted RoboTwin episode directory")
    parser.add_argument(
        "--history",
        type=positive_int,
        default=10,
        metavar="N",
        help="Number of recent TCP samples shown per arm, including current",
    )
    parser.add_argument(
        "--point-cloud-stride",
        type=positive_int,
        default=2,
        metavar="N",
        help="Sample every Nth RGB-D pixel along each image axis",
    )
    parser.add_argument(
        "--no-point-cloud",
        action="store_true",
        help="Do not back-project camera RGB-D",
    )
    parser.add_argument(
        "--no-rgb", action="store_true", help="Do not log the 2D RGB image stream"
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="TCP coordinate-axis length in metres",
    )
    robot_group = parser.add_mutually_exclusive_group()
    robot_group.add_argument(
        "--urdf",
        type=Path,
        default=(
            script_root
            / "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        ),
        help="RoboTwin ARX5 URDF used for robot replay",
    )
    robot_group.add_argument(
        "--no-robot", action="store_true", help="Disable URDF robot replay"
    )
    parser.add_argument(
        "--output", type=Path, help="Write an .rrd recording instead of opening Rerun"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    states, tcp_metadata = load_tcp_states(episode)
    frame_count = len(states["left"])
    camera = str(tcp_metadata.get("camera", "head_view"))

    episode_metadata_path = episode / "metadata.json"
    episode_metadata = (
        json.loads(episode_metadata_path.read_text(encoding="utf-8"))
        if episode_metadata_path.is_file()
        else {}
    )
    fps = float(episode_metadata.get("frequency_hz", 15.0))
    if fps <= 0.0:
        raise SystemExit(f"Invalid frequency_hz: {fps}")
    if args.axis_length <= 0.0:
        raise SystemExit("--axis-length must be positive")
    urdf_path = args.urdf.expanduser().resolve() if args.urdf is not None else None
    if not args.no_robot and (urdf_path is None or not urdf_path.is_file()):
        raise SystemExit(f"RoboTwin URDF not found: {urdf_path}")

    intrinsic_path = episode / "intrinsics" / f"{camera}.npy"
    extrinsic_path = episode / "extrinsics" / f"{camera}.npy"
    if not intrinsic_path.is_file() or not extrinsic_path.is_file():
        raise SystemExit(f"Missing intrinsics/extrinsics for camera {camera!r}")
    intrinsic = np.asarray(
        np.load(intrinsic_path, allow_pickle=False), dtype=np.float64
    )
    if intrinsic.shape != (3, 3):
        raise SystemExit(f"{intrinsic_path} has shape {intrinsic.shape}; expected [3,3]")
    world_to_camera = load_homogeneous_extrinsics(extrinsic_path, frame_count)

    image_paths = sorted((episode / "images" / camera).glob("*.png"))
    depth_paths = sorted((episode / "depths" / camera).glob("*.png"))
    need_rgb = not args.no_rgb or not args.no_point_cloud
    if need_rgb and len(image_paths) != frame_count:
        raise SystemExit(
            f"Expected {frame_count} RGB frames for {camera}, "
            f"found {len(image_paths)}"
        )
    if not args.no_point_cloud and len(depth_paths) != frame_count:
        raise SystemExit(
            f"Expected {frame_count} depth frames for {camera}, "
            f"found {len(depth_paths)}"
        )

    if image_paths:
        with Image.open(image_paths[0]) as first_image:
            width, height = first_image.size
    else:
        width = int(episode_metadata.get("image_width", 320))
        height = int(episode_metadata.get("image_height", 240))

    camera_from_tcp = {
        side: states_to_pose_matrices(value) for side, value in states.items()
    }
    camera_to_world = np.linalg.inv(world_to_camera)
    world_from_tcp = {
        side: camera_to_world @ poses for side, poses in camera_from_tcp.items()
    }
    base_from_world = tcp_converter.world_to_base_transform()
    camera_to_base = base_from_world[None, :, :] @ camera_to_world
    base_from_tcp = {
        side: base_from_world[None, :, :] @ poses
        for side, poses in world_from_tcp.items()
    }
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    robot_state = load_robot_state(episode, frame_count) if not args.no_robot else None

    rr.init(f"robotwin_tcp_{episode.parent.name}_{episode.name}", spawn=False)
    recording = rr.get_global_data_recording()
    if recording is None:
        raise SystemExit("Rerun recording failed to initialize")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output)
    else:
        port = choose_rerun_port()
        if port != DEFAULT_RERUN_PORT:
            print(f"Rerun port {DEFAULT_RERUN_PORT} is occupied; using {port}")
        recording.spawn(port=port)

    log_static_scene(
        episode, camera, intrinsic, width, height, frame_count, fps, args.history
    )
    rr.send_blueprint(make_blueprint(not args.no_rgb, not args.no_robot))

    with tempfile.TemporaryDirectory(prefix="robotwin-tcp-rerun-") as temporary:
        if robot_state is not None:
            assert urdf_path is not None
            log_robot_replay(
                recording,
                robot_state,
                timestamps,
                urdf_path,
                Path(temporary),
            )

        print(
            f"Loading {episode.parent.name}/{episode.name}: {frame_count} frames, "
            f"camera={camera}, history={args.history}"
        )
        for frame_index in range(frame_count):
            rr.set_time(TIMELINE, duration=float(timestamps[frame_index]))
            rgb = None
            if need_rgb:
                with Image.open(image_paths[frame_index]) as image:
                    # Copy before closing PIL's file-backed image buffer.
                    rgb = np.asarray(image.convert("RGB")).copy()
            if not args.no_rgb and rgb is not None:
                rr.log("camera/rgb", rr.Image(rgb))
            if not args.no_point_cloud and rgb is not None:
                with Image.open(depth_paths[frame_index]) as image:
                    depth_mm = np.asarray(image).copy()
                points, colors = backproject_rgbd(
                    depth_mm, rgb, intrinsic, args.point_cloud_stride
                )
                points = transform_points(points, camera_to_base[frame_index])
                rr.log("robot/scene", rr.Points3D(points, colors=colors))

            history_start = max(0, frame_index - args.history + 1)
            for side in ("left", "right"):
                state = states[side][frame_index]
                color = SIDE_COLORS[side]
                current_pose = base_from_tcp[side][frame_index]
                current_position = current_pose[:3, 3]
                history = base_from_tcp[side][
                    history_start : frame_index + 1, :3, 3
                ]
                entity = f"robot/tcp/{side}"
                label = (
                    f"{side.upper()} TCP  camera xyz[m] "
                    f"{state[0]:+.3f} {state[1]:+.3f} {state[2]:+.3f}\n"
                    f"camera rpy[deg] {np.rad2deg(state[3]):+.1f} "
                    f"{np.rad2deg(state[4]):+.1f} "
                    f"{np.rad2deg(state[5]):+.1f}  "
                    f"gripper={'OPEN' if state[6] >= 0.5 else 'CLOSED'}"
                )
                rr.log(
                    f"{entity}/current",
                    rr.Points3D(
                        [current_position],
                        radii=[0.016],
                        colors=[color],
                        labels=[label],
                    ),
                )
                rr.log(
                    f"{entity}/history",
                    rr.LineStrips3D([history], radii=[0.006], colors=[color]),
                )
                history_colors = np.tile(np.append(color, 210), (len(history), 1))
                rr.log(
                    f"{entity}/history_points",
                    rr.Points3D(history, radii=0.007, colors=history_colors),
                )
                rr.log(
                    f"{entity}/axes",
                    rr.Arrows3D(
                        origins=np.repeat(current_position[None, :], 3, axis=0),
                        vectors=current_pose[:3, :3].T * args.axis_length,
                        radii=[0.004] * 3,
                        colors=[
                            [255, 60, 60],
                            [60, 220, 80],
                            [70, 130, 255],
                        ],
                        labels=["X", "Y", "Z"],
                    ),
                )
                rr.log(
                    f"signals/{side}_gripper_open", rr.Scalars(float(state[6]))
                )

        recording.flush()
    if args.output:
        print(f"Saved Rerun recording: {args.output.expanduser().resolve()}")
    else:
        print(f"Loaded in Rerun. Scrub or play the {TIMELINE} timeline.")


if __name__ == "__main__":
    main()
