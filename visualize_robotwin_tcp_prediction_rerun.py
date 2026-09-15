#!/usr/bin/env python3
"""Compare predicted and ground-truth RoboTwin TCP trajectories in Rerun.

The prediction JSON stores left/right TCP poses in the per-frame OpenCV camera
frame.  This script applies the episode extrinsics and the same world-to-robot
base transform used by ``visualize_robotwin_tcp_rerun.py`` so predictions,
ground truth, RGB-D, and the URDF robot replay share one fixed 3D frame.

By default it opens the requested ``beat_block_hammer`` episode:

    conda run -n rerun python visualize_robotwin_tcp_prediction_rerun.py

To save a recording instead of opening the viewer:

    conda run -n rerun python visualize_robotwin_tcp_prediction_rerun.py \
        --output rrd_output/place_dual_shoes_tcp_comparison.rrd
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

import convert_robotwin_tcp as tcp_converter
import visualize_lerobot_rerun as lerobot_viz
import visualize_robotwin_tcp_rerun as base_viz


TIMELINE = base_viz.TIMELINE
DEFAULT_EPISODE = (
    Path(__file__).resolve().parent
    / "4d_datasets/beat_block_hammer/episode_0000000"
)
DEFAULT_PREDICTION_FILE = "tcp_episode.json"
DEFAULT_GROUND_TRUTH_DIR = "TCP_third"

# A pair of related colors is used for each arm.  Ground truth is the darker,
# thicker trajectory and prediction is the brighter, thinner trajectory.
COLORS = {
    "left": {
        "ground_truth": np.array([30, 120, 255], dtype=np.uint8),
        "prediction": np.array([40, 245, 255], dtype=np.uint8),
    },
    "right": {
        "ground_truth": np.array([255, 120, 25], dtype=np.uint8),
        "prediction": np.array([255, 55, 190], dtype=np.uint8),
    },
}
AXIS_COLORS = ([255, 60, 60], [60, 220, 80], [70, 130, 255])


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or a positive integer")
    return parsed


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def load_prediction_states(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Load JSON predictions as [x, y, z, roll, pitch, yaw, gripper]."""
    try:
        document = _require_mapping(
            json.loads(path.read_text(encoding="utf-8")), str(path)
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read prediction JSON {path}: {error}") from error

    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty 'frames' array")
    declared_count = document.get("num_frames")
    if declared_count is not None and int(declared_count) != len(frames):
        raise ValueError(
            f"{path} declares {declared_count} frames but contains {len(frames)}"
        )

    fps = float(document.get("frame_rate_hz", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path} has invalid frame_rate_hz={fps!r}")

    states = {
        side: np.empty((len(frames), 7), dtype=np.float64)
        for side in ("left", "right")
    }
    confidence = {
        side: np.full(len(frames), np.nan, dtype=np.float64)
        for side in ("left", "right")
    }
    timestamps = np.empty(len(frames), dtype=np.float64)

    for expected_index, raw_frame in enumerate(frames):
        frame = _require_mapping(raw_frame, f"frames[{expected_index}]")
        frame_index = int(frame.get("frame_index", expected_index))
        if frame_index != expected_index:
            raise ValueError(
                f"frames[{expected_index}] has frame_index={frame_index}; "
                "frames must be ordered and contiguous"
            )
        timestamps[expected_index] = float(
            frame.get("time_seconds", expected_index / fps)
        )
        for side in ("left", "right"):
            item = _require_mapping(frame.get(side), f"frames[{expected_index}].{side}")
            xyz = np.asarray(item.get("xyz_m"), dtype=np.float64)
            rpy = np.asarray(item.get("rpy_rad"), dtype=np.float64)
            if xyz.shape != (3,) or rpy.shape != (3,):
                raise ValueError(
                    f"frames[{expected_index}].{side} must contain "
                    "xyz_m[3] and rpy_rad[3]"
                )
            if "gripper_probability" in item:
                gripper = float(item["gripper_probability"])
            elif "gripper_open" in item:
                gripper = float(bool(item["gripper_open"]))
            else:
                raise ValueError(
                    f"frames[{expected_index}].{side} has no gripper prediction"
                )
            states[side][expected_index] = np.concatenate((xyz, rpy, [gripper]))
            if "confidence" in item:
                confidence[side][expected_index] = float(item["confidence"])

    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path} timestamps must be finite and strictly increasing")
    if any(not np.all(np.isfinite(value)) for value in states.values()):
        raise ValueError(f"{path} contains non-finite TCP predictions")
    return states, confidence, timestamps, document


def orientation_errors_deg(
    ground_truth_poses: np.ndarray, prediction_poses: np.ndarray
) -> np.ndarray:
    """Return SO(3) geodesic angle errors, independent of RPY wrapping."""
    ground_truth_rotation = ground_truth_poses[:, :3, :3]
    prediction_rotation = prediction_poses[:, :3, :3]
    relative = np.swapaxes(ground_truth_rotation, 1, 2) @ prediction_rotation
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def load_prediction_robot_state(path: Path, frame_count: int) -> np.ndarray:
    """Load and validate a solver-produced RoboTwin [T,14] state."""
    try:
        state = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    except OSError as error:
        raise ValueError(f"Could not read prediction robot state {path}: {error}") from error
    if state.shape != (frame_count, 14):
        raise ValueError(
            f"{path} has shape {state.shape}; expected ({frame_count}, 14)"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{path} contains non-finite values")
    for column in (6, 13):
        if not set(np.unique(state[:, column])).issubset({0.0, 1.0}):
            raise ValueError(f"{path} gripper column {column} is not binary")
    return state


def wrapped_joint_errors(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Return signed joint error in [-pi, pi], independent of angle wrapping."""
    delta = prediction - ground_truth
    return np.arctan2(np.sin(delta), np.cos(delta))


def make_joint_summary(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute per-joint error and prediction continuity statistics."""
    metrics: dict[str, Any] = {}
    errors_deg: dict[str, np.ndarray] = {}
    joint_values_deg: dict[str, np.ndarray] = {}
    for side, prefix, offset in (
        ("left", "fl", 0),
        ("right", "fr", 7),
    ):
        ground_truth_arm = ground_truth[:, offset : offset + 6]
        prediction_arm = prediction[:, offset : offset + 6]
        error = wrapped_joint_errors(ground_truth_arm, prediction_arm)
        error_deg = np.rad2deg(error)
        errors_deg[side] = error_deg
        joint_values_deg[f"{side}_ground_truth"] = np.rad2deg(ground_truth_arm)
        joint_values_deg[f"{side}_prediction"] = np.rad2deg(prediction_arm)

        per_joint: list[dict[str, Any]] = []
        for joint_index in range(6):
            values = error_deg[:, joint_index]
            per_joint.append(
                {
                    "name": f"{prefix}_joint{joint_index + 1}",
                    "mae_deg": float(np.mean(np.abs(values))),
                    "rmse_deg": float(np.sqrt(np.mean(values**2))),
                    "max_abs_deg": float(np.max(np.abs(values))),
                }
            )

        if len(prediction_arm) > 1:
            steps = wrapped_joint_errors(prediction_arm[:-1], prediction_arm[1:])
            max_flat = int(np.argmax(np.abs(steps)))
            step_index, joint_index = np.unravel_index(max_flat, steps.shape)
            step_values = np.abs(steps)
            continuity = {
                "max_abs_step_deg": float(
                    np.rad2deg(step_values[step_index, joint_index])
                ),
                "frame_index": int(step_index + 1),
                "joint_name": f"{prefix}_joint{joint_index + 1}",
                "p95_abs_step_deg": float(np.rad2deg(np.percentile(step_values, 95))),
                "p99_abs_step_deg": float(np.rad2deg(np.percentile(step_values, 99))),
            }
        else:
            continuity = {
                "max_abs_step_deg": 0.0,
                "frame_index": 0,
                "joint_name": f"{prefix}_joint1",
                "p95_abs_step_deg": 0.0,
                "p99_abs_step_deg": 0.0,
            }
        metrics[side] = {
            "mae_deg": float(np.mean(np.abs(error_deg))),
            "rmse_deg": float(np.sqrt(np.mean(error_deg**2))),
            "max_abs_deg": float(np.max(np.abs(error_deg))),
            "per_joint": per_joint,
            "continuity": continuity,
        }
    return metrics, errors_deg, joint_values_deg


def bind_spatial_entities_to_footprint() -> None:
    """Attach all comparison geometry to the URDF footprint frame."""
    entity_paths = ["robot/scene"]
    for side in ("left", "right"):
        for source in ("ground_truth", "prediction"):
            entity_paths.extend(
                f"robot/tcp/{side}/{source}/{leaf}"
                for leaf in ("current", "history", "history_points", "axes")
            )
        entity_paths.append(f"robot/tcp/{side}/error_link")

    for index, entity_path in enumerate(entity_paths):
        frame_name = f"tcp_comparison_base_{index}"
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


def make_summary(
    episode: Path,
    prediction_path: Path,
    prediction_metadata: dict[str, Any],
    camera: str,
    ground_truth_dir: str,
    history: int,
    position_errors_cm: dict[str, np.ndarray],
    orientation_errors: dict[str, np.ndarray],
    gripper_accuracy: dict[str, float],
    prediction_robot_state_path: Path | None,
    robot_source: str,
    joint_metrics: dict[str, Any] | None,
) -> str:
    history_description = (
        "all elapsed samples" if history == 0 else f"last {history} samples"
    )
    lines = [
        f"# {episode.parent.name}/{episode.name}",
        "",
        f"- Prediction: `{prediction_path.name}`",
        f"- Ground truth: `{ground_truth_dir}`",
        f"- Camera frame: `{camera}` (OpenCV RDF)",
        "- Display frame: robot footprint (FLU)",
        f"- URDF replay source: `{robot_source}`",
        f"- Trajectory history: {history_description}",
        f"- Gripper threshold: {float(_gripper_threshold(prediction_metadata)):.3g}",
        "",
        "## Colors",
        "",
        "- Left GT: blue; left prediction: cyan",
        "- Right GT: orange; right prediction: magenta",
        "- Red segment: current GT-to-prediction position error",
        "",
        "## TCP episode metrics",
        "",
        "| Arm | Position mean / max | Orientation mean / max | Gripper accuracy |",
        "|---|---:|---:|---:|",
    ]
    for side in ("left", "right"):
        position = position_errors_cm[side]
        orientation = orientation_errors[side]
        lines.append(
            f"| {side} | {position.mean():.2f} / {position.max():.2f} cm | "
            f"{orientation.mean():.2f} / {orientation.max():.2f} deg | "
            f"{100.0 * gripper_accuracy[side]:.1f}% |"
        )

    if joint_metrics is not None and prediction_robot_state_path is not None:
        lines.extend(
            [
                "",
                "## CuRobo IK joint metrics",
                "",
                f"Prediction state: `{prediction_robot_state_path}`",
                "",
                "Wrapped errors compare prediction-TCP IK with ground-truth joints; "
                "ground truth was not used as per-frame IK seed.",
                "",
                "| Arm | Wrapped MAE | RMSE | Max abs | Max prediction step |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for side in ("left", "right"):
            arm = joint_metrics[side]
            continuity = arm["continuity"]
            lines.append(
                f"| {side} | {arm['mae_deg']:.2f}° | {arm['rmse_deg']:.2f}° | "
                f"{arm['max_abs_deg']:.2f}° | {continuity['max_abs_step_deg']:.2f}° "
                f"({continuity['joint_name']}, frame {continuity['frame_index']}) |"
            )
        lines.extend(
            [
                "",
                "| Joint | MAE | RMSE | Max abs |",
                "|---|---:|---:|---:|",
            ]
        )
        for side in ("left", "right"):
            for joint in joint_metrics[side]["per_joint"]:
                lines.append(
                    f"| {joint['name']} | {joint['mae_deg']:.2f}° | "
                    f"{joint['rmse_deg']:.2f}° | {joint['max_abs_deg']:.2f}° |"
                )
    return "\n".join(lines)


def _gripper_threshold(metadata: dict[str, Any]) -> float:
    gripper = metadata.get("gripper", {})
    if isinstance(gripper, dict):
        return float(gripper.get("binary_threshold", 0.5))
    return 0.5


def log_static_scene(
    intrinsic: np.ndarray,
    width: int,
    height: int,
    summary: str,
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
        rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def make_blueprint(show_rgb: bool, show_joints: bool) -> rrb.Blueprint:
    spatial = rrb.Spatial3DView(
        origin="robot", name="Robot replay: prediction vs ground truth"
    )
    errors = rrb.TimeSeriesView(origin="signals/errors", name="TCP pose errors")
    grippers = rrb.TimeSeriesView(
        origin="signals/grippers", name="Gripper: prediction vs ground truth"
    )
    info = rrb.TextDocumentView(origin="episode_info", name="Comparison summary")
    upper: object
    if show_rgb:
        rgb = rrb.Spatial2DView(origin="camera/rgb", name="Camera RGB")
        upper = rrb.Horizontal(spatial, rgb, column_shares=[2, 1])
    else:
        upper = spatial
    tcp_row = rrb.Horizontal(info, errors, grippers, column_shares=[2, 2, 1])
    if show_joints:
        joint_row = rrb.Horizontal(
            rrb.TimeSeriesView(origin="signals/joints/left", name="Left joints: GT vs IK [deg]"),
            rrb.TimeSeriesView(origin="signals/joints/right", name="Right joints: GT vs IK [deg]"),
            rrb.TimeSeriesView(
                origin="signals/joint_errors/left", name="Left wrapped joint error [deg]"
            ),
            rrb.TimeSeriesView(
                origin="signals/joint_errors/right", name="Right wrapped joint error [deg]"
            ),
            column_shares=[1, 1, 1, 1],
        )
        contents = rrb.Vertical(upper, tcp_row, joint_row, row_shares=[3, 1, 1])
    else:
        contents = rrb.Vertical(upper, tcp_row, row_shares=[3, 1])
    return rrb.Blueprint(
        contents,
        rrb.TimePanel(timeline=TIMELINE, expanded=True),
        auto_views=False,
        collapse_panels=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a RoboTwin episode and compare predicted left/right TCP "
            "trajectories with TCP_third ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        default=DEFAULT_EPISODE,
        help="Converted RoboTwin episode directory",
    )
    parser.add_argument(
        "--prediction-json",
        type=Path,
        help=f"Prediction JSON (default: <episode>/{DEFAULT_PREDICTION_FILE})",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Episode subdirectory containing ground-truth left/right_state.npy",
    )
    parser.add_argument(
        "--camera",
        help="Camera directory; inferred from the prediction JSON when omitted",
    )
    parser.add_argument(
        "--history",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help="Recent trajectory samples to show; 0 shows all elapsed samples",
    )
    parser.add_argument(
        "--point-cloud-stride",
        type=base_viz.positive_int,
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
    parser.add_argument(
        "--prediction-robot-state",
        type=Path,
        help=(
            "CuRobo IK [T,14] state; when omitted, use "
            "<episode>/TCP_prediction_ik/robot_state.npy if it exists"
        ),
    )
    parser.add_argument(
        "--robot-source",
        choices=("ground-truth", "prediction"),
        default="ground-truth",
        help="Joint state used to drive the single URDF replay",
    )
    robot_group = parser.add_mutually_exclusive_group()
    robot_group.add_argument(
        "--urdf",
        type=Path,
        default=(
            Path(__file__).resolve().parent
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
    prediction_path = (
        args.prediction_json.expanduser().resolve()
        if args.prediction_json is not None
        else episode / DEFAULT_PREDICTION_FILE
    )
    if not episode.is_dir():
        raise SystemExit(f"Episode directory not found: {episode}")
    if not prediction_path.is_file():
        raise SystemExit(f"Prediction JSON not found: {prediction_path}")

    try:
        prediction, confidence, timestamps, prediction_metadata = (
            load_prediction_states(prediction_path)
        )
        ground_truth, ground_truth_metadata = base_viz.load_tcp_states(
            episode, args.ground_truth_dir
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    frame_count = len(timestamps)
    if any(len(value) != frame_count for value in ground_truth.values()):
        shapes = {side: value.shape for side, value in ground_truth.items()}
        raise SystemExit(
            f"Prediction has {frame_count} frames but ground truth has {shapes}"
        )

    camera = args.camera or str(prediction_metadata.get("view", ""))
    if not camera:
        raise SystemExit("Camera is missing from the prediction JSON; pass --camera")
    prediction_camera = str(prediction_metadata.get("view", camera))
    ground_truth_camera = str(ground_truth_metadata.get("camera", ""))
    if prediction_camera != camera or ground_truth_camera != camera:
        raise SystemExit(
            "Camera mismatch: "
            f"prediction={prediction_camera!r}, ground_truth={ground_truth_camera!r}, "
            f"requested={camera!r}"
        )

    episode_metadata_path = episode / "metadata.json"
    episode_metadata = (
        json.loads(episode_metadata_path.read_text(encoding="utf-8"))
        if episode_metadata_path.is_file()
        else {}
    )
    prediction_fps = float(prediction_metadata["frame_rate_hz"])
    episode_fps = float(episode_metadata.get("frequency_hz", prediction_fps))
    if not np.isclose(prediction_fps, episode_fps):
        raise SystemExit(
            f"Frame-rate mismatch: prediction={prediction_fps:g} Hz, "
            f"episode={episode_fps:g} Hz"
        )
    if args.axis_length <= 0.0:
        raise SystemExit("--axis-length must be positive")

    urdf_path = args.urdf.expanduser().resolve()
    if not args.no_robot and not urdf_path.is_file():
        raise SystemExit(f"RoboTwin URDF not found: {urdf_path}")

    intrinsic_path = episode / "intrinsics" / f"{camera}.npy"
    extrinsic_path = episode / "extrinsics" / f"{camera}.npy"
    if not intrinsic_path.is_file() or not extrinsic_path.is_file():
        raise SystemExit(f"Missing intrinsics/extrinsics for camera {camera!r}")
    intrinsic = np.asarray(
        np.load(intrinsic_path, allow_pickle=False), dtype=np.float64
    )
    if intrinsic.shape != (3, 3):
        raise SystemExit(
            f"{intrinsic_path} has shape {intrinsic.shape}; expected [3,3]"
        )
    try:
        world_to_camera = base_viz.load_homogeneous_extrinsics(
            extrinsic_path, frame_count
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    image_paths = sorted((episode / "images" / camera).glob("*.png"))
    depth_paths = sorted((episode / "depths" / camera).glob("*.png"))
    need_rgb = not args.no_rgb or not args.no_point_cloud
    if need_rgb and len(image_paths) != frame_count:
        raise SystemExit(
            f"Expected {frame_count} RGB frames for {camera}, found {len(image_paths)}"
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
        "ground_truth": {
            side: base_viz.states_to_pose_matrices(states)
            for side, states in ground_truth.items()
        },
        "prediction": {
            side: base_viz.states_to_pose_matrices(states)
            for side, states in prediction.items()
        },
    }
    camera_to_world = np.linalg.inv(world_to_camera)
    base_from_world = tcp_converter.world_to_base_transform()
    base_from_camera = base_from_world[None, :, :] @ camera_to_world
    base_from_tcp = {
        source: {
            side: base_from_camera @ poses for side, poses in side_poses.items()
        }
        for source, side_poses in camera_from_tcp.items()
    }

    position_errors_cm = {
        side: np.linalg.norm(
            prediction[side][:, :3] - ground_truth[side][:, :3], axis=1
        )
        * 100.0
        for side in ("left", "right")
    }
    orientation_errors = {
        side: orientation_errors_deg(
            camera_from_tcp["ground_truth"][side],
            camera_from_tcp["prediction"][side],
        )
        for side in ("left", "right")
    }
    threshold = _gripper_threshold(prediction_metadata)
    predicted_gripper_open = {
        side: prediction[side][:, 6] >= threshold for side in ("left", "right")
    }
    ground_truth_gripper_open = {
        side: ground_truth[side][:, 6] >= 0.5 for side in ("left", "right")
    }
    gripper_accuracy = {
        side: float(
            np.mean(predicted_gripper_open[side] == ground_truth_gripper_open[side])
        )
        for side in ("left", "right")
    }
    default_prediction_robot_state = (
        episode / "TCP_prediction_ik" / "robot_state.npy"
    )
    prediction_robot_state_path = (
        args.prediction_robot_state.expanduser().resolve()
        if args.prediction_robot_state is not None
        else default_prediction_robot_state if default_prediction_robot_state.is_file() else None
    )
    if args.prediction_robot_state is not None and not prediction_robot_state_path.is_file():
        raise SystemExit(
            f"Prediction robot state not found: {prediction_robot_state_path}"
        )
    try:
        prediction_robot_state = (
            load_prediction_robot_state(prediction_robot_state_path, frame_count)
            if prediction_robot_state_path is not None
            else None
        )
        need_ground_truth_robot_state = (
            not args.no_robot
            or prediction_robot_state is not None
            or args.robot_source == "prediction"
        )
        ground_truth_robot_state = (
            base_viz.load_robot_state(episode, frame_count)
            if need_ground_truth_robot_state
            else None
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.robot_source == "prediction" and prediction_robot_state is None:
        raise SystemExit(
            "--robot-source prediction requires --prediction-robot-state or "
            "<episode>/TCP_prediction_ik/robot_state.npy"
        )

    if prediction_robot_state is not None and ground_truth_robot_state is not None:
        joint_metrics, joint_error_degrees, joint_value_degrees = make_joint_summary(
            ground_truth_robot_state, prediction_robot_state
        )
    else:
        joint_metrics = None
        joint_error_degrees = {}
        joint_value_degrees = {}
    summary = make_summary(
        episode,
        prediction_path,
        prediction_metadata,
        camera,
        args.ground_truth_dir,
        args.history,
        position_errors_cm,
        orientation_errors,
        gripper_accuracy,
        prediction_robot_state_path,
        args.robot_source,
        joint_metrics,
    )
    if args.no_robot:
        robot_state = None
    elif args.robot_source == "prediction":
        robot_state = prediction_robot_state
    else:
        robot_state = ground_truth_robot_state

    rr.init(
        f"robotwin_tcp_comparison_{camera}_{episode.parent.name}_{episode.name}",
        spawn=False,
    )
    recording = rr.get_global_data_recording()
    if recording is None:
        raise SystemExit("Rerun recording failed to initialize")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output)
    else:
        port = base_viz.choose_rerun_port()
        if port != base_viz.DEFAULT_RERUN_PORT:
            print(
                f"Rerun port {base_viz.DEFAULT_RERUN_PORT} is occupied; using {port}"
            )
        recording.spawn(port=port)

    log_static_scene(intrinsic, width, height, summary)
    rr.send_blueprint(
        make_blueprint(not args.no_rgb, prediction_robot_state is not None)
    )

    with tempfile.TemporaryDirectory(prefix="robotwin-tcp-comparison-") as temporary:
        if robot_state is not None:
            base_viz.log_robot_replay(
                recording, robot_state, timestamps, urdf_path, Path(temporary)
            )

        print(
            f"Loading {episode.parent.name}/{episode.name}: {frame_count} frames, "
            f"prediction={prediction_path.name}, ground_truth={args.ground_truth_dir}, "
            f"camera={camera}, robot_source={args.robot_source}"
        )
        for frame_index in range(frame_count):
            rr.set_time(TIMELINE, duration=float(timestamps[frame_index]))
            rgb = None
            if need_rgb:
                with Image.open(image_paths[frame_index]) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
            if not args.no_rgb and rgb is not None:
                rr.log("camera/rgb", rr.Image(rgb))
            if not args.no_point_cloud and rgb is not None:
                with Image.open(depth_paths[frame_index]) as image:
                    depth_mm = np.asarray(image).copy()
                points, colors = base_viz.backproject_rgbd(
                    depth_mm, rgb, intrinsic, args.point_cloud_stride
                )
                points = base_viz.transform_points(
                    points, base_from_camera[frame_index]
                )
                rr.log("robot/scene", rr.Points3D(points, colors=colors))

            history_start = (
                0
                if args.history == 0
                else max(0, frame_index - args.history + 1)
            )
            for side in ("left", "right"):
                gt_position = base_from_tcp["ground_truth"][side][frame_index, :3, 3]
                prediction_position = base_from_tcp["prediction"][side][
                    frame_index, :3, 3
                ]
                position_error = position_errors_cm[side][frame_index]
                orientation_error = orientation_errors[side][frame_index]

                rr.log(
                    f"robot/tcp/{side}/error_link",
                    rr.LineStrips3D(
                        [[gt_position, prediction_position]],
                        radii=[0.003],
                        colors=[[255, 55, 55]],
                        labels=[f"{side} position error: {position_error:.2f} cm"],
                    ),
                )

                for source, states in (
                    ("ground_truth", ground_truth),
                    ("prediction", prediction),
                ):
                    state = states[side][frame_index]
                    pose = base_from_tcp[source][side][frame_index]
                    position = pose[:3, 3]
                    history = base_from_tcp[source][side][
                        history_start : frame_index + 1, :3, 3
                    ]
                    color = COLORS[side][source]
                    gripper_value = state[6]
                    if source == "prediction":
                        is_open = bool(predicted_gripper_open[side][frame_index])
                        confidence_value = confidence[side][frame_index]
                        confidence_text = (
                            f"\nconfidence {confidence_value:.2f}"
                            if np.isfinite(confidence_value)
                            else ""
                        )
                        gripper_text = f"probability {gripper_value:.3f}"
                    else:
                        is_open = bool(
                            ground_truth_gripper_open[side][frame_index]
                        )
                        confidence_text = ""
                        gripper_text = f"state {gripper_value:.0f}"
                    label = (
                        f"{side.upper()} {source.upper()}\n"
                        f"camera xyz [m] {state[0]:+.3f} {state[1]:+.3f} "
                        f"{state[2]:+.3f}\n"
                        f"camera rpy [deg] {np.rad2deg(state[3]):+.1f} "
                        f"{np.rad2deg(state[4]):+.1f} "
                        f"{np.rad2deg(state[5]):+.1f}\n"
                        f"gripper {'OPEN' if is_open else 'CLOSED'} "
                        f"({gripper_text}){confidence_text}"
                    )
                    entity = f"robot/tcp/{side}/{source}"
                    rr.log(
                        f"{entity}/current",
                        rr.Points3D(
                            [position],
                            radii=[0.018 if source == "ground_truth" else 0.013],
                            colors=[color],
                            labels=[label],
                        ),
                    )
                    rr.log(
                        f"{entity}/history",
                        rr.LineStrips3D(
                            [history],
                            radii=[0.006 if source == "ground_truth" else 0.004],
                            colors=[color],
                        ),
                    )
                    alpha = 230 if source == "ground_truth" else 190
                    history_colors = np.tile(
                        np.append(color, alpha), (len(history), 1)
                    )
                    rr.log(
                        f"{entity}/history_points",
                        rr.Points3D(
                            history,
                            radii=0.007 if source == "ground_truth" else 0.005,
                            colors=history_colors,
                        ),
                    )
                    rr.log(
                        f"{entity}/axes",
                        rr.Arrows3D(
                            origins=np.repeat(position[None, :], 3, axis=0),
                            vectors=pose[:3, :3].T * args.axis_length,
                            radii=[0.004 if source == "ground_truth" else 0.0025]
                            * 3,
                            colors=AXIS_COLORS,
                            labels=[
                                f"{source} X",
                                f"{source} Y",
                                f"{source} Z",
                            ],
                        ),
                    )

                rr.log(
                    f"signals/errors/{side}_position_cm",
                    rr.Scalars(float(position_error)),
                )
                rr.log(
                    f"signals/errors/{side}_orientation_deg",
                    rr.Scalars(float(orientation_error)),
                )
                rr.log(
                    f"signals/grippers/{side}_ground_truth",
                    rr.Scalars(float(ground_truth_gripper_open[side][frame_index])),
                )
                rr.log(
                    f"signals/grippers/{side}_prediction_probability",
                    rr.Scalars(float(prediction[side][frame_index, 6])),
                )

            if prediction_robot_state is not None:
                for side in ("left", "right"):
                    for joint_index in range(6):
                        entity = f"signals/joints/{side}/joint_{joint_index + 1}"
                        rr.log(
                            f"{entity}/ground_truth_deg",
                            rr.Scalars(
                                float(
                                    joint_value_degrees[f"{side}_ground_truth"][
                                        frame_index, joint_index
                                    ]
                                )
                            ),
                        )
                        rr.log(
                            f"{entity}/prediction_deg",
                            rr.Scalars(
                                float(
                                    joint_value_degrees[f"{side}_prediction"][
                                        frame_index, joint_index
                                    ]
                                )
                            ),
                        )
                        rr.log(
                            f"signals/joint_errors/{side}/joint_{joint_index + 1}_wrapped_deg",
                            rr.Scalars(
                                float(
                                    joint_error_degrees[side][
                                        frame_index, joint_index
                                    ]
                                )
                            ),
                        )

        recording.flush()

    if args.output:
        print(f"Saved Rerun recording: {args.output.expanduser().resolve()}")
    else:
        print(f"Loaded in Rerun. Scrub or play the {TIMELINE} timeline.")


if __name__ == "__main__":
    main()
