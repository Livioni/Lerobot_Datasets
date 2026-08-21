#!/usr/bin/env python3
"""Visualize a local LeRobot v2.1 dataset with Rerun.

This entry point reuses the renderer from ``visualize_lerobot_rerun.py`` while
adapting the v2.1 JSONL episode metadata and per-episode data/video paths.

Examples:
    conda run -n rerun python visualize_lerobot_rerun_v21.py \
        --root lerobot_datasets_v2.1 --dataset table_clean --episode 0
    conda run -n rerun python visualize_lerobot_rerun_v21.py --list-datasets \
        --root lerobot_datasets_v2.1/w2_datasets
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr

import visualize_lerobot_rerun as common


XYZ_EULER_GRIPPER_COMPONENTS = [
    "x_m",
    "y_m",
    "z_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
    "gripper",
]


def add_known_annotation_component_names(info: dict[str, Any]) -> None:
    """Name well-known v2.1 annotation vectors whose metadata omits names."""
    for feature_key, feature in info.get("features", {}).items():
        if not feature_key.endswith(".xyz_euler_g"):
            continue
        if feature.get("shape") != [len(XYZ_EULER_GRIPPER_COMPONENTS)]:
            print(
                f"Warning: not naming {feature_key}; expected shape "
                f"[{len(XYZ_EULER_GRIPPER_COMPONENTS)}], got {feature.get('shape')}"
            )
            continue
        if feature.get("names") is None:
            feature["names"] = XYZ_EULER_GRIPPER_COMPONENTS.copy()


def read_info(dataset: Path) -> dict[str, Any]:
    """Read and validate LeRobot v2.1 dataset metadata."""
    info_path = dataset / "meta" / "info.json"
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read dataset metadata {info_path}: {error}") from error

    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2.1"):
        raise SystemExit(
            f"{dataset.name} uses LeRobot {version or 'unknown'}; "
            "this script expects the v2.1 layout."
        )

    missing = [key for key in ("data_path", "features", "fps") if key not in info]
    if missing:
        raise SystemExit(
            f"Dataset metadata {info_path} is missing: {', '.join(missing)}"
        )
    if any(
        feature.get("dtype") == "video"
        for feature in info.get("features", {}).values()
    ) and "video_path" not in info:
        raise SystemExit(f"Dataset metadata {info_path} is missing: video_path")
    add_known_annotation_component_names(info)
    return info


def read_episode(dataset: Path, episode_index: int) -> dict[str, Any]:
    """Return one v2.1 episode record from ``meta/episodes.jsonl``."""
    episodes_path = dataset / "meta" / "episodes.jsonl"
    try:
        handle = episodes_path.open("r", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not read episode metadata {episodes_path}: {error}") from error

    available_indices: list[int] = []
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                current_index = int(episode["episode_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise SystemExit(
                    f"Invalid episode metadata at {episodes_path}:{line_number}: {error}"
                ) from error
            available_indices.append(current_index)
            if current_index == episode_index:
                return episode

    if available_indices:
        available = f"{min(available_indices)}..{max(available_indices)}"
    else:
        available = "none"
    raise SystemExit(
        f"Episode {episode_index} does not exist in {dataset.name}; available: {available}"
    )


def episode_path_values(
    info: dict[str, Any], episode_index: int, video_key: str | None = None
) -> dict[str, Any]:
    """Build the placeholders used by LeRobot v2.1 path templates."""
    try:
        chunks_size = int(info.get("chunks_size", 1000))
    except (TypeError, ValueError) as error:
        raise SystemExit("Dataset chunks_size must be an integer") from error
    if chunks_size <= 0:
        raise SystemExit("Dataset chunks_size must be positive")

    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return values


def resolve_episode_path(
    dataset: Path,
    pattern: str,
    info: dict[str, Any],
    episode_index: int,
    video_key: str | None = None,
) -> Path:
    """Resolve a v2.1 data or video path and report bad templates clearly."""
    try:
        relative_path = common.format_dataset_path(
            pattern,
            **episode_path_values(info, episode_index, video_key),
        )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"Unsupported v2.1 path template {pattern!r}: {error}") from error
    return dataset / relative_path


def load_episode_data(
    dataset: Path, info: dict[str, Any], episode: dict[str, Any]
) -> pa.Table:
    """Load the Parquet file dedicated to one v2.1 episode."""
    episode_index = int(episode["episode_index"])
    data_path = resolve_episode_path(
        dataset,
        str(info["data_path"]),
        info,
        episode_index,
    )
    if not data_path.is_file():
        raise SystemExit(f"Episode data file is missing: {data_path}")
    try:
        table = pq.read_table(data_path)
    except Exception as error:
        raise SystemExit(f"Could not read episode data {data_path}: {error}") from error

    if "episode_index" in table.column_names and table.num_rows:
        stored_indices = set(table["episode_index"].to_pylist())
        if stored_indices != {episode_index}:
            raise SystemExit(
                f"Episode data {data_path} contains episode indices "
                f"{sorted(stored_indices)}, expected only {episode_index}"
            )

    expected_length = episode.get("length")
    if expected_length is not None and table.num_rows != int(expected_length):
        print(
            f"Warning: episode metadata says {expected_length} frames, "
            f"but {data_path.name} contains {table.num_rows}"
        )
    return table


def log_videos(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    expected_frame_count: int,
) -> list[tuple[str, str]]:
    """Log the standalone MP4 belonging to each v2.1 video feature."""
    views: list[tuple[str, str]] = []
    episode_index = int(episode["episode_index"])
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    for video_key in video_keys:
        source = resolve_episode_path(
            dataset,
            str(info["video_path"]),
            info,
            episode_index,
            video_key,
        )
        if not source.is_file():
            print(f"Warning: skipping missing video: {source}")
            continue

        camera_name = common.safe_entity_name(video_key.rsplit(".", 1)[-1])
        entity_path = f"cameras/{camera_name}"
        try:
            video_asset = rr.AssetVideo(path=source)
            frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
        except Exception as error:
            print(f"Warning: skipping unreadable video {source}: {error}")
            continue

        if len(frame_timestamps_ns) != expected_frame_count:
            print(
                f"Warning: {source.name} contains {len(frame_timestamps_ns)} frames; "
                f"episode data contains {expected_frame_count}"
            )
        rr.log(entity_path, video_asset, static=True)
        rr.send_columns(
            entity_path,
            indexes=[
                rr.TimeColumn(
                    common.TIMELINE,
                    duration=np.asarray(frame_timestamps_ns, dtype=np.float64) * 1e-9,
                )
            ],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )
        views.append((camera_name, entity_path))
    return views


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Visualize a local LeRobot v2.1 episode using Rerun.",
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
        default=(common.DEFAULT_CAMERA_HEIGHT, common.DEFAULT_CAMERA_WIDTH),
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
        datasets = common.discover_datasets(root)
        compatible: list[Path] = []
        for dataset in datasets:
            try:
                read_info(dataset)
            except SystemExit:
                continue
            compatible.append(dataset)
        if compatible:
            for dataset in compatible:
                print(dataset.name)
        else:
            print(f"No LeRobot v2.1 datasets found under {root}")
        return

    dataset = common.choose_dataset(root, args.dataset)
    info = read_info(dataset)
    if args.episode < 0:
        raise SystemExit("Episode index must be non-negative")

    # v2.1's episodes.jsonl is authoritative. Some converted datasets contain
    # a stale total_episodes value in info.json even though all episode files
    # and episode records are present.
    episode = read_episode(dataset, args.episode)
    table = load_episode_data(dataset, info, episode)
    if not table.num_rows:
        raise SystemExit(f"Episode {args.episode} contains no frames")
    if "timestamp" not in table.column_names:
        raise SystemExit("Episode data does not contain a timestamp column")

    calibrated_camera: common.CameraCalibration | None = None
    if args.camera_calibration is not None:
        if args.no_robot:
            raise SystemExit("--camera-calibration cannot be used with --no-robot")
        try:
            calibrated_camera = common.load_camera_calibration(
                args.camera_calibration,
                args.camera_feature,
                args.camera_resolution[0],
                args.camera_resolution[1],
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(timestamps)):
        raise SystemExit("Episode timestamps contain non-finite values")
    if np.any(np.diff(timestamps) < 0.0):
        raise SystemExit("Episode timestamps are not monotonically increasing")
    timestamps -= timestamps[0]

    app_id = f"lerobot_v21_{common.safe_entity_name(dataset.name)}"
    rr.init(app_id, spawn=False)
    recording = rr.get_global_data_recording()
    if recording is None:
        raise SystemExit("Rerun recording failed to initialize")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output)
    else:
        rerun_port = common.choose_rerun_port()
        if rerun_port != common.DEFAULT_RERUN_PORT:
            print(
                f"Rerun port {common.DEFAULT_RERUN_PORT} is occupied; "
                f"using free port {rerun_port} instead."
            )
        recording.spawn(port=rerun_port)

    print(
        f"Loading {dataset.name}, episode {args.episode} "
        f"({table.num_rows} frames at {info['fps']} FPS, LeRobot v2.1)..."
    )
    common.log_episode_info(dataset, info, episode, table.num_rows)
    signal_views = common.log_signals(info, table, timestamps)

    with tempfile.TemporaryDirectory(prefix="lerobot-rerun-v21-") as temporary:
        temporary_directory = Path(temporary)
        video_views: list[tuple[str, str]] = []
        if not args.no_video:
            video_views = log_videos(dataset, info, episode, table.num_rows)
        robot_replay = common.maybe_log_robot_replay(
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
            common.log_calibrated_camera(calibrated_camera)
        rr.send_blueprint(
            common.make_blueprint(
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
