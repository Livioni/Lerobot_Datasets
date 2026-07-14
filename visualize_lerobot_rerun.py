#!/usr/bin/env python3
"""Visualize a local LeRobot v3 dataset with Rerun.

Examples:
    conda run -n rerun python visualize_lerobot_rerun.py
    conda run -n rerun python visualize_lerobot_rerun.py \
        --dataset m2w-put-mongo-lerobot --episode 12
    conda run -n rerun python visualize_lerobot_rerun.py --list-datasets

When ``--dataset`` is omitted, the script automatically selects the only
compatible dataset, or shows an interactive menu if several are found.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb


TIMELINE = "episode_time"
DEFAULT_RERUN_PORT = 9876


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


def numeric_vector_features(info: dict[str, Any], table: pa.Table) -> list[str]:
    result: list[str] = []
    for key, feature in info.get("features", {}).items():
        if key not in table.column_names or feature.get("dtype") == "video":
            continue
        names = feature.get("names")
        shape = feature.get("shape", [])
        if names and shape and int(shape[0]) == len(names):
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
        for column_index, column_name in enumerate(feature["names"]):
            rr.send_columns(
                f"{root}/{safe_entity_name(column_name)}",
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


def make_blueprint(
    video_views: list[tuple[str, str]], signal_views: list[tuple[str, str]]
) -> rrb.Blueprint:
    top_views: list[Any] = [
        rrb.Spatial2DView(origin=path, name=name) for name, path in video_views
    ]
    if not top_views:
        top_views = [rrb.TextDocumentView(origin="episode_info", name="Episode")]

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
        rrb.Horizontal(*top_views, name="Cameras"),
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
        video_views: list[tuple[str, str]] = []
        if not args.no_video:
            video_views = log_videos(dataset, info, episode, Path(temporary))
        rr.send_blueprint(make_blueprint(video_views, signal_views))
        recording.flush()

    if args.output:
        print(f"Saved Rerun recording: {args.output.expanduser().resolve()}")
    else:
        print("Episode loaded in Rerun. Use the episode_time timeline to scrub or play.")


if __name__ == "__main__":
    main()
