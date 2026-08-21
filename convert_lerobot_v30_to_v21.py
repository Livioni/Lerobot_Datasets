#!/usr/bin/env python3
"""Convert local LeRobot datasets from the v3.0 layout to v2.1.

The conversion is the inverse of LeRobot's official v2.1 -> v3.0 migration:

* consolidated parquet files are split into one file per episode;
* consolidated videos are losslessly split into one MP4 per episode;
* parquet task/episode metadata is written as the v2.1 JSONL files; and
* ``meta/info.json`` is changed to the v2.1 schema.

By default the script discovers every v3.0 dataset below ``lerobot_datasets``
and ``libero`` and writes them below ``lerobot_datasets_v2.1`` while preserving
the two source directory names.

Examples:
    python convert_lerobot_v30_to_v21.py --dry-run
    python convert_lerobot_v30_to_v21.py --jobs 4
    python convert_lerobot_v30_to_v21.py --dataset dump_bowl_lerobot
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit(
        "pyarrow is required. Run this script with the repository environment:\n"
        "  .venv-lerobot/bin/python convert_lerobot_v30_to_v21.py"
    ) from error


V30 = "v3.0"
V21 = "v2.1"
DEFAULT_SOURCE_NAMES = ("lerobot_datasets", "libero")
DEFAULT_OUTPUT_NAME = "lerobot_datasets_v2.1"
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
STATS_PREFIX = "stats/"


@dataclass(frozen=True)
class DatasetJob:
    source: Path
    destination: Path
    display_name: str


@dataclass(frozen=True)
class VideoSplitJob:
    source: Path
    destination_root: Path
    video_key: str
    episodes: tuple[dict[str, Any], ...]
    chunks_size: int
    fps: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert local LeRobot v3.0 datasets to the v2.1 on-disk layout."
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        dest="source_roots",
        help=(
            "directory to scan, or one dataset directory containing meta/info.json; "
            "repeatable (default: ./lerobot_datasets and ./libero)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT_NAME,
        help=f"separate output directory (default: ./{DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="only convert a matching dataset directory name; repeatable",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel ffmpeg/ffprobe workers (default: up to 4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete and recreate matching destination dataset directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show discovered datasets and destinations without writing files",
    )
    parser.add_argument(
        "--skip-video-frame-check",
        action="store_true",
        help="skip the final packet-count check for every generated MP4",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.source_roots is None:
        args.source_roots = [script_dir / name for name in DEFAULT_SOURCE_NAMES]
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=4, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def discover_jobs(
    source_roots: list[Path], output_root: Path, selected_names: set[str]
) -> list[DatasetJob]:
    output_root = output_root.expanduser().resolve()
    jobs: list[DatasetJob] = []
    seen_sources: set[Path] = set()

    for raw_root in source_roots:
        source_root = raw_root.expanduser().resolve()
        if not source_root.is_dir():
            raise ValueError(f"Source root does not exist or is not a directory: {source_root}")

        if (source_root / "meta" / "info.json").is_file():
            candidates = [source_root]
            destination_base = output_root
        else:
            candidates = sorted(
                info_path.parent.parent
                for info_path in source_root.rglob("meta/info.json")
                if output_root not in info_path.parents
            )
            destination_base = output_root / source_root.name

        for source in candidates:
            source = source.resolve()
            if source in seen_sources:
                continue
            seen_sources.add(source)
            if selected_names and source.name not in selected_names:
                continue
            info = load_json(source / "meta" / "info.json")
            if info.get("codebase_version") != V30:
                continue
            relative = Path(source.name) if source == source_root else source.relative_to(source_root)
            destination = destination_base / relative
            jobs.append(
                DatasetJob(
                    source=source,
                    destination=destination,
                    display_name=f"{source_root.name}/{relative.as_posix()}",
                )
            )

    jobs.sort(key=lambda job: job.display_name)
    if selected_names:
        found = {job.source.name for job in jobs}
        missing = sorted(selected_names - found)
        if missing:
            raise ValueError(f"Requested dataset(s) not found as v3.0: {', '.join(missing)}")
    return jobs


def read_episode_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise ValueError(f"No v3.0 episode metadata parquet files found in {root}")
    table = pq.read_table(paths)
    rows = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
    expected = list(range(len(rows)))
    actual = [int(row["episode_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"Episode indices must be contiguous from zero in {root}")
    return rows


def read_tasks(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise ValueError(f"Missing v3.0 tasks metadata: {path}")
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda row: int(row["task_index"]))
    return [{"task_index": int(row["task_index"]), "task": str(row["task"])} for row in rows]


def v21_video_info(info: dict[str, Any]) -> dict[str, Any]:
    """Keep the video metadata fields emitted by LeRobot v2.1."""
    output: dict[str, Any] = {}
    for key, value in info.items():
        if key in {
            "video.height",
            "video.width",
            "video.codec",
            "video.pix_fmt",
            "video.fps",
            "video.channels",
            "has_audio",
        } or key.startswith("audio."):
            output[key] = value
    output["video.is_depth_map"] = bool(
        info.get("video.is_depth_map", info.get("is_depth_map", False))
    )
    return output


def build_v21_info(info: dict[str, Any]) -> dict[str, Any]:
    features = copy.deepcopy(info["features"])
    for feature in features.values():
        if feature.get("dtype") == "video":
            if "info" in feature:
                feature["info"] = v21_video_info(feature["info"])
        else:
            # The official v2.1 -> v3.0 converter adds this field.
            feature.pop("fps", None)

    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    return {
        "codebase_version": V21,
        "robot_type": info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": int(info["total_frames"]),
        "total_tasks": int(info["total_tasks"]),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": math.ceil(total_episodes / chunks_size) if total_episodes else 0,
        "chunks_size": chunks_size,
        "fps": int(info["fps"]),
        "splits": copy.deepcopy(info.get("splits", {"train": f"0:{total_episodes}"})),
        "data_path": V21_DATA_PATH,
        "video_path": V21_VIDEO_PATH if video_keys else None,
        "features": features,
    }


def unflatten_episode_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith(STATS_PREFIX):
            continue
        remainder = key[len(STATS_PREFIX) :]
        try:
            feature_name, stat_name = remainder.rsplit("/", 1)
        except ValueError as error:
            raise ValueError(f"Invalid flattened episode stat key: {key}") from error
        stats.setdefault(feature_name, {})[stat_name] = value
    if not stats:
        raise ValueError(f"Episode {row['episode_index']} has no per-episode statistics")
    return stats


def write_metadata(
    source: Path,
    destination: Path,
    source_info: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    destination_meta = destination / "meta"
    v21_info = build_v21_info(source_info)
    write_json(destination_meta / "info.json", v21_info)
    write_jsonl(destination_meta / "tasks.jsonl", read_tasks(source))
    write_jsonl(
        destination_meta / "episodes.jsonl",
        (
            {
                "episode_index": int(row["episode_index"]),
                "tasks": list(row["tasks"]),
                "length": int(row["length"]),
            }
            for row in episodes
        ),
    )
    write_jsonl(
        destination_meta / "episodes_stats.jsonl",
        (
            {
                "episode_index": int(row["episode_index"]),
                "stats": unflatten_episode_stats(row),
            }
            for row in episodes
        ),
    )
    source_stats = source / "meta" / "stats.json"
    if source_stats.is_file():
        shutil.copy2(source_stats, destination_meta / "stats.json")
    return v21_info


def v21_data_path(root: Path, chunks_size: int, episode_index: int) -> Path:
    return root / V21_DATA_PATH.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def v21_video_path(root: Path, chunks_size: int, video_key: str, episode_index: int) -> Path:
    return root / V21_VIDEO_PATH.format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def write_parquet_atomically(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def existing_parquet_matches(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        return pq.ParquetFile(path).metadata.num_rows == expected_rows
    except Exception:
        return False


def convert_data(
    source: Path,
    destination: Path,
    episodes: list[dict[str, Any]],
    chunks_size: int,
) -> None:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(row)

    converted = 0
    for (chunk_index, file_index), group in sorted(groups.items()):
        source_path = source / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        if not source_path.is_file():
            raise ValueError(f"Missing v3.0 data file: {source_path}")
        table = pq.read_table(source_path)
        if "episode_index" not in table.column_names:
            raise ValueError(f"Data file has no episode_index column: {source_path}")

        for row in group:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            destination_path = v21_data_path(destination, chunks_size, episode_index)
            if existing_parquet_matches(destination_path, length):
                converted += 1
                continue

            mask = pc.equal(table["episode_index"], pa.scalar(episode_index, table["episode_index"].type))
            episode_table = table.filter(mask)
            if episode_table.num_rows != length:
                raise ValueError(
                    f"Episode {episode_index} in {source_path} has {episode_table.num_rows} rows; "
                    f"metadata says {length}"
                )
            write_parquet_atomically(episode_table, destination_path)
            converted += 1
        print(f"    data: {converted}/{len(episodes)} episodes", flush=True)


def build_video_jobs(
    source: Path,
    destination: Path,
    episodes: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
    fps: int,
) -> list[VideoSplitJob]:
    jobs: list[VideoSplitJob] = []
    for video_key in video_keys:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        for row in episodes:
            groups[(int(row[chunk_key]), int(row[file_key]))].append(row)

        for (chunk_index, file_index), group in sorted(groups.items()):
            ordered = tuple(sorted(group, key=lambda row: int(row["episode_index"])))
            indices = [int(row["episode_index"]) for row in ordered]
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(
                    f"Episodes in {video_key} chunk {chunk_index} file {file_index} are not contiguous"
                )
            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            first_timestamp = float(ordered[0][from_key])
            if abs(first_timestamp) > 0.5 / fps:
                raise ValueError(
                    f"First timestamp for {video_key} chunk {chunk_index} file {file_index} "
                    f"is {first_timestamp}, expected zero"
                )
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if abs(float(previous[to_key]) - float(current[from_key])) > 0.5 / fps:
                    raise ValueError(f"Non-contiguous video timestamps in {video_key}")
            source_path = source / V30_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not source_path.is_file():
                raise ValueError(f"Missing v3.0 video file: {source_path}")
            jobs.append(
                VideoSplitJob(
                    source=source_path,
                    destination_root=destination,
                    video_key=video_key,
                    episodes=ordered,
                    chunks_size=chunks_size,
                    fps=fps,
                )
            )
    return jobs


def split_video(job: VideoSplitJob) -> int:
    expected_paths = [
        v21_video_path(
            job.destination_root,
            job.chunks_size,
            job.video_key,
            int(row["episode_index"]),
        )
        for row in job.episodes
    ]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected_paths):
        return len(expected_paths)

    safe_key = job.video_key.replace("/", "_")
    work_dir = (
        job.destination_root
        / ".video-split-work"
        / safe_key
        / f"{job.source.parent.name}-{job.source.stem}"
    )
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    first_episode = int(job.episodes[0]["episode_index"])
    if len(job.episodes) == 1:
        output_path = work_dir / f"episode_{first_episode:06d}.mp4"
        shutil.copy2(job.source, output_path)
    else:
        to_key = f"videos/{job.video_key}/to_timestamp"
        segment_times = ",".join(f"{float(row[to_key]):.9f}" for row in job.episodes[:-1])
        output_pattern = work_dir / "episode_%06d.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(job.source),
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_times",
            segment_times,
            "-segment_time_delta",
            str(0.5 / job.fps),
            "-segment_start_number",
            str(first_episode),
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]
        subprocess.run(command, check=True)

    generated = sorted(work_dir.glob("episode_*.mp4"))
    if len(generated) != len(job.episodes):
        raise ValueError(
            f"ffmpeg produced {len(generated)} files for {job.source}; expected {len(job.episodes)}"
        )
    for temporary, destination_path in zip(generated, expected_paths, strict=True):
        expected_name = destination_path.name
        if temporary.name != expected_name:
            raise ValueError(f"Unexpected ffmpeg output {temporary.name}; expected {expected_name}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination_path)
    shutil.rmtree(work_dir)
    return len(generated)


def convert_videos(
    source: Path,
    destination: Path,
    episodes: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
    fps: int,
    workers: int,
) -> None:
    if not video_keys:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to split v3.0 video files")
    jobs = build_video_jobs(source, destination, episodes, video_keys, chunks_size, fps)
    expected_total = len(episodes) * len(video_keys)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(split_video, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                completed += future.result()
            except Exception as error:
                raise RuntimeError(f"Failed to split {job.source}") from error
            print(f"    videos: {completed}/{expected_total} episode-camera files", flush=True)


def ffprobe_packet_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = result.stdout.strip().splitlines()
    if len(value) != 1 or not value[0].isdigit():
        raise ValueError(f"Could not read video packet count for {path}: {result.stdout!r}")
    return int(value[0])


def validate_dataset(
    destination: Path,
    source_info: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    workers: int,
    check_video_frames: bool,
) -> None:
    info = load_json(destination / "meta" / "info.json")
    if info.get("codebase_version") != V21:
        raise ValueError(f"Output is not tagged as {V21}: {destination}")
    for key in ("total_episodes", "total_frames", "total_tasks"):
        if int(info[key]) != int(source_info[key]):
            raise ValueError(f"Output {key} does not match source in {destination}")

    episodes = read_jsonl(destination / "meta" / "episodes.jsonl")
    episode_stats = read_jsonl(destination / "meta" / "episodes_stats.jsonl")
    tasks = read_jsonl(destination / "meta" / "tasks.jsonl")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"episodes.jsonl row count mismatch in {destination}")
    if len(episode_stats) != len(episodes):
        raise ValueError(f"episodes_stats.jsonl row count mismatch in {destination}")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError(f"tasks.jsonl row count mismatch in {destination}")

    chunks_size = int(info["chunks_size"])
    total_parquet_rows = 0
    for row in source_episodes:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        path = v21_data_path(destination, chunks_size, episode_index)
        if not path.is_file():
            raise ValueError(f"Missing output parquet: {path}")
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        if table.num_rows != length:
            raise ValueError(f"Wrong row count in {path}: {table.num_rows}, expected {length}")
        if pc.any(pc.not_equal(table["episode_index"], episode_index)).as_py():
            raise ValueError(f"Wrong episode_index value in {path}")
        frame_indices = table["frame_index"].to_pylist()
        if frame_indices != list(range(length)):
            raise ValueError(f"Non-contiguous frame_index values in {path}")
        total_parquet_rows += table.num_rows
    if total_parquet_rows != int(info["total_frames"]):
        raise ValueError(f"Total parquet row count mismatch in {destination}")

    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    video_checks: list[tuple[Path, int]] = []
    for row in source_episodes:
        episode_index = int(row["episode_index"])
        for video_key in video_keys:
            path = v21_video_path(destination, chunks_size, video_key, episode_index)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty output video: {path}")
            if check_video_frames:
                video_checks.append((path, int(row["length"])))

    if check_video_frames:
        if shutil.which("ffprobe") is None:
            raise RuntimeError("ffprobe is required for video frame validation")
        checked = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(ffprobe_packet_count, path): (path, expected) for path, expected in video_checks}
            for future in as_completed(futures):
                path, expected = futures[future]
                actual = future.result()
                if actual != expected:
                    raise ValueError(f"Video frame count mismatch in {path}: {actual}, expected {expected}")
                checked += 1
                if checked % 250 == 0 or checked == len(video_checks):
                    print(f"    verify videos: {checked}/{len(video_checks)}", flush=True)

    work_dir = destination / ".video-split-work"
    if work_dir.exists():
        shutil.rmtree(work_dir)


def output_looks_complete(job: DatasetJob, check_video_frames: bool, workers: int) -> bool:
    if not (job.destination / "meta" / "info.json").is_file():
        return False
    try:
        source_info = load_json(job.source / "meta" / "info.json")
        episodes = read_episode_rows(job.source)
        validate_dataset(job.destination, source_info, episodes, workers, check_video_frames)
    except Exception:
        return False
    return True


def convert_dataset(job: DatasetJob, args: argparse.Namespace) -> None:
    source_info = load_json(job.source / "meta" / "info.json")
    if source_info.get("codebase_version") != V30:
        raise ValueError(f"Source is not a LeRobot {V30} dataset: {job.source}")
    episodes = read_episode_rows(job.source)
    if len(episodes) != int(source_info["total_episodes"]):
        raise ValueError(f"Episode metadata count mismatch in {job.source}")
    if sum(int(row["length"]) for row in episodes) != int(source_info["total_frames"]):
        raise ValueError(f"Episode lengths do not sum to total_frames in {job.source}")
    image_keys = [
        key for key, feature in source_info["features"].items() if feature.get("dtype") == "image"
    ]
    if image_keys:
        raise NotImplementedError(
            "External image datasets are not supported by this converter; found: " + ", ".join(image_keys)
        )

    if args.overwrite and job.destination.exists():
        shutil.rmtree(job.destination)
    elif job.destination.exists() and output_looks_complete(
        job, not args.skip_video_frame_check, args.jobs
    ):
        print(f"[skip] {job.display_name}: already complete", flush=True)
        return

    print(f"[convert] {job.display_name}", flush=True)
    print(f"    source:      {job.source}", flush=True)
    print(f"    destination: {job.destination}", flush=True)
    job.destination.mkdir(parents=True, exist_ok=True)
    v21_info = write_metadata(job.source, job.destination, source_info, episodes)
    convert_data(
        job.source,
        job.destination,
        episodes,
        int(v21_info["chunks_size"]),
    )
    video_keys = [
        key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"
    ]
    convert_videos(
        job.source,
        job.destination,
        episodes,
        video_keys,
        int(v21_info["chunks_size"]),
        int(v21_info["fps"]),
        args.jobs,
    )
    print("    validating output", flush=True)
    validate_dataset(
        job.destination,
        source_info,
        episodes,
        args.jobs,
        check_video_frames=not args.skip_video_frame_check,
    )
    print(f"[done] {job.display_name}", flush=True)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    try:
        jobs = discover_jobs(args.source_roots, output_root, set(args.dataset))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not jobs:
        raise SystemExit("No LeRobot v3.0 datasets were found.")

    source_bytes = sum(
        path.stat().st_size
        for job in jobs
        for path in job.source.rglob("*")
        if path.is_file()
    )
    print(f"Discovered {len(jobs)} LeRobot v3.0 dataset(s), {human_size(source_bytes)} total:")
    for job in jobs:
        print(f"  {job.display_name} -> {job.destination}")
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, start=1):
        print(f"\nDataset {index}/{len(jobs)}", flush=True)
        convert_dataset(job, args)
    print(f"\nConverted and validated {len(jobs)} dataset(s) under {output_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
