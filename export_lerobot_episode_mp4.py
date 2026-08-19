#!/usr/bin/env python3
"""Export every LeRobot v3 episode as an independent MP4.

The default ``mosaic`` mode creates one visualization per episode.  With the
three-camera datasets in this repository, ``cam_high`` is shown at full size
on the left and the two wrist cameras are stacked on the right.

Examples:
    conda run -n rerun python export_lerobot_episode_mp4.py
    conda run -n rerun python export_lerobot_episode_mp4.py \
        --dataset m2w-put-mongo-lerobot --episodes 0-4
    conda run -n rerun python export_lerobot_episode_mp4.py \
        --mode both --jobs 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit(
        "pyarrow is required. In this repository, run the script with:\n"
        "  conda run -n rerun python export_lerobot_episode_mp4.py"
    ) from error


@dataclass(frozen=True)
class CameraClip:
    key: str
    name: str
    source: Path
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class ExportJob:
    dataset: Path
    episode_index: int
    tasks: tuple[str, ...]
    frame_count: int
    fps: float
    clips: tuple[CameraClip, ...]
    output_directory: Path


@dataclass(frozen=True)
class ExportOptions:
    ffmpeg: str
    mode: str
    layout: str
    width: int
    crf: int
    preset: str
    overwrite: bool
    dry_run: bool


def discover_datasets(root: Path) -> list[Path]:
    """Find immediate children with a LeRobot metadata file."""
    if (root / "meta" / "info.json").is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    )


def resolve_datasets(root: Path, requested: list[str] | None) -> list[Path]:
    available = discover_datasets(root)
    if not requested:
        return available

    resolved: list[Path] = []
    for value in requested:
        supplied = Path(value).expanduser()
        candidates = (supplied, Path.cwd() / supplied, root / supplied)
        match = next(
            (
                candidate.resolve()
                for candidate in candidates
                if (candidate / "meta" / "info.json").is_file()
            ),
            None,
        )
        if match is None:
            by_name = next((path for path in available if path.name == value), None)
            match = by_name
        if match is None:
            names = ", ".join(path.name for path in available) or "(none)"
            raise SystemExit(f"Dataset not found: {value!r}. Available: {names}")
        if match not in resolved:
            resolved.append(match)
    return resolved


def load_info(dataset: Path) -> dict[str, Any]:
    with (dataset / "meta" / "info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"{dataset.name}: expected LeRobot v3 metadata, found {version or 'unknown'}"
        )
    if not info.get("video_path"):
        raise ValueError(f"{dataset.name}: meta/info.json has no video_path")
    return info


def video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def camera_name(video_key: str) -> str:
    return video_key.rsplit(".", 1)[-1]


def select_video_keys(keys: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return keys
    selected = [
        key
        for key in keys
        if key in requested or camera_name(key) in requested
    ]
    missing = [
        value
        for value in requested
        if not any(value == key or value == camera_name(key) for key in keys)
    ]
    if missing:
        available = ", ".join(camera_name(key) for key in keys) or "(none)"
        raise ValueError(
            f"Unknown camera(s): {', '.join(missing)}. Available: {available}"
        )
    return selected


def format_dataset_path(pattern: str, **values: Any) -> Path:
    return Path(pattern.format(**values))


def read_episode_rows(dataset: Path, keys: list[str]) -> list[dict[str, Any]]:
    metadata_files = sorted((dataset / "meta" / "episodes").glob("**/*.parquet"))
    if not metadata_files:
        raise ValueError(f"{dataset.name}: no meta/episodes parquet files found")

    columns = ["episode_index", "tasks", "length"]
    for key in keys:
        prefix = f"videos/{key}"
        columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )

    rows: list[dict[str, Any]] = []
    for path in metadata_files:
        available_columns = set(pq.read_schema(path).names)
        missing = [column for column in columns if column not in available_columns]
        if missing:
            raise ValueError(
                f"{path}: required episode columns are missing: {', '.join(missing)}"
            )
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def parse_episode_selection(specification: str | None) -> set[int] | None:
    """Parse values such as ``0,2,5-9``."""
    if not specification:
        return None
    selected: set[int] = set()
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"Invalid episode selection {part!r}; use e.g. 0,2,5-9"
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise argparse.ArgumentTypeError(
                f"Invalid descending episode range: {part!r}"
            )
        selected.update(range(start, end + 1))
    return selected


def make_clip(
    dataset: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    key: str,
) -> CameraClip:
    prefix = f"videos/{key}"
    source = dataset / format_dataset_path(
        info["video_path"],
        video_key=key,
        chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"]),
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing source video: {source}")
    start = float(episode[f"{prefix}/from_timestamp"])
    end = float(episode[f"{prefix}/to_timestamp"])
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid timestamps for {dataset.name} episode "
            f"{episode['episode_index']} camera {key}: {start} -> {end}"
        )
    return CameraClip(
        key=key,
        name=camera_name(key),
        source=source,
        start_seconds=start,
        end_seconds=end,
    )


def build_jobs(
    datasets: Iterable[Path],
    output_root: Path,
    requested_episodes: set[int] | None,
    requested_cameras: list[str] | None,
) -> list[ExportJob]:
    jobs: list[ExportJob] = []
    found_episodes: set[int] = set()
    for dataset in datasets:
        info = load_info(dataset)
        keys = select_video_keys(video_keys(info), requested_cameras)
        if not keys:
            raise ValueError(f"{dataset.name}: no video features found")
        fps = float(info["fps"])
        for episode in read_episode_rows(dataset, keys):
            episode_index = int(episode["episode_index"])
            if requested_episodes is not None and episode_index not in requested_episodes:
                continue
            found_episodes.add(episode_index)
            tasks_value = episode.get("tasks") or []
            if isinstance(tasks_value, str):
                tasks_value = [tasks_value]
            jobs.append(
                ExportJob(
                    dataset=dataset,
                    episode_index=episode_index,
                    tasks=tuple(str(task) for task in tasks_value),
                    frame_count=int(episode["length"]),
                    fps=fps,
                    clips=tuple(make_clip(dataset, info, episode, key) for key in keys),
                    output_directory=output_root / dataset.name,
                )
            )

    if requested_episodes is not None:
        missing = requested_episodes - found_episodes
        if missing:
            print(
                "Warning: episode indices not found in any selected dataset: "
                + ", ".join(str(index) for index in sorted(missing)),
                file=sys.stderr,
            )
    return jobs


def even(value: int) -> int:
    """H.264/yuv420p needs even output dimensions."""
    return max(2, value - value % 2)


def camera_order(clip: CameraClip) -> tuple[int, str]:
    name = clip.name.lower()
    if "high" in name or "main" in name or "front" in name:
        return (0, name)
    if "left" in name:
        return (1, name)
    if "right" in name:
        return (2, name)
    return (3, name)


def mosaic_geometry(
    clips: tuple[CameraClip, ...], width: int, layout: str
) -> tuple[list[CameraClip], list[tuple[int, int]], list[tuple[int, int]]]:
    ordered = sorted(clips, key=camera_order)
    count = len(ordered)
    base_width = even(width)
    base_height = even(round(base_width * 3 / 4))

    if layout == "focus" and count == 3:
        small_width = even(base_width // 2)
        small_height = even(base_height // 2)
        sizes = [
            (base_width, base_height),
            (small_width, small_height),
            (small_width, small_height),
        ]
        positions = [(0, 0), (base_width, 0), (base_width, small_height)]
        return ordered, sizes, positions

    if count == 1:
        return ordered, [(base_width, base_height)], [(0, 0)]
    if count == 2:
        sizes = [(base_width, base_height)] * 2
        return ordered, sizes, [(0, 0), (base_width, 0)]

    columns = math.ceil(math.sqrt(count))
    cell_width = even(base_width if count <= 4 else base_width * 2 // 3)
    cell_height = even(round(cell_width * 3 / 4))
    sizes = [(cell_width, cell_height)] * count
    positions = [
        ((index % columns) * cell_width, (index // columns) * cell_height)
        for index in range(count)
    ]
    return ordered, sizes, positions


def ffmpeg_input_arguments(clips: Iterable[CameraClip]) -> list[str]:
    arguments: list[str] = []
    for clip in clips:
        arguments.extend(
            [
                "-ss",
                f"{clip.start_seconds:.9f}",
                "-t",
                f"{clip.duration_seconds:.9f}",
                "-i",
                str(clip.source),
            ]
        )
    return arguments


def escape_metadata(value: str) -> str:
    return value.replace("\x00", " ").replace("\n", " ")


def make_ffmpeg_command(
    job: ExportJob,
    clips: tuple[CameraClip, ...],
    destination: Path,
    options: ExportOptions,
) -> list[str]:
    ordered, sizes, positions = mosaic_geometry(clips, options.width, options.layout)
    filter_parts: list[str] = []
    for index, (size, _) in enumerate(zip(sizes, positions)):
        cell_width, cell_height = size
        filter_parts.append(
            f"[{index}:v:0]"
            f"setpts=PTS-STARTPTS,fps=fps={job.fps:.9f},"
            f"scale=w={cell_width}:h={cell_height}:force_original_aspect_ratio=decrease,"
            f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1[v{index}]"
        )

    if len(ordered) == 1:
        filter_parts.append("[v0]null[outv]")
    else:
        inputs = "".join(f"[v{index}]" for index in range(len(ordered)))
        layout = "|".join(f"{x}_{y}" for x, y in positions)
        filter_parts.append(
            f"{inputs}xstack=inputs={len(ordered)}:layout={layout}:"
            "shortest=1:fill=black[outv]"
        )

    title = f"{job.dataset.name} - episode {job.episode_index:06d}"
    task_text = " | ".join(job.tasks)
    command = [
        options.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        *ffmpeg_input_arguments(ordered),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[outv]",
        "-an",
        "-frames:v",
        str(job.frame_count),
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={escape_metadata(title)}",
        "-metadata",
        f"comment={escape_metadata(task_text)}",
        str(destination),
    ]
    return command


def output_targets(
    job: ExportJob, options: ExportOptions
) -> list[tuple[Path, tuple[CameraClip, ...]]]:
    episode_stem = f"episode_{job.episode_index:06d}"
    targets: list[tuple[Path, tuple[CameraClip, ...]]] = []
    if options.mode in {"mosaic", "both"}:
        targets.append((job.output_directory / f"{episode_stem}.mp4", job.clips))
    if options.mode in {"cameras", "both"}:
        camera_directory = job.output_directory / episode_stem
        for clip in sorted(job.clips, key=camera_order):
            targets.append((camera_directory / f"{clip.name}.mp4", (clip,)))
    return targets


def export_one(job: ExportJob, options: ExportOptions) -> dict[str, Any]:
    targets = output_targets(job, options)
    written: list[str] = []
    skipped: list[str] = []
    if options.dry_run:
        return {
            "dataset": job.dataset.name,
            "episode_index": job.episode_index,
            "tasks": list(job.tasks),
            "frames": job.frame_count,
            "fps": job.fps,
            "outputs": [str(path) for path, _ in targets],
            "status": "planned",
        }

    for destination, clips in targets:
        if destination.exists() and not options.overwrite:
            skipped.append(str(destination))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.stem}.partial.mp4")
        temporary.unlink(missing_ok=True)
        command = make_ffmpeg_command(job, clips, temporary, options)
        try:
            subprocess.run(command, check=True)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        written.append(str(destination))

    status = "written" if written else "skipped"
    return {
        "dataset": job.dataset.name,
        "episode_index": job.episode_index,
        "tasks": list(job.tasks),
        "frames": job.frame_count,
        "fps": job.fps,
        "cameras": [clip.name for clip in job.clips],
        "outputs": written,
        "existing_outputs": skipped,
        "status": status,
    }


def write_manifest(output_root: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.root),
        "mode": args.mode,
        "layout": args.layout,
        "records": sorted(
            records,
            key=lambda item: (str(item["dataset"]), int(item["episode_index"])),
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    temporary = output_root / "manifest.partial.json"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def default_dataset_root(script_directory: Path) -> Path:
    """Follow the dataset directory if it has been renamed in this workspace."""
    for directory_name in ("lerobot_datasets", "w2_datasets"):
        candidate = script_directory / directory_name
        if candidate.is_dir():
            return candidate
    return script_directory / "lerobot_datasets"


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export independent per-episode MP4s from local LeRobot v3 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_dataset_root(script_directory),
        help="LeRobot dataset, or directory containing task datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_directory / "visualization",
        help="Root directory for exported videos",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name/path to export; repeat this option to select several",
    )
    parser.add_argument(
        "--episodes",
        type=parse_episode_selection,
        help="Episode indices/ranges, for example 0,2,5-9; default exports all",
    )
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera key or short name to include; repeat to select several",
    )
    parser.add_argument(
        "--mode",
        choices=("mosaic", "cameras", "both"),
        default="mosaic",
        help="Create one mosaic, individual camera videos, or both",
    )
    parser.add_argument(
        "--layout",
        choices=("focus", "grid"),
        default="focus",
        help="Three-camera mosaic arrangement; focus makes the main camera larger",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of the main camera tile",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent episodes")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality (lower is better)")
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="H.264 encoding speed/efficiency tradeoff",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output MP4s"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List planned outputs without encoding"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.width < 64:
        raise SystemExit("--width must be at least 64")
    if not 0 <= args.crf <= 51:
        raise SystemExit("--crf must be between 0 and 51")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg was not found in PATH")

    root = args.root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    requested_episodes = args.episodes
    datasets = resolve_datasets(root, args.dataset)
    if not datasets:
        raise SystemExit(f"No LeRobot datasets found under {root}")

    try:
        jobs = build_jobs(
            datasets,
            output_root,
            requested_episodes,
            args.camera,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not jobs:
        raise SystemExit("No episodes matched the requested selection")

    options = ExportOptions(
        ffmpeg=ffmpeg,
        mode=args.mode,
        layout=args.layout,
        width=args.width,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        f"Found {len(datasets)} dataset(s), {len(jobs)} episode(s). "
        f"Output: {output_root}"
    )

    records: list[dict[str, Any]] = []
    failures: list[tuple[ExportJob, BaseException]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(export_one, job, options): job for job in jobs}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[{completed_count}/{len(jobs)}] {record['status']:7s} "
                    f"{job.dataset.name}/episode_{job.episode_index:06d}"
                )
            except BaseException as error:
                failures.append((job, error))
                print(
                    f"[{completed_count}/{len(jobs)}] FAILED  "
                    f"{job.dataset.name}/episode_{job.episode_index:06d}: {error}",
                    file=sys.stderr,
                )

    if not args.dry_run:
        manifest_path = write_manifest(output_root, records, args)
        print(f"Manifest: {manifest_path}")
    if failures:
        raise SystemExit(f"{len(failures)} episode(s) failed; see errors above")

    written = sum(record["status"] == "written" for record in records)
    skipped = sum(record["status"] == "skipped" for record in records)
    if args.dry_run:
        print(f"Dry run complete: {len(records)} episode(s) planned")
    else:
        print(f"Done: {written} written, {skipped} already existed")


if __name__ == "__main__":
    main()
