#!/usr/bin/env python3
"""Upload the local LeRobot datasets to a Hugging Face dataset repository.

The contents of the dataset root are uploaded to the repository root, so each
local task directory becomes a top-level directory on the Hub. The large-folder
uploader stores resumable state in the dataset root's ``.cache`` directory;
rerun the same command after an interruption to continue the upload.

Examples:
    # Inspect what will be uploaded without authenticating or installing HF Hub.
    python upload_lerobot_datasets.py --dry-run

    # Authenticate once, then upload with the default settings.
    hf auth login
    python upload_lerobot_datasets.py

    # Alternatively, provide a write token through the environment.
    HF_TOKEN=hf_xxx python upload_lerobot_datasets.py --private
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "HarrisonPENG/World-to-Wrist"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def find_default_dataset_root() -> Path:
    """Support both the requested name and the current workspace name."""
    candidates = (
        SCRIPT_DIRECTORY / "lerobot_datasets",
        SCRIPT_DIRECTORY / "w2_datasets",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


DEFAULT_DATASET_ROOT = find_default_dataset_root()
DEFAULT_IGNORE_PATTERNS = (
    ".DS_Store",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
)


@dataclass(frozen=True)
class FolderSummary:
    file_count: int
    total_bytes: int
    task_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload LeRobot task folders to the Hugging Face dataset "
            f"repository {DEFAULT_REPO_ID}."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"local folder to upload (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        help="target branch or revision (default: main)",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        dest="private",
        action="store_true",
        help="create the repository as private if it does not exist",
    )
    visibility.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="create the repository as public if it does not exist",
    )
    parser.set_defaults(private=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of upload workers (default: chosen by huggingface_hub)",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=60,
        metavar="SECONDS",
        help="seconds between progress reports (default: 60)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="GLOB",
        help="additional glob to exclude; may be supplied more than once",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarize local files without contacting the Hub",
    )
    return parser.parse_args()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def summarize_folder(root: Path) -> FolderSummary:
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")

    task_names = tuple(
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    )
    if not task_names:
        raise SystemExit(
            f"No LeRobot task directories containing meta/info.json found in {root}"
        )

    files = [path for path in root.rglob("*") if path.is_file()]
    return FolderSummary(
        file_count=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
        task_names=task_names,
    )


def print_plan(
    root: Path, repo_id: str, revision: str | None, summary: FolderSummary
) -> None:
    print(f"Local folder : {root}")
    print(f"Destination  : https://huggingface.co/datasets/{repo_id}")
    print(f"Revision     : {revision or 'main'}")
    print(f"Task folders : {', '.join(summary.task_names)}")
    print(f"Payload      : {summary.file_count} files, {human_size(summary.total_bytes)}")


def load_hugging_face_api() -> Any:
    # hf_xet uses this opt-in for maximum upload throughput when it is available.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required for uploading. Install it with:\n"
            '  python -m pip install -U "huggingface_hub[hf_xet]"\n'
            "Then authenticate with `hf auth login` or set HF_TOKEN."
        ) from error
    return HfApi()


def authenticated_user(api: Any) -> str:
    try:
        identity = api.whoami()
    except Exception as error:
        raise SystemExit(
            "Hugging Face authentication failed. Run `hf auth login` or export "
            "a write-enabled HF_TOKEN, then retry.\n"
            f"Details: {error}"
        ) from error
    return str(identity.get("name") or identity.get("fullname") or "unknown")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()

    if args.num_workers is not None and args.num_workers < 1:
        raise SystemExit("--num-workers must be at least 1")
    if args.report_every < 1:
        raise SystemExit("--report-every must be at least 1 second")

    summary = summarize_folder(root)
    print_plan(root, args.repo_id, args.revision, summary)
    if args.dry_run:
        print("Dry run complete; no network requests were made.")
        return

    api = load_hugging_face_api()
    print(f"Authenticated as: {authenticated_user(api)}")
    print("Starting resumable upload. Press Ctrl-C safely; rerun to resume.")

    upload_options: dict[str, Any] = {
        "repo_id": args.repo_id,
        "folder_path": root,
        "repo_type": "dataset",
        "ignore_patterns": [*DEFAULT_IGNORE_PATTERNS, *args.ignore],
        "print_report": True,
        "print_report_every": args.report_every,
    }
    if args.revision:
        upload_options["revision"] = args.revision
    if args.private is not None:
        upload_options["private"] = args.private
    if args.num_workers is not None:
        upload_options["num_workers"] = args.num_workers

    try:
        api.upload_large_folder(**upload_options)
    except KeyboardInterrupt:
        print("\nUpload interrupted. Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as error:
        raise SystemExit(f"Upload failed: {error}") from error

    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
