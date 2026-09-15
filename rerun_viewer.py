"""Shared desktop/Web viewer setup, importable without the Rerun SDK."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    import rerun as rr

DEFAULT_RERUN_PORT = 9876
DEFAULT_WEB_PORT = 9090


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


def has_graphical_display() -> bool:
    """Linux viewers need an X11 or Wayland display; other platforms do not."""
    return not sys.platform.startswith("linux") or any(
        os.environ.get(name)
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET")
    )


def start_web_viewer(recording: rr.RecordingStream, grpc_port: int) -> None:
    import rerun as rr

    server_uri = recording.serve_grpc(
        grpc_port=grpc_port, server_memory_limit="75%"
    )
    web_port = choose_rerun_port(DEFAULT_WEB_PORT)
    if web_port != DEFAULT_WEB_PORT:
        print(
            f"Web port {DEFAULT_WEB_PORT} is occupied; "
            f"using free port {web_port} instead."
        )
    rr.serve_web_viewer(
        web_port=web_port, open_browser=False, connect_to=server_uri
    )
    # connect_to only opens a connection automatically when open_browser=True.
    # Include it in the printed URL so manually opened viewers also receive data.
    viewer_url = f"http://localhost:{web_port}/?url={quote(server_uri, safe=':/')}"
    print(f"Rerun Web viewer: {viewer_url}")
    print(
        "For remote access, run this on your own computer, then open the URL above:\n"
        f"  ssh -N -L {web_port}:127.0.0.1:{web_port} "
        f"-L {grpc_port}:127.0.0.1:{grpc_port} <user>@<server>"
    )


def add_viewer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared Web viewer and recording export options."""
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--web",
        action="store_true",
        help="Use the Web viewer (automatic on Linux without an X11/Wayland display)",
    )
    viewer_group.add_argument(
        "--output",
        type=Path,
        help="Write an .rrd recording instead of opening the Rerun viewer",
    )


def configure_rerun_output(
    recording: rr.RecordingStream, output_path: Path | None, *, web: bool = False
) -> bool:
    """Start the selected recording output and return whether Web is serving."""
    use_web = not output_path and (web or not has_graphical_display())
    if output_path:
        output = output_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output)
    else:
        rerun_port = choose_rerun_port()
        if rerun_port != DEFAULT_RERUN_PORT:
            print(
                f"Rerun port {DEFAULT_RERUN_PORT} is occupied; "
                f"using free port {rerun_port} instead."
            )
        if use_web:
            if not web:
                print("No graphical display detected; starting the Rerun Web viewer.")
            start_web_viewer(recording, rerun_port)
        else:
            recording.spawn(port=rerun_port)
    return use_web


def wait_for_web_viewer(recording: rr.RecordingStream) -> None:
    """Keep the in-process Web server available until Ctrl+C."""
    print("Web viewer is running. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Rerun Web viewer.")
    finally:
        recording.disconnect()
