"""Render replacement robot bodies into RoboTwin third-view RGB frames."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ._composite import compose
from ._data import CAMERA, EMBODIMENTS, load_episode, load_source, load_target


def nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('Must be finite and nonnegative')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--ik-dir', required=True, type=Path,
                        help='IK output directory containing metadata.json and robot_state.npy')
    parser.add_argument('--mode', choices=('no_depth', 'with_depth', 'both'), default='both')
    parser.add_argument('--frames', help='Comma-separated zero-based frame numbers, e.g. 0,120,255; default all')
    parser.add_argument('--depth-tolerance-mm', type=nonnegative_float, default=5.0)
    parser.add_argument('--embodiments-root', type=Path, default=EMBODIMENTS)
    parser.add_argument('--overwrite', action='store_true', help='Replace selected existing output PNGs')
    return parser.parse_args(argv)


def select_frames(value, count):
    if value is None:
        return list(range(count))
    try:
        selected = sorted(set(int(item.strip()) for item in value.split(',')))
    except ValueError as error:
        raise ValueError('--frames must be comma-separated integer frame numbers') from error
    if not selected or selected[0] < 0 or selected[-1] >= count:
        raise ValueError(f'--frames must be in [0, {count - 1}]')
    return selected


def check_outputs(paths, overwrite):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output exists: {existing[0]}; use --overwrite to replace selected frames')


def save_png(path, rgb):
    temporary = path.with_suffix('.png.tmp')
    try:
        Image.fromarray(rgb).save(temporary, format='PNG')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    episode = load_episode(args.episode, args.mode != 'no_depth')
    source = load_source(episode)
    target = load_target(args.ik_dir, episode, args.embodiments_root)
    frames = select_frames(args.frames, episode.count)
    modes = ('no_depth', 'with_depth') if args.mode == 'both' else (args.mode,)
    output = episode.path / 'rerender_images' / CAMERA / target.slug
    destinations = {mode: [output / mode / episode.rgb_paths[i].name for i in frames] for mode in modes}
    check_outputs([p for paths in destinations.values() for p in paths], args.overwrite)
    # Preflight all selected images before writing the first output.
    for index in frames:
        episode.read_frame(index)
    from ._renderer import RobotRenderer
    totals = {mode: {'erased': 0, 'drawn': 0} for mode in modes}
    started = time.monotonic()
    with ExitStack() as stack:
        old_renderer = stack.enter_context(RobotRenderer(source, episode.height, episode.width))
        new_renderer = stack.enter_context(RobotRenderer(target, episode.height, episode.width))
        for mode in modes:
            (output / mode).mkdir(parents=True, exist_ok=True)
        print(f'{target.name}: {len(frames)} frames, {episode.width}x{episode.height}, modes={",".join(modes)}', flush=True)
        for sequence, index in enumerate(frames):
            rgb, scene_depth = episode.read_frame(index)
            old = old_renderer.render(index, episode.intrinsics[index], episode.extrinsics[index])
            new = new_renderer.render(index, episode.intrinsics[index], episode.extrinsics[index])
            for mode in modes:
                result = compose(rgb, old.depth, new.rgb, new.depth,
                                 scene_depth if mode == 'with_depth' else None,
                                 args.depth_tolerance_mm * 0.001)
                save_png(destinations[mode][sequence], result.rgb)
                totals[mode]['erased'] += int(result.erased.sum())
                totals[mode]['drawn'] += int(result.drawn.sum())
            if sequence == 0 or (sequence + 1) % 25 == 0 or sequence + 1 == len(frames):
                print(f'  {sequence + 1}/{len(frames)} (frame {index:06d}), {time.monotonic()-started:.1f}s', flush=True)
    for mode in modes:
        print(f'{mode}: {output / mode} ({len(frames)} PNGs; '
              f'erased={totals[mode]["erased"]}, drawn={totals[mode]["drawn"]} pixels)', flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        raise SystemExit(f'rerender: {error}') from error


if __name__ == '__main__':
    main()
