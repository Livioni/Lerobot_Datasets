#!/usr/bin/env python3
"""Inspect the configured Franka Panda homestate in Rerun, without an episode."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotwin_ik._embodiments import DEFAULT_EMBODIMENTS_ROOT, load_embodiment
from robotwin_ik._kinematics import fk_link, gripper_positions, joint_matrix
from robotwin_ik.visualize_ik_rerun import COLORS, prepare_visual


def openness(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError('Gripper openness must be between 0 (closed) and 1 (open)')
    return parsed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embodiments-root', type=Path, default=DEFAULT_EMBODIMENTS_ROOT)
    parser.add_argument('--embodiment-distance', type=float, default=.6, help='Dual-arm base separation in metres')
    parser.add_argument('--arm', choices=('both', 'left', 'right'), default='both')
    parser.add_argument('--gripper-open', type=openness, default=1., help='Normalized gripper opening')
    parser.add_argument('--output', type=Path, help='Save .rrd instead of opening the Rerun viewer')
    args = parser.parse_args(argv)
    if not np.isfinite(args.embodiment_distance) or args.embodiment_distance <= 0:
        parser.error('--embodiment-distance must be finite and positive')
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    robot = load_embodiment('franka-panda', args.embodiments_root, args.embodiment_distance)
    sides = ('left', 'right') if args.arm == 'both' else (args.arm,)
    config_path = robot.asset_dir / 'config.yml'
    poses, values, descriptions = {}, {}, []
    for side in sides:
        geometry = robot.geometries[side]
        q = robot.homestates[side].copy()
        values[side] = dict(zip(geometry.canonical_joint_names, q))
        values[side].update(gripper_positions(robot.model, robot.config, ('left', 'right').index(side), args.gripper_open))
        root = robot.world_from_root[side]
        poses[side] = {
            'base': root @ geometry.footprint_from_base,
            'tcp': root @ fk_link(robot.model, geometry.ee_link, values[side]) @ geometry.link6_from_tcp,
        }
        lines = [f'## {side.capitalize()} arm', '', '| Joint | rad | deg |', '|---|---:|---:|']
        lines.extend(f'| {name} | {value:.9f} | {np.degrees(value):.4f} |' for name, value in zip(geometry.canonical_joint_names, q))
        lines.extend(['', f"Base XYZ (m): {np.array2string(poses[side]['base'][:3, 3], precision=6)}",
                      f"Closed TCP XYZ (m): {np.array2string(poses[side]['tcp'][:3, 3], precision=6)}", '',
                      'Finger joints (m): ' + ', '.join(f'{name}={value:.6f}' for name, value in values[side].items() if name not in geometry.canonical_joint_names)])
        descriptions.append('\n'.join(lines))
        print(f'{side} homestate (rad): {q.tolist()}')
        print(f"  base XYZ (m): {poses[side]['base'][:3, 3].tolist()}")
        print(f"  closed TCP XYZ (m): {poses[side]['tcp'][:3, 3].tolist()}")

    # Optional viewer imports stay lazy so --help and argument checks need no Rerun/CUDA.
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init('franka_panda_config_homestate', spawn=args.output is None)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output)
    recording = rr.get_global_data_recording()
    rr.log('world', rr.CoordinateFrame('world'), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    info = ('# Franka Panda — configured homestate\n\n'
            f'Source: `{config_path}` → `homestate`\n\n'
            f'Base separation: {robot.distance:g} m; gripper opening: {args.gripper_open:g}.\n\n'
            'Axes: red X, green Y, blue Z. TCP marks the virtual closed fingertip midpoint.\n\n'
            + '\n\n'.join(descriptions))
    rr.log('info', rr.TextDocument(info, media_type=rr.MediaType.MARKDOWN), static=True)
    selected_positions = np.array([poses[side]['tcp'][:3, 3] for side in sides])
    look_target = selected_positions.mean(axis=0)
    eye = look_target + [1.2, 1.4, .8]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', contents=['world/**', *[f'{side}_robot/**' for side in sides]],
                              name='Franka Panda homestate',
                              eye_controls=rrb.EyeControls3D(position=eye.tolist(), look_target=look_target.tolist(), eye_up=[0, 0, 1])),
            rrb.TextDocumentView(origin='info', name='Homestate joint angles'),
            column_shares=[3, 2]),
        auto_views=False))

    def axes(path: str, transform: np.ndarray, label: str, length: float) -> None:
        rr.log(path, rr.CoordinateFrame('world'), static=True)
        position = transform[:3, 3]
        rr.log(path, rr.Arrows3D(origins=np.repeat(position[None], 3, axis=0),
                                vectors=transform[:3, :3].T*length,
                                colors=[[255, 60, 60], [60, 220, 80], [70, 130, 255]], radii=.002), static=True)
        rr.log(path+'/label', rr.CoordinateFrame('world'), static=True)
        rr.log(path+'/label', rr.Points3D([position], labels=[label], radii=.004), static=True)

    axes('world/origin', np.eye(4), 'World', .15)
    with tempfile.TemporaryDirectory(prefix='franka_homestate_') as temporary:
        prepared = Path(temporary) / 'franka_visual.urdf'
        prepare_visual(robot.urdf_path, prepared, 'franka-panda')
        for side in sides:
            prefix = side + '/'
            tree = rr.urdf.UrdfTree.from_file_path(prepared, entity_path_prefix=f'{side}_robot',
                                                  frame_prefix=prefix,
                                                  static_transform_entity_path=f'{side}_robot_static_tf')
            # Log geometry once, then provide every joint transform at homestate.
            # Excluding URDF zero-position transforms avoids competing static frames.
            recording.send_chunks(tree.stream(include_joint_transforms=False))
            root = robot.world_from_root[side]
            rr.log(f'transforms/{side}/root', rr.Transform3D(
                translation=root[:3, 3], mat3x3=root[:3, :3],
                parent_frame='world', child_frame=prefix+robot.model.root_link), static=True)
            for name, joint in robot.model.joints_by_name.items():
                if joint.joint_type != 'fixed' and name not in values[side]:
                    raise ValueError(f'Missing homestate value for movable joint {name}')
                transform = joint_matrix(joint, values[side].get(name, 0.))
                rr.log(f'transforms/{side}/{name}', rr.Transform3D(
                    translation=transform[:3, 3], mat3x3=transform[:3, :3],
                    parent_frame=prefix+joint.parent, child_frame=prefix+joint.child), static=True)
            axes(f'world/{side}/base', poses[side]['base'], f'{side} base', .12)
            axes(f'world/{side}/tcp', poses[side]['tcp'], f'{side} closed TCP', .08)
            rr.log(f'world/{side}/tcp/point', rr.CoordinateFrame('world'), static=True)
            rr.log(f'world/{side}/tcp/point', rr.Points3D([poses[side]['tcp'][:3, 3]], colors=[COLORS[side]], radii=.007), static=True)
        recording.flush()
    print(f'Source: {config_path} -> homestate')
    print(f'Saved {output}' if args.output else 'Sent homestate to the Rerun viewer')


if __name__ == '__main__':
    main()
