#!/usr/bin/env python3
"""Inspect the solver's built-in initial joint states without an episode or CUDA."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerun_viewer import add_viewer_arguments, configure_rerun_output, wait_for_web_viewer
from robotwin_ik._aloha import load_aloha
from robotwin_ik._embodiments import DEFAULT_EMBODIMENTS_ROOT, load_embodiment
from robotwin_ik._initial_state import BUILTIN_INITIAL_JOINTS, load_initial_state
from robotwin_ik._kinematics import fk_link, gripper_positions, joint_matrix
from robotwin_ik.visualize_ik_rerun import COLORS, prepare_visual
from robotwin_ik.visualize_franka_homestate_rerun import openness


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--robot', choices=('all', *BUILTIN_INITIAL_JOINTS), default='all')
    parser.add_argument('--arm', choices=('both', 'left', 'right'), default='both')
    parser.add_argument('--embodiments-root', type=Path, default=DEFAULT_EMBODIMENTS_ROOT)
    parser.add_argument('--embodiment-distance', type=float, default=.6,
                        help='Base separation in metres (not applied to Aloha)')
    parser.add_argument('--gripper-open', type=openness, default=None,
                        help='Override display opening; default is the built-in initial opening')
    add_viewer_arguments(parser)
    args = parser.parse_args(argv)
    if not np.isfinite(args.embodiment_distance) or args.embodiment_distance <= 0:
        parser.error('--embodiment-distance must be finite and positive')
    return args


def load_state(name, args):
    if name == 'aloha-agilex':
        robot = load_aloha(SimpleNamespace(
            robotwin_root=None, embodiments_root=args.embodiments_root,
            urdf=None, left_config=None, right_config=None))
    else:
        robot = load_embodiment(name, args.embodiments_root, args.embodiment_distance)
    initial, provenance = load_initial_state(
        name, robot.model, {s: g.canonical_joint_names for s, g in robot.geometries.items()})
    return robot, initial, provenance


def main(argv=None):
    args = parse_args(argv)
    names = list(BUILTIN_INITIAL_JOINTS) if args.robot == 'all' else [args.robot]
    # Validate all inputs before opening a viewer. This uses the same loader as IK.
    states = [load_state(name, args) for name in names]
    sides = ('left', 'right') if args.arm == 'both' else (args.arm,)

    import rerun as rr
    import rerun.blueprint as rrb

    rr.init('robotwin_builtin_initial_states', spawn=False)
    recording = rr.get_global_data_recording()
    if recording is None:
        raise RuntimeError('Rerun recording failed to initialize')
    use_web = configure_rerun_output(recording, args.output, web=args.web)
    views = []

    def axes(path, transform, label, length=.1):
        pos = transform[:3, 3]
        rr.log(path, rr.CoordinateFrame('world'), rr.Arrows3D(
            origins=np.repeat(pos[None], 3, axis=0), vectors=transform[:3, :3].T*length,
            colors=[[255, 60, 60], [60, 220, 80], [70, 130, 255]], radii=.002), static=True)
        rr.log(path+'/label', rr.CoordinateFrame('world'),
               rr.Points3D([pos], labels=[label], radii=.004), static=True)

    with tempfile.TemporaryDirectory(prefix='robotwin_initial_states_') as temporary:
        for robot, initial, provenance in states:
            name = robot.name
            scene = f'{name}/world'
            rr.log(scene, rr.CoordinateFrame('world'), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            axes(scene+'/origin', np.eye(4), 'World', .15)
            info = [f'# {name} — solver initial state', '',
                    'Source: `robotwin_ik/_initial_state.py` → `BUILTIN_INITIAL_JOINTS`.', '',
                    'IK seed and first-frame soft preference; not a fixed first output frame.', '',
                    'Axes: red X, green Y, blue Z. TCP: virtual closed fingertip midpoint.', '']
            positions = []
            contents = [scene+'/**']
            for side in sides:
                geometry = robot.geometries[side]
                q = initial[side]
                opening = provenance['gripper_open'][side] if args.gripper_open is None else args.gripper_open
                values = dict(zip(geometry.canonical_joint_names, q))
                values.update(gripper_positions(robot.model, robot.config, ('left', 'right').index(side), opening))
                root = robot.world_from_root[side]
                base = root @ geometry.footprint_from_base
                tcp = root @ fk_link(robot.model, geometry.ee_link, values) @ geometry.link6_from_tcp
                positions.extend([base[:3, 3], tcp[:3, 3]])
                print(f'{name}/{side}: joints (rad) = {q.tolist()}')
                print(f'  base XYZ = {base[:3, 3].tolist()}; closed TCP XYZ = {tcp[:3, 3].tolist()}')
                info.extend([f'## {side} arm', '', '| Joint | rad | deg |', '|---|---:|---:|'])
                info.extend(f'| {joint} | {value:.8f} | {np.degrees(value):.4f} |'
                            for joint, value in zip(geometry.canonical_joint_names, q))
                info.extend(['', f'Gripper opening: {opening:g}',
                             f'Base XYZ (m): {np.round(base[:3, 3], 6).tolist()}',
                             f'Closed TCP XYZ (m): {np.round(tcp[:3, 3], 6).tolist()}', ''])
                prepared = Path(temporary) / f'{name}_{side}.urdf'
                prepare_visual(robot.urdf_path, prepared, name, side)
                # UrdfTree treats entity_path_prefix as one path segment, escaping '/'.
                # Keep it flat so the blueprint filter matches emitted mesh entities.
                entity = f'{name}_{side}_robot'
                prefix = f'{name}/{side}/'
                contents.append(entity+'/**')
                tree = rr.urdf.UrdfTree.from_file_path(
                    prepared, entity_path_prefix=entity, frame_prefix=prefix,
                    static_transform_entity_path=entity+'_static_tf')
                recording.send_chunks(tree.stream(include_joint_transforms=False))
                rr.log(f'{name}/transforms/{side}/root', rr.Transform3D(
                    translation=root[:3, 3], mat3x3=root[:3, :3], parent_frame='world',
                    child_frame=prefix+robot.model.root_link), static=True)
                for joint_name, joint in robot.model.joints_by_name.items():
                    # Aloha's full URDF also contains wheels and inactive rear arms;
                    # retain their URDF zero pose, as in the trajectory viewer.
                    if name != 'aloha-agilex' and joint.joint_type != 'fixed' and joint_name not in values:
                        raise ValueError(f'Missing initial value for {name}/{joint_name}')
                    t = joint_matrix(joint, values.get(joint_name, 0.))
                    rr.log(f'{name}/transforms/{side}/{joint_name}', rr.Transform3D(
                        translation=t[:3, 3], mat3x3=t[:3, :3],
                        parent_frame=prefix+joint.parent, child_frame=prefix+joint.child), static=True)
                axes(f'{scene}/{side}/base', base, f'{side} base', .12)
                axes(f'{scene}/{side}/tcp', tcp, f'{side} closed TCP', .08)
                rr.log(f'{scene}/{side}/tcp/point', rr.CoordinateFrame('world'),
                       rr.Points3D([tcp[:3, 3]], colors=[COLORS[side]], radii=.007), static=True)
            rr.log(name+'/info', rr.TextDocument('\n'.join(info), media_type=rr.MediaType.MARKDOWN), static=True)
            target = np.mean(positions, axis=0)
            views.append(rrb.Horizontal(
                rrb.Spatial3DView(origin=scene, contents=contents, name=name,
                    eye_controls=rrb.EyeControls3D(position=(target+[1.2, 1.4, .9]).tolist(),
                                                  look_target=target.tolist(), eye_up=[0, 0, 1])),
                rrb.TextDocumentView(origin=name+'/info', name='Initial joint angles'),
                column_shares=[3, 2], name=name))
        rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*views), auto_views=False))
        recording.flush()
    print(f'Saved {args.output.expanduser().resolve()}' if args.output else 'Initial states sent to Rerun.')
    if use_web:
        wait_for_web_viewer(recording)


if __name__ == '__main__':
    main()
