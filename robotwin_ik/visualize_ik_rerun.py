#!/usr/bin/env python3
"""Replay a target embodiment and compare its closed TCP with predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_robotwin_tcp as urdf
from robotwin_ik._io import load_prediction, load_extrinsics
from robotwin_ik._embodiments import DEFAULT_EMBODIMENTS_ROOT, sha256
from robotwin_ik._kinematics import fk_link, joint_matrix, gripper_positions
from robotwin_ik._solver import rotation_error_radians


COLORS = {'left': [40, 185, 255], 'right': [255, 150, 40]}


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--ik-dir', required=True, type=Path)
    p.add_argument('--embodiments-root', type=Path, default=DEFAULT_EMBODIMENTS_ROOT)
    p.add_argument('--output', type=Path, help='Save .rrd instead of opening viewer')
    p.add_argument('--no-rgb', action='store_true')
    p.add_argument('--no-point-cloud', action='store_true')
    p.add_argument('--point-cloud-stride', type=positive_int, default=4)
    p.add_argument('--history', type=positive_int, default=30)
    p.add_argument('--show-candidate', action='store_true', help='Replay an explicitly failed trajectory candidate')
    return p.parse_args(argv)


def load_replay(ik_dir, embodiments_root=DEFAULT_EMBODIMENTS_ROOT, show_candidate=False):
    meta = json.loads((ik_dir / 'metadata.json').read_text())
    if meta.get('format') not in ('robotwin_closed_tcp_ik_v2', 'robotwin_closed_tcp_trajectory_v3'):
        raise ValueError('Expected v2/v3 output; use the original viewer for historical Aloha v1 output')
    filename = 'robot_state.npy'
    if meta.get('schema_version') == 3:
        failed = meta['status'] == 'failed'
        if failed and not show_candidate:
            raise ValueError('Whole trajectory failed validation; use --show-candidate to inspect it')
        if show_candidate and not failed:
            raise ValueError('This result has no failed candidate')
        filename = 'robot_state_candidate.npy' if failed else 'robot_state.npy'
        role = 'failed_candidate' if failed else 'solution'
        if meta['output'].get('state_file') != filename or meta['output'].get('role') != role:
            raise ValueError('State filename/role disagrees with trajectory status')
    elif show_candidate:
        raise ValueError('--show-candidate requires a v3 failed candidate')
    state = np.load(ik_dir / filename, allow_pickle=False)
    if list(state.shape) != meta['output']['shape'] or state.shape[0] != meta['frame_count'] or not np.isfinite(state).all():
        raise ValueError('Robot state shape or values disagree with metadata')
    for arm in meta['arms'].values():
        if not set(np.unique(state[:, arm['gripper_column']])).issubset({0., 1.}):
            raise ValueError('Non-binary gripper state')
    # Old recordings may name an external RoboTwin checkout; resolve from the
    # selected local asset directory and verify it is the identical model.
    asset_dir = Path(embodiments_root).expanduser().resolve() / meta['embodiment']
    local_path = (asset_dir / meta['robot_config_snapshot']['urdf_path']).resolve()
    recorded_path = Path(meta['inputs']['urdf']).expanduser()
    path = local_path if local_path.is_file() and sha256(local_path) == meta['inputs']['urdf_sha256'] else recorded_path
    if sha256(path) != meta['inputs']['urdf_sha256']:
        raise ValueError('URDF changed since solving; replay requires the same kinematic model')
    meta['inputs']['urdf'] = str(path)
    meta['inputs']['embodiments_root'] = str(asset_dir.parent)
    return meta, state, urdf.load_robot_model(str(path))


def prepare_visual(source, destination, name, side=None):
    """Only alter a temporary visualization copy; preserve all kinematics."""
    tree = ET.parse(source)
    if name == 'aloha-agilex' and side is not None:
        own, other = ('fl_', 'fr_') if side == 'left' else ('fr_', 'fl_')
        for link in tree.getroot().findall('link'):
            label = link.get('name', '')
            remove = label.startswith(other) or (side == 'right' and not label.startswith(own) and label != 'right_camera')
            remove |= label == ('right_camera' if side == 'left' else 'left_camera')
            if remove:
                for visual in link.findall('visual'):
                    link.remove(visual)
    # UR5-WSG has duplicate limit tags; keep the first positional limits,
    # matching our URDF reader, and fill only missing attributes from later tags.
    for joint in tree.getroot().findall('joint'):
        limits = joint.findall('limit')
        if len(limits) > 1:
            for extra in limits[1:]:
                for key, value in extra.attrib.items():
                    if key not in limits[0].attrib:
                        limits[0].set(key, value)
                joint.remove(extra)
    for link in tree.getroot().findall('link'):
        for collision in list(link.findall('collision')):
            link.remove(collision)
        for mesh in link.findall('./visual/geometry/mesh'):
            filename = mesh.get('filename', '')
            if name == 'piper' and link.get('name') == 'link8':
                filename = 'meshes/link8.STL'  # DAE has a stray ~24 cm triangle.
            if not filename.startswith(('/', 'package://')):
                filename = str((source.parent / filename).resolve())
            mesh.set('filename', filename)
            if name == 'franka-panda' and link.get('name') == 'camera' and Path(filename).name == 'd435.dae':
                # Rerun's DAE import ignores COLLADA's millimeter unit. Make
                # the unit explicit in URDF, leaving mount origins in meters.
                ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
                unit = ET.parse(filename).find('./c:asset/c:unit', ns)
                meters = float(unit.get('meter', '1')) if unit is not None else 1.0
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                mesh.set('scale', ' '.join(format(v * meters, '.12g') for v in scale))
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def replay_positions(meta, state, model):
    """Compute actual closed TCPs independently of the Rerun transform engine."""
    actual = {}
    for side, arm in meta['arms'].items():
        names = arm['active_joint_names']
        root = np.asarray(arm['world_from_root'])
        tip = np.asarray(arm['ee_from_tcp'])
        actual[side] = np.asarray([root @ fk_link(model, arm['ee_link'], dict(zip(names, q))) @ tip
                                  for q in state[:, arm['joint_columns']]])
    return actual


def main(argv=None):
    args = parse_args(argv)
    # Keep help and input validation independent of the optional Rerun SDK.
    meta, state, model = load_replay(args.ik_dir.expanduser().resolve(), args.embodiments_root, args.show_candidate)
    episode = args.episode.expanduser().resolve()
    prediction_path = Path(meta['inputs']['prediction_json'])
    if sha256(prediction_path) != meta['inputs']['prediction_sha256']:
        raise ValueError('Prediction JSON changed since solving')
    prediction = load_prediction(prediction_path)
    count = meta['frame_count']
    timestamps = np.asarray(meta['timestamps_seconds'])
    if len(prediction.timestamps) != count or not np.allclose(prediction.timestamps, timestamps):
        raise ValueError('Prediction timestamps changed since solving')
    camera = meta['camera']
    extrinsics_path = episode / 'extrinsics' / f'{camera}.npy'
    if sha256(extrinsics_path) != meta['inputs']['extrinsics_sha256']:
        raise ValueError('Camera extrinsics differ from those used for IK')
    extrinsics = load_extrinsics(extrinsics_path, count)
    world_from_camera = np.linalg.inv(extrinsics)
    targets = {side: world_from_camera @ prediction.camera_tcp[side] for side in meta['arms']}
    actual = replay_positions(meta, state, model)
    diagnostics = json.loads((args.ik_dir / 'diagnostics.json').read_text())
    if len(diagnostics['frames']) != count:
        raise ValueError('Diagnostics frame count differs from state')
    import rerun as rr
    import rerun.blueprint as rrb
    from PIL import Image
    rr.init('robotwin_closed_tcp_' + meta['embodiment'], spawn=args.output is None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.output)
    recording = rr.get_global_data_recording()
    rr.log('world', rr.CoordinateFrame('world'), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Explicit frame IDs do not inherit from entity-path parents in Rerun 0.35.
    spatial_paths = ['world/scene']
    for side in meta['arms']:
        spatial_paths.append(f'world/tcp/{side}/error')
        for label in ('target', 'achieved'):
            spatial_paths.extend(f'world/tcp/{side}/{label}{suffix}' for suffix in ('', '/axes', '/history'))
    for path in spatial_paths:
        rr.log(path, rr.CoordinateFrame('world'), static=True)
    distance = meta['embodiment_distance_m']
    layout = f' | base separation {distance} m' if distance is not None else ''
    rr.log('info', rr.TextDocument(
        f"{meta['embodiment']} | {count} frames{layout}\n"
        'Target: white; achieved: arm color; failed: red. TCP is the virtual closed fingertip contact midpoint.\n'
        f"Status: {meta['status']} | {'FAILED CANDIDATE' if args.show_candidate else 'recorded result'}\n{json.dumps(meta['counts'], ensure_ascii=False)}",
        media_type=rr.MediaType.TEXT), static=True)
    spatial = rrb.Spatial3DView(origin='world', contents=['world/**', 'left_robot/**', 'right_robot/**'],
        name='Robot + target / achieved closed TCP',
        eye_controls=rrb.EyeControls3D(position=[1.4, 1.4, 1.8], look_target=[0, -.2, .9], eye_up=[0, 0, 1]))
    upper = spatial if args.no_rgb else rrb.Horizontal(spatial, rrb.Spatial2DView(origin='rgb', name='Source RGB'), column_shares=[3, 1])
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(upper, rrb.Horizontal(
        rrb.TimeSeriesView(origin='signals', name='TCP error / success / gripper'),
        rrb.TimeSeriesView(origin='joints', name='Joint angles (rad)'),
        rrb.TextDocumentView(origin='info')), row_shares=[3, 1]),
        rrb.TimePanel(timeline='time', expanded=True), auto_views=False))
    frames_rgb = sorted((episode / 'images' / camera).glob('*.png'))
    frames_depth = sorted((episode / 'depths' / camera).glob('*.png'))
    if (not args.no_rgb or not args.no_point_cloud) and len(frames_rgb) != count:
        raise ValueError('RGB count differs from state; use --no-rgb --no-point-cloud for robot-only replay')
    if not args.no_point_cloud:
        if len(frames_depth) != count:
            raise ValueError('Depth count differs from state; use --no-point-cloud')
        intrinsics = np.load(episode / 'intrinsics' / f'{camera}.npy', allow_pickle=False)
        if intrinsics.shape == (3, 3):
            intrinsics = np.broadcast_to(intrinsics, (count, 3, 3))
        if intrinsics.shape != (count, 3, 3):
            raise ValueError('Expected camera intrinsics [3,3] or [T,3,3]')
    with tempfile.TemporaryDirectory(prefix='robotwin_ik_replay_') as temporary:
        prepared = Path(temporary) / 'visual.urdf'
        prepare_visual(Path(meta['inputs']['urdf']), prepared, meta['embodiment'])
        for side, arm in meta['arms'].items():
            if meta['embodiment'] == 'aloha-agilex':
                prepared = Path(temporary) / f'visual_{side}.urdf'
                prepare_visual(Path(meta['inputs']['urdf']), prepared, meta['embodiment'], side)
            prefix = side + '/'
            robot_tree = rr.urdf.UrdfTree.from_file_path(prepared,
                entity_path_prefix=f'{side}_robot', frame_prefix=prefix,
                static_transform_entity_path=f'{side}_robot_static_tf')
            robot_tree.log_urdf_to_recording(recording)
            t = np.asarray(arm['world_from_root'])
            rr.log(f'transforms/{side}/root', rr.Transform3D(translation=t[:3, 3], mat3x3=t[:3, :3],
                   parent_frame='world', child_frame=prefix+model.root_link), static=True)
        for i, seconds in enumerate(timestamps):
            rr.set_time('time', duration=float(seconds))
            if not args.no_rgb or not args.no_point_cloud:
                rgb = np.asarray(Image.open(frames_rgb[i]).convert('RGB'))
                if not args.no_rgb:
                    rr.log('rgb', rr.Image(rgb))
                if not args.no_point_cloud:
                    depth = np.asarray(Image.open(frames_depth[i]), dtype=float)[::args.point_cloud_stride, ::args.point_cloud_stride]*.001
                    vv, uu = np.mgrid[0:rgb.shape[0]:args.point_cloud_stride, 0:rgb.shape[1]:args.point_cloud_stride]
                    k = intrinsics[i]; valid = np.isfinite(depth) & (depth > 0)
                    xyz = np.column_stack(((uu[valid]-k[0, 2])*depth[valid]/k[0, 0],
                                           (vv[valid]-k[1, 2])*depth[valid]/k[1, 1], depth[valid]))
                    t = world_from_camera[i]
                    rr.log('world/scene', rr.Points3D(xyz@t[:3, :3].T+t[:3, 3],
                           colors=rgb[::args.point_cloud_stride, ::args.point_cloud_stride][valid]))
            for index, (side, arm) in enumerate(meta['arms'].items()):
                values = dict(zip(arm['active_joint_names'], state[i, arm['joint_columns']]))
                values.update(gripper_positions(model, meta['robot_config_snapshot'], index, state[i, arm['gripper_column']]))
                for name, value in values.items():
                    joint = model.joints_by_name[name]
                    t = joint_matrix(joint, value)
                    # Explicit origin @ motion fixes SDK versions that rotate prismatic axes incorrectly.
                    rr.log(f'transforms/{side}/{name}', rr.Transform3D(translation=t[:3, 3], mat3x3=t[:3, :3],
                           parent_frame=f'{side}/{joint.parent}', child_frame=f'{side}/{joint.child}'))
                success = bool(diagnostics['frames'][i][side]['success'])
                for label, poses, color in [('target', targets[side], [235, 235, 235]),
                                             ('achieved', actual[side], COLORS[side] if success else [255, 35, 35])]:
                    pose = poses[i]; pos = pose[:3, 3]; path = f'world/tcp/{side}/{label}'
                    rr.log(path, rr.Points3D([pos], colors=[color], radii=.005, labels=[f'{side} {label}']))
                    rr.log(path+'/history', rr.LineStrips3D([poses[max(0, i-args.history+1):i+1, :3, 3]], colors=[color], radii=.0015))
                    rr.log(path+'/axes', rr.Arrows3D(origins=np.repeat(pos[None], 3, axis=0),
                           vectors=pose[:3, :3].T*.035, colors=[[255, 60, 60], [60, 220, 80], [70, 130, 255]], radii=.001))
                rr.log(f'world/tcp/{side}/error', rr.LineStrips3D([[targets[side][i, :3, 3], actual[side][i, :3, 3]]],
                       colors=[[100, 240, 100] if success else [255, 35, 35]], radii=.001))
                rr.log(f'signals/{side}/position_error_mm', rr.Scalars(float(np.linalg.norm(targets[side][i, :3, 3]-actual[side][i, :3, 3])*1000)))
                rr.log(f'signals/{side}/rotation_error_deg', rr.Scalars(float(np.degrees(rotation_error_radians(targets[side][i], actual[side][i])))))
                rr.log(f'signals/{side}/success', rr.Scalars(float(success)))
                frame = diagnostics['frames'][i][side]
                for key in ('joint_step_max_abs_rad', 'joint_velocity_rad_s', 'joint_acceleration_rad_s2'):
                    value = frame.get(key)
                    if value is not None:
                        rr.log(f'signals/{side}/{key}', rr.Scalars(float(np.max(np.abs(value)))))
                if frame.get('failure_reasons'):
                    rr.log(f'diagnostics/{side}', rr.TextLog(', '.join(frame['failure_reasons'])))
                rr.log(f'signals/{side}/gripper_open', rr.Scalars(float(state[i, arm['gripper_column']])))
                for name in arm['active_joint_names']:
                    rr.log(f'joints/{side}/{name}', rr.Scalars(float(values[name])))
    recording.flush()
    print(f'Replayed {meta["embodiment"]}: {count} frames' + (f' -> {args.output}' if args.output else ''))


if __name__ == '__main__':
    main()
