"""SAPIEN 3 rasterization of URDF visuals, with no simulation stepping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from ._data import Robot

# Camera entities use +X forward, +Y left, +Z up. Episode cameras use OpenCV.
CV_FROM_SAPIEN = np.array([[0., -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])


def resolve_asset(filename, source):
    if filename.startswith('package://'):
        relative = filename[len('package://'):]
        candidates = []
        for directory in (source.parent, *source.parents):
            candidates.append(directory / relative)
            if directory.name == relative.split('/')[0]:
                candidates.append(directory / relative.partition('/')[2])
    elif filename.startswith('file://'):
        candidates = [Path(filename[len('file://'):])]
    else:
        candidates = [source.parent / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'{source}: missing visual asset {filename}')


def prepare_visual(robot, destination):
    """Keep original origins, axes and scales; only adapt a temporary XML copy."""
    tree = ET.parse(robot.urdf_path)
    root = tree.getroot()
    for joint in root.findall('joint'):
        limits = joint.findall('limit')
        for extra in limits[1:]:
            for key, value in extra.attrib.items():
                if key not in limits[0].attrib:
                    limits[0].set(key, value)
            joint.remove(extra)
    for link in root.findall('link'):
        for collision in list(link.findall('collision')):
            link.remove(collision)
        for mesh in link.findall('./visual/geometry/mesh'):
            filename = mesh.get('filename', '')
            if robot.name == 'piper' and link.get('name') == 'link8':
                filename = 'meshes/link8.STL'  # Existing replay fix for a stray DAE triangle.
            mesh.set('filename', str(resolve_asset(filename, robot.urdf_path)))
    for texture in root.findall('.//material/texture'):
        texture.set('filename', str(resolve_asset(texture.get('filename', ''), robot.urdf_path)))
    # COLLADA units/node transforms are handled by SAPIEN's mesh importer.
    tree.write(destination, encoding='utf-8', xml_declaration=True)


@dataclass
class Render:
    rgb: np.ndarray
    depth: np.ndarray


class RobotRenderer:
    def __init__(self, robot: Robot, height: int, width: int):
        try:
            import sapien
        except ImportError as error:
            raise RuntimeError('SAPIEN is required: use conda run -n RoboTwin python -m rerender') from error
        self.sapien = sapien
        self.robot = robot
        self.scene = None
        self.camera = None
        self.instances = []
        self.temporary = tempfile.TemporaryDirectory(prefix='robotwin_rerender_')
        try:
            prepared = Path(self.temporary.name) / 'visual.urdf'
            prepare_visual(robot, prepared)
            sapien.render.set_camera_shader_dir('default')
            sapien.render.set_msaa(1)
            self.scene = sapien.Scene()
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.add_directional_light([0, 0, -1], [1, 1, 1], shadow=False)
            self.camera = self.scene.add_camera('third_views', width, height, 1.0, 0.01, 100.0)
            groups = [('left', 'right')] if robot.config.get('dual_arm', False) else [('left',), ('right',)]
            for sides in groups:
                root = np.asarray(robot.arms[sides[0]]['world_from_root'])
                if any(not np.allclose(root, robot.arms[s]['world_from_root'], atol=1e-6) for s in sides):
                    raise ValueError('Shared dual-arm URDF must have a common world_from_root')
                loader = self.scene.create_urdf_loader()
                loader.fix_root_link = True
                builder = loader.load_file_as_articulation_builder(str(prepared))
                builder.set_initial_pose(sapien.Pose(root))
                articulation = builder.build()
                names = [joint.name for joint in articulation.get_active_joints()]
                column = {name: i for i, name in enumerate(names)}
                driven = set().union(*(robot.values(s, 0, include_locks=False) for s in sides))
                if not driven.issubset(column):
                    raise ValueError(f'URDF renderer lacks driven joints: {driven - column.keys()}')
                locked = {}
                for side in sides:
                    for name, value in robot.locks.get(side, {}).items():
                        # Per-arm solver locks must not freeze the other observed arm.
                        if name in driven:
                            continue
                        if name in locked and not np.isclose(locked[name], value):
                            raise ValueError(f'Conflicting shared joint lock: {name}')
                        locked[name] = value
                qpos = np.zeros((len(robot.state), len(names)), dtype=np.float32)
                for name, value in locked.items():
                    if name in column:
                        qpos[:, column[name]] = value
                for frame in range(len(qpos)):
                    assigned = {}
                    for side in sides:
                        for name, value in robot.values(side, frame, include_locks=False).items():
                            if name in assigned and not np.isclose(assigned[name], value):
                                raise ValueError(f'Conflicting driven joint: {name}')
                            assigned[name] = value
                            qpos[frame, column[name]] = value
                self.instances.append((articulation, qpos))
        except BaseException:
            self.close()
            raise

    def render(self, frame, intrinsic, world_to_camera):
        self.camera.set_perspective_parameters(
            0.01, 100.0, float(intrinsic[0, 0]), float(intrinsic[1, 1]),
            float(intrinsic[0, 2]), float(intrinsic[1, 2]), float(intrinsic[0, 1]))
        self.camera.entity.set_pose(self.sapien.Pose(np.linalg.inv(world_to_camera) @ CV_FROM_SAPIEN))
        for articulation, qpos in self.instances:
            articulation.set_qpos(qpos[frame])
        self.scene.update_render()
        self.camera.take_picture()
        position = self.camera.get_picture('Position')
        depth = -position[..., 2]
        valid = (position[..., 3] < 1) & np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0).astype(np.float32)
        color = self.camera.get_picture('Color')[..., :3]
        rgb = np.rint(np.clip(color, 0, 1) * 255).astype(np.uint8)
        return Render(rgb, depth)

    def close(self):
        self.instances.clear()
        self.camera = None
        self.scene = None
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
