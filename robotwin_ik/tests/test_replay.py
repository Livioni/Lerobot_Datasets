"""Verify visualization URDFs preserve kinematics and load in Rerun."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from robotwin_ik._embodiments import NAMES, load_embodiment
from robotwin_ik.visualize_ik_rerun import prepare_visual, prepare_collada
from robotwin_ik._aloha import DEFAULT_URDF
import convert_robotwin_tcp as tc


class ReplayTests(unittest.TestCase):
    def test_aloha_collada_materials_preserve_geometry(self):
        ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
        original = ET.parse(DEFAULT_URDF)
        original_meshes = {
            link.get('name'): (DEFAULT_URDF.parent / link.find('./visual/geometry/mesh').get('filename')).resolve()
            for link in original.findall('link') if link.find('./visual/geometry/mesh') is not None
        }
        repaired = set()
        with tempfile.TemporaryDirectory() as temporary:
            for side in ('left', 'right'):
                prepared = Path(temporary) / f'{side}.urdf'
                prepare_visual(DEFAULT_URDF, prepared, 'aloha-agilex', side)
                self.assertEqual(tc.load_robot_model(str(prepared)).joints_by_name,
                                 tc.load_robot_model(str(DEFAULT_URDF)).joints_by_name)
                for link in ET.parse(prepared).findall('link'):
                    for mesh in link.findall('./visual/geometry/mesh'):
                        path = Path(mesh.get('filename'))
                        if path.suffix.lower() != '.dae':
                            continue
                        source = original_meshes[link.get('name')]
                        before = source.read_bytes()
                        output = ET.parse(path)
                        self.assertTrue(all(t.get('texcoord') for t in output.findall('.//c:texture', ns)))
                        if path != source:
                            repaired.add(source)
                            for tag in ('library_geometries', 'library_visual_scenes', 'asset'):
                                self.assertEqual(ET.tostring(output.find(f'c:{tag}', ns)),
                                                 ET.tostring(ET.parse(source).find(f'c:{tag}', ns)))
                        self.assertEqual(source.read_bytes(), before)
            self.assertGreater(len(repaired), 0)

    def test_valid_collada_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'valid.dae'
            source.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
                              '<texture texture="sampler" texcoord="UVMap"/></COLLADA>')
            self.assertEqual(prepare_collada(source, Path(temporary)), source)

    def test_visual_copy_preserves_joints_and_loads(self):
        try:
            import rerun as rr
            from rerun import urdf as _rerun_urdf  # Older solver environments may lack URDF support.
        except ImportError:
            rr = None
        with tempfile.TemporaryDirectory() as temporary:
            for name in NAMES:
                with self.subTest(name=name):
                    robot=load_embodiment(name)
                    prepared=Path(temporary)/f'{name}.urdf'
                    prepare_visual(robot.urdf_path,prepared,name)
                    output=tc.load_robot_model(str(prepared))
                    self.assertEqual(output.joints_by_name,robot.model.joints_by_name)
                    xml=ET.parse(prepared).getroot()
                    self.assertTrue(all(len(j.findall('limit'))<=1 for j in xml.findall('joint')))
                    self.assertFalse(xml.findall('./link/collision'))
                    if name == 'franka-panda':
                        camera = xml.find("./link[@name='camera']/visual")
                        original = ET.parse(robot.urdf_path).find("./link[@name='camera']/visual")
                        self.assertEqual(camera.find('origin').attrib, original.find('origin').attrib)
                        mesh = camera.find('./geometry/mesh')
                        scale = np.fromstring(mesh.get('scale'), sep=' ')
                        np.testing.assert_allclose(scale, [.001, .001, .001])
                        if rr is not None:
                            import trimesh
                            scene = trimesh.load(mesh.get('filename'), force='scene')
                            # Physical D435 body is roughly 90 x 25 x 25 mm.
                            np.testing.assert_allclose(scene.extents * scale, [.090, .025, .025], atol=.001)
                    if rr is not None:
                        left=rr.urdf.UrdfTree.from_file_path(prepared,entity_path_prefix='left_robot',frame_prefix='left/')
                        right=rr.urdf.UrdfTree.from_file_path(prepared,entity_path_prefix='right_robot',frame_prefix='right/')
                        self.assertNotEqual(left.frame_prefix,right.frame_prefix)
                        paths=left.get_visual_geometry_paths(robot.geometries['left'].ee_link)
                        self.assertTrue(all(path.startswith('/left_robot/') for path in paths),paths)


if __name__=='__main__':unittest.main()
