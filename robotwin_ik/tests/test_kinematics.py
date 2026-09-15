"""Run with: python -m unittest discover -s robotwin_ik/tests -v."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from robotwin_ik._embodiments import DEFAULT_EMBODIMENTS_ROOT, NAMES, load_embodiment, resolve_distance
from robotwin_ik._kinematics import fk_link, joint_matrix, nearest_valid_branch, gripper_positions
from robotwin_ik._common import build_targets
from robotwin_ik._io import PredictionData
from robotwin_ik._solver import independent_fk_tcp
import convert_robotwin_tcp as tc


class KinematicsTests(unittest.TestCase):
    def test_models_and_layout(self):
        for name in NAMES:
            with self.subTest(name=name):
                robot = load_embodiment(name)
                self.assertEqual(robot.asset_dir, DEFAULT_EMBODIMENTS_ROOT / name)
                self.assertTrue((robot.asset_dir / 'curobo_tmp.yml').is_file())
                dof = 7 if name == 'franka-panda' else 6
                self.assertEqual(len(robot.geometries['left'].canonical_joint_names), dof)
                self.assertAlmostEqual(robot.world_from_root['left'][0, 3], -.3)
                self.assertAlmostEqual(robot.world_from_root['right'][0, 3], .3)
                cs = robot.runtime.robot_cfg['kinematics']['cspace']
                for key in ('retract_config', 'null_space_weight', 'cspace_distance_weight'):
                    self.assertEqual(len(cs[key]), len(cs['joint_names']))
        robot = load_embodiment('ur5-wsg')
        self.assertAlmostEqual(robot.world_from_root['left'][2, 3], robot.config['robot_pose'][0][2])
        self.assertAlmostEqual(robot.world_from_root['right'][2, 3], robot.config['robot_pose'][1][2])

    def test_task_config_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'task.yml'
            path.write_text('embodiment: [piper, piper, 0.8]\n')
            self.assertEqual(resolve_distance('piper', None, path), .8)
            self.assertEqual(resolve_distance('piper', .7, path), .7)
            with self.assertRaises(ValueError):resolve_distance('ARX-X5', None, path)
        for value in [0, -.1, float('nan')]:
            with self.assertRaises(ValueError):resolve_distance('piper', value, None)

    def test_camera_world_base_roundtrip(self):
        for name in NAMES:
            robot = load_embodiment(name, distance=.72)
            camera_from_world = np.eye(4)
            camera_from_world[:3,:3] = tc.rotation_matrix_from_rpy([.3,-.6,.8])
            camera_from_world[:3,3] = [.2,.1,-.7]
            tcp = {side: camera_from_world @ robot.world_from_root[side] @ independent_fk_tcp(
                robot.model, geometry, robot.homestates[side]) for side,geometry in robot.geometries.items()}
            prediction = PredictionData({s:np.stack([v,v]) for s,v in tcp.items()},
                {s:np.ones(2) for s in tcp},np.array([0,.1]),10,'test',{})
            _, targets = build_targets(robot,prediction,np.stack([camera_from_world]*2))
            for side, geometry in robot.geometries.items():
                expected = np.linalg.inv(geometry.footprint_from_base) @ fk_link(robot.model, geometry.ee_link,
                    dict(zip(geometry.canonical_joint_names,robot.homestates[side])))
                np.testing.assert_allclose(targets[side][1],np.stack([expected]*2),atol=1e-10)

    def test_limit_aware_branch_and_no_fk_clamping(self):
        robot = load_embodiment('ARX-X5');names=robot.geometries['left'].canonical_joint_names
        q = np.zeros(6); q[-1] = -3.0
        reference = np.zeros(6);reference[-1] = 3.0
        result = nearest_valid_branch(robot.model,names,q,reference)
        self.assertAlmostEqual(result[-1],-3.0)  # +3.283 would violate the URDF upper bound.
        joint=robot.model.joints_by_name[names[-1]]
        self.assertFalse(np.allclose(joint_matrix(joint,4),joint_matrix(joint,joint.upper)))

    def test_gripper_mapping_and_prismatic_rotation(self):
        for name in NAMES:
            robot=load_embodiment(name)
            closed=gripper_positions(robot.model,robot.config,0,0)
            opened=gripper_positions(robot.model,robot.config,0,1)
            self.assertNotEqual(closed,opened)
            for joint_name,value in opened.items():
                j=robot.model.joints_by_name[joint_name]
                self.assertGreaterEqual(value,j.lower);self.assertLessEqual(value,j.upper)
                displacement=joint_matrix(j,value)[:3,3]-joint_matrix(j,0)[:3,3]
                expected=tc.rotation_matrix_from_rpy(j.origin_rpy)@np.asarray(j.axis)*value/np.linalg.norm(j.axis)
                np.testing.assert_allclose(displacement,expected,atol=1e-12)

    def test_contact_surface_calibration(self):
        try:import trimesh
        except ImportError:self.skipTest('trimesh required for mesh calibration verification')
        specifications={
            'franka-panda':(['panda_leftfinger','panda_rightfinger'],[1,1],[-1,-1],[-.0001,-.0001],1e-4),
            'ARX-X5':(['link7','link8'],[1,1],[-1,1],[-.0244944,.0244944],2e-6),
            'piper':(['link7','link8'],[2,2],[1,1],[0,0],2e-6),
            'ur5-wsg':(['finger_left','finger_right'],[0,0],[1,1],[.003,.003],2e-6)}
        for name,(links,axes,signs,planes,tolerance) in specifications.items():
            robot=load_embodiment(name);tree=ET.parse(robot.urdf_path).getroot();g=robot.geometries['left']
            q=dict(zip(g.canonical_joint_names,robot.homestates['left']))
            q.update(gripper_positions(robot.model,robot.config,0,0))
            inv=np.linalg.inv(fk_link(robot.model,g.ee_link,q));points=[]
            for name_link,axis,sign,plane in zip(links,axes,signs,planes):
                link=next(x for x in tree.findall('link') if x.get('name')==name_link)
                collision=link.find('collision');mesh=trimesh.load(robot.urdf_path.parent/collision.find('./geometry/mesh').get('filename'),force='mesh',process=False)
                centers=mesh.triangles_center;mask=(mesh.face_normals[:,axis]*sign>.999)&(np.abs(centers[:,axis]-plane)<tolerance)
                if name=='franka-panda':mask &= centers[:,2]>.035
                self.assertTrue(mask.any())
                center=np.average(centers[mask],axis=0,weights=mesh.area_faces[mask])
                origin=collision.find('origin');transform=inv@fk_link(robot.model,name_link,q)
                if origin is not None:
                    local=np.eye(4);local[:3,:3]=tc.rotation_matrix_from_rpy(np.fromstring(origin.get('rpy','0 0 0'),sep=' '));local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');transform=transform@local
                points.append(transform[:3,:3]@center+transform[:3,3])
            np.testing.assert_allclose(np.mean(points,axis=0),g.link6_from_tcp[:3,3],atol=2e-9)



if __name__=='__main__':unittest.main()
