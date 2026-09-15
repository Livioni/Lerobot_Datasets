"""Aloha geometry/configuration adapter for the shared whole-trajectory solver."""
from __future__ import annotations
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import yaml
import convert_robotwin_tcp as tcp_converter
from ._io import _mapping
from ._kinematics import ArmGeometry
from ._embodiments import DEFAULT_EMBODIMENTS_ROOT

DEFAULT_URDF = DEFAULT_EMBODIMENTS_ROOT / 'aloha-agilex/urdf/arx5_description_isaac.urdf'
DEFAULT_LEFT_CONFIG = DEFAULT_EMBODIMENTS_ROOT / 'aloha-agilex/curobo_left_tmp.yml'
DEFAULT_RIGHT_CONFIG = DEFAULT_EMBODIMENTS_ROOT / 'aloha-agilex/curobo_right_tmp.yml'

@dataclass(frozen=True)
class RuntimeRobotConfig:
    robot_cfg: dict[str, Any]
    source_document: dict[str, Any]
    collision_path: Path
    base_link: str
    ee_link: str
    lock_joints: dict[str, float]
    retract_config: list[float]

def fixed_link_transform(model: tcp_converter.RobotModel, target_link: str) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    for joint in tcp_converter.joint_chain(model, target_link):
        if joint.joint_type != "fixed":
            raise ValueError(
                f"URDF chain footprint->{target_link} unexpectedly contains movable {joint.name}"
            )
        pose = pose @ tcp_converter.joint_transform(joint, 0.0)
    return pose


def arm_geometry(
    model: tcp_converter.RobotModel, side: str, base_link: str, ee_link: str
) -> ArmGeometry:
    prefix = "fl" if side == "left" else "fr"
    expected_base = f"{prefix}_base_link"
    expected_ee = f"{prefix}_link6"
    if base_link != expected_base or ee_link != expected_ee:
        raise ValueError(
            f"{side} config uses base={base_link!r}, ee={ee_link!r}; "
            f"expected {expected_base!r}/{expected_ee!r}"
        )

    finger_joints = [model.joints_by_name[f"{prefix}_joint{i}"] for i in (7, 8)]
    for joint in finger_joints:
        if joint.parent != ee_link:
            raise ValueError(f"{joint.name} parent is {joint.parent!r}, expected {ee_link!r}")
        if not np.allclose(joint.origin_rpy, 0.0, atol=1e-10):
            raise ValueError(
                f"{joint.name} origin rotation is nonzero; pure-translation TCP assumption invalid"
            )
    finger_origins = np.asarray([joint.origin_xyz for joint in finger_joints])
    link6_from_tcp = np.eye(4, dtype=np.float64)
    link6_from_tcp[:3, 3] = finger_origins.mean(axis=0) + np.asarray(
        tcp_converter.ARX5_FINGERTIP_CONTACT_OFFSET_METERS, dtype=np.float64
    )
    return ArmGeometry(
        side=side,
        base_link=base_link,
        ee_link=ee_link,
        canonical_joint_names=tuple(f"{prefix}_joint{i}" for i in range(1, 7)),
        footprint_from_base=fixed_link_transform(model, base_link),
        link6_from_tcp=link6_from_tcp,
    )


def resolve_collision_path(config_path: Path, configured_value: object) -> Path:
    if not isinstance(configured_value, str) or not configured_value:
        raise ValueError(f"{config_path} has no collision_spheres path")
    local_sibling = config_path.parent / Path(configured_value).name
    expanded = Path(os.path.expandvars(configured_value)).expanduser()
    candidates = [local_sibling, expanded]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve collision_spheres={configured_value!r} from {config_path}; "
        f"tried {', '.join(str(path) for path in candidates)}"
    )


def load_runtime_robot_config(
    config_path: Path, urdf_path: Path
) -> RuntimeRobotConfig:
    try:
        document = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), str(config_path)
        )
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read CuRobo config {config_path}: {error}") from error
    robot_cfg = copy.deepcopy(_mapping(document.get("robot_cfg"), f"{config_path}.robot_cfg"))
    kinematics = _mapping(robot_cfg.get("kinematics"), f"{config_path}.kinematics")
    collision_path = resolve_collision_path(config_path, kinematics.get("collision_spheres"))
    kinematics["urdf_path"] = str(urdf_path.resolve())
    kinematics["asset_root_path"] = str(urdf_path.resolve().parent)
    kinematics["collision_spheres"] = str(collision_path)

    base_link = str(kinematics.get("base_link", ""))
    ee_link = str(kinematics.get("ee_link", ""))
    lock_joints_raw = _mapping(
        kinematics.get("lock_joints", {}), f"{config_path}.kinematics.lock_joints"
    )
    lock_joints = {str(name): float(value) for name, value in lock_joints_raw.items()}
    cspace = _mapping(kinematics.get("cspace"), f"{config_path}.kinematics.cspace")
    retract_config = [float(value) for value in cspace.get("retract_config", [])]
    if not base_link or not ee_link or not lock_joints or not retract_config:
        raise ValueError(f"{config_path} is missing base/ee/locked/retract configuration")
    return RuntimeRobotConfig(
        robot_cfg=robot_cfg,
        source_document=document,
        collision_path=collision_path,
        base_link=base_link,
        ee_link=ee_link,
        lock_joints=lock_joints,
        retract_config=retract_config,
    )


def load_aloha(args):
    assets = ((args.robotwin_root.expanduser() / 'assets/embodiments') if args.robotwin_root
              else args.embodiments_root.expanduser()) / 'aloha-agilex'
    assets = assets.resolve()
    config = yaml.safe_load((assets / 'config.yml').read_text())
    path = (args.urdf or assets / config['urdf_path']).expanduser().resolve()
    model = tcp_converter.load_robot_model(str(path))
    runtimes, geometries = {}, {}
    for side in ('left', 'right'):
        config_path = getattr(args, side+'_config') or assets / f'curobo_{side}_tmp.yml'
        runtime = load_runtime_robot_config(config_path.expanduser().resolve(), path)
        runtimes[side] = runtime
        geometries[side] = arm_geometry(model, side, runtime.base_link, runtime.ee_link)
    root = np.linalg.inv(tcp_converter.world_to_base_transform())
    # Both arms belong to a single dual-arm URDF; do not add a second separation.
    return SimpleNamespace(name='aloha-agilex', asset_dir=assets, urdf_path=path, model=model,
                           config=config, runtime_configs=runtimes, geometries=geometries,
                           world_from_root={s: root.copy() for s in geometries},
                           calibration={'source': 'legacy_aloha_closed_fingertip_geometry'},
                           distance=None)


def main(argv=None):
    from ._common import main as solve
    return solve('aloha-agilex', argv)
