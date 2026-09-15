"""Pixel compositing with camera-axis depths in meters (no renderer dependency)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Composite:
    rgb: np.ndarray
    erased: np.ndarray
    drawn: np.ndarray


def compose(rgb, old_depth, new_rgb, new_depth, scene_depth=None, tolerance_m=0.005):
    """Zero/nonfinite rendered depth means no robot; invalid scene depth is preserved.

    A removed old robot is not an occluder. No background is invented in the
    resulting hole: it stays black unless the new robot covers it.
    """
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('Depth tolerance must be nonnegative and finite')
    shape = rgb.shape[:2]
    if rgb.shape != (*shape, 3) or new_rgb.shape != rgb.shape:
        raise ValueError('Expected matching HxWx3 RGB arrays')
    if old_depth.shape != shape or new_depth.shape != shape:
        raise ValueError('Rendered depths must match RGB resolution')
    old = np.isfinite(old_depth) & (old_depth > 0)
    new = np.isfinite(new_depth) & (new_depth > 0)
    if scene_depth is None:
        erased, drawn = old, new
    else:
        if scene_depth.shape != shape:
            raise ValueError('Scene depth must match RGB resolution')
        valid = np.isfinite(scene_depth) & (scene_depth > 0)
        erased = old & valid & (old_depth <= scene_depth + tolerance_m)
        drawn = new & valid & (erased | (new_depth <= scene_depth + tolerance_m))
    output = rgb.copy()
    output[erased] = 0
    output[drawn] = new_rgb[drawn]
    return Composite(output, erased, drawn)
