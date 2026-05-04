from __future__ import annotations

from typing import Tuple

import numpy as np


def zyxzyx_to_zyxdhw(boxes: np.ndarray) -> np.ndarray:
    """Convert [z1,y1,x1,z2,y2,x2] -> [zc,yc,xc,d,h,w]."""
    if boxes.size == 0:
        return boxes.reshape(-1, 6).astype(np.float32, copy=False)
    out = boxes.astype(np.float32, copy=True)
    z1, y1, x1, z2, y2, x2 = [out[:, i] for i in range(6)]
    out[:, 0] = (z1 + z2) / 2.0
    out[:, 1] = (y1 + y2) / 2.0
    out[:, 2] = (x1 + x2) / 2.0
    out[:, 3] = z2 - z1
    out[:, 4] = y2 - y1
    out[:, 5] = x2 - x1
    return out


def zyxdhw_to_zyxzyx(boxes: np.ndarray) -> np.ndarray:
    """Convert [zc,yc,xc,d,h,w] -> [z1,y1,x1,z2,y2,x2]."""
    if boxes.size == 0:
        return boxes.reshape(-1, 6).astype(np.float32, copy=False)
    out = boxes.astype(np.float32, copy=True)
    zc, yc, xc, d, h, w = [out[:, i] for i in range(6)]
    out[:, 0] = zc - d / 2.0
    out[:, 1] = yc - h / 2.0
    out[:, 2] = xc - w / 2.0
    out[:, 3] = zc + d / 2.0
    out[:, 4] = yc + h / 2.0
    out[:, 5] = xc + w / 2.0
    return out


def clip_boxes_zyxzyx(boxes: np.ndarray, patch_size_zyx: Tuple[int, int, int]) -> np.ndarray:
    """Clip boxes to [0, size] in each axis.

    boxes: [N,6] in [z1,y1,x1,z2,y2,x2]
    """
    if boxes.size == 0:
        return boxes
    pd, ph, pw = patch_size_zyx
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, float(pd))
    out[:, 3] = np.clip(out[:, 3], 0.0, float(pd))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(ph))
    out[:, 4] = np.clip(out[:, 4], 0.0, float(ph))
    out[:, 2] = np.clip(out[:, 2], 0.0, float(pw))
    out[:, 5] = np.clip(out[:, 5], 0.0, float(pw))
    return out


def drop_invalid_boxes_zyxzyx(
    boxes: np.ndarray,
    labels: np.ndarray,
    min_size_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Drop boxes that are empty after cropping/clipping.

    A box is kept if:
      z2 > z1 + min_d, y2 > y1 + min_h, x2 > x1 + min_w
    """
    if boxes.size == 0:
        return boxes, labels
    min_d, min_h, min_w = min_size_zyx
    dz = boxes[:, 3] - boxes[:, 0]
    dy = boxes[:, 4] - boxes[:, 1]
    dx = boxes[:, 5] - boxes[:, 2]
    keep = (dz > min_d) & (dy > min_h) & (dx > min_w)
    return boxes[keep], labels[keep]
