from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def make_anchor_points_3d(
    feat_shape_dhw: Sequence[int],
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create anchor points (cell centers) in voxel coordinates.

    Returns:
      anchor_points_zyx: [N,3] where order is (z,y,x)
    """
    d, h, w = int(feat_shape_dhw[0]), int(feat_shape_dhw[1]), int(feat_shape_dhw[2])
    z = (torch.arange(d, device=device, dtype=dtype) + 0.5) * float(stride)
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * float(stride)
    x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * float(stride)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)


def decode_dfl_distances_3d(reg_dfl: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Decode DFL logits to expected distances in bins.

    reg_dfl: [B,N,6*(reg_max+1)]
    Returns: [B,N,6]
    """
    b, n, c = reg_dfl.shape
    bins = reg_max + 1
    if c != 6 * bins:
        raise ValueError(f"Expected channels {6*bins}, got {c}")
    x = reg_dfl.view(b, n, 6, bins)
    prob = F.softmax(x, dim=3)
    proj = torch.arange(bins, device=reg_dfl.device, dtype=prob.dtype)
    dist = (prob * proj).sum(dim=3)
    return dist


def distances_to_boxes_zyxzyx(anchor_points_zyx: torch.Tensor, dist_6: torch.Tensor) -> torch.Tensor:
    """Convert distances to zyxzyx boxes.

    anchor_points_zyx: [B,N,3]
    dist_6: [B,N,6] order: [dz1, dz2, dy1, dy2, dx1, dx2] in voxels
    """
    pz, py, px = anchor_points_zyx[..., 0], anchor_points_zyx[..., 1], anchor_points_zyx[..., 2]
    dz1, dz2, dy1, dy2, dx1, dx2 = [dist_6[..., i] for i in range(6)]
    z1 = pz - dz1
    z2 = pz + dz2
    y1 = py - dy1
    y2 = py + dy2
    x1 = px - dx1
    x2 = px + dx2
    return torch.stack([z1, y1, x1, z2, y2, x2], dim=-1)


def _volumes(boxes: torch.Tensor) -> torch.Tensor:
    d = (boxes[..., 3] - boxes[..., 0]).clamp_min(0.0)
    h = (boxes[..., 4] - boxes[..., 1]).clamp_min(0.0)
    w = (boxes[..., 5] - boxes[..., 2]).clamp_min(0.0)
    return d * h * w


def boxes_iou3d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Pairwise IoU for 3D axis-aligned boxes in zyxzyx.

    a: [M,6], b: [N,6]
    returns: [M,N]
    """
    m = a.shape[0]
    n = b.shape[0]
    if m == 0 or n == 0:
        return torch.zeros((m, n), device=a.device, dtype=a.dtype)

    a1 = a[:, None, :3]
    a2 = a[:, None, 3:]
    b1 = b[None, :, :3]
    b2 = b[None, :, 3:]

    inter1 = torch.maximum(a1, b1)
    inter2 = torch.minimum(a2, b2)
    inter = (inter2 - inter1).clamp_min(0.0)
    inter_vol = inter[..., 0] * inter[..., 1] * inter[..., 2]

    va = _volumes(a)[:, None]
    vb = _volumes(b)[None, :]
    union = (va + vb - inter_vol).clamp_min(eps)
    return inter_vol / union


def boxes_iou3d_aligned(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """逐元素对齐计算 3D IoU。

    输入张量形状：
      - `a`: [P, 6]
      - `b`: [P, 6]

    返回：
      - `iou`: [P]

    这个函数专门服务于正样本质量监督场景：
      - 第 i 个预测框只和第 i 个目标框做 IoU
      - 避免构造完整 `[P, P]` 两两 IoU 矩阵，显存和计算量都更省
    """
    if a.shape != b.shape:
        raise ValueError(f"boxes_iou3d_aligned expects same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return torch.zeros((0,), device=a.device, dtype=a.dtype)

    inter1 = torch.maximum(a[:, :3], b[:, :3])
    inter2 = torch.minimum(a[:, 3:], b[:, 3:])
    inter = (inter2 - inter1).clamp_min(0.0)
    inter_vol = inter[:, 0] * inter[:, 1] * inter[:, 2]

    va = _volumes(a)
    vb = _volumes(b)
    union = (va + vb - inter_vol).clamp_min(eps)
    return inter_vol / union


def boxes_giou3d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """GIoU for aligned pairs (same length) of 3D boxes.

    a: [P,6], b: [P,6]
    returns: [P]
    """
    if a.numel() == 0:
        return torch.zeros((0,), device=a.device, dtype=a.dtype)
    inter1 = torch.maximum(a[:, :3], b[:, :3])
    inter2 = torch.minimum(a[:, 3:], b[:, 3:])
    inter = (inter2 - inter1).clamp_min(0.0)
    inter_vol = inter[:, 0] * inter[:, 1] * inter[:, 2]

    va = _volumes(a)
    vb = _volumes(b)
    union = (va + vb - inter_vol).clamp_min(eps)
    iou = inter_vol / union

    c1 = torch.minimum(a[:, :3], b[:, :3])
    c2 = torch.maximum(a[:, 3:], b[:, 3:])
    c = (c2 - c1).clamp_min(0.0)
    c_vol = (c[:, 0] * c[:, 1] * c[:, 2]).clamp_min(eps)
    giou = iou - (c_vol - union) / c_vol
    return giou
