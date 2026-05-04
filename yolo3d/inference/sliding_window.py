from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from yolo3d.inference.decode import decode_predictions_3d
from yolo3d.inference.fusion import weighted_boxes_fusion_3d
from yolo3d.inference.nms3d import nms3d
from yolo3d.utils.patch_util import generate_coords_map


def _gen_starts(size: int, window: int, step: int) -> List[int]:
    if window >= size:
        return [0]
    starts = list(range(0, size - window + 1, step))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def _axis_keep_bounds(origin: int, valid_size: int, full_size: int, margin: int) -> Tuple[float, float]:
    lo = float(margin) if origin > 0 else 0.0
    hi = float(valid_size - margin) if (origin + valid_size) < full_size else float(valid_size)
    if hi < lo:
        mid = float(valid_size) / 2.0
        lo = min(lo, mid)
        hi = max(hi, mid)
    return lo, hi


def _pad_patch_to_size_zyx(
    patch_cdhw: torch.Tensor,
    target_size_zyx: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Pad [C,D,H,W] patch to target size on tail sides and return original valid size."""
    if patch_cdhw.ndim != 4:
        raise ValueError(f"Expected patch [C,D,H,W], got {tuple(patch_cdhw.shape)}")
    valid_d, valid_h, valid_w = [int(v) for v in patch_cdhw.shape[1:]]
    target_d, target_h, target_w = [int(v) for v in target_size_zyx]
    pad_d = max(0, target_d - valid_d)
    pad_h = max(0, target_h - valid_h)
    pad_w = max(0, target_w - valid_w)
    if pad_d or pad_h or pad_w:
        patch_cdhw = F.pad(patch_cdhw, (0, pad_w, 0, pad_h, 0, pad_d))
    return patch_cdhw, (valid_d, valid_h, valid_w)


def _compute_border_weights(
    centers_zyx: torch.Tensor,
    origin_zyx: Tuple[int, int, int],
    valid_size_zyx: Tuple[int, int, int],
    full_size_zyx: Tuple[int, int, int],
    decay_margin_zyx: Sequence[int],
) -> torch.Tensor:
    """为每个框中心计算窗口边界衰减权重。

    输入：
      - `centers_zyx`: [K, 3]，坐标仍处于当前窗口的局部坐标系内
      - `origin_zyx`: 当前窗口在整幅体数据中的左上前角起点
      - `valid_size_zyx`: 当前窗口真实有效区域大小（未 pad 前）
      - `full_size_zyx`: 整个体数据大小
      - `decay_margin_zyx`: 距离窗口内部边界多少 voxel 内开始衰减

    返回：
      - `weights`: [K]，范围 [0, 1]

    设计细节：
      - 只惩罚“窗口内部的重叠边界”，不惩罚贴着整幅 volume 边界的那一侧
      - 三个轴分别计算归一化距离，再取最小值，表示“离任一危险边界有多近”
    """
    if centers_zyx.numel() == 0:
        return torch.zeros((0,), device=centers_zyx.device, dtype=centers_zyx.dtype)

    z0, y0, x0 = origin_zyx
    valid_d, valid_h, valid_w = valid_size_zyx
    full_d, full_h, full_w = full_size_zyx
    margin_z, margin_y, margin_x = [int(v) for v in decay_margin_zyx]

    weights = torch.ones((centers_zyx.shape[0],), device=centers_zyx.device, dtype=centers_zyx.dtype)
    axis_cfg = [
        (0, z0, valid_d, full_d, margin_z),
        (1, y0, valid_h, full_h, margin_y),
        (2, x0, valid_w, full_w, margin_x),
    ]

    for axis, origin, valid_size, full_size, margin in axis_cfg:
        if margin <= 0:
            continue

        center = centers_zyx[:, axis]
        axis_weight = torch.ones_like(center)

        if origin > 0:
            axis_weight = torch.minimum(axis_weight, (center / float(margin)).clamp(0.0, 1.0))
        if (origin + valid_size) < full_size:
            axis_weight = torch.minimum(
                axis_weight,
                ((float(valid_size) - center) / float(margin)).clamp(0.0, 1.0),
            )
        weights = torch.minimum(weights, axis_weight)

    return weights.clamp(0.0, 1.0)


def _classwise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thr: float,
    max_dets: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_all = []
    for c in labels.unique().tolist():
        mask = labels == int(c)
        keep = nms3d(boxes[mask], scores[mask], iou_thr=iou_thr, topk=max_dets)
        idx = mask.nonzero(as_tuple=False).squeeze(1)[keep]
        keep_all.append(idx)

    keep_all = (
        torch.cat(keep_all, dim=0)
        if keep_all
        else torch.zeros((0,), device=boxes.device, dtype=torch.long)
    )
    if keep_all.numel() > max_dets:
        _, idx = torch.topk(scores[keep_all], k=max_dets)
        keep_all = keep_all[idx]

    return boxes[keep_all], scores[keep_all], labels[keep_all]


@torch.no_grad()
def sliding_window_inference_3d(
    model,
    volume_zyx: torch.Tensor,
    window_size_zyx: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    strides_zyx: Sequence[int] = (8, 16, 32),
    reg_max: int = 16,
    score_thr: float = 0.25,
    pre_nms_topk: int = 300,
    nms_iou_thr: float = 0.5,
    max_dets: int = 300,
    window_border_margin_zyx: Sequence[int] = (0, 0, 0),
    min_box_size_zyx: Sequence[float] = (0.0, 0.0, 0.0),
    min_box_volume: float = 0.0,
    add_coords_channels: bool = False,
    qa_alpha: float = 0.0,
    qa_alpha_per_level: Sequence[float] | None = None,
    fusion_method: str = "nms",
    fusion_iou_thr: float | None = None,
    border_score_decay: bool = False,
    border_decay_margin_zyx: Sequence[int] = (0, 0, 0),
    use_quality_fusion: bool = False,
) -> Dict[str, torch.Tensor]:
    """单个 3D 体数据的滑窗推理。

    该实现同时支持两条后处理路径：
      - `fusion_method="nms"`：保持原始全局 NMS 流程
      - `fusion_method="wbf"`：先跨窗口做 3D WBF，再追加一次轻量 NMS

    默认参数全部关闭时，行为与原版实现保持一致。
    """
    if volume_zyx.ndim != 4:
        raise ValueError(f"Expected [C, D, H, W], got {tuple(volume_zyx.shape)}")
    if fusion_method not in ("nms", "wbf"):
        raise ValueError(f"Unsupported fusion_method='{fusion_method}'. Expected 'nms' or 'wbf'.")

    _, D, H, W = volume_zyx.shape
    wz, wy, wx = [int(v) for v in window_size_zyx]
    margin_z, margin_y, margin_x = [int(v) for v in window_border_margin_zyx]
    min_d, min_h, min_w = [float(v) for v in min_box_size_zyx]
    min_box_volume = float(min_box_volume)
    fusion_iou_thr = float(nms_iou_thr if fusion_iou_thr is None else fusion_iou_thr)

    sz = max(1, int(round(wz * (1.0 - float(overlap)))))
    sy = max(1, int(round(wy * (1.0 - float(overlap)))))
    sx = max(1, int(round(wx * (1.0 - float(overlap)))))

    zs = _gen_starts(D, wz, sz)
    ys = _gen_starts(H, wy, sy)
    xs = _gen_starts(W, wx, sx)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_cls_scores = []
    all_quality_scores = []
    all_border_weights = []

    model.eval()
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                patch = volume_zyx[:, z0 : z0 + wz, y0 : y0 + wy, x0 : x0 + wx]
                patch, (valid_d, valid_h, valid_w) = _pad_patch_to_size_zyx(
                    patch,
                    target_size_zyx=(wz, wy, wx),
                )

                if add_coords_channels:
                    if patch.shape[0] == 1:
                        coords_map = generate_coords_map(
                            patch_coords=(z0, y0, x0),
                            image_size=(D, H, W),
                            patch_size=(wz, wy, wx),
                            device=patch.device,
                            dtype=patch.dtype,
                        )
                        max_z = float(max(0, D - 1)) / float(max(1, D))
                        max_y = float(max(0, H - 1)) / float(max(1, H))
                        max_x = float(max(0, W - 1)) / float(max(1, W))
                        coords_map[0].clamp_(0.0, max_z)
                        coords_map[1].clamp_(0.0, max_y)
                        coords_map[2].clamp_(0.0, max_x)
                        model_patch = torch.cat([patch, coords_map], dim=0)  # [4, D, H, W]
                    elif patch.shape[0] == 4:
                        model_patch = patch
                    else:
                        raise ValueError(
                            f"add_coords_channels=True expects input C=1 or C=4, got C={int(patch.shape[0])}"
                        )
                else:
                    model_patch = patch if patch.shape[0] == 1 else patch[:1]

                out = model(model_patch.unsqueeze(0))
                boxes, scores, labels, aux = decode_predictions_3d(
                    out,
                    strides_zyx=strides_zyx,
                    reg_max=reg_max,
                    score_thr=score_thr,
                    topk=pre_nms_topk,
                    qa_alpha=qa_alpha,
                    qa_alpha_per_level=qa_alpha_per_level,
                    return_aux=True,
                )
                if boxes.numel() == 0:
                    continue

                keep = torch.ones((boxes.shape[0],), device=boxes.device, dtype=torch.bool)
                box_d = boxes[:, 3] - boxes[:, 0]
                box_h = boxes[:, 4] - boxes[:, 1]
                box_w = boxes[:, 5] - boxes[:, 2]

                if min_d > 0.0:
                    keep &= box_d >= min_d
                if min_h > 0.0:
                    keep &= box_h >= min_h
                if min_w > 0.0:
                    keep &= box_w >= min_w
                if min_box_volume > 0.0:
                    keep &= (box_d * box_h * box_w) >= min_box_volume

                center_z = (boxes[:, 0] + boxes[:, 3]) * 0.5
                center_y = (boxes[:, 1] + boxes[:, 4]) * 0.5
                center_x = (boxes[:, 2] + boxes[:, 5]) * 0.5
                local_centers = torch.stack([center_z, center_y, center_x], dim=1)  # [K, 3]

                if margin_z > 0 or margin_y > 0 or margin_x > 0:
                    lo_z, hi_z = _axis_keep_bounds(z0, valid_d, D, margin_z)
                    lo_y, hi_y = _axis_keep_bounds(y0, valid_h, H, margin_y)
                    lo_x, hi_x = _axis_keep_bounds(x0, valid_w, W, margin_x)
                    keep &= (center_z >= lo_z) & (center_z <= hi_z)
                    keep &= (center_y >= lo_y) & (center_y <= hi_y)
                    keep &= (center_x >= lo_x) & (center_x <= hi_x)

                if not bool(keep.any()):
                    continue

                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                cls_scores = aux["cls_scores"][keep]
                quality_scores = aux["quality_scores"][keep]
                local_centers = local_centers[keep]

                if border_score_decay:
                    border_weights = _compute_border_weights(
                        centers_zyx=local_centers,
                        origin_zyx=(z0, y0, x0),
                        valid_size_zyx=(valid_d, valid_h, valid_w),
                        full_size_zyx=(D, H, W),
                        decay_margin_zyx=border_decay_margin_zyx,
                    )
                else:
                    border_weights = torch.ones_like(scores)

                shift = torch.tensor([z0, y0, x0, z0, y0, x0], device=boxes.device, dtype=boxes.dtype)
                boxes = boxes + shift[None, :]
                boxes[:, 0] = boxes[:, 0].clamp(0.0, float(D))
                boxes[:, 3] = boxes[:, 3].clamp(0.0, float(D))
                boxes[:, 1] = boxes[:, 1].clamp(0.0, float(H))
                boxes[:, 4] = boxes[:, 4].clamp(0.0, float(H))
                boxes[:, 2] = boxes[:, 2].clamp(0.0, float(W))
                boxes[:, 5] = boxes[:, 5].clamp(0.0, float(W))

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
                all_cls_scores.append(cls_scores)
                all_quality_scores.append(quality_scores)
                all_border_weights.append(border_weights)

    if not all_boxes:
        device = volume_zyx.device
        return {
            "boxes_zyxzyx": torch.zeros((0, 6), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), device=device, dtype=torch.long),
        }

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls_scores = torch.cat(all_cls_scores, dim=0)
    quality_scores = torch.cat(all_quality_scores, dim=0)
    border_weights = torch.cat(all_border_weights, dim=0)

    if fusion_method == "wbf":
        if use_quality_fusion:
            fusion_weights = cls_scores * quality_scores * border_weights
        else:
            fusion_weights = scores * border_weights

        fused = weighted_boxes_fusion_3d(
            boxes=boxes,
            scores=scores,
            labels=labels,
            weights=fusion_weights,
            iou_thr=fusion_iou_thr,
            skip_box_thr=score_thr,
            topk=max_dets,
        )
        boxes = fused["boxes_zyxzyx"]
        scores = fused["scores"]
        labels = fused["labels"]

        if boxes.numel() > 0:
            boxes, scores, labels = _classwise_nms(
                boxes=boxes,
                scores=scores,
                labels=labels,
                iou_thr=nms_iou_thr,
                max_dets=max_dets,
            )
    else:
        if border_score_decay:
            scores = scores * border_weights

        boxes, scores, labels = _classwise_nms(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_thr=nms_iou_thr,
            max_dets=max_dets,
        )

    return {"boxes_zyxzyx": boxes, "scores": scores, "labels": labels}
