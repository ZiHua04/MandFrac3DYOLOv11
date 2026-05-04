from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from yolo3d.utils.qa_fusion import resolve_qa_alpha_per_level
from yolo3d.utils.box3d_torch import (
    decode_dfl_distances_3d,
    distances_to_boxes_zyxzyx,
    make_anchor_points_3d,
)


@torch.no_grad()
def decode_predictions_3d(
    outputs: List[Dict[str, torch.Tensor]],
    strides_zyx: Sequence[int] = (8, 16, 32),
    reg_max: int = 16,
    score_thr: float = 0.25,
    topk: int = 500,
    qa_alpha: float = 0.0,
    qa_alpha_per_level: Sequence[float] | None = None,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """把单张样本的网络输出解码成 3D 检测框。

    参数说明：
      - `outputs`: 每个尺度层一个 dict，至少包含
          - `cls_logits`: [1, nc, D, H, W]
          - `reg_dfl`:    [1, 6*(reg_max+1), D, H, W]
        可选包含：
          - `quality_logits`: [1, 1, D, H, W]
      - `qa_alpha`: 质量分数融合系数
          - 0.0 -> 完全退化为原始 `sigmoid(cls)` 排序
          - 1.0 -> 完全由质量分支决定排序
          - (0,1) -> `cls^(1-a) * quality^a`
      - `return_aux=True` 时会额外返回分类分数和质量分数，供滑窗融合模块使用
    """
    device = outputs[0]["cls_logits"].device
    dtype = outputs[0]["cls_logits"].dtype
    alpha_per_level = resolve_qa_alpha_per_level(
        qa_alpha=qa_alpha,
        qa_alpha_per_level=qa_alpha_per_level,
        num_levels=len(outputs),
    )

    all_boxes = []
    all_scores = []
    all_labels = []
    all_cls_scores = []
    all_quality_scores = []

    for level, out in enumerate(outputs):
        level_alpha = float(alpha_per_level[level])
        cls = out["cls_logits"]  # [1, nc, D, H, W]
        reg = out["reg_dfl"]  # [1, 6*(reg_max+1), D, H, W]
        quality = out.get("quality_logits")  # [1, 1, D, H, W] or None
        if cls.shape[0] != 1:
            raise ValueError("decode_predictions_3d expects batch size 1")

        _, nc, d, h, w = cls.shape
        stride = int(strides_zyx[level])
        ap = make_anchor_points_3d((d, h, w), stride=stride, device=device, dtype=dtype)  # [N, 3]
        n = ap.shape[0]

        cls_flat = cls.permute(0, 2, 3, 4, 1).reshape(1, n, nc)[0]  # [N, nc]
        reg_flat = reg.permute(0, 2, 3, 4, 1).reshape(1, n, -1)[0]  # [N, 6*(reg_max+1)]
        cls_scores = cls_flat.sigmoid()

        if quality is not None:
            # quality 只有 1 个通道，因此先展平成 [N, 1]，再广播到 [N, nc]，
            # 表示“同一锚点位置下，不同类别共用同一个定位质量估计”。
            quality_flat = quality.permute(0, 2, 3, 4, 1).reshape(1, n, 1)[0]
            quality_scores = quality_flat.sigmoid().expand(-1, nc)
        else:
            quality_scores = torch.ones_like(cls_scores)

        # P2/P3/P4/P5 can use different QA fusion strengths.
        # This keeps dense low-level candidates recall-friendly while still
        # allowing higher levels to benefit from quality-aware re-ranking.
        if level_alpha > 0.0 and quality is not None:
            fused_scores = (cls_scores.clamp_min(1e-9) ** (1.0 - level_alpha)) * (
                quality_scores.clamp_min(1e-9) ** level_alpha
            )
        else:
            fused_scores = cls_scores

        keep = fused_scores > float(score_thr)
        if not keep.any():
            continue

        keep_idx = keep.nonzero(as_tuple=False)  # [K, 2] -> (anchor_index, class_index)
        a_idx = keep_idx[:, 0]
        c_idx = keep_idx[:, 1]
        s = fused_scores[a_idx, c_idx]
        cls_s = cls_scores[a_idx, c_idx]
        quality_s = quality_scores[a_idx, c_idx]

        # 只对保留下来的锚点解码，减少无效计算。
        dist_bins = decode_dfl_distances_3d(reg_flat[a_idx][None, ...], reg_max=reg_max)[0]  # [K, 6]
        dist_vox = dist_bins * float(stride)
        boxes = distances_to_boxes_zyxzyx(ap[a_idx][None, ...], dist_vox[None, ...])[0]  # [K, 6]

        all_boxes.append(boxes)
        all_scores.append(s)
        all_labels.append(c_idx.to(torch.long))
        all_cls_scores.append(cls_s)
        all_quality_scores.append(quality_s)

    if not all_boxes:
        empty_boxes = torch.zeros((0, 6), device=device, dtype=dtype)
        empty_scores = torch.zeros((0,), device=device, dtype=dtype)
        empty_labels = torch.zeros((0,), device=device, dtype=torch.long)
        if return_aux:
            aux = {
                "cls_scores": empty_scores,
                "quality_scores": empty_scores,
            }
            return empty_boxes, empty_scores, empty_labels, aux
        return empty_boxes, empty_scores, empty_labels

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls_scores = torch.cat(all_cls_scores, dim=0)
    quality_scores = torch.cat(all_quality_scores, dim=0)

    if boxes.shape[0] > topk:
        scores, idx = torch.topk(scores, k=topk, largest=True)
        boxes = boxes[idx]
        labels = labels[idx]
        cls_scores = cls_scores[idx]
        quality_scores = quality_scores[idx]

    if return_aux:
        aux = {
            "cls_scores": cls_scores,
            "quality_scores": quality_scores,
        }
        return boxes, scores, labels, aux
    return boxes, scores, labels
