from __future__ import annotations

from typing import Dict, List, Optional

import torch

from yolo3d.utils.box3d_torch import boxes_iou3d


def _fuse_single_cluster(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """融合单个聚类中的所有 3D 框。

    输入张量：
      - `boxes`:   [K, 6]
      - `scores`:  [K]
      - `weights`: [K]

    输出：
      - `box`:   [6]
      - `score`: [1] 标量

    说明：
      - 框坐标使用加权平均
      - 分数使用加权平均，第一版更稳，便于和 NMS 做公平对比
    """
    weights = weights.clamp_min(1e-6)
    fused_box = (boxes * weights[:, None]).sum(dim=0) / weights.sum()
    fused_score = (scores * weights).sum() / weights.sum()
    return {"box": fused_box, "score": fused_score}


def cluster_boxes_3d_classwise(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thr: float = 0.25,
    weights: Optional[torch.Tensor] = None,
) -> List[Dict[str, torch.Tensor]]:
    """按类别聚类 3D 框，为 WBF 做准备。

    聚类规则：
      1. 先按类别分组，避免不同类别互相融合
      2. 每类内部按 `scores` 从高到低遍历
      3. 如果当前框与已有簇的代表框 IoU > `iou_thr`，则并入该簇
      4. 否则新建一个簇

    每个簇内部保存：
      - `indices`: 原始框索引
      - `box`:     当前簇的融合代表框 [6]
      - `score`:   当前簇的融合代表分数
      - `label`:   当前类别 id
    """
    if boxes.numel() == 0:
        return []

    if weights is None:
        weights = scores

    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]
    original_indices = order

    clusters: List[Dict[str, torch.Tensor]] = []
    for cls in labels.unique(sorted=True).tolist():
        cls_mask = labels == int(cls)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_weights = weights[cls_mask]
        cls_indices = original_indices[cls_mask]

        cls_clusters: List[Dict[str, torch.Tensor]] = []
        for i in range(cls_boxes.shape[0]):
            box_i = cls_boxes[i : i + 1]  # [1, 6]
            merged = False

            if cls_clusters:
                rep_boxes = torch.stack([c["box"] for c in cls_clusters], dim=0)  # [M, 6]
                ious = boxes_iou3d(box_i, rep_boxes)[0]  # [M]
                best_iou, best_idx = torch.max(ious, dim=0)
                if float(best_iou) > float(iou_thr):
                    cluster = cls_clusters[int(best_idx)]
                    cluster["indices"] = torch.cat(
                        [cluster["indices"], cls_indices[i : i + 1]],
                        dim=0,
                    )
                    cluster["boxes"] = torch.cat([cluster["boxes"], cls_boxes[i : i + 1]], dim=0)
                    cluster["scores"] = torch.cat([cluster["scores"], cls_scores[i : i + 1]], dim=0)
                    cluster["weights"] = torch.cat([cluster["weights"], cls_weights[i : i + 1]], dim=0)
                    fused = _fuse_single_cluster(cluster["boxes"], cluster["scores"], cluster["weights"])
                    cluster["box"] = fused["box"]
                    cluster["score"] = fused["score"]
                    merged = True

            if not merged:
                cls_clusters.append(
                    {
                        "label": torch.tensor(int(cls), device=boxes.device, dtype=labels.dtype),
                        "indices": cls_indices[i : i + 1],
                        "boxes": cls_boxes[i : i + 1],
                        "scores": cls_scores[i : i + 1],
                        "weights": cls_weights[i : i + 1],
                        "box": cls_boxes[i],
                        "score": cls_scores[i],
                    }
                )

        clusters.extend(cls_clusters)

    return clusters


@torch.no_grad()
def weighted_boxes_fusion_3d(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    iou_thr: float = 0.25,
    skip_box_thr: float = 0.0,
    topk: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """3D 版 Weighted Boxes Fusion。

    输入：
      - `boxes`:   [N, 6]
      - `scores`:  [N]
      - `labels`:  [N]
      - `weights`: [N]，用于控制坐标融合时各框的话语权

    输出：
      - `boxes_zyxzyx`: [M, 6]
      - `scores`:       [M]
      - `labels`:       [M]
    """
    if boxes.numel() == 0:
        device = boxes.device
        return {
            "boxes_zyxzyx": torch.zeros((0, 6), device=device, dtype=boxes.dtype),
            "scores": torch.zeros((0,), device=device, dtype=scores.dtype),
            "labels": torch.zeros((0,), device=device, dtype=labels.dtype),
        }

    keep = scores >= float(skip_box_thr)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if weights is None:
        weights = scores
    else:
        weights = weights[keep]

    if boxes.numel() == 0:
        device = scores.device
        return {
            "boxes_zyxzyx": torch.zeros((0, 6), device=device, dtype=boxes.dtype),
            "scores": torch.zeros((0,), device=device, dtype=scores.dtype),
            "labels": torch.zeros((0,), device=device, dtype=labels.dtype),
        }

    clusters = cluster_boxes_3d_classwise(
        boxes=boxes,
        scores=scores,
        labels=labels,
        iou_thr=float(iou_thr),
        weights=weights,
    )
    if not clusters:
        device = scores.device
        return {
            "boxes_zyxzyx": torch.zeros((0, 6), device=device, dtype=boxes.dtype),
            "scores": torch.zeros((0,), device=device, dtype=scores.dtype),
            "labels": torch.zeros((0,), device=device, dtype=labels.dtype),
        }

    fused_boxes = torch.stack([c["box"] for c in clusters], dim=0)
    fused_scores = torch.stack([c["score"] for c in clusters], dim=0)
    fused_labels = torch.stack([c["label"] for c in clusters], dim=0)

    order = torch.argsort(fused_scores, descending=True)
    if topk is not None and int(topk) > 0:
        order = order[: int(topk)]

    return {
        "boxes_zyxzyx": fused_boxes[order],
        "scores": fused_scores[order],
        "labels": fused_labels[order],
    }
