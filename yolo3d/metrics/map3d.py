from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from yolo3d.utils.box3d_torch import boxes_iou3d


def _ap_from_pr(precision: torch.Tensor, recall: torch.Tensor) -> float:
    # VOC-style 101-point interpolation.
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    # sample at 101 points
    xs = torch.linspace(0, 1, 101, device=recall.device)
    ap = 0.0
    for x in xs:
        ap += float(mpre[mrec >= x].max())
    return ap / 101.0


def _report_pr_stats(
    results: Dict[str, float],
    *,
    thr: float,
    precision: torch.Tensor,
    recall: torch.Tensor,
    pred_scores: torch.Tensor,
    pr_score_thr: float,
) -> None:
    """Export precision/recall/F1 summaries for a given IoU threshold."""
    thr_name = f"{float(thr):.1f}"
    results[f"precision@{thr_name}"] = float(precision[-1]) if precision.numel() else 0.0
    results[f"recall@{thr_name}"] = float(recall[-1]) if recall.numel() else 0.0

    pr_t = float(pr_score_thr)
    if pred_scores.numel():
        k = int((pred_scores >= pr_t).sum().item())
    else:
        k = 0
    if k > 0:
        results[f"precision@{thr_name}_score{pr_t:g}"] = float(precision[k - 1])
        results[f"recall@{thr_name}_score{pr_t:g}"] = float(recall[k - 1])
    else:
        results[f"precision@{thr_name}_score{pr_t:g}"] = 0.0
        results[f"recall@{thr_name}_score{pr_t:g}"] = 0.0

    if precision.numel():
        f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1e-9)
        best_idx = int(torch.argmax(f1).item())
        results[f"best_f1@{thr_name}"] = float(f1[best_idx])
        results[f"best_precision@{thr_name}"] = float(precision[best_idx])
        results[f"best_recall@{thr_name}"] = float(recall[best_idx])
        results[f"best_score_thr@{thr_name}"] = float(pred_scores[best_idx])
    else:
        results[f"best_f1@{thr_name}"] = 0.0
        results[f"best_precision@{thr_name}"] = 0.0
        results[f"best_recall@{thr_name}"] = 0.0
        results[f"best_score_thr@{thr_name}"] = 0.0


@torch.no_grad()
def evaluate_map3d_single_class(
    all_pred: List[Dict[str, torch.Tensor]],
    all_gt: List[Dict[str, torch.Tensor]],
    iou_thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    pr_score_thr: float = 0.5,
) -> Dict[str, float]:
    """Compute single-class 3D mAP + precision/recall at best-effort.

    Expected per-image dict:
      pred: {boxes_zyxzyx [K,6], scores [K]}
      gt: {boxes_zyxzyx [M,6]}
    """
    device = all_pred[0]["scores"].device if all_pred else torch.device("cpu")
    iou_thresholds = [float(t) for t in iou_thresholds]

    # Flatten predictions across dataset while keeping image id.
    pred_boxes = []
    pred_scores = []
    pred_img = []
    gt_boxes = []
    gt_img = []

    for i, (p, g) in enumerate(zip(all_pred, all_gt)):
        pb = p["boxes_zyxzyx"]
        ps = p["scores"]
        if pb.numel():
            pred_boxes.append(pb)
            pred_scores.append(ps)
            pred_img.append(torch.full((pb.shape[0],), i, device=pb.device, dtype=torch.long))
        gb = g["boxes_zyxzyx"]
        if gb.numel():
            gt_boxes.append(gb)
            gt_img.append(torch.full((gb.shape[0],), i, device=gb.device, dtype=torch.long))

    if not gt_boxes:
        return {
            "mAP": 0.0,
            "AP@0.1": 0.0,
            "AP@0.5": 0.0,
            "precision@0.1": 0.0,
            "recall@0.1": 0.0,
            "precision@0.5": 0.0,
            "recall@0.5": 0.0,
        }

    pred_boxes = torch.cat(pred_boxes, dim=0) if pred_boxes else torch.zeros((0, 6), device=device)
    pred_scores = torch.cat(pred_scores, dim=0) if pred_scores else torch.zeros((0,), device=device)
    pred_img = torch.cat(pred_img, dim=0) if pred_img else torch.zeros((0,), device=device, dtype=torch.long)
    gt_boxes = torch.cat(gt_boxes, dim=0)
    gt_img = torch.cat(gt_img, dim=0)

    # Sort predictions by score descending.
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_img = pred_img[order]

    results: Dict[str, float] = {}
    aps = []

    for thr in iou_thresholds:
        matched = torch.zeros((gt_boxes.shape[0],), device=device, dtype=torch.bool)
        tp = torch.zeros((pred_boxes.shape[0],), device=device)
        fp = torch.zeros((pred_boxes.shape[0],), device=device)

        for i in range(pred_boxes.shape[0]):
            img_id = pred_img[i]
            cand = (gt_img == img_id).nonzero(as_tuple=False).squeeze(1)
            if cand.numel() == 0:
                fp[i] = 1.0
                continue
            ious = boxes_iou3d(gt_boxes[cand], pred_boxes[i : i + 1]).squeeze(1)  # [num_gt_img]
            best_iou, best_j = torch.max(ious, dim=0)
            gi = cand[best_j]
            if float(best_iou) >= thr and not bool(matched[gi]):
                matched[gi] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall = tp_cum / float(gt_boxes.shape[0])
        precision = tp_cum / (tp_cum + fp_cum).clamp_min(1e-9)
        ap = _ap_from_pr(precision, recall)
        aps.append(ap)
        results[f"AP@{thr:.1f}"] = float(ap)

        # NOTE:
        # precision[-1]/recall[-1] corresponds to "use all predictions" (score threshold -> -inf).
        # This is not a clean operating point when upstream score prefilter is loose (e.g. 0.01),
        # but we still export it for backward compatibility, together with fixed-threshold and best-F1 stats.
        if abs(thr - 0.1) < 1e-6 or abs(thr - 0.5) < 1e-6:
            _report_pr_stats(
                results,
                thr=thr,
                precision=precision,
                recall=recall,
                pred_scores=pred_scores,
                pr_score_thr=pr_score_thr,
            )

    results["mAP"] = float(sum(aps) / max(1, len(aps)))
    return results
