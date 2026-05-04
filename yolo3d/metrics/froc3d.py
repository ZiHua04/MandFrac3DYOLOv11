from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from yolo3d.utils.box3d_torch import boxes_iou3d


@torch.no_grad()
def evaluate_froc3d_single_class(
    all_pred: List[Dict[str, torch.Tensor]],
    all_gt: List[Dict[str, torch.Tensor]],
    iou_thr: float = 0.1,
    fp_per_scan_points: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    max_thresholds: int = 200,
) -> Dict[str, float]:
    """Compute a simple IoU-based FROC summary for single-class detection.

    This is a pragmatic implementation intended for tracking progress:
      - A prediction is a TP if it matches an unmatched GT in the same scan with IoU >= iou_thr.
      - FPs are unmatched predictions.
      - Sensitivity = TP / total_gt, FP/scan = FP / num_scans
      - For each requested FP/scan point, report the best sensitivity achievable at FP/scan <= point.

    Notes:
      - Results depend on which predictions are provided (e.g., any score pre-filtering upstream).
      - For full FROC curves, you can increase max_thresholds or export raw predictions.
    """
    if not all_gt:
        return {f"froc_sens@{p}fp": 0.0 for p in fp_per_scan_points} | {"froc_auc": 0.0}

    num_scans = len(all_gt)
    device = (
        all_pred[0]["scores"].device
        if all_pred and "scores" in all_pred[0]
        else (all_gt[0]["boxes_zyxzyx"].device if all_gt else torch.device("cpu"))
    )

    # Gather all scores to build a global threshold list.
    scores_all = []
    total_gt = 0
    for p, g in zip(all_pred, all_gt):
        ps = p.get("scores", None)
        if ps is not None and ps.numel():
            scores_all.append(ps.detach())
        gb = g.get("boxes_zyxzyx", None)
        if gb is not None:
            total_gt += int(gb.shape[0])

    if total_gt == 0:
        return {f"froc_sens@{p}fp": 0.0 for p in fp_per_scan_points} | {"froc_auc": 0.0}

    if not scores_all:
        # No predictions anywhere -> zero sensitivity at all FP rates.
        out = {f"froc_sens@{p}fp": 0.0 for p in fp_per_scan_points}
        out["froc_auc"] = 0.0
        return out

    scores_cat = torch.cat(scores_all, dim=0).to(device=device)
    scores_cat = torch.unique(scores_cat)
    scores_cat = torch.sort(scores_cat, descending=True).values

    # Subsample thresholds for speed if needed.
    if scores_cat.numel() > int(max_thresholds):
        idx = torch.linspace(0, scores_cat.numel() - 1, steps=int(max_thresholds), device=device)
        idx = idx.round().to(torch.long).clamp(0, scores_cat.numel() - 1)
        thresholds = torch.unique(scores_cat[idx])
        thresholds = torch.sort(thresholds, descending=True).values
    else:
        thresholds = scores_cat

    # Add a threshold above max score to represent the "no predictions" point.
    thresholds = torch.cat([thresholds.new_tensor([float(thresholds[0]) + 1e-6]), thresholds], dim=0)

    fp_per_scan_points = [float(x) for x in fp_per_scan_points]
    best_sens_at = {p: 0.0 for p in fp_per_scan_points}

    # Evaluate each threshold independently (simple, fast enough at current scales).
    for t in thresholds.tolist():
        tp = 0
        fp = 0

        for p, g in zip(all_pred, all_gt):
            pb = p.get("boxes_zyxzyx", None)
            ps = p.get("scores", None)
            gb = g.get("boxes_zyxzyx", None)
            if gb is None:
                continue

            if pb is None or ps is None or pb.numel() == 0 or ps.numel() == 0:
                continue

            keep = ps >= float(t)
            if not bool(keep.any()):
                continue

            pbk = pb[keep]
            psk = ps[keep]

            # Sort predictions by score descending (standard for FROC thresholding).
            order = torch.argsort(psk, descending=True)
            pbk = pbk[order]

            if gb.numel() == 0:
                fp += int(pbk.shape[0])
                continue

            matched = torch.zeros((gb.shape[0],), device=device, dtype=torch.bool)
            # Greedy matching: for each prediction, match best unmatched GT above IoU threshold.
            for i in range(pbk.shape[0]):
                ious = boxes_iou3d(gb, pbk[i : i + 1]).squeeze(1)  # [M]
                best_iou, best_j = torch.max(ious, dim=0)
                j = int(best_j.item())
                if float(best_iou) >= float(iou_thr) and not bool(matched[j]):
                    matched[j] = True
                    tp += 1
                else:
                    fp += 1

        fp_per_scan = fp / float(max(1, num_scans))
        sens = tp / float(max(1, total_gt))

        for p in fp_per_scan_points:
            if fp_per_scan <= p:
                if sens > best_sens_at[p]:
                    best_sens_at[p] = float(sens)

    out = {f"froc_sens@{p}fp": float(best_sens_at[p]) for p in fp_per_scan_points}
    out["froc_auc"] = float(sum(best_sens_at[p] for p in fp_per_scan_points) / max(1, len(fp_per_scan_points)))
    return out
