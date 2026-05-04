from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class TaskAlignedAssigner3D:
    """3D TaskAlignedAssigner (anchor-free).

    This is a pragmatic 3D adaptation:
      - restrict candidates to anchor points inside gt box
      - per-gt select topk by a stable alignment metric based on distance to gt center
      - resolve conflicts by choosing gt with max metric per anchor

    Inputs are per-image (no padding): gt boxes/labels can be empty.
    """

    topk: int = 10
    alpha: float = 1.0
    beta: float = 6.0
    p2_max_gt_min_side: float = 16.0
    p2_scale_rule: str = "equiv_side"
    p2_max_gt_equiv_side: float = 8.0
    p2_max_gt_volume: float = 0.0
    p2_max_pos_per_gt: int = 2
    p2_reserve_non_p2: bool = False

    def _allow_p2_for_gt(self, gt_boxes_zyxzyx: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """返回每个 GT 是否允许被 P2(anchor stride 4) 匹配。"""
        gt_sizes = (gt_boxes_zyxzyx[:, 3:6] - gt_boxes_zyxzyx[:, 0:3]).clamp_min(0.0)
        rule = str(self.p2_scale_rule).lower()

        if rule == "equiv_side":
            thr = float(self.p2_max_gt_equiv_side)
            if thr <= 0.0:
                return torch.ones((gt_boxes_zyxzyx.shape[0], 1), device=gt_boxes_zyxzyx.device, dtype=torch.bool)
            gt_equiv_side = (gt_sizes.prod(dim=1).clamp_min(float(eps))).pow(1.0 / 3.0)
            return (gt_equiv_side <= thr).unsqueeze(1)

        if rule == "volume":
            thr = float(self.p2_max_gt_volume)
            if thr <= 0.0:
                return torch.ones((gt_boxes_zyxzyx.shape[0], 1), device=gt_boxes_zyxzyx.device, dtype=torch.bool)
            gt_volume = gt_sizes.prod(dim=1)
            return (gt_volume <= thr).unsqueeze(1)

        # Backward-compatible fallback.
        thr = float(self.p2_max_gt_min_side)
        if thr <= 0.0:
            return torch.ones((gt_boxes_zyxzyx.shape[0], 1), device=gt_boxes_zyxzyx.device, dtype=torch.bool)
        gt_min_side = gt_sizes.min(dim=1).values
        return (gt_min_side <= thr).unsqueeze(1)

    @torch.no_grad()
    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_boxes_zyxzyx: torch.Tensor,
        anchor_points_zyx: torch.Tensor,
        anchor_strides: torch.Tensor | None,
        gt_boxes_zyxzyx: torch.Tensor,
        gt_labels: torch.Tensor,
        eps: float = 1e-9,
    ) -> Dict[str, torch.Tensor]:
        """Assign targets for one image.

        Args:
          pred_scores: [N, nc] sigmoid probabilities
          pred_boxes_zyxzyx: [N, 6]
          anchor_points_zyx: [N, 3] in voxel coords
          gt_boxes_zyxzyx: [M, 6]
          gt_labels: [M]

        Returns dict with:
          - fg_mask: [N] bool
          - assigned_labels: [N] long (background=-1)
          - assigned_boxes: [N, 6] float
          - assigned_scores: [N, nc] float (quality-weighted onehot)
          - assigned_gt_idx: [N] long (or -1)
        """
        device = pred_scores.device
        n, nc = pred_scores.shape
        m = gt_boxes_zyxzyx.shape[0]

        assigned_labels = torch.full((n,), -1, device=device, dtype=torch.long)
        assigned_gt_idx = torch.full((n,), -1, device=device, dtype=torch.long)
        assigned_boxes = torch.zeros((n, 6), device=device, dtype=pred_boxes_zyxzyx.dtype)
        assigned_scores = torch.zeros((n, nc), device=device, dtype=pred_scores.dtype)
        fg_mask = torch.zeros((n,), device=device, dtype=torch.bool)

        if m == 0 or n == 0:
            return {
                "fg_mask": fg_mask,
                "assigned_labels": assigned_labels,
                "assigned_boxes": assigned_boxes,
                "assigned_scores": assigned_scores,
                "assigned_gt_idx": assigned_gt_idx,
            }

        # Candidate mask: anchor point must be inside gt.
        pz, py, px = anchor_points_zyx[:, 0], anchor_points_zyx[:, 1], anchor_points_zyx[:, 2]
        z1, y1, x1, z2, y2, x2 = (
            gt_boxes_zyxzyx[:, 0:1],
            gt_boxes_zyxzyx[:, 1:2],
            gt_boxes_zyxzyx[:, 2:3],
            gt_boxes_zyxzyx[:, 3:4],
            gt_boxes_zyxzyx[:, 4:5],
            gt_boxes_zyxzyx[:, 5:6],
        )
        inside = (
            (pz[None, :] >= z1)
            & (pz[None, :] <= z2)
            & (py[None, :] >= y1)
            & (py[None, :] <= y2)
            & (px[None, :] >= x1)
            & (px[None, :] <= x2)
        )  # [M,N]

        # First-pass scale-aware gating for the extra P2 level.
        # When stride-4 anchors are enabled they can overwhelm assignment simply
        # because they are denser and sit closer to every GT center. Restricting
        # P2 anchors to smaller objects keeps the higher-resolution branch from
        # hijacking medium/large GTs.
        if (
            anchor_strides is not None
            and anchor_strides.numel() == n
        ):
            small_gt = self._allow_p2_for_gt(gt_boxes_zyxzyx, eps=eps)  # [M, 1]
            is_p2_anchor = anchor_strides.view(1, n) <= 4.0 + 1e-6
            inside = inside & (~is_p2_anchor | small_gt)

        # Stable alignment metric (center distance) to avoid early-training collapse.
        # Using IoU(pred_box, gt_box) in the metric is brittle because random-initialized
        # boxes have near-zero IoU, which can prevent any useful positives from being
        # selected, and can also de-correlate classification score from localization.
        cz = (z1 + z2) / 2.0
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        dz = (pz[None, :] - cz).to(dtype=pred_scores.dtype)
        dy = (py[None, :] - cy).to(dtype=pred_scores.dtype)
        dx = (px[None, :] - cx).to(dtype=pred_scores.dtype)
        dist2 = dz * dz + dy * dy + dx * dx  # [M,N]
        metric = inside.to(dtype=pred_scores.dtype) / (dist2 + float(eps))  # [M,N], higher is better

        topk = min(self.topk, n)
        has_p2_anchors = bool(
            anchor_strides is not None
            and anchor_strides.numel() == n
            and bool((anchor_strides <= 4.0 + 1e-6).any().item())
        )
        p2_anchor_mask = (
            (anchor_strides <= 4.0 + 1e-6)
            if has_p2_anchors
            else torch.zeros((n,), device=device, dtype=torch.bool)
        )

        cand_gt_parts = []
        cand_idx_parts = []
        cand_metric_parts = []
        max_p2 = int(max(0, self.p2_max_pos_per_gt))
        reserve_non_p2 = bool(self.p2_reserve_non_p2)

        # For each gt: pick topk anchors by metric.
        # When P2 is enabled, additionally cap how many stride-4 positives a GT can receive,
        # otherwise the denser P2 layer can dominate assignment for thin / tiny boxes.
        for gi in range(m):
            row_metric = metric[gi]
            valid_idx = torch.nonzero(row_metric > 0, as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            if has_p2_anchors and max_p2 > 0:
                valid_is_p2 = p2_anchor_mask[valid_idx]
                p2_idx = valid_idx[valid_is_p2]
                non_p2_idx = valid_idx[~valid_is_p2]

                p2_limit = min(max_p2, topk)
                if reserve_non_p2 and non_p2_idx.numel() > 0 and topk > 1:
                    p2_limit = min(p2_limit, topk - 1)

                selected_idx_parts = []

                if p2_idx.numel() > 0 and p2_limit > 0:
                    p2_scores = row_metric[p2_idx]
                    p2_take = min(p2_limit, p2_idx.numel())
                    _, p2_order = torch.topk(p2_scores, k=p2_take, largest=True)
                    selected_idx_parts.append(p2_idx[p2_order])

                selected_so_far = sum(int(x.numel()) for x in selected_idx_parts)
                remaining = max(0, topk - selected_so_far)

                if non_p2_idx.numel() > 0 and remaining > 0:
                    non_p2_scores = row_metric[non_p2_idx]
                    non_p2_take = min(remaining, non_p2_idx.numel())
                    _, non_p2_order = torch.topk(non_p2_scores, k=non_p2_take, largest=True)
                    selected_idx_parts.append(non_p2_idx[non_p2_order])

                if not selected_idx_parts:
                    continue
                selected_idx = torch.cat(selected_idx_parts, dim=0)
            else:
                take = min(topk, valid_idx.numel())
                _, order = torch.topk(row_metric[valid_idx], k=take, largest=True)
                selected_idx = valid_idx[order]

            cand_gt_parts.append(torch.full((selected_idx.numel(),), gi, device=device, dtype=torch.long))
            cand_idx_parts.append(selected_idx)
            cand_metric_parts.append(row_metric[selected_idx])

        if cand_idx_parts:
            cand_gt = torch.cat(cand_gt_parts, dim=0)
            cand_idx = torch.cat(cand_idx_parts, dim=0)
            cand_metric = torch.cat(cand_metric_parts, dim=0)
        else:
            cand_gt = torch.zeros((0,), device=device, dtype=torch.long)
            cand_idx = torch.zeros((0,), device=device, dtype=torch.long)
            cand_metric = torch.zeros((0,), device=device, dtype=pred_scores.dtype)

        # If no valid candidates at all, keep all negative.
        if cand_idx.numel() == 0:
            return {
                "fg_mask": fg_mask,
                "assigned_labels": assigned_labels,
                "assigned_boxes": assigned_boxes,
                "assigned_scores": assigned_scores,
                "assigned_gt_idx": assigned_gt_idx,
            }

        # Resolve conflicts: for each anchor choose gt with max metric.
        best_metric = torch.full((n,), -1.0, device=device, dtype=pred_scores.dtype)
        best_gt = torch.full((n,), -1, device=device, dtype=torch.long)
        order = torch.argsort(cand_metric, descending=True)
        cand_gt = cand_gt[order]
        cand_idx = cand_idx[order]
        cand_metric = cand_metric[order]
        seen = torch.zeros((n,), device=device, dtype=torch.bool)
        for i in range(cand_idx.numel()):
            a = cand_idx[i]
            if not seen[a]:
                seen[a] = True
                best_metric[a] = cand_metric[i]
                best_gt[a] = cand_gt[i]

        fg_mask = best_gt >= 0
        assigned_gt_idx[fg_mask] = best_gt[fg_mask]
        assigned_labels[fg_mask] = gt_labels[best_gt[fg_mask]]
        assigned_boxes[fg_mask] = gt_boxes_zyxzyx[best_gt[fg_mask]]

        # Classification targets for selected positives.
        # Using raw IoU as the class target is fragile in early training because
        # random initial boxes have near-zero IoU, which teaches the classifier
        # that positives should also score near zero. For this lightweight 3D
        # detector, a hard positive target is much more stable.
        sel_a = torch.nonzero(fg_mask, as_tuple=False).squeeze(1)
        assigned_scores[sel_a, assigned_labels[sel_a].clamp_min(0)] = 1.0

        return {
            "fg_mask": fg_mask,
            "assigned_labels": assigned_labels,
            "assigned_boxes": assigned_boxes,
            "assigned_scores": assigned_scores,
            "assigned_gt_idx": assigned_gt_idx,
        }
