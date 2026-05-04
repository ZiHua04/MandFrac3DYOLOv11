from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo3d.assigner.task_aligned_3d import TaskAlignedAssigner3D
from yolo3d.utils.box3d_torch import (
    boxes_giou3d,
    boxes_iou3d_aligned,
    decode_dfl_distances_3d,
    distances_to_boxes_zyxzyx,
    make_anchor_points_3d,
)


@dataclass(frozen=True)
class LossWeights3D:
    box: float = 7.5
    dfl: float = 1.5
    cls: float = 0.5
    quality: float = 0.25


@dataclass(frozen=True)
class VarifocalLoss3DConfig:
    """Varifocal 风格分类加权配置。"""

    alpha: float = 0.75
    gamma: float = 2.0


class YOLOv11Loss3D(nn.Module):
    """YOLOv11 风格 3D 检测损失。

    模型输出是一个 list，每个尺度层输出一个 dict：
      - `cls_logits`:     [B, nc, D, H, W]
      - `reg_dfl`:        [B, 6*(reg_max+1), D, H, W]
      - `quality_logits`: [B, 1, D, H, W]，仅在启用 QA head 时存在

    目标来自数据集：
      targets: list[dict], len=B
        - `boxes_zyxzyx`: [Ni, 6]
        - `labels`:       [Ni]
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        strides_zyx: Sequence[int] = (8, 16, 32),
        assigner: TaskAlignedAssigner3D | None = None,
        weights: LossWeights3D = LossWeights3D(),
        vfl: VarifocalLoss3DConfig = VarifocalLoss3DConfig(),
        quality_neg_weight: float = 0.05,
        quality_neg_sample_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides_zyx = tuple(int(s) for s in strides_zyx)
        self.assigner = assigner or TaskAlignedAssigner3D(topk=10, alpha=1.0, beta=6.0)
        self.weights = weights
        self.vfl = vfl
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.quality_neg_weight = float(max(0.0, quality_neg_weight))
        self.quality_neg_sample_ratio = float(max(0.0, quality_neg_sample_ratio))

    def _flatten_outputs(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        cls_all = []
        reg_all = []
        quality_all = []
        anchor_points = []
        anchor_strides = []
        has_quality = any("quality_logits" in out for out in outputs)

        for level, out in enumerate(outputs):
            cls = out["cls_logits"]  # [B, nc, D, H, W]
            reg = out["reg_dfl"]  # [B, 6*(reg_max+1), D, H, W]
            b, nc, d, h, w = cls.shape
            stride = self.strides_zyx[level]
            ap = make_anchor_points_3d((d, h, w), stride=stride, device=cls.device, dtype=cls.dtype)

            cls_all.append(cls.permute(0, 2, 3, 4, 1).reshape(b, -1, nc))
            reg_all.append(reg.permute(0, 2, 3, 4, 1).reshape(b, -1, reg.shape[1]))
            anchor_points.append(ap)
            anchor_strides.append(torch.full((ap.shape[0],), float(stride), device=cls.device, dtype=cls.dtype))

            if has_quality:
                quality = out.get("quality_logits")
                if quality is None:
                    # 为了兼容混合输出结构，这里在缺失 quality 分支时补零。
                    quality = torch.zeros((b, 1, d, h, w), device=cls.device, dtype=cls.dtype)
                quality_all.append(quality.permute(0, 2, 3, 4, 1).reshape(b, -1, 1))

        cls_all_t = torch.cat(cls_all, dim=1)  # [B, N, nc]
        reg_all_t = torch.cat(reg_all, dim=1)  # [B, N, 6*(reg_max+1)]
        quality_all_t = torch.cat(quality_all, dim=1) if has_quality else None  # [B, N, 1]
        anchor_points_t = torch.cat(anchor_points, dim=0)  # [N, 3]
        anchor_strides_t = torch.cat(anchor_strides, dim=0)  # [N]
        return cls_all_t, reg_all_t, quality_all_t, anchor_points_t, anchor_strides_t

    def forward(self, outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        cls_logits, reg_dfl, quality_logits, anchor_points, anchor_strides = self._flatten_outputs(outputs)
        bsz, _, _ = cls_logits.shape
        device = cls_logits.device

        # 1) 先把 DFL 距离分布解码成连续距离，再还原成 3D 盒子。
        pred_dist = decode_dfl_distances_3d(reg_dfl, reg_max=self.reg_max)  # [B, N, 6]
        pred_dist_vox = pred_dist * anchor_strides[None, :, None]  # [B, N, 6]
        pred_boxes = distances_to_boxes_zyxzyx(anchor_points[None, :, :].expand(bsz, -1, -1), pred_dist_vox)
        pred_scores = cls_logits.sigmoid()

        fg_masks = []
        assigned_boxes = []
        assigned_scores = []
        for bi in range(bsz):
            gt_boxes = targets[bi]["boxes_zyxzyx"].to(device=device, dtype=pred_boxes.dtype)
            gt_labels = targets[bi]["labels"].to(device=device, dtype=torch.long)
            res = self.assigner.assign(
                pred_scores=pred_scores[bi],
                pred_boxes_zyxzyx=pred_boxes[bi],
                anchor_points_zyx=anchor_points,
                anchor_strides=anchor_strides,
                gt_boxes_zyxzyx=gt_boxes,
                gt_labels=gt_labels,
            )
            fg_masks.append(res["fg_mask"])
            assigned_boxes.append(res["assigned_boxes"])
            assigned_scores.append(res["assigned_scores"])

        fg_mask = torch.stack(fg_masks, dim=0)  # [B, N]
        tgt_boxes = torch.stack(assigned_boxes, dim=0)  # [B, N, 6]
        tgt_scores = torch.stack(assigned_scores, dim=0).to(dtype=cls_logits.dtype)  # [B, N, nc]
        pos_counts_by_stride = {
            f"num_pos_s{int(stride)}": (fg_mask & (anchor_strides.view(1, -1) == float(stride))).sum().to(dtype=cls_logits.dtype)
            for stride in self.strides_zyx
        }

        # 2) 分类损失：保留现有 Varifocal 风格加权，避免稠密 3D 网格中负样本完全主导。
        target_score_sum = tgt_scores.sum().clamp_min(1.0)
        p = pred_scores.detach()
        neg_w = float(self.vfl.alpha) * (p.clamp(0.0, 1.0) ** float(self.vfl.gamma))
        vfl_w = torch.where(tgt_scores > 0, tgt_scores, neg_w).to(dtype=cls_logits.dtype)
        cls_loss = (self.bce(cls_logits, tgt_scores) * vfl_w).sum() / target_score_sum

        num_pos = int(fg_mask.sum().item())
        zero = torch.tensor(0.0, device=device)
        if num_pos == 0:
            total = self.weights.cls * cls_loss
            out = {
                "loss": total,
                "loss_cls": cls_loss.detach(),
                "loss_box": zero,
                "loss_dfl": zero,
                "loss_quality": zero,
                "loss_quality_pos": zero,
                "loss_quality_neg": zero,
                "quality_neg_samples": zero,
                "num_pos": zero,
            }
            out.update({k: v.detach() for k, v in pos_counts_by_stride.items()})
            return out

        pos_idx = fg_mask
        pos_pred_boxes = pred_boxes[pos_idx]  # [P, 6]
        pos_tgt_boxes = tgt_boxes[pos_idx]  # [P, 6]

        giou = boxes_giou3d(pos_pred_boxes, pos_tgt_boxes).clamp(-1.0, 1.0)
        box_loss = (1.0 - giou).mean()

        # 3) DFL 损失：
        #    每个正样本都要回归 6 个方向的距离分布：
        #    [pz-z1, z2-pz, py-y1, y2-py, px-x1, x2-px]
        ap = anchor_points[None, :, :].expand(bsz, -1, -1)
        ap_pos = ap[pos_idx]  # [P, 3]
        stride_pos = anchor_strides[None, :, None].expand(bsz, -1, 6)[pos_idx]  # [P, 6]

        z1y1x1 = pos_tgt_boxes[:, 0:3]
        z2y2x2 = pos_tgt_boxes[:, 3:6]
        dist_vox = torch.stack(
            [
                ap_pos[:, 0] - z1y1x1[:, 0],
                z2y2x2[:, 0] - ap_pos[:, 0],
                ap_pos[:, 1] - z1y1x1[:, 1],
                z2y2x2[:, 1] - ap_pos[:, 1],
                ap_pos[:, 2] - z1y1x1[:, 2],
                z2y2x2[:, 2] - ap_pos[:, 2],
            ],
            dim=1,
        ).clamp_min(0.0)
        dist_bins = (dist_vox / stride_pos).clamp(0.0, float(self.reg_max) - 1e-3)

        reg_dim = self.reg_max + 1
        pos_reg = reg_dfl[pos_idx].view(-1, 6, reg_dim)  # [P, 6, reg_max+1]
        dfl_loss = distribution_focal_loss(pos_reg, dist_bins).mean()

        # 4) 质量损失：
        #    只对正样本监督，避免海量负样本把 quality 分支拉成“全 0 分类器”。
        if quality_logits is not None:
            with torch.no_grad():
                quality_target = boxes_iou3d_aligned(pos_pred_boxes, pos_tgt_boxes).clamp(0.0, 1.0)
            quality_logits_pos = quality_logits[pos_idx].reshape(-1)
            quality_loss_pos = F.binary_cross_entropy_with_logits(
                quality_logits_pos,
                quality_target.to(dtype=quality_logits_pos.dtype),
                reduction="mean",
            )
            quality_loss_neg = zero
            quality_neg_samples = zero
            if self.quality_neg_weight > 0.0 and self.quality_neg_sample_ratio > 0.0:
                quality_logits_neg = quality_logits[~fg_mask].reshape(-1)
                if quality_logits_neg.numel() > 0:
                    max_neg = int(max(1, round(float(quality_logits_pos.numel()) * self.quality_neg_sample_ratio)))
                    if quality_logits_neg.numel() > max_neg:
                        neg_idx = torch.randperm(quality_logits_neg.numel(), device=device)[:max_neg]
                        quality_logits_neg = quality_logits_neg[neg_idx]
                    quality_loss_neg = F.binary_cross_entropy_with_logits(
                        quality_logits_neg,
                        torch.zeros_like(quality_logits_neg),
                        reduction="mean",
                    )
                    quality_neg_samples = torch.tensor(float(quality_logits_neg.numel()), device=device)
            quality_loss = quality_loss_pos + self.quality_neg_weight * quality_loss_neg
        else:
            quality_loss = zero
            quality_loss_pos = zero
            quality_loss_neg = zero
            quality_neg_samples = zero

        total = (
            self.weights.box * box_loss
            + self.weights.dfl * dfl_loss
            + self.weights.cls * cls_loss
            + self.weights.quality * quality_loss
        )
        out = {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_dfl": dfl_loss.detach(),
            "loss_quality": quality_loss.detach(),
            "loss_quality_pos": quality_loss_pos.detach(),
            "loss_quality_neg": quality_loss_neg.detach(),
            "quality_neg_samples": quality_neg_samples.detach(),
            "num_pos": torch.tensor(float(num_pos), device=device),
        }
        out.update({k: v.detach() for k, v in pos_counts_by_stride.items()})
        return out


def distribution_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """连续目标的 Distribution Focal Loss。

    参数：
      - `pred_logits`: [P, 6, reg_max+1]
      - `target`:      [P, 6]

    返回：
      - `loss`:        [P, 6]
    """
    p, sides, bins = pred_logits.shape
    t = target.clamp(0.0, float(bins - 1) - 1e-6)
    t0 = t.floor().to(torch.long)
    t1 = (t0 + 1).clamp_max(bins - 1)
    w1 = t - t0.to(t.dtype)
    w0 = 1.0 - w1

    logp = F.log_softmax(pred_logits.view(-1, bins), dim=1)
    idx = torch.arange(p * sides, device=pred_logits.device)
    logp0 = logp[idx, t0.view(-1)].view(p, sides)
    logp1 = logp[idx, t1.view(-1)].view(p, sides)
    return -(logp0 * w0 + logp1 * w1)
