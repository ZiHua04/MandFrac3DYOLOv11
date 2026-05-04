from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from yolo3d.inference import sliding_window_inference_3d
from yolo3d.metrics import evaluate_froc3d_single_class, evaluate_map3d_single_class


@dataclass(frozen=True)
class Trainer3DConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_clip_norm: Optional[float] = None
    # 必须与模型检测层输出 stride 一一对应：
    # - baseline: (8, 16, 32)
    # - 开启 P2:  (4, 8, 16, 32)
    strides_zyx: Sequence[int] = (8, 16, 32)
    reg_max: int = 16
    score_thr: float = 0.25
    pre_nms_topk: int = 300
    nms_iou_thr: float = 0.5
    window_size_zyx: Sequence[int] = (96, 96, 96)
    overlap: float = 0.5
    max_dets: int = 300
    window_border_margin_zyx: Sequence[int] = (0, 0, 0)
    min_box_size_zyx: Sequence[float] = (0.0, 0.0, 0.0)
    min_box_volume: float = 0.0
    use_coord_channels: bool = False
    qa_alpha: float = 0.0
    qa_alpha_per_level: Optional[Sequence[float]] = None
    fusion_method: str = "nms"
    fusion_iou_thr: Optional[float] = None
    border_score_decay: bool = False
    border_decay_margin_zyx: Sequence[int] = (0, 0, 0)
    use_quality_fusion: bool = False


def _autocast_enabled(device: str, amp: bool) -> bool:
    return bool(amp and device.startswith("cuda") and torch.cuda.is_available())


def fit_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    cfg: Trainer3DConfig = Trainer3DConfig(),
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.train()
    if hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(epoch)

    total_loss = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_dfl = 0.0
    total_quality = 0.0
    total_num_pos = 0.0
    extra_totals: Dict[str, float] = {}
    steps = 0

    use_amp = _autocast_enabled(cfg.device, cfg.amp)
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            losses = loss_fn(outputs, targets)
            loss = losses["loss"]

        if scaler is not None and use_amp:
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            did_step = float(scaler.get_scale()) >= scale_before
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
            optimizer.step()
            did_step = True

        if scheduler is not None and did_step:
            scheduler.step()

        total_loss += float(losses["loss"].detach().cpu())
        total_cls += float(losses["loss_cls"].detach().cpu())
        total_box += float(losses["loss_box"].detach().cpu())
        total_dfl += float(losses["loss_dfl"].detach().cpu())
        total_quality += float(losses.get("loss_quality", torch.tensor(0.0)).detach().cpu())
        total_num_pos += float(losses.get("num_pos", torch.tensor(0.0)).detach().cpu())
        for key, value in losses.items():
            if key in {"loss", "loss_cls", "loss_box", "loss_dfl", "loss_quality", "num_pos"}:
                continue
            if not torch.is_tensor(value):
                continue
            extra_totals[key] = extra_totals.get(key, 0.0) + float(value.detach().cpu())
        steps += 1

    denom = max(1, steps)
    out = {
        "loss": total_loss / denom,
        "loss_cls": total_cls / denom,
        "loss_box": total_box / denom,
        "loss_dfl": total_dfl / denom,
        "loss_quality": total_quality / denom,
        "num_pos": total_num_pos / denom,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    out.update({key: value / denom for key, value in sorted(extra_totals.items())})
    return out


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    cfg: Trainer3DConfig = Trainer3DConfig(),
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()

    preds = []
    gts = []
    for images, targets in dataloader:
        for bi in range(images.shape[0]):
            vol = images[bi].to(device)
            pred = sliding_window_inference_3d(
                model=model,
                volume_zyx=vol,
                window_size_zyx=cfg.window_size_zyx,
                overlap=cfg.overlap,
                strides_zyx=cfg.strides_zyx,
                reg_max=cfg.reg_max,
                score_thr=cfg.score_thr,
                pre_nms_topk=cfg.pre_nms_topk,
                nms_iou_thr=cfg.nms_iou_thr,
                max_dets=cfg.max_dets,
                window_border_margin_zyx=cfg.window_border_margin_zyx,
                min_box_size_zyx=cfg.min_box_size_zyx,
                min_box_volume=cfg.min_box_volume,
                add_coords_channels=cfg.use_coord_channels,
                qa_alpha=cfg.qa_alpha,
                qa_alpha_per_level=cfg.qa_alpha_per_level,
                fusion_method=cfg.fusion_method,
                fusion_iou_thr=cfg.fusion_iou_thr,
                border_score_decay=cfg.border_score_decay,
                border_decay_margin_zyx=cfg.border_decay_margin_zyx,
                use_quality_fusion=cfg.use_quality_fusion,
            )
            preds.append(pred)
            gts.append({"boxes_zyxzyx": targets[bi]["boxes_zyxzyx"].to(device)})

    out = evaluate_map3d_single_class(preds, gts)
    out.update(evaluate_froc3d_single_class(preds, gts, iou_thr=0.1))
    return out
