from __future__ import annotations

import argparse
import csv
import json
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from yolo3d.data import build_detection_dataloader
from yolo3d.assigner import TaskAlignedAssigner3D
from yolo3d.engine import Trainer3DConfig, fit_one_epoch, validate_one_epoch
from yolo3d.losses import LossWeights3D, VarifocalLoss3DConfig, YOLOv11Loss3D
from yolo3d.model import YOLOv11_3D
from yolo3d.utils.qa_fusion import coerce_qa_alpha_per_level


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-json", type=str, default="train.json")
    p.add_argument("--val-json", type=str, default="val.json")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--coord_attention",
        type=str,
        default="none",
        choices=["none", "se", "cbam"],
        help="attention after coord-fused C3k2 blocks in multi_scale mode",
    )
    p.add_argument(
        "--feature-attention",
        type=str,
        default="none",
        choices=["none", "se", "cbam"],
        help="optional shallow feature attention for PAN neck blocks",
    )
    p.add_argument(
        "--feature-attention-scope",
        type=str,
        default="none",
        choices=["none", "neck_p2", "neck_p2_p3", "shallow_pan", "all_pan"],
        help="where to apply feature attention inside the PAN neck",
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=100,
        help="if >0, sample with replacement to get this many training steps per epoch (useful for tiny datasets)",
    )
    p.add_argument(
        "--positive-crop-prob",
        type=float,
        default=0.6,
        help="probability to crop around a GT box when GT exists (lower -> more background patches)",
    )
    p.add_argument(
        "--background-crop-prob",
        type=float,
        default=0.4,
        help="when GT exists and positive crop isn't selected, probability to force a pure background crop (no GT overlap)",
    )
    p.add_argument(
        "--background-margin",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="margin [D H W] added around GT boxes when sampling background crops (keeps background patches away from lesions)",
    )
    p.add_argument(
        "--background-max-tries",
        type=int,
        default=100,
        help="max random tries to find a background patch; falls back to random crop if not found",
    )
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--width-mult", type=float, default=0.5, help="model width multiplier")
    p.add_argument("--depth-mult", type=float, default=0.5, help="model depth multiplier")
    p.add_argument("--qa-head", action="store_true", help="enable QA-Head3D quality prediction branch")
    p.add_argument(
        "--qa-alpha",
        type=float,
        default=0.0,
        help="quality-aware decode fusion exponent; default 0 keeps QA as training-only unless explicitly enabled",
    )
    p.add_argument(
        "--qa-alpha-per-level",
        type=float,
        nargs="+",
        default=None,
        help="optional per-level QA fusion exponents; length must match detection levels, e.g. P2/P3/P4/P5",
    )
    p.add_argument("--use-p2", action="store_true", help="enable extra P2 detection level with stride 4")
    p.add_argument(
        "--neck-type",
        type=str,
        default="pan",
        choices=["pan", "bifpn"],
        help="neck type: original PAN or lightweight weighted BiFPN",
    )
    p.add_argument(
        "--neck-channels",
        type=int,
        default=0,
        help="neck output channels for BiFPN; <=0 means auto using scaled P3 width",
    )
    p.add_argument("--bifpn-repeats", type=int, default=1, help="number of repeated BiFPN blocks")
    p.add_argument("--neck-lite", action="store_true", help="use depthwise-separable conv in BiFPN blocks")
    p.add_argument(
        "--directional-reg-head",
        action="store_true",
        help="replace regression head 3D convs with tri-plane directional conv blocks",
    )
    p.add_argument(
        "--directional-shallow-p2p3",
        action="store_true",
        help="use directional conv inside shallow P2/P3 backbone-neck CSP blocks",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=0.0, help="gradient clipping max norm; 0 disables")
    p.add_argument("--loss-cls-weight", type=float, default=0.5, help="classification loss weight")
    p.add_argument("--loss-box-weight", type=float, default=7.5, help="box regression loss weight")
    p.add_argument("--loss-dfl-weight", type=float, default=1.5, help="distribution focal loss weight")
    p.add_argument("--quality-loss-weight", type=float, default=0.25, help="QA head quality loss weight")
    p.add_argument(
        "--quality-neg-weight",
        type=float,
        default=0.05,
        help="extra weight for sampled negative quality BCE; 0 disables negative calibration for QA head",
    )
    p.add_argument(
        "--quality-neg-sample-ratio",
        type=float,
        default=1.0,
        help="sample up to this many negative quality anchors per positive anchor",
    )
    p.add_argument("--vfl-alpha", type=float, default=0.75, help="varifocal negative weight alpha")
    p.add_argument("--vfl-gamma", type=float, default=2.0, help="varifocal negative focusing gamma")
    p.add_argument("--assigner-topk", type=int, default=10, help="task-aligned assigner top-k anchors per gt")
    p.add_argument("--assigner-alpha", type=float, default=1.0, help="task-aligned assigner alpha")
    p.add_argument("--assigner-beta", type=float, default=6.0, help="task-aligned assigner beta")
    p.add_argument(
        "--assigner-p2-max-gt-min-side",
        type=float,
        default=16.0,
        help="when P2 is enabled, only allow stride-4 anchors to match GT whose min side is <= this value; <=0 disables",
    )
    p.add_argument(
        "--assigner-p2-scale-rule",
        type=str,
        default="equiv_side",
        choices=["min_side", "equiv_side", "volume"],
        help="scale rule used to decide whether a GT is small enough for stride-4 P2 assignment",
    )
    p.add_argument(
        "--assigner-p2-max-gt-equiv-side",
        type=float,
        default=8.0,
        help="when P2 is enabled and scale rule is equiv_side, only allow stride-4 anchors to match GT whose equivalent side is <= this value; <=0 disables",
    )
    p.add_argument(
        "--assigner-p2-max-gt-volume",
        type=float,
        default=0.0,
        help="when P2 is enabled and scale rule is volume, only allow stride-4 anchors to match GT whose volume is <= this value; <=0 disables",
    )
    p.add_argument(
        "--assigner-p2-max-pos-per-gt",
        type=int,
        default=2,
        help="when P2 is enabled, cap how many stride-4 positive anchors a GT can receive; <=0 disables",
    )
    p.add_argument(
        "--assigner-p2-reserve-non-p2",
        action="store_true",
        help="when P2 is enabled, reserve at least one non-P2 positive per GT when possible",
    )
    p.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    # AMP is helpful, but for 3D training it can overflow early; keep it opt-in.
    p.add_argument("--amp", action="store_true", help="enable AMP mixed precision on CUDA")
    p.add_argument("--no-augment", action="store_true", help="disable training-time flip/intensity augmentation")
    p.add_argument("--no-scheduler", action="store_true", help="disable LR scheduler (keep LR constant)")
    from datetime import datetime
    
    # 生成当前时间字符串，例如：20260319_153045
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument("--save-dir", type=str, default=f"runs/{time_str}")
    
    p.add_argument("--val-interval", type=int, default=1, help="run validation every N epochs (0 disables periodic val; still validates on last epoch)")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint .pt to resume from (e.g. runs/exp/last.pt)")
    p.add_argument(
        "--best-metric",
        type=str,
        default="froc_auc",
        help="validation metric used to decide whether to update best.pt (e.g. froc_auc, mAP, froc_sens@1.0fp)",
    )
    p.add_argument(
        "--resume-model-only",
        action="store_true",
        help="when resuming, load only model weights and reinitialize optimizer/scheduler/scaler (useful for finetune with new LR)",
    )
    p.add_argument("--eval-only", action="store_true", help="run validation once and exit without training")
    p.add_argument(
        "--val-window-size",
        type=int,
        nargs=3,
        default=None,
        help="validation sliding-window size [D H W]; defaults to --patch-size",
    )
    p.add_argument("--val-overlap", type=float, default=0.25, help="validation sliding-window overlap fraction")
    p.add_argument("--val-score-thr", type=float, default=0.01, help="validation score threshold before NMS")
    p.add_argument("--val-pre-nms-topk", type=int, default=200, help="per-window top-k predictions kept before merging")
    p.add_argument("--val-nms-iou-thr", type=float, default=0.25, help="validation NMS IoU threshold")
    p.add_argument("--val-max-dets", type=int, default=200, help="max detections kept per validation volume after NMS")
    p.add_argument(
        "--no-coord-channels",
        action="store_true",
        default=True,
        help="disable 3 coordinate channels; use intensity-only 1-channel input",
    )
    p.add_argument(
        "--coord-fusion-mode",
        type=str,
        default="input_only",
        choices=["input_only", "multi_scale"],
        help="coordinate usage mode when coord channels are enabled: input_only | multi_scale",
    )
    p.add_argument(
        "--val-window-border-margin",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="discard boxes whose centers fall inside overlapping window borders [D H W] unless the window touches the volume edge",
    )
    p.add_argument(
        "--val-border-decay-margin",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="soft border score decay margin [D H W]; only affects internal overlapping borders",
    )
    p.add_argument(
        "--val-border-score-decay",
        action="store_true",
        help="softly downweight boxes near overlapping window borders before global fusion",
    )
    p.add_argument(
        "--val-fusion-method",
        type=str,
        default="nms",
        choices=["nms", "wbf"],
        help="global window merge method during validation",
    )
    p.add_argument(
        "--val-fusion-iou-thr",
        type=float,
        default=None,
        help="IoU threshold used by WBF clustering; defaults to --val-nms-iou-thr when omitted",
    )
    p.add_argument(
        "--val-use-quality-fusion",
        action="store_true",
        help="when using WBF, weight fusion by cls*quality*border_weight instead of fused decode score",
    )
    p.add_argument(
        "--val-min-box-size",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="discard predicted boxes smaller than [D H W] voxels before global NMS",
    )
    p.add_argument(
        "--val-min-box-volume",
        type=float,
        default=0.0,
        help="discard predicted boxes whose volume is smaller than this many voxels before global NMS",
    )
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_effective_args(
    args: argparse.Namespace,
    *,
    use_coord_channels: bool,
    coord_fusion_mode: str,
    strides_zyx: tuple[int, ...],
) -> Dict[str, Any]:
    effective_args = dict(vars(args))
    qa_alpha_per_level = coerce_qa_alpha_per_level(args.qa_alpha_per_level)
    if qa_alpha_per_level is not None and len(qa_alpha_per_level) != len(strides_zyx):
        raise ValueError(
            f"--qa-alpha-per-level expects {len(strides_zyx)} values for strides {strides_zyx}, "
            f"got {len(qa_alpha_per_level)}: {qa_alpha_per_level}"
        )
    effective_args["no_coord_channels"] = not bool(use_coord_channels)
    effective_args["use_coord_channels"] = bool(use_coord_channels)
    effective_args["coord_fusion_mode"] = str(coord_fusion_mode)
    effective_args["in_channels"] = 4 if use_coord_channels else 1
    effective_args["strides_zyx"] = [int(v) for v in strides_zyx]
    effective_args["num_levels"] = len(strides_zyx)
    effective_args["qa_alpha_per_level"] = list(qa_alpha_per_level) if qa_alpha_per_level is not None else None
    effective_args["assigner_p2_max_gt_min_side"] = (
        float(args.assigner_p2_max_gt_min_side) if bool(args.use_p2) else 0.0
    )
    effective_args["assigner_p2_scale_rule"] = str(args.assigner_p2_scale_rule) if bool(args.use_p2) else "min_side"
    effective_args["assigner_p2_max_gt_equiv_side"] = (
        float(args.assigner_p2_max_gt_equiv_side) if bool(args.use_p2) else 0.0
    )
    effective_args["assigner_p2_max_gt_volume"] = (
        float(args.assigner_p2_max_gt_volume) if bool(args.use_p2) else 0.0
    )
    effective_args["assigner_p2_max_pos_per_gt"] = (
        int(args.assigner_p2_max_pos_per_gt) if bool(args.use_p2) else 0
    )
    effective_args["assigner_p2_reserve_non_p2"] = bool(args.assigner_p2_reserve_non_p2) if bool(args.use_p2) else False
    return effective_args


def _save_config(save_dir: Path, args: argparse.Namespace, effective_args: Dict[str, Any]) -> None:
    cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "effective_args": effective_args,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _append_csv(save_dir: Path, row: Dict[str, Any]) -> None:
    path = save_dir / "metrics.csv"
    is_new = not path.exists()
    row_keys = list(row.keys())
    if is_new:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row_keys)
            w.writeheader()
            w.writerow(row)
        return

    # If the CSV schema evolved (new columns), rewrite the file with a union header.
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old_fieldnames = list(r.fieldnames or [])
        old_rows = list(r)

    if old_fieldnames and all(k in old_fieldnames for k in row_keys):
        # No new columns; keep the existing order.
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=old_fieldnames)
            w.writerow(row)
        return

    # Add any new columns at the end, preserving existing order.
    union_fieldnames = old_fieldnames[:] if old_fieldnames else []
    for k in row_keys:
        if k not in union_fieldnames:
            union_fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=union_fieldnames)
        w.writeheader()
        for orow in old_rows:
            w.writerow(orow)
        w.writerow(row)


def _try_load_state(ckpt: Dict[str, Any], key: str, obj) -> None:
    if key in ckpt and ckpt[key] is not None:
        try:
            obj.load_state_dict(ckpt[key])
        except Exception as e:
            print(f"[warn] failed to load {key} state_dict: {e}")


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()
    _set_seed(int(args.seed))
    use_coord_channels = not bool(args.no_coord_channels)
    coord_fusion_mode = str(args.coord_fusion_mode) if use_coord_channels else "none"
    strides_zyx = (4, 8, 16, 32) if bool(args.use_p2) else (8, 16, 32)
    effective_args = _build_effective_args(
        args,
        use_coord_channels=use_coord_channels,
        coord_fusion_mode=coord_fusion_mode,
        strides_zyx=strides_zyx,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _save_config(save_dir, args, effective_args)

    train_loader = build_detection_dataloader(
        args.train_json,
        split_key="training",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        num_samples_per_epoch=(int(args.steps_per_epoch) * int(args.batch_size)) if int(args.steps_per_epoch) > 0 else None,
        patch_size_zyx=args.patch_size,
        positive_crop_prob=float(args.positive_crop_prob),
        background_crop_prob=float(args.background_crop_prob),
        background_margin_zyx=tuple(int(v) for v in args.background_margin),
        background_max_tries=int(args.background_max_tries),
        intensity_norm_mode="ct_window",
        augment=not bool(args.no_augment),
        base_seed=int(args.seed),
        add_coords_channels=use_coord_channels,
    )
    val_loader = build_detection_dataloader(
        args.val_json,
        split_key="training",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        patch_size_zyx=None,
        intensity_norm_mode="ct_window",
        augment=False,
        base_seed=int(args.seed),
        add_coords_channels=False,
    )

    model = YOLOv11_3D(
        num_classes=1,
        reg_max=16,
        width_mult=float(args.width_mult),
        depth_mult=float(args.depth_mult),
        in_channels=4 if use_coord_channels else 1,
        coord_fusion_mode=coord_fusion_mode,
        coord_attention=str(args.coord_attention),
        feature_attention=str(args.feature_attention),
        feature_attention_scope=str(args.feature_attention_scope),
        qa_head=bool(args.qa_head),
        use_p2=bool(args.use_p2),
        neck_type=str(args.neck_type),
        neck_channels=(int(args.neck_channels) if int(args.neck_channels) > 0 else None),
        bifpn_repeats=int(args.bifpn_repeats),
        neck_lite=bool(args.neck_lite),
        directional_reg_head=bool(args.directional_reg_head),
        directional_shallow_p2p3=bool(args.directional_shallow_p2p3),
    )
    model.to(args.device)
    loss_fn = YOLOv11Loss3D(
        num_classes=1,
        reg_max=16,
        strides_zyx=strides_zyx,
        assigner=TaskAlignedAssigner3D(
            topk=int(args.assigner_topk),
            alpha=float(args.assigner_alpha),
            beta=float(args.assigner_beta),
            p2_max_gt_min_side=(float(args.assigner_p2_max_gt_min_side) if bool(args.use_p2) else 0.0),
            p2_scale_rule=(str(args.assigner_p2_scale_rule) if bool(args.use_p2) else "min_side"),
            p2_max_gt_equiv_side=(float(args.assigner_p2_max_gt_equiv_side) if bool(args.use_p2) else 0.0),
            p2_max_gt_volume=(float(args.assigner_p2_max_gt_volume) if bool(args.use_p2) else 0.0),
            p2_max_pos_per_gt=(int(args.assigner_p2_max_pos_per_gt) if bool(args.use_p2) else 0),
            p2_reserve_non_p2=(bool(args.assigner_p2_reserve_non_p2) if bool(args.use_p2) else False),
        ),
        weights=LossWeights3D(
            box=float(args.loss_box_weight),
            dfl=float(args.loss_dfl_weight),
            cls=float(args.loss_cls_weight),
            quality=float(args.quality_loss_weight),
        ),
        vfl=VarifocalLoss3DConfig(
            alpha=float(args.vfl_alpha),
            gamma=float(args.vfl_gamma),
        ),
        quality_neg_weight=float(args.quality_neg_weight),
        quality_neg_sample_ratio=float(args.quality_neg_sample_ratio),
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None if bool(args.no_scheduler) else CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and args.device.startswith("cuda") and torch.cuda.is_available()))
    val_window_size = tuple(args.val_window_size) if args.val_window_size is not None else tuple(args.patch_size)
    cfg = Trainer3DConfig(
        device=args.device,
        amp=bool(args.amp),
        grad_clip_norm=(float(args.grad_clip_norm) if float(args.grad_clip_norm) > 0 else None),
        window_size_zyx=val_window_size,
        overlap=float(args.val_overlap),
        score_thr=float(args.val_score_thr),
        pre_nms_topk=int(args.val_pre_nms_topk),
        nms_iou_thr=float(args.val_nms_iou_thr),
        max_dets=int(args.val_max_dets),
        window_border_margin_zyx=tuple(int(v) for v in args.val_window_border_margin),
        border_decay_margin_zyx=tuple(int(v) for v in args.val_border_decay_margin),
        border_score_decay=bool(args.val_border_score_decay),
        min_box_size_zyx=tuple(float(v) for v in args.val_min_box_size),
        min_box_volume=float(args.val_min_box_volume),
        strides_zyx=strides_zyx,
        use_coord_channels=use_coord_channels,
        qa_alpha=float(args.qa_alpha),
        qa_alpha_per_level=(
            tuple(float(v) for v in effective_args["qa_alpha_per_level"])
            if effective_args["qa_alpha_per_level"] is not None
            else None
        ),
        fusion_method=str(args.val_fusion_method),
        fusion_iou_thr=(float(args.val_fusion_iou_thr) if args.val_fusion_iou_thr is not None else None),
        use_quality_fusion=bool(args.val_use_quality_fusion),
    )

    start_epoch = 0
    best_metric_value = -1.0
    best_map = -1.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        ckpt = torch.load(resume_path, map_location=args.device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        if not bool(args.resume_model_only):
            _try_load_state(ckpt, "optimizer", optimizer)
            _try_load_state(ckpt, "scheduler", scheduler)
            _try_load_state(ckpt, "scaler", scaler)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            ckpt_best_metric_name = str(ckpt.get("best_metric_name", "mAP"))
            if ckpt_best_metric_name == str(args.best_metric):
                best_metric_value = float(ckpt.get("best_metric_value", ckpt.get("best_map", best_metric_value)))
            else:
                # Metric changed across runs; restart best tracking for the new metric.
                best_metric_value = -1.0
                print(
                    f"[resume] checkpoint best metric is '{ckpt_best_metric_name}', "
                    f"but current --best-metric is '{args.best_metric}'. Reset best_{args.best_metric} to -1.0."
                )
            best_map = float(ckpt.get("best_map", best_map))
            print(
                f"[resume] loaded {args.resume} start_epoch={start_epoch} "
                f"best_{args.best_metric}={best_metric_value}"
            )
        else:
            # Treat as a fresh run with pretrained weights.
            start_epoch = 0
            best_metric_value = -1.0
            best_map = -1.0
            print(f"[resume] loaded model weights only from {args.resume}; optimizer/scheduler reset")

    if bool(args.eval_only):
        val_stats = validate_one_epoch(model=model, dataloader=val_loader, cfg=cfg)
        print(f"eval_only val={val_stats}")
        csv_row = {
            "epoch": start_epoch - 1 if args.resume else -1,
            "lr": None,
            "loss": None,
            "loss_cls": None,
            "loss_box": None,
            "loss_dfl": None,
            "loss_quality": None,
            "mAP": val_stats.get("mAP", None),
            "AP@0.1": val_stats.get("AP@0.1", None),
            "AP@0.5": val_stats.get("AP@0.5", None),
            "precision@0.1": val_stats.get("precision@0.1", None),
            "recall@0.1": val_stats.get("recall@0.1", None),
            "precision@0.1_score0.5": val_stats.get("precision@0.1_score0.5", None),
            "recall@0.1_score0.5": val_stats.get("recall@0.1_score0.5", None),
            "best_f1@0.1": val_stats.get("best_f1@0.1", None),
            "best_precision@0.1": val_stats.get("best_precision@0.1", None),
            "best_recall@0.1": val_stats.get("best_recall@0.1", None),
            "best_score_thr@0.1": val_stats.get("best_score_thr@0.1", None),
            "precision@0.5": val_stats.get("precision@0.5", None),
            "recall@0.5": val_stats.get("recall@0.5", None),
            "precision@0.5_score0.5": val_stats.get("precision@0.5_score0.5", None),
            "recall@0.5_score0.5": val_stats.get("recall@0.5_score0.5", None),
            "best_f1@0.5": val_stats.get("best_f1@0.5", None),
            "best_precision@0.5": val_stats.get("best_precision@0.5", None),
            "best_recall@0.5": val_stats.get("best_recall@0.5", None),
            "best_score_thr@0.5": val_stats.get("best_score_thr@0.5", None),
            "froc_sens@0.125fp": val_stats.get("froc_sens@0.125fp", None),
            "froc_sens@0.25fp": val_stats.get("froc_sens@0.25fp", None),
            "froc_sens@0.5fp": val_stats.get("froc_sens@0.5fp", None),
            "froc_sens@1.0fp": val_stats.get("froc_sens@1.0fp", None),
            "froc_sens@2.0fp": val_stats.get("froc_sens@2.0fp", None),
            "froc_sens@4.0fp": val_stats.get("froc_sens@4.0fp", None),
            "froc_sens@8.0fp": val_stats.get("froc_sens@8.0fp", None),
            "froc_auc": val_stats.get("froc_auc", None),
        }
        _append_csv(save_dir, csv_row)
        return

    for epoch in range(start_epoch, args.epochs):
        train_stats = fit_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg,
        )

        # Validation (sliding window) can be very slow for 3D, so we run it periodically.
        run_val = (epoch == args.epochs - 1) or (args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0))
        val_stats: Dict[str, Any] = {}
        if run_val:
            val_stats = validate_one_epoch(model=model, dataloader=val_loader, cfg=cfg)
            print(f"epoch {epoch+1}/{args.epochs} train={train_stats} val={val_stats}")
        else:
            print(f"epoch {epoch+1}/{args.epochs} train={train_stats} val=skipped")

        csv_row = {
            "epoch": epoch,
            "lr": train_stats.get("lr", None),
            "loss": train_stats.get("loss", None),
            "loss_cls": train_stats.get("loss_cls", None),
            "loss_box": train_stats.get("loss_box", None),
            "loss_dfl": train_stats.get("loss_dfl", None),
            "loss_quality": train_stats.get("loss_quality", None),
            "loss_quality_pos": train_stats.get("loss_quality_pos", None),
            "loss_quality_neg": train_stats.get("loss_quality_neg", None),
            "quality_neg_samples": train_stats.get("quality_neg_samples", None),
            "num_pos": train_stats.get("num_pos", None),
            "mAP": val_stats.get("mAP", None),
            "AP@0.1": val_stats.get("AP@0.1", None),
            "AP@0.5": val_stats.get("AP@0.5", None),
            "precision@0.1": val_stats.get("precision@0.1", None),
            "recall@0.1": val_stats.get("recall@0.1", None),
            "precision@0.1_score0.5": val_stats.get("precision@0.1_score0.5", None),
            "recall@0.1_score0.5": val_stats.get("recall@0.1_score0.5", None),
            "best_f1@0.1": val_stats.get("best_f1@0.1", None),
            "best_precision@0.1": val_stats.get("best_precision@0.1", None),
            "best_recall@0.1": val_stats.get("best_recall@0.1", None),
            "best_score_thr@0.1": val_stats.get("best_score_thr@0.1", None),
            "precision@0.5": val_stats.get("precision@0.5", None),
            "recall@0.5": val_stats.get("recall@0.5", None),
            "precision@0.5_score0.5": val_stats.get("precision@0.5_score0.5", None),
            "recall@0.5_score0.5": val_stats.get("recall@0.5_score0.5", None),
            "best_f1@0.5": val_stats.get("best_f1@0.5", None),
            "best_precision@0.5": val_stats.get("best_precision@0.5", None),
            "best_recall@0.5": val_stats.get("best_recall@0.5", None),
            "best_score_thr@0.5": val_stats.get("best_score_thr@0.5", None),
            "froc_sens@0.125fp": val_stats.get("froc_sens@0.125fp", None),
            "froc_sens@0.25fp": val_stats.get("froc_sens@0.25fp", None),
            "froc_sens@0.5fp": val_stats.get("froc_sens@0.5fp", None),
            "froc_sens@1.0fp": val_stats.get("froc_sens@1.0fp", None),
            "froc_sens@2.0fp": val_stats.get("froc_sens@2.0fp", None),
            "froc_sens@4.0fp": val_stats.get("froc_sens@4.0fp", None),
            "froc_sens@8.0fp": val_stats.get("froc_sens@8.0fp", None),
            "froc_auc": val_stats.get("froc_auc", None),
        }
        for key, value in train_stats.items():
            if key.startswith("num_pos_s"):
                csv_row[key] = value
        _append_csv(save_dir, csv_row)

        improved = False
        improved_map = False
        if run_val:
            current_metric = float(val_stats.get(args.best_metric, 0.0))
            improved = current_metric > best_metric_value
            if improved:
                best_metric_value = current_metric
            # Track best mAP separately for backward compatibility / reporting.
            cur_map = float(val_stats.get("mAP", 0.0))
            improved_map = cur_map > best_map
            if improved_map:
                best_map = cur_map

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "best_metric_name": args.best_metric,
            "best_metric_value": best_metric_value,
            # Keep legacy key for backward compatibility with older checkpoints/tools.
            "best_map": best_map,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if improved:
            torch.save(ckpt, save_dir / "best.pt")
        # Always keep a best-by-mAP checkpoint in addition to the user-selected --best-metric.
        # This helps when you optimize training for FROC but later want the best localization/AP behavior.
        if run_val and improved_map:
            torch.save(ckpt, save_dir / "best_map.pt")


if __name__ == "__main__":
    main()
