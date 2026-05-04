from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo3d.model.layers import C3k2_3D, ConvBNAct3D, SPPF3D, Upsample3D, make_spatial_conv3d
from yolo3d.model.neck_bifpn3d import BiFPN3D
from yolo3d.utils.patch_util import generate_downsampled_coords_map


@dataclass(frozen=True)
class YOLO3DConfig:
    num_classes: int = 1
    reg_max: int = 16
    width_mult: float = 0.5
    depth_mult: float = 0.5
    coord_fusion_mode: str = "none"  # "none" | "input_only" | "multi_scale"
    coord_attention: str = "none"  # "none" | "se" | "cbam"
    feature_attention: str = "none"  # "none" | "se" | "cbam"
    feature_attention_scope: str = "none"  # "none" | "neck_p2" | "neck_p2_p3" | "shallow_pan" | "all_pan"
    qa_head: bool = False
    use_p2: bool = False
    neck_type: str = "pan"  # "pan" | "bifpn"
    neck_channels: Optional[int] = None
    bifpn_repeats: int = 1
    neck_lite: bool = False
    directional_reg_head: bool = False
    directional_shallow_p2p3: bool = False


def _make_divisible(v: int, divisor: int = 8) -> int:
    return max(divisor, int((v + divisor / 2) // divisor * divisor))


def _scale_channels(c: int, width_mult: float) -> int:
    return _make_divisible(max(8, int(c * width_mult)))


def _scale_depth(n: int, depth_mult: float) -> int:
    return max(1, int(round(n * depth_mult)))


class DetectionHead3D(nn.Module):
    """解耦式 3D 检测头。

    每个 level 的输入特征均为 `[B, C, D, H, W]`。
    对于每个尺度层，头部输出包含：
      - `cls_logits`:     `[B, nc, D, H, W]`
      - `reg_dfl`:        `[B, 6 * (reg_max + 1), D, H, W]`
      - `quality_logits`: `[B, 1, D, H, W]`（仅在 `qa_head=True` 时输出）

    这里保持原有 decoupled head 结构不变，只在回归分支后面增设一个质量分支：
      - 分类分支继续负责“是不是目标”
      - 回归分支继续负责 6 个面的距离分布
      - 质量分支显式预测当前锚点回归框的定位可靠性
    """

    def __init__(
        self,
        in_channels: List[int],
        nc: int,
        reg_max: int = 16,
        qa_head: bool = False,
        reg_conv_type: str = "standard",
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.qa_head = bool(qa_head)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.quality_preds = nn.ModuleList() if self.qa_head else nn.ModuleList()

        for c in in_channels:
            cls_c = max(32, int(c))
            reg_c = max(32, int(c))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNAct3D(int(c), cls_c, k=3, s=1),
                    ConvBNAct3D(cls_c, cls_c, k=3, s=1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    make_spatial_conv3d(int(c), reg_c, kind=reg_conv_type, s=1),
                    make_spatial_conv3d(reg_c, reg_c, kind=reg_conv_type, s=1),
                )
            )
            self.cls_preds.append(nn.Conv3d(cls_c, self.nc, kernel_size=1))
            self.reg_preds.append(nn.Conv3d(reg_c, 6 * (self.reg_max + 1), kernel_size=1))
            if self.qa_head:
                self.quality_preds.append(nn.Conv3d(reg_c, 1, kernel_size=1))

    def forward(self, feats: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        outputs: List[Dict[str, torch.Tensor]] = []
        for level, (x, cls_conv, reg_conv, cls_pred, reg_pred) in enumerate(
            zip(feats, self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds)
        ):
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)

            out = {
                "cls_logits": cls_pred(cls_feat),
                "reg_dfl": reg_pred(reg_feat),
            }
            if self.qa_head:
                # 质量分支直接复用回归分支特征，原因是：
                # 1) 定位质量与几何回归特征天然更相关；
                # 2) 不额外新建中间卷积层时更稳、更省参数；
                # 3) 每个空间位置输出 1 个质量 logit，对应当前锚点中心的框质量估计。
                out["quality_logits"] = self.quality_preds[level](reg_feat)
            outputs.append(out)
        return outputs


class YOLOv11_3D(nn.Module):
    """3D YOLOv11 风格检测器。

    默认行为与原始实现保持一致：
      - 输出层级：`P3/P4/P5`
      - Neck：`PAN`
      - 无显式 QA head

    开启改进项后支持：
      - `qa_head=True`  : 每个检测层增加质量分支
      - `use_p2=True`   : Head/Neck 扩展到 `P2/P3/P4/P5`
      - `neck_type=bifpn`: 使用轻量加权融合 Neck
    """

    def __init__(
        self,
        num_classes: int = 1,
        reg_max: int = 16,
        width_mult: float = 0.5,
        depth_mult: float = 0.5,
        in_channels: int = 1,
        coord_fusion_mode: str = "none",
        coord_attention: str = "none",
        feature_attention: str = "none",
        feature_attention_scope: str = "none",
        qa_head: bool = False,
        use_p2: bool = False,
        neck_type: str = "pan",
        neck_channels: Optional[int] = None,
        bifpn_repeats: int = 1,
        neck_lite: bool = False,
        directional_reg_head: bool = False,
        directional_shallow_p2p3: bool = False,
    ) -> None:
        super().__init__()
        if coord_fusion_mode not in ("none", "input_only", "multi_scale"):
            raise ValueError(
                f"Unsupported coord_fusion_mode='{coord_fusion_mode}'. "
                "Expected one of: 'none', 'input_only', 'multi_scale'."
            )
        if coord_attention not in ("none", "se", "cbam"):
            raise ValueError(
                f"Unsupported coord_attention='{coord_attention}'. "
                "Expected one of: 'none', 'se', 'cbam'."
            )
        if feature_attention not in ("none", "se", "cbam"):
            raise ValueError(
                f"Unsupported feature_attention='{feature_attention}'. "
                "Expected one of: 'none', 'se', 'cbam'."
            )
        if feature_attention_scope not in ("none", "neck_p2", "neck_p2_p3", "shallow_pan", "all_pan"):
            raise ValueError(
                f"Unsupported feature_attention_scope='{feature_attention_scope}'. "
                "Expected one of: 'none', 'neck_p2', 'neck_p2_p3', 'shallow_pan', 'all_pan'."
            )
        if neck_type not in ("pan", "bifpn"):
            raise ValueError(f"Unsupported neck_type='{neck_type}'. Expected one of: 'pan', 'bifpn'.")
        if coord_fusion_mode != "none" and int(in_channels) < 4:
            raise ValueError(
                f"coord_fusion_mode='{coord_fusion_mode}' requires input with intensity+3 coords (C>=4), got C={in_channels}."
            )
        if neck_type != "pan" and feature_attention_scope != "none":
            raise ValueError("feature_attention_scope is only supported when neck_type='pan'.")
        if not use_p2 and feature_attention_scope in ("neck_p2", "neck_p2_p3"):
            raise ValueError(f"feature_attention_scope='{feature_attention_scope}' requires use_p2=True.")

        self.cfg = YOLO3DConfig(
            num_classes=num_classes,
            reg_max=reg_max,
            width_mult=width_mult,
            depth_mult=depth_mult,
            coord_fusion_mode=coord_fusion_mode,
            coord_attention=coord_attention,
            feature_attention=feature_attention,
            feature_attention_scope=feature_attention_scope,
            qa_head=qa_head,
            use_p2=use_p2,
            neck_type=neck_type,
            neck_channels=neck_channels,
            bifpn_repeats=bifpn_repeats,
            neck_lite=neck_lite,
            directional_reg_head=directional_reg_head,
            directional_shallow_p2p3=directional_shallow_p2p3,
        )
        self.coord_fusion_mode = coord_fusion_mode
        self.coord_attention = coord_attention
        self.feature_attention = str(feature_attention)
        self.feature_attention_scope = str(feature_attention_scope)
        self.qa_head = bool(qa_head)
        self.use_p2 = bool(use_p2)
        self.neck_type = str(neck_type)
        self.directional_reg_head = bool(directional_reg_head)
        self.directional_shallow_p2p3 = bool(directional_shallow_p2p3)
        self.output_strides_zyx = (4, 8, 16, 32) if self.use_p2 else (8, 16, 32)

        c1 = _scale_channels(32, width_mult)
        c2 = _scale_channels(64, width_mult)
        c3 = _scale_channels(128, width_mult)
        c4 = _scale_channels(256, width_mult)
        c5 = _scale_channels(384, width_mult)

        n1 = _scale_depth(2, depth_mult)
        n2 = _scale_depth(2, depth_mult)
        n3 = _scale_depth(2, depth_mult)
        n4 = _scale_depth(2, depth_mult)

        # Backbone
        # - none/input_only: stem 直接吃完整输入通道
        # - multi_scale: stem 只吃强度图，坐标图在更深层逐级拼接
        stem_in_channels = 1 if self.coord_fusion_mode == "multi_scale" else in_channels
        fusion_attention = self.coord_attention if self.coord_fusion_mode == "multi_scale" else "none"
        neck_p4_attention = self._resolve_feature_attention("neck_p4")
        neck_p3_attention = self._resolve_feature_attention("neck_p3")
        neck_p2_attention = self._resolve_feature_attention("neck_p2")
        pan_p3_attention = self._resolve_feature_attention("pan_p3")
        pan_p4_attention = self._resolve_feature_attention("pan_p4")
        pan_p5_attention = self._resolve_feature_attention("pan_p5")
        stage2_in_channels = c2
        stage3_in_channels = c3 + 3 if self.coord_fusion_mode == "multi_scale" else c3
        stage4_in_channels = c4 + 3 if self.coord_fusion_mode == "multi_scale" else c4
        stage5_in_channels = c5 + 3 if self.coord_fusion_mode == "multi_scale" else c5

        shallow_spatial_conv = "directional" if self.directional_shallow_p2p3 else "standard"

        self.stem = ConvBNAct3D(stem_in_channels, c1, k=3, s=2)  # [B, c1, D/2,  H/2,  W/2]
        self.stage2_down = ConvBNAct3D(c1, c2, k=3, s=2)  # [B, c2, D/4,  H/4,  W/4]
        self.stage2 = C3k2_3D(stage2_in_channels, c2, n=n1, spatial_conv=shallow_spatial_conv)
        self.stage3_down = ConvBNAct3D(c2, c3, k=3, s=2)  # [B, c3, D/8,  H/8,  W/8]
        self.stage3 = C3k2_3D(
            stage3_in_channels,
            c3,
            n=n2,
            attention=fusion_attention,
            spatial_conv=shallow_spatial_conv,
        )
        self.stage4_down = ConvBNAct3D(c3, c4, k=3, s=2)  # [B, c4, D/16, H/16, W/16]
        self.stage4 = C3k2_3D(stage4_in_channels, c4, n=n3, attention=fusion_attention)
        self.stage5_down = ConvBNAct3D(c4, c5, k=3, s=2)  # [B, c5, D/32, H/32, W/32]
        self.stage5 = C3k2_3D(stage5_in_channels, c5, n=n4, attention=fusion_attention)
        self.sppf = SPPF3D(c5, c5)

        if self.neck_type == "pan":
            self.up1 = Upsample3D(scale_factor=2)
            self.neck_p4 = C3k2_3D(c5 + c4, c4, n=n2, shortcut=False, attention=neck_p4_attention)
            self.up2 = Upsample3D(scale_factor=2)
            self.neck_p3 = C3k2_3D(
                c4 + c3,
                c3,
                n=n2,
                shortcut=False,
                attention=neck_p3_attention,
                spatial_conv=shallow_spatial_conv,
            )

            if self.use_p2:
                self.up3 = Upsample3D(scale_factor=2)
                self.neck_p2 = C3k2_3D(
                    c3 + c2,
                    c2,
                    n=n2,
                    shortcut=False,
                    attention=neck_p2_attention,
                    spatial_conv=shallow_spatial_conv,
                )
                self.down_p3 = ConvBNAct3D(c2, c2, k=3, s=2)
                self.pan_p3 = C3k2_3D(
                    c2 + c3,
                    c3,
                    n=n2,
                    shortcut=False,
                    attention=pan_p3_attention,
                    spatial_conv=shallow_spatial_conv,
                )
            else:
                self.up3 = None
                self.neck_p2 = None
                self.down_p3 = None
                self.pan_p3 = None

            self.down_p4 = ConvBNAct3D(c3, c3, k=3, s=2)
            self.pan_p4 = C3k2_3D(c3 + c4, c4, n=n2, shortcut=False, attention=pan_p4_attention)
            self.down_p5 = ConvBNAct3D(c4, c4, k=3, s=2)
            self.pan_p5 = C3k2_3D(c4 + c5, c5, n=n2, shortcut=False, attention=pan_p5_attention)

            detect_in_channels = [c2, c3, c4, c5] if self.use_p2 else [c3, c4, c5]
            self.bifpn = None
        else:
            neck_out_channels = int(neck_channels) if neck_channels is not None else c3
            bifpn_in_channels = [c2, c3, c4, c5] if self.use_p2 else [c3, c4, c5]
            self.bifpn = BiFPN3D(
                in_channels=bifpn_in_channels,
                channels=neck_out_channels,
                repeats=int(bifpn_repeats),
                use_p2=self.use_p2,
                lite=bool(neck_lite),
            )
            detect_in_channels = [neck_out_channels] * len(bifpn_in_channels)

        self.detect = DetectionHead3D(
            detect_in_channels,
            nc=num_classes,
            reg_max=reg_max,
            qa_head=self.qa_head,
            reg_conv_type=("directional" if self.directional_reg_head else "standard"),
        )

    def _resolve_feature_attention(self, target: str) -> str:
        if self.feature_attention == "none" or self.feature_attention_scope == "none":
            return "none"
        if self.feature_attention_scope == "neck_p2":
            return self.feature_attention if target == "neck_p2" else "none"
        if self.feature_attention_scope == "neck_p2_p3":
            return self.feature_attention if target in ("neck_p2", "pan_p3") else "none"
        if self.feature_attention_scope == "shallow_pan":
            return self.feature_attention if target in ("neck_p3", "neck_p2", "pan_p3") else "none"
        if self.feature_attention_scope == "all_pan":
            return (
                self.feature_attention
                if target in ("neck_p4", "neck_p3", "neck_p2", "pan_p3", "pan_p4", "pan_p5")
                else "none"
            )
        return "none"

    @staticmethod
    def _downsample_coords_like(coords_map: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """把坐标图下采样到与特征层完全一致的空间尺寸。"""
        coords_map = generate_downsampled_coords_map(coords_map)
        if tuple(coords_map.shape[2:]) != tuple(feat.shape[2:]):
            coords_map = F.interpolate(coords_map, size=feat.shape[2:], mode="trilinear", align_corners=True)
        return coords_map

    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向 Backbone，显式返回 `p2/p3/p4/p5`。

        尺寸约定：
          - 输入 `x`: [B, C, D, H, W]
          - 输出 `p2`: [B, C2, D/4,  H/4,  W/4 ]
          - 输出 `p3`: [B, C3, D/8,  H/8,  W/8 ]
          - 输出 `p4`: [B, C4, D/16, H/16, W/16]
          - 输出 `p5`: [B, C5, D/32, H/32, W/32]
        """
        if self.coord_fusion_mode == "multi_scale":
            if x.shape[1] < 4:
                raise ValueError(
                    "coord_fusion_mode='multi_scale' expects input channels >=4 "
                    f"(intensity+3 coords), got {int(x.shape[1])}."
                )
            img = x[:, :1, ...]
            coords_map = x[:, 1:4, ...]

            x = self.stem(img)
            x = self.stage2_down(x)
            coords_map = self._downsample_coords_like(coords_map, x)
            p2 = self.stage2(x)

            x = self.stage3_down(p2)
            coords_map = self._downsample_coords_like(coords_map, x)
            p3 = self.stage3(torch.cat([x, coords_map], dim=1))

            x = self.stage4_down(p3)
            coords_map = self._downsample_coords_like(coords_map, x)
            p4 = self.stage4(torch.cat([x, coords_map], dim=1))

            x = self.stage5_down(p4)
            coords_map = self._downsample_coords_like(coords_map, x)
            p5 = self.sppf(self.stage5(torch.cat([x, coords_map], dim=1)))
            return p2, p3, p4, p5

        x = self.stem(x)
        p2 = self.stage2(self.stage2_down(x))
        p3 = self.stage3(self.stage3_down(p2))
        p4 = self.stage4(self.stage4_down(p3))
        p5 = self.sppf(self.stage5(self.stage5_down(p4)))
        return p2, p3, p4, p5

    def forward_neck(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Neck 前向。

        - `PAN` 路径保留原有 concat + C3 逻辑，并在 `use_p2=True` 时扩展到四层输出
        - `BiFPN` 路径把不同尺度先统一通道，再使用 weighted sum 做双向融合
        """
        if self.neck_type == "bifpn":
            feats = [p2, p3, p4, p5] if self.use_p2 else [p3, p4, p5]
            return self.bifpn(feats)

        p5_up = self.up1(p5)
        n4 = self.neck_p4(torch.cat([p5_up, p4], dim=1))

        n4_up = self.up2(n4)
        n3 = self.neck_p3(torch.cat([n4_up, p3], dim=1))

        if self.use_p2:
            n3_up = self.up3(n3)
            # `n2`: [B, C2, D/4, H/4, W/4]
            # 这是新增的小目标检测层，用于承接更高分辨率的细节信息。
            n2 = self.neck_p2(torch.cat([n3_up, p2], dim=1))

            n2_down = self.down_p3(n2)
            n3_out = self.pan_p3(torch.cat([n2_down, n3], dim=1))

            n3_down = self.down_p4(n3_out)
            n4_out = self.pan_p4(torch.cat([n3_down, n4], dim=1))
            n4_down = self.down_p5(n4_out)
            n5_out = self.pan_p5(torch.cat([n4_down, p5], dim=1))
            return [n2, n3_out, n4_out, n5_out]

        n3_down = self.down_p4(n3)
        n4_out = self.pan_p4(torch.cat([n3_down, n4], dim=1))

        n4_down = self.down_p5(n4_out)
        n5_out = self.pan_p5(torch.cat([n4_down, p5], dim=1))
        return [n3, n4_out, n5_out]

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        p2, p3, p4, p5 = self.forward_backbone(x)
        feats = self.forward_neck(p2, p3, p4, p5)
        return self.detect(feats)
