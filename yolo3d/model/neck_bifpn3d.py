from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from yolo3d.model.layers import ConvBNAct3D, Upsample3D


class WeightedFuse3D(nn.Module):
    """3D 加权融合模块。

    这里的输入是多个已经对齐到同一空间尺寸、同一通道数的 3D 特征图：
      - 每个输入张量形状都为 [B, C, D, H, W]
      - 通过可学习标量权重学习“这一层更该信哪一路特征”

    设计动机：
      - 原始 PAN 使用 concat，会直接把通道数叠加起来，显存开销更大
      - 这里改成 weighted sum，可以在保持信息交互的同时把通道数固定住
      - 使用 ReLU 保证权重非负，再归一化为和为 1，训练更稳定
    """

    def __init__(self, n_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        if int(n_inputs) <= 0:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        self.weights = nn.Parameter(torch.ones(int(n_inputs), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(xs) != int(self.weights.numel()):
            raise ValueError(
                f"WeightedFuse3D expects {int(self.weights.numel())} inputs, got {len(xs)}"
            )
        if len(xs) == 0:
            raise ValueError("WeightedFuse3D received an empty input list")

        w = torch.relu(self.weights)
        w = w / (w.sum() + self.eps)

        # 所有特征图已经在调用前对齐到同一 [B, C, D, H, W]，
        # 因此这里可以逐项做加权求和。
        out = xs[0] * w[0].to(dtype=xs[0].dtype)
        for wi, xi in zip(w[1:], xs[1:]):
            out = out + xi * wi.to(dtype=xi.dtype)
        return out


class BiFPNConvBlock3D(nn.Module):
    """BiFPN 中每个融合节点后的轻量卷积块。

    输入/输出张量尺寸：
      - 输入:  [B, C, D, H, W]
      - 输出:  [B, C, D, H, W]

    `lite=False` 时使用普通 3x3x3 卷积；
    `lite=True` 时使用 depthwise 3D 卷积 + pointwise 1x1x1 卷积，
    在 3D 场景下可以显著减小参数量与显存占用。
    """

    def __init__(self, channels: int, lite: bool = False) -> None:
        super().__init__()
        if lite:
            self.block = nn.Sequential(
                ConvBNAct3D(channels, channels, k=3, s=1, g=channels),
                ConvBNAct3D(channels, channels, k=1, s=1),
            )
        else:
            self.block = ConvBNAct3D(channels, channels, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BiFPNBlock3D(nn.Module):
    """单个 BiFPN 重复块。

    这里实现的是 3D 版的双向多尺度融合，支持两种层级集合：
      - `use_p2=False`: [P3, P4, P5]
      - `use_p2=True`:  [P2, P3, P4, P5]

    每一层特征在进入该模块前已经通过 1x1x1 卷积统一为相同通道数，
    因此加权融合时只需要保证空间尺寸一致即可。
    """

    def __init__(self, channels: int, use_p2: bool = False, lite: bool = False) -> None:
        super().__init__()
        self.use_p2 = bool(use_p2)
        self.up = Upsample3D(scale_factor=2)

        self.fuse_td4 = WeightedFuse3D(2)
        self.fuse_td3 = WeightedFuse3D(2)
        self.post_td4 = BiFPNConvBlock3D(channels, lite=lite)
        self.post_td3 = BiFPNConvBlock3D(channels, lite=lite)

        if self.use_p2:
            self.fuse_td2 = WeightedFuse3D(2)
            self.post_td2 = BiFPNConvBlock3D(channels, lite=lite)
            self.down_from_p2 = ConvBNAct3D(channels, channels, k=3, s=2)
            self.fuse_out3 = WeightedFuse3D(3)
        else:
            self.fuse_out3 = WeightedFuse3D(2)

        self.down_from_p3 = ConvBNAct3D(channels, channels, k=3, s=2)
        self.down_from_p4 = ConvBNAct3D(channels, channels, k=3, s=2)
        self.fuse_out4 = WeightedFuse3D(3)
        self.fuse_out5 = WeightedFuse3D(2)

        self.post_out3 = BiFPNConvBlock3D(channels, lite=lite)
        self.post_out4 = BiFPNConvBlock3D(channels, lite=lite)
        self.post_out5 = BiFPNConvBlock3D(channels, lite=lite)

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.use_p2:
            if len(feats) != 4:
                raise ValueError(f"BiFPNBlock3D(use_p2=True) expects 4 inputs, got {len(feats)}")
            p2, p3, p4, p5 = feats
        else:
            if len(feats) != 3:
                raise ValueError(f"BiFPNBlock3D(use_p2=False) expects 3 inputs, got {len(feats)}")
            p3, p4, p5 = feats
            p2 = None

        # 自顶向下路径：
        #   P5 -> P4 -> P3 -> (P2)
        # 逐层把高语义、低分辨率信息传播到更高分辨率层。
        td4 = self.post_td4(self.fuse_td4([p4, self.up(p5)]))
        td3 = self.post_td3(self.fuse_td3([p3, self.up(td4)]))

        if self.use_p2 and p2 is not None:
            td2 = self.post_td2(self.fuse_td2([p2, self.up(td3)]))
            out3 = self.post_out3(self.fuse_out3([p3, td3, self.down_from_p2(td2)]))
        else:
            td2 = None
            out3 = self.post_out3(self.fuse_out3([p3, td3]))

        # 自底向上路径：
        #   (P2) -> P3 -> P4 -> P5
        # 把高分辨率层的细节再反馈回更深层，形成双向融合闭环。
        out4 = self.post_out4(self.fuse_out4([p4, td4, self.down_from_p3(out3)]))
        out5 = self.post_out5(self.fuse_out5([p5, self.down_from_p4(out4)]))

        if self.use_p2 and td2 is not None:
            return [td2, out3, out4, out5]
        return [out3, out4, out5]


class BiFPN3D(nn.Module):
    """轻量 3D BiFPN Neck。

    参数说明：
      - `in_channels`: 各输入层的通道数列表
      - `channels`:    统一后的 Neck 通道数
      - `repeats`:     BiFPN 重复次数
      - `use_p2`:      是否接入 P2 层
      - `lite`:        是否启用 depthwise-separable 轻量卷积

    输入特征张量尺寸示例：
      - use_p2=False: [P3, P4, P5]
      - use_p2=True:  [P2, P3, P4, P5]
    其中每个张量都为 [B, C_i, D_i, H_i, W_i]。
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        repeats: int = 1,
        use_p2: bool = False,
        lite: bool = False,
    ) -> None:
        super().__init__()
        self.use_p2 = bool(use_p2)
        expected_inputs = 4 if self.use_p2 else 3
        if len(in_channels) != expected_inputs:
            raise ValueError(
                f"BiFPN3D expects {expected_inputs} input channels, got {len(in_channels)}"
            )

        self.input_proj = nn.ModuleList(
            [ConvBNAct3D(int(cin), int(channels), k=1, s=1) for cin in in_channels]
        )
        self.blocks = nn.ModuleList(
            [BiFPNBlock3D(int(channels), use_p2=self.use_p2, lite=lite) for _ in range(max(1, int(repeats)))]
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != len(self.input_proj):
            raise ValueError(f"BiFPN3D expects {len(self.input_proj)} inputs, got {len(feats)}")

        # 先用 1x1x1 卷积把不同层的通道数统一成同一宽度，
        # 这样后续每一个加权求和节点都能直接逐元素融合。
        outs = [proj(x) for proj, x in zip(self.input_proj, feats)]
        for block in self.blocks:
            outs = block(outs)
        return list(outs)
