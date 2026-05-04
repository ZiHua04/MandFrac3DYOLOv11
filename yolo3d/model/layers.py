from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int | Tuple[int, int, int], p: Optional[int | Tuple[int, int, int]] = None):
    if p is not None:
        return p
    if isinstance(k, int):
        return k // 2
    return tuple(kk // 2 for kk in k)


def _split_channels(total: int, parts: int) -> Tuple[int, ...]:
    base = int(total) // int(parts)
    rem = int(total) % int(parts)
    return tuple(base + (1 if i < rem else 0) for i in range(int(parts)))


class ConvBNAct3D(nn.Module):
    """Conv3d + BatchNorm3d + SiLU.

    Input/Output shape: [B, C, D, H, W]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | Tuple[int, int, int] = 3,
        s: int | Tuple[int, int, int] = 1,
        p: Optional[int | Tuple[int, int, int]] = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DirectionalConvBNAct3D(nn.Module):
    """Tri-plane anisotropic conv for thin / elongated 3D structures.

    The block keeps the same input/output contract as `ConvBNAct3D`, but
    replaces a single isotropic 3D kernel with three plane-aware branches:
    `1x3x3`, `3x1x3`, and `3x3x1`, followed by a `1x1x1` fusion.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        s: int | Tuple[int, int, int] = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        splits = _split_channels(int(c2), parts=3)
        kernels = ((1, 3, 3), (3, 1, 3), (3, 3, 1))
        self.branches = nn.ModuleList(
            [ConvBNAct3D(c1, split_c, k=kernel, s=s, act=True) for split_c, kernel in zip(splits, kernels)]
        )
        self.fuse = ConvBNAct3D(sum(splits), c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(branch_feats, dim=1))


def make_spatial_conv3d(
    c1: int,
    c2: int,
    *,
    kind: str = "standard",
    s: int | Tuple[int, int, int] = 1,
    act: bool = True,
) -> nn.Module:
    if kind == "standard":
        return ConvBNAct3D(c1, c2, k=3, s=s, act=act)
    if kind == "directional":
        return DirectionalConvBNAct3D(c1, c2, s=s, act=act)
    raise ValueError(f"Unsupported spatial conv kind='{kind}'. Expected 'standard' or 'directional'.")


class Bottleneck3D(nn.Module):
    """Standard bottleneck with optional shortcut."""

    def __init__(self, c: int, shortcut: bool = True, e: float = 0.5, spatial_conv: str = "standard") -> None:
        super().__init__()
        c_ = max(1, int(c * e))
        self.cv1 = ConvBNAct3D(c, c_, k=1, s=1)
        self.cv2 = make_spatial_conv3d(c_, c, kind=spatial_conv, s=1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class SE3D(nn.Module):
    """Squeeze-and-Excitation for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc2(self.act(self.fc1(w)))
        return x * self.gate(w)


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sigmoid()

        if spatial_kernel_size not in (3, 7):
            raise ValueError("CBAM3D spatial_kernel_size must be 3 or 7.")
        spatial_padding = spatial_kernel_size // 2
        self.spatial = nn.Conv3d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_padding, bias=False)
        self.spatial_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: pooled descriptors -> shared MLP -> sigmoid gate
        ch_attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        x = x * self.channel_gate(ch_attn)

        # Spatial attention: aggregate over channels then predict a spatial mask
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * self.spatial_gate(sp_attn)


class C3k2_3D(nn.Module):
    """CSP-style block approximating YOLOv11 C3k2 in 3D.

    This is a pragmatic 3D translation:
      - split -> bottleneck stack -> concat -> fuse
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 2,
        shortcut: bool = True,
        e: float = 0.5,
        attention: str = "none",
        spatial_conv: str = "standard",
        se_reduction: int = 16,
        cbam_spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if attention not in ("none", "se", "cbam"):
            raise ValueError(f"Unsupported attention='{attention}'. Expected one of: 'none', 'se', 'cbam'.")
        c_ = max(1, int(c2 * e))
        self.cv1 = ConvBNAct3D(c1, c_, k=1, s=1)
        self.cv2 = ConvBNAct3D(c1, c_, k=1, s=1)
        self.m = nn.Sequential(
            *[Bottleneck3D(c_, shortcut=shortcut, e=1.0, spatial_conv=spatial_conv) for _ in range(n)]
        )
        self.cv3 = ConvBNAct3D(2 * c_, c2, k=1, s=1)
        if attention == "se":
            self.attn = SE3D(c2, reduction=se_reduction)
        elif attention == "cbam":
            self.attn = CBAM3D(c2, reduction=se_reduction, spatial_kernel_size=cbam_spatial_kernel_size)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attn(x)


class SPPF3D(nn.Module):
    """3D SPPF: maxpool k=5 repeated 3 times then concat."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = max(1, c1 // 2)
        self.cv1 = ConvBNAct3D(c1, c_, k=1, s=1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNAct3D(c_ * 4, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class Upsample3D(nn.Module):
    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
