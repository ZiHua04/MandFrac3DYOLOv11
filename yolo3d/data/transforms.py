from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from yolo3d.utils.box_ops import clip_boxes_zyxzyx, drop_invalid_boxes_zyxzyx


def pad_volume_zyx_to_size(
    volume_zyx: np.ndarray,
    target_size_zyx: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad a [D,H,W] volume to target size using tail-side constant padding."""
    if volume_zyx.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got shape={volume_zyx.shape}")
    td, th, tw = [int(v) for v in target_size_zyx]
    d, h, w = [int(v) for v in volume_zyx.shape]
    if td <= 0 or th <= 0 or tw <= 0:
        raise ValueError(f"target_size_zyx must be positive, got {target_size_zyx}")

    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume_zyx

    return np.pad(
        volume_zyx,
        pad_width=((0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=float(pad_value),
    )


@dataclass(frozen=True)
class RandomFlip3D:
    """Randomly flip a 3D patch along z/y/x axes and update boxes.

    volume: [D,H,W] (z,y,x)
    boxes: [N,6] in [z1,y1,x1,z2,y2,x2]
    """

    pz: float = 0.0
    py: float = 0.5
    px: float = 0.5

    def __call__(
        self,
        volume_zyx: np.ndarray,
        boxes_zyxzyx: np.ndarray,
        coords_map_zyx: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        d, h, w = volume_zyx.shape
        vol = volume_zyx
        boxes = boxes_zyxzyx.astype(np.float32, copy=True) if boxes_zyxzyx is not None else boxes_zyxzyx
        coords = coords_map_zyx

        if boxes is None:
            boxes = np.zeros((0, 6), dtype=np.float32)
        if coords is not None:
            if coords.ndim != 4 or coords.shape[0] != 3:
                raise ValueError(f"Expected coords_map_zyx shape [3,D,H,W], got {coords.shape}")
            if tuple(coords.shape[1:]) != tuple(vol.shape):
                raise ValueError(
                    f"coords_map_zyx spatial shape {coords.shape[1:]} does not match volume shape {vol.shape}"
                )

        if self.pz > 0 and float(rng.random()) < self.pz:
            vol = vol[::-1, :, :]
            if coords is not None:
                coords = coords[:, ::-1, :, :]
            if len(boxes):
                z1 = boxes[:, 0].copy()
                z2 = boxes[:, 3].copy()
                boxes[:, 0] = (d - z2)
                boxes[:, 3] = (d - z1)

        if self.py > 0 and float(rng.random()) < self.py:
            vol = vol[:, ::-1, :]
            if coords is not None:
                coords = coords[:, :, ::-1, :]
            if len(boxes):
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 4].copy()
                boxes[:, 1] = (h - y2)
                boxes[:, 4] = (h - y1)

        if self.px > 0 and float(rng.random()) < self.px:
            vol = vol[:, :, ::-1]
            if coords is not None:
                coords = coords[:, :, :, ::-1]
            if len(boxes):
                x1 = boxes[:, 2].copy()
                x2 = boxes[:, 5].copy()
                boxes[:, 2] = (w - x2)
                boxes[:, 5] = (w - x1)

        if coords is None:
            return vol, boxes
        return vol, boxes, coords


@dataclass(frozen=True)
class RandomIntensityJitterCT:
    """Simple CT intensity jitter on the cropped patch.

    Operations:
      - scale (contrast-like)
      - shift (brightness-like)
      - optional gaussian noise
    """

    scale_range: Tuple[float, float] = (0.9, 1.1)
    shift_range: Tuple[float, float] = (-50.0, 50.0)
    noise_std: float = 5.0
    p_noise: float = 0.3

    def __call__(self, volume_zyx: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        v = volume_zyx.astype(np.float32, copy=False)
        s = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        b = float(rng.uniform(self.shift_range[0], self.shift_range[1]))
        v = v * s + b
        if self.p_noise > 0 and float(rng.random()) < self.p_noise and self.noise_std > 0:
            v = v + rng.normal(0.0, float(self.noise_std), size=v.shape).astype(np.float32)
        return v


@dataclass(frozen=True)
class Compose3D:
    """Compose spatial transforms that take (vol, boxes) and return (vol, boxes)."""

    t1: any
    t2: Optional[any] = None
    t3: Optional[any] = None

    def __call__(
        self,
        volume_zyx: np.ndarray,
        boxes_zyxzyx: np.ndarray,
        coords_map_zyx: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if coords_map_zyx is None:
            vol, boxes = self.t1(volume_zyx, boxes_zyxzyx, rng=rng)
            if self.t2 is not None:
                vol, boxes = self.t2(vol, boxes, rng=rng)
            if self.t3 is not None:
                vol, boxes = self.t3(vol, boxes, rng=rng)
            return vol, boxes

        out = self.t1(volume_zyx, boxes_zyxzyx, coords_map_zyx=coords_map_zyx, rng=rng)
        if not isinstance(out, tuple):
            raise TypeError("Compose3D transform must return a tuple")
        if len(out) == 3:
            vol, boxes, coords = out
        else:
            raise ValueError("Compose3D transforms must return (volume_zyx, boxes_zyxzyx, coords_map_zyx)")

        if self.t2 is not None:
            out = self.t2(vol, boxes, coords_map_zyx=coords, rng=rng)
            if not isinstance(out, tuple):
                raise TypeError("Compose3D transform must return a tuple")
            if len(out) == 3:
                vol, boxes, coords = out
            else:
                raise ValueError("Compose3D transforms must return (volume_zyx, boxes_zyxzyx, coords_map_zyx)")
        if self.t3 is not None:
            out = self.t3(vol, boxes, coords_map_zyx=coords, rng=rng)
            if not isinstance(out, tuple):
                raise TypeError("Compose3D transform must return a tuple")
            if len(out) == 3:
                vol, boxes, coords = out
            else:
                raise ValueError("Compose3D transforms must return (volume_zyx, boxes_zyxzyx, coords_map_zyx)")
        return vol, boxes, coords


@dataclass(frozen=True)
class RandomCropAroundBoxes3D:
    """Crop a fixed-size patch around a randomly selected GT box.

    Coordinate convention:
      - volume: numpy array shaped [D, H, W] matching (z, y, x)
      - boxes: numpy array shaped [N, 6] in [z1, y1, x1, z2, y2, x2] voxel coords
    """

    patch_size_zyx: Tuple[int, int, int] = (96, 96, 96)
    # When GT exists, sample a positive crop with this probability.
    positive_crop_prob: float = 0.75
    # When GT exists and positive crop is not selected, optionally sample a *background* crop that is
    # forced to not intersect any GT (with an optional margin). This is important when all scans
    # are "positive volumes" (contain at least one lesion), otherwise random crops can still
    # frequently contain partial lesions and the model never learns clean background.
    background_crop_prob: float = 0.0
    background_margin_zyx: Tuple[int, int, int] = (0, 0, 0)
    background_max_tries: int = 50
    # Maximum random shift applied around chosen box center, in voxels.
    center_jitter_zyx: Tuple[int, int, int] = (12, 12, 12)
    # Drop boxes that become too tiny after cropping (in voxels)
    min_box_size_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def _patch_intersects_any_box(
        origin_zyx: Tuple[int, int, int],
        patch_size_zyx: Tuple[int, int, int],
        boxes_zyxzyx: np.ndarray,
    ) -> bool:
        """Return True if the patch [oz,oz+pd) x [oy,oy+ph) x [ox,ox+pw) intersects any box."""
        oz, oy, ox = origin_zyx
        pd, ph, pw = patch_size_zyx
        if boxes_zyxzyx is None or boxes_zyxzyx.size == 0:
            return False
        z1, y1, x1, z2, y2, x2 = [boxes_zyxzyx[:, i] for i in range(6)]
        inter = (float(oz) < z2) & (float(oz + pd) > z1) & (float(oy) < y2) & (float(oy + ph) > y1) & (float(ox) < x2) & (float(ox + pw) > x1)
        return bool(np.any(inter))

    def __call__(
        self,
        volume_zyx: np.ndarray,
        boxes_zyxzyx: np.ndarray,
        labels: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        if volume_zyx.ndim != 3:
            raise ValueError(f"Expected volume [D,H,W], got shape={volume_zyx.shape}")

        d, h, w = volume_zyx.shape
        target_pd, target_ph, target_pw = [int(v) for v in self.patch_size_zyx]
        pd, ph, pw = target_pd, target_ph, target_pw
        if pd > d or ph > h or pw > w:
            # If the volume is smaller than patch size, crop available region first
            # then pad to target size after cropping.
            pd = min(pd, d)
            ph = min(ph, h)
            pw = min(pw, w)

        boxes_exist = boxes_zyxzyx is not None and len(boxes_zyxzyx) > 0

        # Choose crop mode: positive (around GT) vs forced-background vs random.
        use_positive_crop = boxes_exist and float(rng.random()) < float(self.positive_crop_prob)
        use_background_crop = (
            boxes_exist
            and (not use_positive_crop)
            and float(self.background_crop_prob) > 0.0
            and float(rng.random()) < float(self.background_crop_prob)
        )

        if use_background_crop:
            # Try to find a patch origin that doesn't intersect any GT (optionally with margin).
            margin_z, margin_y, margin_x = [int(v) for v in self.background_margin_zyx]
            if margin_z or margin_y or margin_x:
                expanded = boxes_zyxzyx.astype(np.float32, copy=True)
                expanded[:, 0] -= float(margin_z)
                expanded[:, 3] += float(margin_z)
                expanded[:, 1] -= float(margin_y)
                expanded[:, 4] += float(margin_y)
                expanded[:, 2] -= float(margin_x)
                expanded[:, 5] += float(margin_x)
                expanded = clip_boxes_zyxzyx(expanded, (d, h, w))
            else:
                expanded = boxes_zyxzyx

            max_tries = max(1, int(self.background_max_tries))
            found = False
            oz = oy = ox = 0
            for _ in range(max_tries):
                oz = int(rng.integers(0, max(1, d - pd + 1)))
                oy = int(rng.integers(0, max(1, h - ph + 1)))
                ox = int(rng.integers(0, max(1, w - pw + 1)))
                if not self._patch_intersects_any_box((oz, oy, ox), (pd, ph, pw), expanded):
                    found = True
                    break
            if not found:
                # Fall back to random crop if we couldn't find a clean background patch.
                use_background_crop = False

        if use_positive_crop:
            bi = int(rng.integers(0, len(boxes_zyxzyx)))
            z1, y1, x1, z2, y2, x2 = boxes_zyxzyx[bi].astype(np.float32)
            cz, cy, cx = (z1 + z2) / 2.0, (y1 + y2) / 2.0, (x1 + x2) / 2.0
            jz, jy, jx = self.center_jitter_zyx
            cz += float(rng.integers(-jz, jz + 1)) if jz > 0 else 0.0
            cy += float(rng.integers(-jy, jy + 1)) if jy > 0 else 0.0
            cx += float(rng.integers(-jx, jx + 1)) if jx > 0 else 0.0

            # Convert center to top-left-front (origin) and clamp to image bounds.
            oz = int(round(cz - pd / 2.0))
            oy = int(round(cy - ph / 2.0))
            ox = int(round(cx - pw / 2.0))
            oz = max(0, min(oz, d - pd))
            oy = max(0, min(oy, h - ph))
            ox = max(0, min(ox, w - pw))
        elif not use_background_crop:
            # Pure random crop.
            oz = int(rng.integers(0, max(1, d - pd + 1)))
            oy = int(rng.integers(0, max(1, h - ph + 1)))
            ox = int(rng.integers(0, max(1, w - pw + 1)))

        patch = volume_zyx[oz : oz + pd, oy : oy + ph, ox : ox + pw]
        patch = pad_volume_zyx_to_size(
            patch,
            target_size_zyx=(target_pd, target_ph, target_pw),
            pad_value=0.0,
        )
        crop_origin_zyx = np.array([oz, oy, ox], dtype=np.int32)

        if boxes_zyxzyx is None or len(boxes_zyxzyx) == 0:
            return patch, boxes_zyxzyx, labels, crop_origin_zyx

        # Shift boxes into patch frame and clip to patch extents.
        shifted = boxes_zyxzyx.astype(np.float32).copy()
        shifted[:, [0, 3]] -= float(oz)  # z1,z2
        shifted[:, [1, 4]] -= float(oy)  # y1,y2
        shifted[:, [2, 5]] -= float(ox)  # x1,x2

        shifted = clip_boxes_zyxzyx(shifted, (pd, ph, pw))
        shifted, labels = drop_invalid_boxes_zyxzyx(
            shifted,
            labels,
            min_size_zyx=self.min_box_size_zyx,
        )
        return patch, shifted, labels, crop_origin_zyx
