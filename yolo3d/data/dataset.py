from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import nibabel as nib

from yolo3d.data.transforms import RandomCropAroundBoxes3D
from yolo3d.utils.box_ops import clip_boxes_zyxzyx, drop_invalid_boxes_zyxzyx
from yolo3d.utils.patch_util import generate_coords_map


def _load_json_list(json_path: str | Path, split_key: str = "training") -> List[Dict[str, Any]]:
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if split_key not in data:
        raise KeyError(f"JSON missing key '{split_key}'. Available keys={list(data.keys())}")
    if not isinstance(data[split_key], list):
        raise TypeError(f"Expected '{split_key}' to be a list, got {type(data[split_key]).__name__}")
    return data[split_key]


@dataclass(frozen=True)
class CTIntensityNorm:
    """Simple per-patch normalization (kept minimal for Phase 1)."""

    mode: str = "none"  # "none" | "zscore" | "ct_window"
    ct_window: Tuple[float, float] = (-1000.0, 2000.0)  # (low, high)

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32, copy=False)
        if self.mode == "none":
            return vol
        if self.mode == "zscore":
            m = float(vol.mean())
            s = float(vol.std())
            s = s if s > 1e-6 else 1.0
            return (vol - m) / s
        if self.mode == "ct_window":
            lo, hi = self.ct_window
            vol = np.clip(vol, lo, hi)
            # scale to roughly [-1, 1]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 if (hi - lo) != 0 else 1.0
            return (vol - mid) / half
        raise ValueError(f"Unknown norm mode: {self.mode}")


class CT3DDetectionDataset(Dataset):
    """3D detection dataset for mandibular fracture CT patches.

    Expected JSON item format:
      {
        "image": ".../reset_image.nii.gz",
        "boxes": [[z1,y1,x1,z2,y2,x2], ...],
        "labels": [0, ...]
      }

    Returns:
      image: FloatTensor [C, D, H, W], where C=1 (intensity only) or C=4 (intensity + z/y/x coords)
      target: dict
        - boxes_zyxzyx: FloatTensor [N, 6]
        - labels: LongTensor [N]
        - image_path: str
        - crop_origin_zyx: IntTensor [3] (only when patching enabled)
    """

    def __init__(
        self,
        json_path: str | Path,
        split_key: str = "training",
        patch_crop: Optional[RandomCropAroundBoxes3D] = None,
        intensity_norm: Optional[CTIntensityNorm] = None,
        spatial_aug: Optional[Any] = None,
        intensity_aug: Optional[Any] = None,
        base_seed: int = 0,
        strict_paths: bool = True,
        add_coords_channels: bool = False,
    ) -> None:
        self.items = _load_json_list(json_path, split_key=split_key)
        self.patch_crop = patch_crop
        self.intensity_norm = intensity_norm or CTIntensityNorm(mode="none")
        self.spatial_aug = spatial_aug
        self.intensity_aug = intensity_aug
        self.base_seed = int(base_seed)
        self.epoch = 0
        self._sample_call_count = 0
        self.strict_paths = strict_paths
        self.add_coords_channels = bool(add_coords_channels)
        self.num_image_channels = 4 if self.add_coords_channels else 1

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """Allow external training loop to advance RNG deterministically."""
        self.epoch = int(epoch)
        self._sample_call_count = 0

    @staticmethod
    def _sanitize_boxes_against_volume(
        boxes_zyxzyx: np.ndarray,
        labels: np.ndarray,
        volume_shape_zyx: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if boxes_zyxzyx.size == 0:
            return boxes_zyxzyx, labels
        clipped = clip_boxes_zyxzyx(boxes_zyxzyx.astype(np.float32, copy=False), volume_shape_zyx)
        clipped, labels = drop_invalid_boxes_zyxzyx(clipped, labels, min_size_zyx=(1.0, 1.0, 1.0))
        return clipped, labels

    def _load_volume_zyx(self, image_path: str) -> np.ndarray:
        p = Path(image_path)
        if self.strict_paths and not p.exists():
            raise FileNotFoundError(f"NIfTI not found: {image_path}")
        img = nib.load(str(p))
        # IMPORTANT:
        # - Project convention for both volumes and boxes is voxel order [z, y, x] (D, H, W).
        # - nibabel returns the raw NIfTI data array in stored axis order, which for typical
        #   medical CT NIfTI is [x, y, z]. (SimpleITK's GetArrayFromImage is [z, y, x].)
        # If boxes in JSON were produced in [z,y,x], we must transpose nibabel's output.
        vol_xyz = np.asarray(img.dataobj)
        vol = np.transpose(vol_xyz, (2, 1, 0))  # [x,y,z] -> [z,y,x]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got shape={vol.shape} from {image_path}")
        return vol

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        item = self.items[idx]
        image_path = item["image"]
        boxes = np.asarray(item.get("boxes", []), dtype=np.float32)
        labels = np.asarray(item.get("labels", []), dtype=np.int64)

        if len(boxes) != len(labels):
            raise ValueError(f"boxes/labels length mismatch at idx={idx}: {len(boxes)} vs {len(labels)}")
        if boxes.size > 0 and boxes.shape[-1] != 6:
            raise ValueError(f"Expected boxes shape [N,6], got {boxes.shape} at idx={idx}")

        vol = self._load_volume_zyx(image_path)
        full_volume_shape_zyx = tuple(int(v) for v in vol.shape)
        boxes, labels = self._sanitize_boxes_against_volume(boxes, labels, full_volume_shape_zyx)

        crop_origin_zyx = None
        # Include a per-process call counter so repeated sampling of the same idx within
        # one epoch still gets different random crops / augmentations.
        call_id = self._sample_call_count
        self._sample_call_count += 1
        rng = np.random.default_rng(seed=self.base_seed + idx + 1_000_003 * self.epoch + 9_176 * call_id)
        if self.patch_crop is not None:
            vol, boxes, labels, crop_origin_zyx = self.patch_crop(vol, boxes, labels, rng=rng)

        coords_map_zyx: Optional[np.ndarray] = None
        if self.add_coords_channels:
            patch_origin_zyx = (0, 0, 0) if crop_origin_zyx is None else tuple(int(v) for v in crop_origin_zyx.tolist())
            patch_size_zyx = tuple(int(v) for v in vol.shape)
            coords_map_zyx = generate_coords_map(
                patch_coords=patch_origin_zyx,
                image_size=full_volume_shape_zyx,
                patch_size=patch_size_zyx,
                device=None,
                dtype=torch.float32,
            ).cpu().numpy()
            # When crop gets padded (e.g. tiny depth volume), padded voxels should
            # not produce out-of-range normalized coordinates.
            max_z = float(max(0, full_volume_shape_zyx[0] - 1)) / float(max(1, full_volume_shape_zyx[0]))
            max_y = float(max(0, full_volume_shape_zyx[1] - 1)) / float(max(1, full_volume_shape_zyx[1]))
            max_x = float(max(0, full_volume_shape_zyx[2] - 1)) / float(max(1, full_volume_shape_zyx[2]))
            coords_map_zyx[0] = np.clip(coords_map_zyx[0], 0.0, max_z)
            coords_map_zyx[1] = np.clip(coords_map_zyx[1], 0.0, max_y)
            coords_map_zyx[2] = np.clip(coords_map_zyx[2], 0.0, max_x)

        if self.spatial_aug is not None and boxes is not None:
            if coords_map_zyx is None:
                vol, boxes = self.spatial_aug(vol, boxes, rng=rng)
            else:
                aug_out = self.spatial_aug(vol, boxes, rng=rng, coords_map_zyx=coords_map_zyx)
                if not isinstance(aug_out, tuple):
                    raise TypeError("spatial_aug must return a tuple when coords_map_zyx is provided")
                if len(aug_out) == 3:
                    vol, boxes, coords_map_zyx = aug_out
                else:
                    raise ValueError(
                        "When coords_map_zyx is provided, spatial_aug must return "
                        "(volume_zyx, boxes_zyxzyx, coords_map_zyx)."
                    )

        if self.intensity_aug is not None:
            vol = self.intensity_aug(vol, rng=rng)

        vol = self.intensity_norm(vol)

        vol = np.ascontiguousarray(vol.astype(np.float32, copy=False))
        x = torch.from_numpy(vol)[None, ...]  # [1, D, H, W]
        if coords_map_zyx is not None:
            coords_map_zyx = np.ascontiguousarray(coords_map_zyx.astype(np.float32, copy=False))
            x = torch.cat([x, torch.from_numpy(coords_map_zyx)], dim=0)  # [4, D, H, W]

        target: Dict[str, Any] = {
            "boxes_zyxzyx": torch.from_numpy(boxes.astype(np.float32, copy=False)),
            "labels": torch.from_numpy(labels.astype(np.int64, copy=False)),
            "image_path": str(image_path),
        }
        if crop_origin_zyx is not None:
            target["crop_origin_zyx"] = torch.from_numpy(crop_origin_zyx.astype(np.int32, copy=False))
        return x, target
