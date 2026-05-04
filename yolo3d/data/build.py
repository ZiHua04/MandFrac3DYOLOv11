from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from yolo3d.data.collate import detection_collate_fn
from yolo3d.data.dataset import CT3DDetectionDataset, CTIntensityNorm
from yolo3d.data.transforms import RandomCropAroundBoxes3D, RandomFlip3D, RandomIntensityJitterCT


def build_detection_dataloader(
    json_path: str | Path,
    split_key: str = "training",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    num_samples_per_epoch: Optional[int] = None,
    patch_size_zyx: Optional[Sequence[int]] = (96, 96, 96),
    positive_crop_prob: float = 0.75,
    background_crop_prob: float = 0.0,
    background_margin_zyx: Sequence[int] = (0, 0, 0),
    background_max_tries: int = 50,
    intensity_norm_mode: str = "ct_window",
    augment: bool = True,
    base_seed: int = 0,
    add_coords_channels: bool = False,
) -> DataLoader:
    patch_crop = None
    if patch_size_zyx is not None:
        patch_crop = RandomCropAroundBoxes3D(
            patch_size_zyx=tuple(int(v) for v in patch_size_zyx),
            positive_crop_prob=float(positive_crop_prob),
            background_crop_prob=float(background_crop_prob),
            background_margin_zyx=tuple(int(v) for v in background_margin_zyx),
            background_max_tries=int(background_max_tries),
        )

    spatial_aug = RandomFlip3D(pz=0.0, py=0.5, px=0.5) if augment else None
    intensity_aug = RandomIntensityJitterCT() if augment else None

    dataset = CT3DDetectionDataset(
        json_path=json_path,
        split_key=split_key,
        patch_crop=patch_crop,
        intensity_norm=CTIntensityNorm(mode=intensity_norm_mode),
        spatial_aug=spatial_aug,
        intensity_aug=intensity_aug,
        base_seed=base_seed,
        add_coords_channels=add_coords_channels,
    )

    sampler = None
    if num_samples_per_epoch is not None:
        # Useful for tiny datasets / overfit-debug: draw with replacement to get more
        # optimizer steps per epoch even when len(dataset) is very small.
        sampler = RandomSampler(dataset, replacement=True, num_samples=int(num_samples_per_epoch))
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
