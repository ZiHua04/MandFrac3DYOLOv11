from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collate for variable-length 3D boxes.

    Returns:
      images: FloatTensor [B, C, D, H, W]
      targets: list of dict, len=B. Each dict contains:
        - boxes_zyxzyx: FloatTensor [Ni, 6]
        - labels: LongTensor [Ni]
        - ... (optional debug keys)
    """
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = [x[1] for x in batch]
    return images, targets

