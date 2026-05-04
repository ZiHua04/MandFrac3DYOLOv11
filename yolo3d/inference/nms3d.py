from __future__ import annotations

from typing import Tuple

import torch

from yolo3d.utils.box3d_torch import boxes_iou3d


@torch.no_grad()
def nms3d(
    boxes_zyxzyx: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.5,
    topk: int | None = None,
) -> torch.Tensor:
    """Axis-aligned 3D NMS.

    Args:
      boxes_zyxzyx: [N,6]
      scores: [N]
    Returns:
      keep indices (LongTensor)
    """
    if boxes_zyxzyx.numel() == 0:
        return torch.zeros((0,), device=boxes_zyxzyx.device, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    if topk is not None:
        order = order[: int(topk)]

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = boxes_iou3d(boxes_zyxzyx[i : i + 1], boxes_zyxzyx[rest]).squeeze(0)
        order = rest[ious <= float(iou_thr)]

    return torch.tensor(keep, device=boxes_zyxzyx.device, dtype=torch.long)

