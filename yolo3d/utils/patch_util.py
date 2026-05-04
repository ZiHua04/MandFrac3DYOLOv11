import torch.nn.functional as F
import torch


def generate_coords_map(patch_coords, image_size, patch_size, device=None, dtype=torch.float32):
    """
    z1, y1, x1: Patch 起始偏移量
    image_size: 图像的尺寸 (d, h, w)
    patch_size: Patch 的尺寸 (pd, ph, pw)
    """
    z1, y1, x1 = patch_coords
    image_d, image_h, image_w = image_size
    pd, ph, pw = patch_size
    # 1. 在原尺度下生成基础坐标网格 (Shape: 3, D, H, W)
    # 注意: torch.meshgrid 使用 indexing='ij' 对应 (z, y, x)
    # 归一化到[0,1]范围
    z_coords = torch.arange(z1, z1 + pd, device=device, dtype=dtype) / float(image_d)
    y_coords = torch.arange(y1, y1 + ph, device=device, dtype=dtype) / float(image_h)
    x_coords = torch.arange(x1, x1 + pw, device=device, dtype=dtype) / float(image_w)

    grid = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    grid = torch.stack(grid, dim=0)  # Shape: (3, pd, ph, pw)
    return grid


def generate_downsampled_coords_map(coords_map):
    """

    :param coords_map: (b, 3, d, h, w) or None
    :return: (b, 3, d/2, h/2, w/2) or None
    """
    if coords_map is None:
        return None

    _, _, pd, ph, pw = coords_map.shape

    scale = 0.5
    # 计算目标尺寸
    target_d = int(pd * scale)
    target_h = int(ph * scale)
    target_w = int(pw * scale)

    # 2. 使用三线性插值下采样
    # align_corners=True 确保边界坐标（起始和结束点）对齐
    downsampled_grid = F.interpolate(
        coords_map,
        size=(target_d, target_h, target_w),
        mode='trilinear',
        align_corners=True
    )
    return downsampled_grid
