"""瓦片网格：划分、边界计算、变更检测。

瓦片按行主序编号：index = row * cols + col。
边缘瓦片按图像实际边界裁剪（不越界、不填充）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def grid_dims(width: int, height: int, tile_size: int) -> Tuple[int, int]:
    """返回 (cols, rows)。"""
    cols = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    return cols, rows


def tile_count(width: int, height: int, tile_size: int) -> int:
    cols, rows = grid_dims(width, height, tile_size)
    return cols * rows


def tile_bounds(index: int, cols: int, tile_size: int,
                width: int, height: int) -> Tuple[int, int, int, int]:
    """返回瓦片的 (x0, y0, x1, y1)，x1/y1 为开边界。"""
    row, col = divmod(index, cols)
    x0 = col * tile_size
    y0 = row * tile_size
    x1 = min(x0 + tile_size, width)
    y1 = min(y0 + tile_size, height)
    return x0, y0, x1, y1


def changed_tiles(canvas: np.ndarray, frame: np.ndarray,
                  cols: int, tile_size: int,
                  width: int, height: int) -> List[int]:
    """返回与当前画布相比发生变化的瓦片索引列表（严格升序）。"""
    diff = np.any(canvas != frame, axis=2)
    out: List[int] = []
    cols_n, rows_n = grid_dims(width, height, tile_size)
    total = cols_n * rows_n
    for idx in range(total):
        x0, y0, x1, y1 = tile_bounds(idx, cols, tile_size, width, height)
        if diff[y0:y1, x0:x1].any():
            out.append(idx)
    return out
