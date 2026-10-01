"""脏矩形局部区域检测与处理。

当画面仅有一部分面积发生变化时，提取发生变化的矩形区域 (x, y, w, h)。
在下一帧中，编码端仅压缩并保存该区域的画面数据，解码端直接覆盖到对应坐标。
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


def find_dirty_bbox(
    canvas: np.ndarray, frame: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """计算画面变化的最小外接矩形 (x, y, w, h)。

    若画面完全一致，返回 None。
    """
    if canvas.shape != frame.shape:
        raise ValueError("canvas 与 frame 形状不一致")

    diff = np.any(canvas != frame, axis=2)
    if not diff.any():
        return None

    rows = np.any(diff, axis=1)
    cols = np.any(diff, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    x = int(xmin)
    y = int(ymin)
    w = int(xmax - xmin + 1)
    h = int(ymax - ymin + 1)
    return x, y, w, h


def compute_change_ratio(canvas: np.ndarray, frame: np.ndarray) -> float:
    """计算变化像素占总像素的百分比 (0.0 ~ 1.0)。"""
    diff = np.any(canvas != frame, axis=2)
    return float(np.count_nonzero(diff)) / float(diff.size)
