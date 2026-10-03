"""脏矩形局部区域检测与多矩形智能聚类。

当画面仅有一部分面积发生变化时，提取发生变化的矩形区域 (x, y, w, h)。
支持：
1. 单一最小外接矩形检测 (Single Bounding Box)；
2. 多脏矩形智能聚类与分割 (Multi-Dirty-Rect Clustering)：
   针对多处分散变动的场景（例如左上角有通知弹窗，右下角同时有鼠标移动），
   自动分解为多个紧凑的不相交矩形，避免单个大外接框包含过多无变化的空白区域。
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


def find_dirty_bbox(
    canvas: np.ndarray, frame: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """计算画面变化的全局单一最小外接矩形 (x, y, w, h)。

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


def find_dirty_rects(
    canvas: np.ndarray,
    frame: np.ndarray,
    block_size: int = 16,
    max_rects: int = 8,
    area_efficiency_threshold: float = 0.65,
) -> List[Tuple[int, int, int, int]]:
    """智能提取多脏矩形列表 [(x, y, w, h), ...]。

    :param block_size: 网格检测粒度 (默认 16 像素)
    :param max_rects: 允许的最大独立矩形数 (默认最多 8 个，避免包头开销过大)
    :param area_efficiency_threshold: 拆分效率阈值。只有当多矩形总面积小于单矩形面积的此比例时才拆分。
    """
    single_bbox = find_dirty_bbox(canvas, frame)
    if single_bbox is None:
        return []

    x_all, y_all, w_all, h_all = single_bbox
    single_area = w_all * h_all

    # 如果单一矩形本来就很小（例如小于 64x64），直接返回单矩形
    if single_area <= (block_size * 2) * (block_size * 2):
        return [single_bbox]

    height, width = canvas.shape[:2]
    diff = np.any(canvas != frame, axis=2)

    # 1. 构建粗粒度变动网格
    grid_w = (width + block_size - 1) // block_size
    grid_h = (height + block_size - 1) // block_size
    grid = np.zeros((grid_h, grid_w), dtype=bool)

    # 快速统计每个 block 是否有变动 (只在 single_bbox 范围内遍历)
    bx0 = x_all // block_size
    by0 = y_all // block_size
    bx1 = (x_all + w_all + block_size - 1) // block_size
    by1 = (y_all + h_all + block_size - 1) // block_size

    for by in range(by0, by1):
        y0 = by * block_size
        y1 = min(y0 + block_size, height)
        for bx in range(bx0, bx1):
            x0 = bx * block_size
            x1 = min(x0 + block_size, width)
            if diff[y0:y1, x0:x1].any():
                grid[by, bx] = True

    # 2. 8-邻域连通分量标记 (轻量级 BFS，网格规模通常很小)
    visited = np.zeros_like(grid, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    for by in range(by0, by1):
        for bx in range(bx0, bx1):
            if grid[by, bx] and not visited[by, bx]:
                comp = []
                queue = [(by, bx)]
                visited[by, bx] = True
                while queue:
                    cy, cx = queue.pop(0)
                    comp.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < grid_h and 0 <= nx < grid_w:
                                if grid[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                components.append(comp)

    if len(components) <= 1:
        return [single_bbox]

    # 3. 将每个连通分量转换为像素坐标下的紧凑外接矩形
    rects: List[Tuple[int, int, int, int]] = []
    for comp in components:
        # 获取该组件包含的 block 范围
        min_by = min(c[0] for c in comp)
        max_by = max(c[0] for c in comp)
        min_bx = min(c[1] for c in comp)
        max_bx = max(c[1] for c in comp)

        # 映射到像素并在该区域内做微调边界
        px0 = min_bx * block_size
        py0 = min_by * block_size
        px1 = min((max_bx + 1) * block_size, width)
        py1 = min((max_by + 1) * block_size, height)

        # 在这个连通块范围内提取精确像素外接矩形
        sub_diff = diff[py0:py1, px0:px1]
        s_rows = np.any(sub_diff, axis=1)
        s_cols = np.any(sub_diff, axis=0)
        if s_rows.any() and s_cols.any():
            rymin, rymax = np.where(s_rows)[0][[0, -1]]
            rxmin, rxmax = np.where(s_cols)[0][[0, -1]]
            rects.append((
                px0 + int(rxmin),
                py0 + int(rymin),
                int(rxmax - rxmin + 1),
                int(rymax - rymin + 1),
            ))

    # 4. 贪心合并过多的矩形，使其数量不超过 max_rects
    def rect_union(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        nx0 = min(x1, x2)
        ny0 = min(y1, y2)
        nx1 = max(x1 + w1, x2 + w2)
        ny1 = max(y1 + h1, y2 + h2)
        return nx0, ny0, nx1 - nx0, ny1 - ny0

    def added_area(r1, r2):
        u = rect_union(r1, r2)
        return (u[2] * u[3]) - (r1[2] * r1[3] + r2[2] * r2[3])

    while len(rects) > max_rects:
        # 寻找合并后多余空白面积增加最小的两个矩形进行合并
        best_pair = None
        min_cost = float("inf")
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                cost = added_area(rects[i], rects[j])
                if cost < min_cost:
                    min_cost = cost
                    best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            new_r = rect_union(rects[i], rects[j])
            rects.pop(j)
            rects.pop(i)
            rects.append(new_r)
        else:
            break

    # 5. 校验拆分收益：若多矩形总面积未能明显小于单大矩形，则保持单大矩形以减少包头
    total_multi_area = sum(r[2] * r[3] for r in rects)
    if total_multi_area > single_area * area_efficiency_threshold:
        return [single_bbox]

    return rects
