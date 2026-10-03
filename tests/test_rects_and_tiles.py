import numpy as np
import pytest

from dyv.rects import compute_change_ratio, find_dirty_bbox, find_dirty_rects
from dyv.tiles import changed_tiles, grid_dims, tile_bounds, tile_count


def test_multi_dirty_rects_clustering():
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    frame = canvas.copy()

    # 在左上角 (10, 10, 20, 20) 和 右下角 (160, 160, 20, 20) 各画一个色块
    frame[10:30, 10:30] = 255
    frame[160:180, 160:180] = 180

    # 单外接框面积会是 ~170x170 = 28900
    single = find_dirty_bbox(canvas, frame)
    assert single is not None
    assert single[2] * single[3] > 20000

    # 多矩形聚类应该识别出两个独立紧凑的矩形
    rects = find_dirty_rects(canvas, frame, block_size=16)
    assert len(rects) == 2
    # 两个小矩形面积和仅为 400 + 400 = 800，比单框节省 95% 以上的无用像素面积！
    total_area = sum(r[2] * r[3] for r in rects)
    assert total_area < 2000
    assert total_area < (single[2] * single[3] * 0.1)


def test_grid_dims_and_bounds():
    # 100 x 60 图像，tile_size=32
    # cols = ceil(100/32) = 4, rows = ceil(60/32) = 2
    cols, rows = grid_dims(100, 60, 32)
    assert cols == 4
    assert rows == 2
    assert tile_count(100, 60, 32) == 8

    # 检查角落瓦片越界截断
    # 瓦片 0 (第一行第一列): (0, 0, 32, 32)
    assert tile_bounds(0, cols, 32, 100, 60) == (0, 0, 32, 32)
    # 瓦片 3 (第一行最后一列): x0=96, x1=min(128, 100)=100
    assert tile_bounds(3, cols, 32, 100, 60) == (96, 0, 100, 32)
    # 瓦片 7 (第二行最后一列): y0=32, y1=min(64, 60)=60
    assert tile_bounds(7, cols, 32, 100, 60) == (96, 32, 100, 60)


def test_dirty_bbox_detection():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = canvas.copy()

    # 无变化
    assert find_dirty_bbox(canvas, frame) is None
    assert compute_change_ratio(canvas, frame) == 0.0

    # 在 [20:35, 40:60] 画一个矩形
    frame[20:35, 40:60] = 255
    bbox = find_dirty_bbox(canvas, frame)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 40
    assert y == 20
    assert w == 20
    assert h == 15

    ratio = compute_change_ratio(canvas, frame)
    assert ratio == pytest.approx((15 * 20) / (100 * 100))


def test_changed_tiles():
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    frame = canvas.copy()
    # 划分为 32x32 瓦片：2x2=4 块
    cols = 2
    # 仅修改瓦片 3 (右下角 [32:64, 32:64])
    frame[40:50, 40:50] = 128
    tiles = changed_tiles(canvas, frame, cols, 32, 64, 64)
    assert tiles == [3]
