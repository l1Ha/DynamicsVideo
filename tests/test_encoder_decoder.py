import io
import numpy as np
import pytest

from dyv.decoder import DyvReader
from dyv.encoder import DyvWriter
from dyv.format import FrameType, PixelFormat, TileCodec, Transform


def test_roundtrip_lossless_rect_mode():
    """测试脏矩形局部刷新的无损端到端往返解码精度。"""
    width, height = 128, 96
    frames = []

    # 帧 0: 全黑背景
    f0 = np.zeros((height, width, 3), dtype=np.uint8)
    frames.append(f0)

    # 帧 1: 仅在 (20, 30) 处出现一个 15x15 的红色方块
    f1 = f0.copy()
    f1[30:45, 20:35, 0] = 255
    frames.append(f1)

    # 帧 2: 方块移动到 (60, 40)，原方块抹平为绿色
    f2 = f1.copy()
    f2[30:45, 20:35] = [0, 255, 0]  # 原区域变绿
    f2[40:55, 60:75, 2] = 255       # 新区域加蓝
    frames.append(f2)

    # 帧 3: 画面完全静止（应自动生成空帧）
    f3 = f2.copy()
    frames.append(f3)

    bio = io.BytesIO()
    with DyvWriter(
        bio,
        width=width,
        height=height,
        tile_size=32,
        refresh_mode="rect",
        tile_codec=TileCodec.ZLIB,
        transform=Transform.XOR_DELTA,
    ) as writer:
        for i, frame in enumerate(frames):
            writer.append_frame(frame, pts=i * 33)

    bio.seek(0)
    with DyvReader(bio) as reader:
        assert reader.width == width
        assert reader.height == height

        decoded_frames = list(reader.iter_frames())
        assert len(decoded_frames) == 4

        # 验证各帧类型
        assert decoded_frames[0].is_key
        assert decoded_frames[1].frame_type == FrameType.DELTA_RECTS
        assert decoded_frames[2].frame_type == FrameType.DELTA_RECTS
        assert decoded_frames[3].frame_type == FrameType.EMPTY

        # 逐像素验证 100% 绝对一致
        for orig, dec in zip(frames, decoded_frames):
            assert np.array_equal(orig, dec.image), f"第 {dec.ordinal} 帧像素不匹配"


def test_roundtrip_tile_mode():
    """测试瓦片网格局部刷新模式。"""
    width, height = 64, 64
    f0 = np.zeros((height, width, 3), dtype=np.uint8)
    f1 = f0.copy()
    # 仅修改右下角 [32:64, 32:64]
    f1[40:50, 40:50] = 200

    bio = io.BytesIO()
    with DyvWriter(
        bio,
        width=width,
        height=height,
        tile_size=32,
        refresh_mode="tile",
        tile_codec=TileCodec.ZLIB,
    ) as w:
        w.append_frame(f0, pts=0)
        w.append_frame(f1, pts=50)

    bio.seek(0)
    with DyvReader(bio) as r:
        dec0 = r.read_frame()
        dec1 = r.read_frame()
        assert dec0 is not None and dec0.is_key
        assert dec1 is not None and dec1.frame_type == FrameType.DELTA_TILES
        assert dec1.changed_tiles == [3]  # 仅第 3 块瓦片刷新
        assert np.array_equal(f0, dec0.image)
        assert np.array_equal(f1, dec1.image)


def test_keyframe_seeking():
    """测试基于关键帧索引的随机定位 (Seek)。"""
    width, height = 64, 64
    bio = io.BytesIO()
    ground_truth = []

    with DyvWriter(
        bio,
        width=width,
        height=height,
        tile_size=32,
        keyframe_interval=4,  # 每 4 帧一个关键帧
        build_index=True,
    ) as w:
        for i in range(12):
            f = np.full((height, width, 3), i * 20, dtype=np.uint8)
            ground_truth.append(f)
            w.append_frame(f, pts=i * 100)

    bio.seek(0)
    with DyvReader(bio) as r:
        assert r.total_frames == 12
        assert r.keyframe_index is not None
        assert len(r.keyframe_index) == 3  # 0, 4, 8 为关键帧

        # 定位到第 6 帧 (PTS=600)
        f_seek = r.seek_pts(600)
        assert f_seek is not None
        assert f_seek.pts == 600
        assert np.array_equal(ground_truth[6], f_seek.image)

        # 定位到第 9 帧 (PTS=900)
        f_seek9 = r.seek_pts(900)
        assert f_seek9 is not None
        assert f_seek9.pts == 900
        assert np.array_equal(ground_truth[9], f_seek9.image)
