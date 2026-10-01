import io
import numpy as np
import pytest

from dyv.decoder import DyvReader
from dyv.encoder import DyvWriter


def test_variable_frame_rate():
    """测试任意动态帧率（VFR）：非等间距时间戳与高时间精度。"""
    width, height = 32, 32
    bio = io.BytesIO()

    # 模拟真实动态帧率事件流：
    # 0.0s (0ms) -> 0.05s (50ms) -> 0.12s (120ms) -> 2.50s (2500ms，静止等待) -> 2.52s (2520ms)
    time_points = [0.0, 0.05, 0.12, 2.50, 2.52]
    frames = []

    with DyvWriter(bio, width=width, height=height, timescale=1000) as w:
        for i, t in enumerate(time_points):
            f = np.full((height, width, 3), i * 30, dtype=np.uint8)
            frames.append(f)
            w.append_time(f, seconds=t)

    bio.seek(0)
    with DyvReader(bio) as r:
        dec_list = list(r.iter_frames())
        assert len(dec_list) == len(time_points)

        for orig_t, dec in zip(time_points, dec_list):
            assert dec.timestamp == pytest.approx(orig_t, abs=0.002)

        # 测试 seek_time
        # 定位到 2.0s（应该定位到 0.12s 之后、2.50s 之前的画面状态）
        f_mid = r.seek_time(2.0)
        assert f_mid is not None
        assert f_mid.timestamp <= 2.0
        assert np.array_equal(frames[2], f_mid.image)

        # 定位到 2.51s
        f_late = r.seek_time(2.51)
        assert f_late is not None
        assert np.array_equal(frames[3], f_late.image)
