import os
import tempfile
import numpy as np
import pytest

from PIL import Image
from dyv.converter import dyv_to_gif, gif_to_dyv
from dyv.decoder import DyvReader


def test_gif_and_dyv_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = os.path.join(tmpdir, "test.gif")
        dyv_path = os.path.join(tmpdir, "test.dyv")
        gif_out_path = os.path.join(tmpdir, "out.gif")

        # 1. 创建一个 3 帧的测试动图
        frames = []
        for i in range(3):
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            arr[i * 8 : (i + 1) * 8, :, 0] = 200
            frames.append(Image.fromarray(arr))

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=[100, 150, 200],
            loop=0,
        )

        # 2. 测试 gif_to_dyv
        n = gif_to_dyv(gif_path, dyv_path)
        assert n == 3
        assert os.path.exists(dyv_path)

        with DyvReader(dyv_path) as r:
            assert r.width == 32
            assert r.height == 32
            dec_frames = list(r.iter_frames())
            assert len(dec_frames) == 3
            assert dec_frames[1].pts == 100

        # 3. 测试 dyv_to_gif
        n_out = dyv_to_gif(dyv_path, gif_out_path)
        assert n_out == 3
        assert os.path.exists(gif_out_path)
