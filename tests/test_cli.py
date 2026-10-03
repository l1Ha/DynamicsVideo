import os
import tempfile
import numpy as np
import pytest

from PIL import Image
from dyv.cli import main
from dyv.encoder import DyvWriter


def test_cli_info_and_bench(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        dyv_path = os.path.join(tmpdir, "test.dyv")
        with DyvWriter(dyv_path, width=64, height=64) as w:
            f0 = np.zeros((64, 64, 3), dtype=np.uint8)
            f1 = f0.copy()
            f1[10:20, 10:20] = 255
            w.append_frame(f0, pts=0)
            w.append_frame(f1, pts=33)

        # 测试 info 命令
        ret = main(["info", dyv_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "64 x 64" in captured.out
        assert "DyV 视频信息" in captured.out

        # 测试 verify 命令
        ret_veri = main(["verify", dyv_path])
        assert ret_veri == 0

        # 测试 bench 命令
        ret = main(["bench", dyv_path])
        assert ret == 0
        captured = capsys.readouterr()
        assert "解码完成: 共 2 帧" in captured.out
        assert "FPS" in captured.out


def test_cli_dump_and_encode(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        # 生成两张测试图片
        f0_path = os.path.join(tmpdir, "img_00.png")
        f1_path = os.path.join(tmpdir, "img_01.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(f0_path)
        Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8)).save(f1_path)

        out_dyv = os.path.join(tmpdir, "out.dyv")
        # 测试 encode
        ret = main(["encode", os.path.join(tmpdir, "img_*.png"), out_dyv])
        assert ret == 0
        assert os.path.exists(out_dyv)

        # 测试 dump
        dump_dir = os.path.join(tmpdir, "dump")
        ret = main(["dump", out_dyv, "-o", dump_dir])
        assert ret == 0
        assert os.path.exists(os.path.join(dump_dir, "frame_00000_pts_0.png"))
