"""通用格式与 DyV 互转适配器工具。

支持将动图 (GIF, APNG, WebP) 以及常见视频文件 (MP4, MKV 等若有 OpenCV/imageio) 转换为 DyV，
或将 DyV 转换为 GIF / APNG 动图。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .codec import HAS_PIL
from .decoder import DyvReader
from .encoder import DyvWriter
from .format import TileCodec, Transform

if HAS_PIL:
    from PIL import Image, ImageSequence


def gif_to_dyv(
    gif_path: str,
    dyv_path: str,
    refresh_mode: str = "auto",
    tile_codec: int = TileCodec.ZLIB,
    transform: int = Transform.NONE,
) -> int:
    """将 GIF 动图转换为 DyV 视频格式。"""
    if not HAS_PIL:
        raise RuntimeError("转换功能需要安装 Pillow (pip install Pillow)")

    img = Image.open(gif_path)
    width, height = img.size
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(img):
        frame_rgb = frame.convert("RGB")
        arr = np.asarray(frame_rgb, dtype=np.uint8)
        frames.append(arr)
        # 获取每帧持续时间（毫秒，默认 100ms）
        duration_ms = frame.info.get("duration", 100)
        durations.append(max(duration_ms, 10))

    with DyvWriter(
        dyv_path,
        width=width,
        height=height,
        timescale=1000,
        refresh_mode=refresh_mode,
        tile_codec=tile_codec,
        transform=transform,
    ) as writer:
        current_pts = 0
        for i, (f, dur) in enumerate(zip(frames, durations)):
            writer.append_frame(f, pts=current_pts)
            current_pts += dur

    return len(frames)


def dyv_to_gif(dyv_path: str, gif_path: str, max_fps: float = 30.0) -> int:
    """将 DyV 视频导出为高质量 GIF 动图。"""
    if not HAS_PIL:
        raise RuntimeError("转换功能需要安装 Pillow (pip install Pillow)")

    pil_frames = []
    durations = []

    with DyvReader(dyv_path) as reader:
        last_pts = None
        for frame in reader.iter_frames():
            img = Image.fromarray(frame.image)
            pil_frames.append(img)

            if last_pts is not None:
                diff_ms = int(round((frame.pts - last_pts) * 1000 / reader.timescale))
                # 限制在合理范围内
                durations.append(max(diff_ms, int(1000 / max_fps)))
            last_pts = frame.pts

        if pil_frames:
            durations.append(durations[-1] if durations else 100)
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
                optimize=True,
            )

    return len(pil_frames)
