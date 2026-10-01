"""DyV 编码器。

支持：
1. 动态帧率（VFR）：显式精确时间戳（PTS）与自适应时间步长；
2. 局部刷新（Partial Refresh）：
   - 当画面零变化时：写入纯时间戳空帧（EMPTY，仅 2~3 字节）；
   - 当局部面积变化时：仅保存发生变化的矩形区域（Dirty Rect，仅存该区域像素）；
   - 瓦片网格模式（Tile Grid）：适合分散多处变动场景；
   - 自动决策模式（Auto）：智能选取开销最小的局部刷新策略；
3. 无损/有损可插拔编码器（zlib, zstd, raw, jpeg, webp）；
4. XOR delta 变换，提升局部刷新的残差压缩率。
"""

from __future__ import annotations

from typing import BinaryIO, List, Optional, Union
import numpy as np

from .codec import compress_tile
from .format import (
    BOX_ENCODER,
    BOX_TOTAL_TICKS,
    FLAG_HAS_INDEX,
    FrameType,
    FRAME_END,
    Header,
    IndexEntry,
    PIXEL_CHANNELS,
    PixelFormat,
    PROFILE_LOSSLESS,
    PROFILE_LOSSY,
    RectPatch,
    TileCodec,
    Transform,
    write_boxes,
    write_delta_rects_packet,
    write_delta_tiles_packet,
    write_empty_packet,
    write_header,
    write_key_packet,
    write_tail,
)
from .rects import find_dirty_bbox
from .tiles import changed_tiles, grid_dims, tile_bounds, tile_count


class DyvWriter:
    """流式写入 DyV 文件的编码器。"""

    def __init__(
        self,
        target: Union[str, BinaryIO],
        width: int,
        height: int,
        tile_size: int = 64,
        timescale: int = 1000,
        pixel_format: int = PixelFormat.RGB24,
        tile_codec: int = TileCodec.ZLIB,
        transform: int = Transform.NONE,
        refresh_mode: str = "auto",  # 'auto', 'rect', 'tile'
        keyframe_interval: int = 0,
        quality: int = 85,
        build_index: bool = True,
        boxes: Optional[List[tuple[int, bytes]]] = None,
    ) -> None:
        """
        :param refresh_mode: 局部刷新模式
            - 'auto': 自动选择（无变化存空帧；局部变化优先存脏矩形，分散变化存瓦片）
            - 'rect': 脏矩形局部刷新（仅保存变化面积的矩形数据）
            - 'tile': 瓦片网格局部刷新（仅保存变化的瓦片块）
        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.timescale = timescale
        self.pixel_format = pixel_format
        self.tile_codec = tile_codec
        self.transform = transform
        self.refresh_mode = refresh_mode.lower()
        self.keyframe_interval = keyframe_interval
        self.quality = quality
        self.build_index = build_index

        self.cols, self.rows = grid_dims(width, height, tile_size)
        self.total_tiles = tile_count(width, height, tile_size)
        self.channels = PIXEL_CHANNELS[pixel_format]

        if isinstance(target, str):
            self._file = open(target, "wb")
            self._owns_file = True
        else:
            self._file = target
            self._owns_file = False

        self._closed = False
        self._frame_count = 0
        self._last_pts = -1
        self._last_keyframe_ordinal = -1
        self._index_entries: List[IndexEntry] = []

        # 解码端画布镜像
        self._canvas = np.zeros((height, width, self.channels), dtype=np.uint8)

        # 写入文件头
        profile = (
            PROFILE_LOSSY
            if tile_codec in (TileCodec.JPEG, TileCodec.WEBP)
            else PROFILE_LOSSLESS
        )
        flags = FLAG_HAS_INDEX if build_index else 0
        header_boxes = list(boxes or [])
        header_boxes.append((BOX_ENCODER, b"DyV Reference Encoder v1.0"))

        header = Header(
            width=width,
            height=height,
            tile_size=tile_size,
            timescale=timescale,
            pixel_format=pixel_format,
            tile_codec=tile_codec,
            transform=transform,
            profile=profile,
            flags=flags,
            boxes=header_boxes,
        )
        write_header(self._file, header)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append_frame(
        self,
        frame: np.ndarray,
        pts: Optional[int] = None,
        duration: Optional[int] = None,
        force_key: bool = False,
    ) -> None:
        """追加一帧。支持动态 PTS 与局部刷新。

        :param frame: HxWxC uint8 图像数组
        :param pts: timescale ticks 时间戳。若为 None，自增
        :param duration: 若未指定 pts，则 pts = last_pts + duration
        :param force_key: 强制此帧为关键帧
        """
        if self._closed:
            raise RuntimeError("DyvWriter 已关闭")
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"帧尺寸不匹配: 期望 {(self.height, self.width, self.channels)}, 得到 {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError("帧数组类型必须是 uint8")

        # 动态帧率：时间戳处理
        if pts is None:
            if self._frame_count == 0:
                pts = 0
            else:
                delta = duration if duration is not None else 1
                pts = self._last_pts + delta
        if pts < 0:
            raise ValueError("PTS 不能为负数")
        if self._frame_count > 0 and pts < self._last_pts:
            raise ValueError(f"PTS 必须单调递增 (当前 {pts} < 上一帧 {self._last_pts})")

        # 判断是否为关键帧
        is_key = False
        if self._frame_count == 0 or force_key:
            is_key = True
        elif self.keyframe_interval > 0 and (
            self._frame_count - self._last_keyframe_ordinal
        ) >= self.keyframe_interval:
            is_key = True

        packet_offset = self._file.tell() if hasattr(self._file, "tell") else 0

        if is_key:
            self._write_keyframe(frame, pts)
            self._last_keyframe_ordinal = self._frame_count
            if self.build_index and hasattr(self._file, "tell"):
                self._index_entries.append(
                    IndexEntry(self._frame_count, pts, packet_offset)
                )
        else:
            self._write_delta_frame(frame, pts)

        self._last_pts = pts
        self._frame_count += 1

    def append_time(
        self,
        frame: np.ndarray,
        seconds: float,
        force_key: bool = False,
    ) -> None:
        """根据秒数添加帧（按 timescale 转换为 ticks）。"""
        pts = int(round(seconds * self.timescale))
        self.append_frame(frame, pts=pts, force_key=force_key)

    def _write_keyframe(self, frame: np.ndarray, pts: int) -> None:
        payloads: List[bytes] = []
        for idx in range(self.total_tiles):
            x0, y0, x1, y1 = tile_bounds(
                idx, self.cols, self.tile_size, self.width, self.height
            )
            tile_data = frame[y0:y1, x0:x1]
            payload = compress_tile(self.tile_codec, tile_data, quality=self.quality)
            payloads.append(payload)

        write_key_packet(self._file, pts, payloads)
        self._canvas[:] = frame

    def _write_delta_frame(self, frame: np.ndarray, pts: int) -> None:
        # 1. 检查画面是否完全无变化 -> 空帧（零画面数据，仅推进时间戳）
        bbox = find_dirty_bbox(self._canvas, frame)
        if bbox is None:
            write_empty_packet(self._file, pts)
            return

        x, y, w, h = bbox
        dirty_area = w * h
        total_pixels = self.width * self.height

        # 2. 决定使用“矩形局部刷新”还是“瓦片网格局部刷新”
        use_rect = False
        if self.refresh_mode == "rect":
            use_rect = True
        elif self.refresh_mode == "tile":
            use_rect = False
        else:  # auto
            # 当变化集中在某区域且面积不超过画面的 85% 时，使用精确脏矩形；
            # 否则比较瓦片数
            if dirty_area <= total_pixels * 0.85:
                use_rect = True
            else:
                use_rect = False

        if use_rect:
            # === 矩形局部刷新：仅保存变化地方的画面数据 ===
            patch_data = frame[y : y + h, x : x + w]
            if (
                self.transform == Transform.XOR_DELTA
                and self.tile_codec not in (TileCodec.JPEG, TileCodec.WEBP)
            ):
                ref_patch = self._canvas[y : y + h, x : x + w]
                diff_data = np.bitwise_xor(patch_data, ref_patch)
                payload = compress_tile(
                    self.tile_codec, diff_data, quality=self.quality
                )
            else:
                payload = compress_tile(
                    self.tile_codec, patch_data, quality=self.quality
                )

            rect_patch = RectPatch(x=x, y=y, w=w, h=h, payload=payload)
            write_delta_rects_packet(self._file, pts, [rect_patch])
            self._canvas[y : y + h, x : x + w] = patch_data
        else:
            # === 瓦片局部刷新：仅保存变化的瓦片 ===
            indices = changed_tiles(
                self._canvas, frame, self.cols, self.tile_size, self.width, self.height
            )
            payloads: List[bytes] = []
            for idx in indices:
                x0, y0, x1, y1 = tile_bounds(
                    idx, self.cols, self.tile_size, self.width, self.height
                )
                tile_data = frame[y0:y1, x0:x1]
                if (
                    self.transform == Transform.XOR_DELTA
                    and self.tile_codec not in (TileCodec.JPEG, TileCodec.WEBP)
                ):
                    ref_data = self._canvas[y0:y1, x0:x1]
                    diff_data = np.bitwise_xor(tile_data, ref_data)
                    payload = compress_tile(
                        self.tile_codec, diff_data, quality=self.quality
                    )
                else:
                    payload = compress_tile(
                        self.tile_codec, tile_data, quality=self.quality
                    )
                payloads.append(payload)

            write_delta_tiles_packet(self._file, pts, indices, payloads)
            for idx in indices:
                x0, y0, x1, y1 = tile_bounds(
                    idx, self.cols, self.tile_size, self.width, self.height
                )
                self._canvas[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

    def close(self) -> None:
        """写出帧流终止哨兵、尾部元数据盒与索引，关闭文件。"""
        if self._closed:
            return
        self._closed = True

        # 写入 0xFF 帧流结束哨兵
        self._file.write(bytes([FRAME_END]))

        # 写 Trailer 盒（例如总时长）
        trailer_boxes: List[tuple[int, bytes]] = []
        if self._last_pts >= 0:
            import io
            from .format import write_varint

            bio = io.BytesIO()
            write_varint(bio, self._last_pts)
            trailer_boxes.append((BOX_TOTAL_TICKS, bio.getvalue()))
        write_boxes(self._file, trailer_boxes)

        # 写入尾部及可选索引
        can_seek = hasattr(self._file, "seek") and hasattr(self._file, "tell")
        if can_seek and self.build_index and self._index_entries:
            index_offset = self._file.tell()
            write_tail(
                self._file,
                total_frames=self._frame_count,
                index_offset=index_offset,
                index=self._index_entries,
            )
        elif can_seek:
            write_tail(
                self._file,
                total_frames=self._frame_count,
                index_offset=0,
                index=None,
            )

        self._file.flush()
        if self._owns_file:
            self._file.close()

    def __enter__(self) -> "DyvWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
