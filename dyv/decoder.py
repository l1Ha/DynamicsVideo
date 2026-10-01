"""DyV 解码器。

支持顺序流式读取、精确帧时间戳解析、瓦片/脏矩形/空帧局部刷新回放，
以及基于尾部索引的毫秒级关键帧随机定位 (seek)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import BinaryIO, Iterator, List, Optional, Tuple, Union
import numpy as np

from .codec import decompress_tile
from .format import (
    BOX_TOTAL_TICKS,
    FRAME_END,
    FrameType,
    Header,
    IndexEntry,
    PIXEL_CHANNELS,
    RawFramePacket,
    RectPatch,
    Tail,
    TileCodec,
    Transform,
    read_boxes,
    read_frame_packet,
    read_header,
    read_tail,
)
from .tiles import grid_dims, tile_bounds, tile_count


@dataclass
class DecodedFrame:
    ordinal: int
    pts: int
    timestamp: float
    is_key: bool
    frame_type: int
    image: np.ndarray
    changed_tiles: List[int] = field(default_factory=list)
    changed_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)


class DyvReader:
    """DyV 文件读取与解码器。"""

    def __init__(self, source: Union[str, BinaryIO]) -> None:
        if isinstance(source, str):
            self._file = open(source, "rb")
            self._owns_file = True
        else:
            self._file = source
            self._owns_file = False

        self.header: Header = read_header(self._file)
        self.width = self.header.width
        self.height = self.header.height
        self.tile_size = self.header.tile_size
        self.timescale = self.header.timescale
        self.pixel_format = self.header.pixel_format
        self.tile_codec = self.header.tile_codec
        self.transform = self.header.transform

        self.cols, self.rows = grid_dims(self.width, self.height, self.tile_size)
        self.total_tiles = tile_count(self.width, self.height, self.tile_size)
        self.channels = PIXEL_CHANNELS[self.pixel_format]

        self._first_frame_offset = self._file.tell() if hasattr(self._file, "tell") else 0

        # 当前画布与解码状态
        self._canvas = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        self._current_ordinal = 0
        self._last_pts = -1
        self._reached_end = False

        # 尝试读取尾部索引
        self.tail: Optional[Tail] = None
        self.trailer_boxes: List[Tuple[int, bytes]] = []
        if hasattr(self._file, "seek") and hasattr(self._file, "tell"):
            cur = self._file.tell()
            self.tail = read_tail(self._file)
            self._file.seek(cur)

    @property
    def total_frames(self) -> Optional[int]:
        if self.tail is not None and self.tail.total_frames > 0:
            return self.tail.total_frames
        return None

    @property
    def keyframe_index(self) -> Optional[List[IndexEntry]]:
        if self.tail is not None:
            return self.tail.index
        return None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.tail and self.tail.index:
            last_entry = self.tail.index[-1]
            return last_entry.pts / self.timescale
        return None

    def get_canvas(self) -> np.ndarray:
        """获取当前画布的副本。"""
        return self._canvas.copy()

    def reset(self) -> None:
        """重置回文件开头第一帧。"""
        if not hasattr(self._file, "seek"):
            raise RuntimeError("非可寻址流无法 reset()")
        self._file.seek(self._first_frame_offset)
        self._canvas.fill(0)
        self._current_ordinal = 0
        self._last_pts = -1
        self._reached_end = False

    def read_frame(self) -> Optional[DecodedFrame]:
        """读取并解码下一帧。到达末尾返回 None。"""
        if self._reached_end:
            return None

        packet: Optional[RawFramePacket] = read_frame_packet(self._file, self.total_tiles)
        if packet is None:
            self._reached_end = True
            try:
                self.trailer_boxes = read_boxes(self._file)
            except Exception:
                pass
            return None

        ftype = packet.frame_type
        pts = packet.pts
        is_key = (ftype == FrameType.KEY)
        changed_tiles_out: List[int] = []
        changed_rects_out: List[Tuple[int, int, int, int]] = []

        if ftype == FrameType.EMPTY:
            # 空帧：画面完全无变动，保持当前画布
            pass

        elif ftype == FrameType.KEY or ftype == FrameType.DELTA_TILES:
            # 瓦片刷新（包含全屏关键帧与瓦片局部刷新）
            changed_tiles_out = packet.tile_indices
            for i, idx in enumerate(packet.tile_indices):
                x0, y0, x1, y1 = tile_bounds(
                    idx, self.cols, self.tile_size, self.width, self.height
                )
                h = y1 - y0
                w = x1 - x0
                payload = packet.tile_payloads[i]
                tile_arr = decompress_tile(self.tile_codec, payload, (h, w, self.channels))

                if (
                    not is_key
                    and self.transform == Transform.XOR_DELTA
                    and self.tile_codec not in (TileCodec.JPEG, TileCodec.WEBP)
                ):
                    ref = self._canvas[y0:y1, x0:x1]
                    tile_arr = np.bitwise_xor(ref, tile_arr)

                self._canvas[y0:y1, x0:x1] = tile_arr

        elif ftype == FrameType.DELTA_RECTS:
            # 脏矩形局部刷新：仅覆盖发生变化地方的画面数据
            for r in packet.rects:
                patch = decompress_tile(self.tile_codec, r.payload, (r.h, r.w, self.channels))
                if (
                    self.transform == Transform.XOR_DELTA
                    and self.tile_codec not in (TileCodec.JPEG, TileCodec.WEBP)
                ):
                    ref = self._canvas[r.y : r.y + r.h, r.x : r.x + r.w]
                    patch = np.bitwise_xor(ref, patch)

                self._canvas[r.y : r.y + r.h, r.x : r.x + r.w] = patch
                changed_rects_out.append((r.x, r.y, r.w, r.h))

        frame = DecodedFrame(
            ordinal=self._current_ordinal,
            pts=pts,
            timestamp=pts / self.timescale,
            is_key=is_key,
            frame_type=ftype,
            image=self._canvas.copy(),
            changed_tiles=changed_tiles_out,
            changed_rects=changed_rects_out,
        )
        self._current_ordinal += 1
        self._last_pts = pts
        return frame

    def iter_frames(self) -> Iterator[DecodedFrame]:
        """按顺序迭代所有帧。"""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def seek_pts(self, target_pts: int) -> Optional[DecodedFrame]:
        """精确或向前对齐定位到指定 PTS 的帧。

        若有索引，跳转到 target_pts 之前最近的关键帧，并向后解码到目标帧。
        若无索引，从头顺序解码到目标帧。
        """
        if not hasattr(self._file, "seek"):
            raise RuntimeError("流不支持 seek")

        if target_pts < 0:
            target_pts = 0

        # 如果已有索引，找到 <= target_pts 的最后关键帧
        start_offset = self._first_frame_offset
        start_ordinal = 0
        if self.tail and self.tail.index:
            best_entry: Optional[IndexEntry] = None
            for entry in self.tail.index:
                if entry.pts <= target_pts:
                    best_entry = entry
                else:
                    break
            if best_entry is not None:
                start_offset = best_entry.offset
                start_ordinal = best_entry.ordinal

        # 跳转至关键帧
        self._file.seek(start_offset)
        self._current_ordinal = start_ordinal
        self._reached_end = False

        # 解码并前进至最接近或等于 target_pts 的帧
        target_frame: Optional[DecodedFrame] = None
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            if frame.pts <= target_pts:
                target_frame = frame
            if frame.pts >= target_pts:
                if target_frame is None or frame.pts == target_pts:
                    target_frame = frame
                break

        return target_frame

    def seek_time(self, seconds: float) -> Optional[DecodedFrame]:
        """按秒定位。"""
        target_pts = int(round(seconds * self.timescale))
        return self.seek_pts(target_pts)

    def close(self) -> None:
        if self._owns_file:
            self._file.close()

    def __enter__(self) -> "DyvReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
