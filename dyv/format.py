"""DyV (Dynamic Video) v1 容器层。

负责常量定义、varint 编码、文件头 / 元数据盒 / 帧包 / 尾部索引的二进制读写。
字节序统一为小端（little-endian），整数变长编码采用 LEB128 无符号 varint。

文件总体布局::

    ┌──────────────────────────────┐
    │ Header (26 字节固定部分)      │
    │ Header 元数据盒 (TLV, 0 结尾) │
    ├──────────────────────────────┤
    │ Frame 0 (关键帧)             │
    │ Frame 1..n (增量帧/关键帧)    │
    │   - DELTA_RECTS: 任意矩形刷新 │
    │   - DELTA_TILES: 瓦片局部刷新 │
    │   - EMPTY: 纯时间戳空帧       │
    │ 0xFF 帧流结束哨兵             │
    │ Trailer 元数据盒 (TLV, 0 结尾)│
    ├──────────────────────────────┤
    │ [可选] 关键帧索引             │
    │ total_frames u64             │
    │ index_offset u64 (0=无索引)   │
    │ 'DYVE' 尾部魔数               │
    └──────────────────────────────┘
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, List, Optional, Tuple

MAGIC = b"DYV "
FOOTER_MAGIC = b"DYVE"
VERSION = 1


class DyvError(Exception):
    """DyV 通用错误。"""


class DyvFormatError(DyvError):
    """文件格式损坏或不符合规范。"""


class PixelFormat:
    RGB24 = 0
    RGBA32 = 1
    GRAY8 = 2
    GRAY8A = 3


PIXEL_FORMAT_NAMES = {
    "rgb24": PixelFormat.RGB24,
    "rgba32": PixelFormat.RGBA32,
    "gray8": PixelFormat.GRAY8,
    "gray8a": PixelFormat.GRAY8A,
}
PIXEL_CHANNELS = {
    PixelFormat.RGB24: 3,
    PixelFormat.RGBA32: 4,
    PixelFormat.GRAY8: 1,
    PixelFormat.GRAY8A: 2,
}
PIXEL_FORMAT_LABELS = {
    PixelFormat.RGB24: "rgb24",
    PixelFormat.RGBA32: "rgba32",
    PixelFormat.GRAY8: "gray8",
    PixelFormat.GRAY8A: "gray8a",
}


class TileCodec:
    RAW = 0
    ZLIB = 1
    ZSTD = 2
    JPEG = 3
    WEBP = 4


CODEC_NAMES = {
    TileCodec.RAW: "raw",
    TileCodec.ZLIB: "zlib",
    TileCodec.ZSTD: "zstd",
    TileCodec.JPEG: "jpeg",
    TileCodec.WEBP: "webp",
}
NAME_TO_CODEC = {v: k for k, v in CODEC_NAMES.items()}
LOSSY_CODECS = frozenset({TileCodec.JPEG, TileCodec.WEBP})


class Transform:
    NONE = 0
    XOR_DELTA = 1


TRANSFORM_NAMES = {Transform.NONE: "none", Transform.XOR_DELTA: "xor"}
NAME_TO_TRANSFORM = {v: k for k, v in TRANSFORM_NAMES.items()}


class FrameType:
    KEY = 0          # 全帧关键帧（独立编码全部瓦片）
    DELTA_TILES = 1  # 瓦片网格局部刷新（仅编码发生变化的瓦片）
    DELTA_RECTS = 2  # 任意脏矩形局部刷新（仅编码发生变化的矩形区域 (x, y, w, h)）
    EMPTY = 3        # 空帧（画面零变化，仅步进时间戳 PTS）


FRAME_END = 0xFF  # 帧流结束哨兵

FLAG_HAS_INDEX = 1 << 0
FLAG_LOOP = 1 << 1
FLAG_HAS_CRC32 = 1 << 2  # 帧尾/流校验扩展标志
FLAG_HAS_AUDIO = 1 << 3  # 音频轨道扩展标志

PROFILE_LOSSLESS = 0
PROFILE_LOSSY = 1


class Profile:
    LOSSLESS = PROFILE_LOSSLESS
    LOSSY = PROFILE_LOSSY


# ---- 元数据盒类型（TLV 结构）----
BOX_TITLE = 1          # utf-8 标题
BOX_LOOP = 2           # u32 循环次数（0 = 无限循环）
BOX_ICC = 3            # ICC 色彩配置文件原始字节
BOX_CREATED = 4        # u64 unix 时间戳
BOX_ENCODER = 5        # utf-8 编码器描述
BOX_KV = 6             # varint 键长 + utf-8 键 + 值
BOX_DEFAULT_TICKS = 7  # varint 默认帧间隔（ticks），供 CFR 播放器回退
BOX_TOTAL_TICKS = 8    # varint 总时长（ticks）
BOX_AUDIO_HEADER = 20  # 音频头扩展（采样率、通道数、编码类型：Opus/AAC/FLAC/PCM）
BOX_AUDIO_CHUNK = 21   # 音频交织流数据包 (PTS + 音频载荷)


@dataclass
class Header:
    width: int
    height: int
    tile_size: int
    timescale: int
    pixel_format: int
    tile_codec: int
    transform: int
    profile: int = PROFILE_LOSSLESS
    flags: int = 0
    boxes: List[Tuple[int, bytes]] = field(default_factory=list)


@dataclass
class IndexEntry:
    ordinal: int  # 帧序号（从 0 计）
    pts: int      # 时间戳（timescale ticks）
    offset: int   # 帧包在文件中的绝对偏移


@dataclass
class Tail:
    total_frames: int
    index_offset: int          # 0 表示无索引
    index: Optional[List[IndexEntry]] = None


@dataclass
class RectPatch:
    """脏矩形局部刷新块。"""
    x: int
    y: int
    w: int
    h: int
    payload: bytes


# ---------------- varint ----------------

def write_varint(f: BinaryIO, value: int) -> None:
    if value < 0:
        raise DyvError("varint 只支持非负整数")
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            f.write(bytes([b | 0x80]))
        else:
            f.write(bytes([b]))
            return


def read_varint(f: BinaryIO) -> int:
    result = 0
    shift = 0
    while True:
        data = f.read(1)
        if not data:
            raise DyvFormatError("varint 越过文件末尾")
        b = data[0]
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result
        shift += 7
        if shift > 70:
            raise DyvFormatError("varint 过长")


# ---------------- 元数据盒 ----------------

def write_boxes(f: BinaryIO, boxes: List[Tuple[int, bytes]]) -> None:
    for box_type, payload in boxes:
        if not 0 < box_type < 256 or box_type == FRAME_END:
            raise DyvError(f"非法盒类型 {box_type}")
        f.write(bytes([box_type]))
        write_varint(f, len(payload))
        f.write(payload)
    f.write(b"\x00")


def read_boxes(f: BinaryIO) -> List[Tuple[int, bytes]]:
    boxes = []
    while True:
        data = f.read(1)
        if not data:
            raise DyvFormatError("元数据盒未正常终止")
        box_type = data[0]
        if box_type == 0:
            return boxes
        length = read_varint(f)
        payload = f.read(length)
        if len(payload) != length:
            raise DyvFormatError("元数据盒数据不完整")
        boxes.append((box_type, payload))


# ---------------- 文件头 ----------------

_HEADER_STRUCT = struct.Struct("<BBHIIHIBBBB")


def write_header(f: BinaryIO, header: Header) -> None:
    f.write(MAGIC)
    f.write(_HEADER_STRUCT.pack(
        VERSION, header.profile, header.flags,
        header.width, header.height, header.tile_size, header.timescale,
        header.pixel_format, header.tile_codec, header.transform, 0))
    write_boxes(f, header.boxes)


def read_header(f: BinaryIO) -> Header:
    magic = f.read(4)
    if magic != MAGIC:
        raise DyvFormatError(f"不是 DyV 文件（魔数 {magic!r}）")
    fields = _HEADER_STRUCT.unpack(f.read(_HEADER_STRUCT.size))
    (version, profile, flags, width, height, tile_size, timescale,
     pixel_format, tile_codec, transform, _reserved) = fields
    if version != VERSION:
        raise DyvFormatError(f"不支持的 DyV 版本 {version}")
    if width == 0 or height == 0 or timescale == 0 or tile_size == 0:
        raise DyvFormatError("文件头字段非法")
    if pixel_format not in PIXEL_CHANNELS:
        raise DyvFormatError(f"未实现的像素格式 {pixel_format}")
    if tile_codec not in CODEC_NAMES:
        raise DyvFormatError(f"未实现的瓦片编码 {tile_codec}")
    if transform not in TRANSFORM_NAMES:
        raise DyvFormatError(f"未实现的变换 {transform}")
    boxes = read_boxes(f)
    return Header(width, height, tile_size, timescale, pixel_format,
                  tile_codec, transform, profile, flags, boxes)


# ---------------- 帧包读写 ----------------

def read_frame_payloads(f: BinaryIO, count: int) -> List[bytes]:
    payloads = []
    for _ in range(count):
        length = read_varint(f)
        data = f.read(length)
        if len(data) != length:
            raise DyvFormatError("载荷数据不完整")
        payloads.append(data)
    return payloads


def write_key_packet(f: BinaryIO, pts: int, payloads: List[bytes]) -> None:
    """写关键帧（包含所有瓦片数据）。"""
    f.write(bytes([FrameType.KEY]))
    write_varint(f, pts)
    for payload in payloads:
        write_varint(f, len(payload))
        f.write(payload)


def write_empty_packet(f: BinaryIO, pts: int) -> None:
    """写空帧（画面完全无变化，仅更新时间戳 PTS）。"""
    f.write(bytes([FrameType.EMPTY]))
    write_varint(f, pts)


def write_delta_tiles_packet(f: BinaryIO, pts: int,
                             indices: List[int], payloads: List[bytes]) -> None:
    """写瓦片网格局部刷新帧（仅保存发生变化的瓦片）。"""
    f.write(bytes([FrameType.DELTA_TILES]))
    write_varint(f, pts)
    write_varint(f, len(indices))
    prev = None
    for i in indices:
        if prev is None:
            write_varint(f, i)
        else:
            write_varint(f, i - prev - 1)
        prev = i
    for payload in payloads:
        write_varint(f, len(payload))
        f.write(payload)


def write_delta_rects_packet(f: BinaryIO, pts: int,
                             rects: List[RectPatch]) -> None:
    """写任意矩形局部刷新帧（仅保存发生变化的矩形区域像素）。"""
    f.write(bytes([FrameType.DELTA_RECTS]))
    write_varint(f, pts)
    write_varint(f, len(rects))
    for r in rects:
        write_varint(f, r.x)
        write_varint(f, r.y)
        write_varint(f, r.w)
        write_varint(f, r.h)
        write_varint(f, len(r.payload))
        f.write(r.payload)


@dataclass
class RawFramePacket:
    frame_type: int
    pts: int
    # 瓦片帧字段
    tile_indices: List[int] = field(default_factory=list)
    tile_payloads: List[bytes] = field(default_factory=list)
    # 矩形帧字段
    rects: List[RectPatch] = field(default_factory=list)


def read_frame_packet(f: BinaryIO, total_tiles: int) -> Optional[RawFramePacket]:
    """读取一个完整帧包。遇 0xFF 哨兵返回 None。"""
    head = f.read(1)
    if not head:
        raise DyvFormatError("帧流意外结束")
    ftype = head[0]
    if ftype == FRAME_END:
        return None

    if ftype not in (FrameType.KEY, FrameType.DELTA_TILES,
                     FrameType.DELTA_RECTS, FrameType.EMPTY):
        raise DyvFormatError(f"未知帧类型 0x{ftype:02X}")

    pts = read_varint(f)

    if ftype == FrameType.EMPTY:
        return RawFramePacket(frame_type=ftype, pts=pts)

    if ftype == FrameType.KEY:
        payloads = read_frame_payloads(f, total_tiles)
        return RawFramePacket(
            frame_type=ftype,
            pts=pts,
            tile_indices=list(range(total_tiles)),
            tile_payloads=payloads,
        )

    if ftype == FrameType.DELTA_TILES:
        count = read_varint(f)
        indices: List[int] = []
        prev = None
        for _ in range(count):
            gap = read_varint(f)
            idx = gap if prev is None else prev + gap + 1
            indices.append(idx)
            prev = idx
        payloads = read_frame_payloads(f, count)
        return RawFramePacket(
            frame_type=ftype,
            pts=pts,
            tile_indices=indices,
            tile_payloads=payloads,
        )

    if ftype == FrameType.DELTA_RECTS:
        count = read_varint(f)
        rect_list: List[RectPatch] = []
        for _ in range(count):
            rx = read_varint(f)
            ry = read_varint(f)
            rw = read_varint(f)
            rh = read_varint(f)
            rlen = read_varint(f)
            rdata = f.read(rlen)
            if len(rdata) != rlen:
                raise DyvFormatError("矩形载荷数据不完整")
            rect_list.append(RectPatch(x=rx, y=ry, w=rw, h=rh, payload=rdata))
        return RawFramePacket(
            frame_type=ftype,
            pts=pts,
            rects=rect_list,
        )

    raise DyvFormatError(f"未处理的帧类型 {ftype}")


# ---------------- 尾部与索引 ----------------

def write_tail(f: BinaryIO, total_frames: int, index_offset: int,
               index: Optional[List[IndexEntry]] = None) -> None:
    if index is not None:
        f.write(struct.pack("<I", len(index)))
        for e in index:
            f.write(struct.pack("<QQQ", e.ordinal, e.pts, e.offset))
    f.write(struct.pack("<QQ", total_frames, index_offset))
    f.write(FOOTER_MAGIC)


def read_tail(f: BinaryIO) -> Optional[Tail]:
    """从可 seek 文件末尾解析尾部。不可用返回 None。"""
    try:
        f.seek(-4, 2)
    except (OSError, ValueError):
        return None
    if f.read(4) != FOOTER_MAGIC:
        return None
    f.seek(-20, 2)
    total_frames, index_offset = struct.unpack("<QQ", f.read(16))
    index = None
    if index_offset:
        f.seek(index_offset)
        (count,) = struct.unpack("<I", f.read(4))
        index = []
        for _ in range(count):
            ordinal, pts, offset = struct.unpack("<QQQ", f.read(24))
            index.append(IndexEntry(ordinal, pts, offset))
    return Tail(total_frames, index_offset, index)
