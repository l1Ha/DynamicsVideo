"""DyV (Dynamic Video) - 支持动态帧率与局部刷新的高效轻量级视频格式。"""

from .codec import HAS_PIL, HAS_ZSTD, compress_tile, decompress_tile
from .decoder import DecodedFrame, DyvReader
from .encoder import DyvWriter
from .format import (
    CODEC_NAMES,
    NAME_TO_CODEC,
    NAME_TO_TRANSFORM,
    PIXEL_CHANNELS,
    PIXEL_FORMAT_LABELS,
    PIXEL_FORMAT_NAMES,
    TRANSFORM_NAMES,
    DyvError,
    DyvFormatError,
    FrameType,
    Header,
    IndexEntry,
    PixelFormat,
    Profile,
    RawFramePacket,
    RectPatch,
    Tail,
    TileCodec,
    Transform,
)
from .rects import compute_change_ratio, find_dirty_bbox
from .tiles import changed_tiles, grid_dims, tile_bounds, tile_count

__version__ = "1.0.0"

__all__ = [
    "DyvWriter",
    "DyvReader",
    "DecodedFrame",
    "PixelFormat",
    "TileCodec",
    "Transform",
    "FrameType",
    "Header",
    "Tail",
    "IndexEntry",
    "RectPatch",
    "RawFramePacket",
    "DyvError",
    "DyvFormatError",
    "find_dirty_bbox",
    "compute_change_ratio",
    "grid_dims",
    "tile_count",
    "tile_bounds",
    "changed_tiles",
    "compress_tile",
    "decompress_tile",
    "HAS_ZSTD",
    "HAS_PIL",
]
