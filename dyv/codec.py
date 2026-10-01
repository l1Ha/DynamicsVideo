"""瓦片载荷编解码：RAW / ZLIB / ZSTD / JPEG / WEBP。

无损编码（raw/zlib/zstd）作用于瓦片的原始字节；有损编码（jpeg/webp）
直接对瓦片图像编码，忽略变换。zstandard 为可选依赖。
"""

from __future__ import annotations

import io
import zlib
from typing import Tuple

import numpy as np

from .format import DyvError, TileCodec

try:
    import zstandard as _zstd
    HAS_ZSTD = True
except ImportError:  # pragma: no cover - 环境可选
    _zstd = None
    HAS_ZSTD = False

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:  # pragma: no cover - 环境可选
    _PILImage = None
    HAS_PIL = False


def _require_pil(codec_name: str) -> None:
    if not HAS_PIL:
        raise DyvError(f"编码 {codec_name} 需要 Pillow")


def compress_tile(codec: int, arr: np.ndarray, quality: int = 85) -> bytes:
    """把 HxWxC uint8 瓦片数组编码为载荷字节。"""
    if arr.dtype != np.uint8:
        raise DyvError("瓦片数组必须是 uint8")
    if codec == TileCodec.RAW:
        return arr.tobytes()
    if codec == TileCodec.ZLIB:
        return zlib.compress(arr.tobytes(), 6)
    if codec == TileCodec.ZSTD:
        if not HAS_ZSTD:
            raise DyvError("zstd 编码需要 zstandard 库（pip install zstandard）")
        return _zstd.ZstdCompressor(level=9).compress(arr.tobytes())
    if codec in (TileCodec.JPEG, TileCodec.WEBP):
        _require_pil("jpeg/webp")
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        img = _PILImage.fromarray(arr, mode)
        bio = io.BytesIO()
        if codec == TileCodec.JPEG:
            img.save(bio, "JPEG", quality=quality)
        else:
            img.save(bio, "WEBP", quality=quality)
        return bio.getvalue()
    raise DyvError(f"未知瓦片编码 {codec}")


def decompress_tile(codec: int, payload: bytes, shape: Tuple[int, int, int]) -> np.ndarray:
    """把载荷字节还原为 shape=(h,w,c) 的 uint8 数组。"""
    h, w, c = shape
    if codec == TileCodec.RAW:
        expected = h * w * c
        if len(payload) != expected:
            raise DyvError(f"raw 瓦片长度 {len(payload)} != 期望 {expected}")
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c).copy()
    if codec == TileCodec.ZLIB:
        data = zlib.decompress(payload)
        if len(data) != h * w * c:
            raise DyvError("zlib 瓦片解压后长度不符")
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, c).copy()
    if codec == TileCodec.ZSTD:
        if not HAS_ZSTD:
            raise DyvError("zstd 解码需要 zstandard 库")
        data = _zstd.ZstdDecompressor().decompress(payload)
        if len(data) != h * w * c:
            raise DyvError("zstd 瓦片解压后长度不符")
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, c).copy()
    if codec in (TileCodec.JPEG, TileCodec.WEBP):
        _require_pil("jpeg/webp")
        img = _PILImage.open(io.BytesIO(payload))
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape != (h, w, c):
            raise DyvError(f"有损瓦片尺寸 {arr.shape} != 期望 {(h, w, c)}")
        return arr.copy()
    raise DyvError(f"未知瓦片编码 {codec}")
