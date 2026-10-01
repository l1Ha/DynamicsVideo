import numpy as np
import pytest

from dyv.codec import HAS_PIL, HAS_ZSTD, compress_tile, decompress_tile
from dyv.format import TileCodec


def test_codec_raw():
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    payload = compress_tile(TileCodec.RAW, arr)
    assert len(payload) == 32 * 32 * 3
    rec = decompress_tile(TileCodec.RAW, payload, (32, 32, 3))
    assert np.array_equal(arr, rec)


def test_codec_zlib():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:20, 10:20] = 255  # 一些非零数据
    payload = compress_tile(TileCodec.ZLIB, arr)
    assert len(payload) < 64 * 64 * 3  # 大幅压缩
    rec = decompress_tile(TileCodec.ZLIB, payload, (64, 64, 3))
    assert np.array_equal(arr, rec)


@pytest.mark.skipif(not HAS_PIL, reason="需要 Pillow")
def test_codec_jpeg_webp():
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    jpeg_bytes = compress_tile(TileCodec.JPEG, arr, quality=90)
    rec_jpeg = decompress_tile(TileCodec.JPEG, jpeg_bytes, (32, 32, 3))
    assert rec_jpeg.shape == (32, 32, 3)

    webp_bytes = compress_tile(TileCodec.WEBP, arr, quality=90)
    rec_webp = decompress_tile(TileCodec.WEBP, webp_bytes, (32, 32, 3))
    assert rec_webp.shape == (32, 32, 3)
