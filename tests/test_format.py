import io
import pytest

from dyv.format import (
    BOX_TITLE,
    DyvFormatError,
    FrameType,
    Header,
    IndexEntry,
    PixelFormat,
    RectPatch,
    Tail,
    TileCodec,
    Transform,
    read_boxes,
    read_frame_packet,
    read_header,
    read_tail,
    read_varint,
    write_boxes,
    write_delta_rects_packet,
    write_delta_tiles_packet,
    write_empty_packet,
    write_header,
    write_key_packet,
    write_tail,
    write_varint,
)


def test_varint_roundtrip():
    for val in [0, 1, 127, 128, 255, 300, 16384, 1_000_000, 2**32 - 1]:
        bio = io.BytesIO()
        write_varint(bio, val)
        bio.seek(0)
        assert read_varint(bio) == val


def test_header_and_boxes_roundtrip():
    bio = io.BytesIO()
    h = Header(
        width=1920,
        height=1080,
        tile_size=64,
        timescale=1000,
        pixel_format=PixelFormat.RGB24,
        tile_codec=TileCodec.ZLIB,
        transform=Transform.NONE,
        boxes=[(BOX_TITLE, "测试视频".encode("utf-8"))],
    )
    write_header(bio, h)
    bio.seek(0)
    h2 = read_header(bio)
    assert h2.width == 1920
    assert h2.height == 1080
    assert h2.tile_size == 64
    assert h2.timescale == 1000
    assert len(h2.boxes) == 1
    assert h2.boxes[0][1].decode("utf-8") == "测试视频"


def test_packets_roundtrip():
    bio = io.BytesIO()
    # 1. 关键帧 (2 个瓦片)
    write_key_packet(bio, pts=0, payloads=[b"tile0", b"tile1"])
    # 2. 瓦片局部刷新帧 (更新索引 1 的瓦片)
    write_delta_tiles_packet(bio, pts=33, indices=[1], payloads=[b"new_tile1"])
    # 3. 脏矩形局部刷新帧 (更新 10,20,30,40 的矩形)
    write_delta_rects_packet(
        bio,
        pts=66,
        rects=[RectPatch(x=10, y=20, w=30, h=40, payload=b"patch_data")],
    )
    # 4. 空帧
    write_empty_packet(bio, pts=100)

    # 5. 哨兵
    bio.write(b"\xFF")

    bio.seek(0)
    p0 = read_frame_packet(bio, total_tiles=2)
    assert p0.frame_type == FrameType.KEY
    assert p0.pts == 0
    assert p0.tile_payloads == [b"tile0", b"tile1"]

    p1 = read_frame_packet(bio, total_tiles=2)
    assert p1.frame_type == FrameType.DELTA_TILES
    assert p1.pts == 33
    assert p1.tile_indices == [1]
    assert p1.tile_payloads == [b"new_tile1"]

    p2 = read_frame_packet(bio, total_tiles=2)
    assert p2.frame_type == FrameType.DELTA_RECTS
    assert p2.pts == 66
    assert len(p2.rects) == 1
    assert p2.rects[0].x == 10
    assert p2.rects[0].y == 20
    assert p2.rects[0].w == 30
    assert p2.rects[0].h == 40
    assert p2.rects[0].payload == b"patch_data"

    p3 = read_frame_packet(bio, total_tiles=2)
    assert p3.frame_type == FrameType.EMPTY
    assert p3.pts == 100

    pend = read_frame_packet(bio, total_tiles=2)
    assert pend is None


def test_tail_and_index():
    bio = io.BytesIO()
    # 先写入一些占位数据
    bio.write(b"video_data_stream_content")
    idx = [IndexEntry(0, 0, 100), IndexEntry(30, 1000, 5000)]
    idx_offset = bio.tell()
    write_tail(bio, total_frames=60, index_offset=idx_offset, index=idx)

    tail = read_tail(bio)
    assert tail is not None
    assert tail.total_frames == 60
    assert len(tail.index) == 2
    assert tail.index[1].ordinal == 30
    assert tail.index[1].pts == 1000
