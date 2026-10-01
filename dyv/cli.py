"""DyV 命令行工具。

支持 info / dump / bench / encode 等命令。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List

import numpy as np

from .codec import HAS_PIL
from .decoder import DyvReader
from .encoder import DyvWriter
from .format import (
    CODEC_NAMES,
    NAME_TO_CODEC,
    NAME_TO_TRANSFORM,
    PIXEL_FORMAT_LABELS,
    PIXEL_FORMAT_NAMES,
    TRANSFORM_NAMES,
    PixelFormat,
    TileCodec,
    Transform,
)

if HAS_PIL:
    from PIL import Image


def cmd_info(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}", file=sys.stderr)
        return 1

    file_size = os.path.getsize(path)
    with DyvReader(path) as r:
        h = r.header
        print("=" * 50)
        print(f"DyV 视频信息: {os.path.basename(path)}")
        print("=" * 50)
        print(f"文件大小:      {file_size:,} 字节 ({file_size / 1024:.2f} KB)")
        print(f"分辨率:        {h.width} x {h.height}")
        print(f"瓦片大小:      {h.tile_size} x {h.tile_size} ({r.cols} 列 x {r.rows} 行, 共 {r.total_tiles} 块)")
        print(f"时间基 (基数): {h.timescale} ticks/秒")
        print(f"像素格式:      {PIXEL_FORMAT_LABELS.get(h.pixel_format, str(h.pixel_format))}")
        print(f"瓦片压缩:      {CODEC_NAMES.get(h.tile_codec, str(h.tile_codec))}")
        print(f"瓦片变换:      {TRANSFORM_NAMES.get(h.transform, str(h.transform))}")
        print(f"配置特征:      {'有损 (Lossy)' if h.profile == 1 else '无损 (Lossless)'}")
        print(f"元数据盒数:    {len(h.boxes)}")
        for b_type, b_data in h.boxes:
            print(f"  - 盒类型 0x{b_type:02X} (长度 {len(b_data)} 字节)")

        if r.tail:
            print("-" * 50)
            print(f"总帧数:        {r.tail.total_frames}")
            if r.tail.index:
                print(f"关键帧索引数:  {len(r.tail.index)}")
                for i, entry in enumerate(r.tail.index[:5]):
                    print(f"  * 关键帧 #{entry.ordinal}: PTS={entry.pts} ({entry.pts / h.timescale:.3f}s), 偏移={entry.offset}")
                if len(r.tail.index) > 5:
                    print(f"  ... 另有 {len(r.tail.index) - 5} 个关键帧")
        else:
            print("尾部索引:      无 (未写入尾部索引或流式不可寻址)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}", file=sys.stderr)
        return 1

    print(f"正在对 {path} 进行解码基准测试...")
    t0 = time.perf_counter()
    count = 0
    empty_frames = 0
    rect_frames = 0
    tile_frames = 0
    key_frames = 0
    total_dirty_area = 0

    with DyvReader(path) as r:
        total_screen_area = r.width * r.height
        for f in r.iter_frames():
            count += 1
            if f.is_key:
                key_frames += 1
                total_dirty_area += total_screen_area
            elif f.frame_type == 3:  # EMPTY
                empty_frames += 1
            elif f.frame_type == 2:  # DELTA_RECTS
                rect_frames += 1
                for rx, ry, rw, rh in f.changed_rects:
                    total_dirty_area += rw * rh
            elif f.frame_type == 1:  # DELTA_TILES
                tile_frames += 1
                total_dirty_area += len(f.changed_tiles) * (r.tile_size * r.tile_size)

    elapsed = time.perf_counter() - t0
    fps = count / elapsed if elapsed > 0 else float("inf")
    avg_ms = (elapsed / count * 1000) if count > 0 else 0
    print(f"解码完成: 共 {count} 帧, 耗时 {elapsed:.4f} 秒")
    print(f"平均速度: {fps:.1f} FPS (每帧平均 {avg_ms:.2f} ms)")
    print(f"帧构成统计:")
    print(f"  - 关键帧 (KEY):         {key_frames} 帧")
    print(f"  - 脏矩形刷新 (RECT):    {rect_frames} 帧")
    print(f"  - 瓦片局部刷新 (TILE):  {tile_frames} 帧")
    print(f"  - 零开销空帧 (EMPTY):   {empty_frames} 帧")
    if count > 0 and total_screen_area > 0:
        avg_dirty_ratio = (total_dirty_area / (count * total_screen_area)) * 100
        print(f"平均每帧更新面积比例:   {avg_dirty_ratio:.2f}% (局部刷新节省率 {(100 - avg_dirty_ratio):.2f}%)")
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    if not HAS_PIL:
        print("错误: dump 命令需要 Pillow 库支持", file=sys.stderr)
        return 1
    path = args.file
    out_dir = args.out_dir or "dumped_frames"
    os.makedirs(out_dir, exist_ok=True)

    print(f"导出帧至 {out_dir} ...")
    count = 0
    with DyvReader(path) as r:
        for f in r.iter_frames():
            out_name = os.path.join(out_dir, f"frame_{f.ordinal:05d}_pts_{f.pts}.png")
            img = Image.fromarray(f.image)
            img.save(out_name)
            count += 1
    print(f"成功导出 {count} 帧。")
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    if not HAS_PIL:
        print("错误: encode 命令需要 Pillow 库支持", file=sys.stderr)
        return 1

    inputs = sorted(glob.glob(args.input_pattern))
    if not inputs:
        print(f"未找到匹配的输入文件: {args.input_pattern}", file=sys.stderr)
        return 1

    first_img = Image.open(inputs[0]).convert("RGB")
    width, height = first_img.size
    tile_size = args.tile_size
    timescale = args.timescale
    codec = NAME_TO_CODEC.get(args.codec.lower(), TileCodec.ZLIB)
    transform = NAME_TO_TRANSFORM.get(args.transform.lower(), Transform.NONE)
    fps = args.fps
    step_pts = int(round(timescale / fps))

    print(f"正在编码 {len(inputs)} 帧到 {args.output} ...")
    print(f"尺寸: {width}x{height}, 瓦片: {tile_size}, 编码: {args.codec}, 变换: {args.transform}")

    with DyvWriter(
        args.output,
        width=width,
        height=height,
        tile_size=tile_size,
        timescale=timescale,
        tile_codec=codec,
        transform=transform,
        refresh_mode=args.refresh_mode,
        keyframe_interval=args.keyframe_interval,
    ) as w:
        pts = 0
        for i, fpath in enumerate(inputs):
            img = Image.open(fpath).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            w.append_frame(arr, pts=pts)
            pts += step_pts

    file_size = os.path.getsize(args.output)
    print(f"编码完成: 输出 {file_size:,} 字节 ({file_size / 1024:.2f} KB)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dyv",
        description="DyV (Dynamic Video) 命令行实用工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = subparsers.add_parser("info", help="查看 DyV 文件元数据与结构")
    p_info.add_argument("file", help="DyV 视频文件路径")

    # bench
    p_bench = subparsers.add_parser("bench", help="测试解码速度与性能")
    p_bench.add_argument("file", help="DyV 视频文件路径")

    # dump
    p_dump = subparsers.add_parser("dump", help="提取视频各帧保存为图片")
    p_dump.add_argument("file", help="DyV 视频文件路径")
    p_dump.add_argument("--out-dir", "-o", default=None, help="输出目录")

    # encode
    p_enc = subparsers.add_parser("encode", help="将图片序列编码为 DyV 文件")
    p_enc.add_argument("input_pattern", help="图片匹配通配符，如 'frames/*.png'")
    p_enc.add_argument("output", help="输出 DyV 文件路径")
    p_enc.add_argument("--tile-size", type=int, default=64, help="瓦片大小 (默认 64)")
    p_enc.add_argument("--fps", type=float, default=30.0, help="基准帧率 (默认 30)")
    p_enc.add_argument("--timescale", type=int, default=1000, help="时间基 (默认 1000)")
    p_enc.add_argument(
        "--codec",
        choices=["raw", "zlib", "zstd", "jpeg", "webp"],
        default="zlib",
        help="瓦片压缩格式 (默认 zlib)",
    )
    p_enc.add_argument(
        "--transform",
        choices=["none", "xor"],
        default="none",
        help="瓦片预处理变换 (默认 none)",
    )
    p_enc.add_argument(
        "--refresh-mode",
        choices=["auto", "rect", "tile"],
        default="auto",
        help="局部刷新模式 (默认 auto: 优先脏矩形刷新，分散变动切瓦片)",
    )
    p_enc.add_argument(
        "--keyframe-interval",
        type=int,
        default=60,
        help="关键帧间隔 (默认 60 帧)",
    )

    # convert
    p_conv = subparsers.add_parser("convert", help="将 GIF 与 DyV 进行相互转换")
    p_conv.add_argument("input", help="输入文件路径 (.gif 或 .dyv)")
    p_conv.add_argument("output", help="输出文件路径 (.dyv 或 .gif)")

    args = parser.parse_args(argv)
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "bench":
        return cmd_bench(args)
    elif args.command == "dump":
        return cmd_dump(args)
    elif args.command == "encode":
        return cmd_encode(args)
    elif args.command == "convert":
        from .converter import gif_to_dyv, dyv_to_gif
        in_path = args.input
        out_path = args.output
        if in_path.lower().endswith(".gif") and out_path.lower().endswith(".dyv"):
            n = gif_to_dyv(in_path, out_path)
            print(f"成功将 GIF ({n} 帧) 转换为 DyV: {out_path}")
            return 0
        elif in_path.lower().endswith(".dyv") and out_path.lower().endswith(".gif"):
            n = dyv_to_gif(in_path, out_path)
            print(f"成功将 DyV ({n} 帧) 转换为 GIF: {out_path}")
            return 0
        else:
            print("错误: 目前支持 .gif <=> .dyv 互转", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
