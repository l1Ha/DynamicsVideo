# DyV (Dynamic Video)

**DyV** 是一种专为**动态帧率 (Variable Frame Rate, VFR)** 与**局部刷新 (Partial Area Refresh)** 优化设计的高效、轻量级现代视频格式。

---

## 🎯 解决的核心痛点

在**屏幕录制、UI/交互原型、代码终端、远程桌面、物联网监视屏、车机仪表盘**等场景中，传统视频编码（如 H.264/H.265/AV1）面临严重的冗余浪费：
1. **画面绝大部分静止**：当画面中只有局部（如鼠标移动、文字键入、局部闪烁）变化时，传统编码器仍需对全图宏块计算残差与预测，消耗大量计算和存储。
2. **缺乏真正的弹性动态帧率**：静止 10 秒时必须不断塞入重复帧或依靠复杂的容器时间戳适配；而突发高刷动效时又受限于固定帧率产生卡顿。
3. **解码算力门槛高**：无法让轻量级终端（如 Web 页面 Canvas、微控制器 MCU）直接提取“发生变化的局部矩形”进行局部贴图。

**DyV 的核心设计：**
- **局部刷新 (Partial Refresh)**：**当画面中只有一部分面积的内容变化时，在下一帧只用保存变化地方的画面数据**（通过脏矩形 `Dirty Rect` 或瓦片网格 `Tile Grid` 仅编码变动区域），解码端直接覆盖局部参考画布。
- **原生动态帧率 (Native VFR)**：基于 `Timescale`（时间基）与显式 `PTS`（显示时间戳），两帧间隔可从 1 毫秒自由跨越到数小时，画面静止时零开销挂起。
- **超高压缩比与极速解码**：在典型屏幕/交互录制场景中，局部刷新使体积缩减 **95% ~ 98%**，解码速度提升至 **5000+ FPS**！

---

## 📊 实测基准数据 (Benchmark)

在 640×480 分辨率、包含 100 帧（模拟键盘打字、停顿思考、鼠标移动，总计 10.69 秒）的典型场景下实测：

| 编码模式 | 文件大小 | 相对原数据 | 对比全帧节省率 | 解码吞吐速度 | 像素还原度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **原始未压缩 (Raw RGB)** | 87.89 MB | 100.0% | - | - | 100% |
| **全屏关键帧模式 (关闭局部刷新)** | 368.5 KB | 0.41% | 基准 | 1,177 FPS | 100% 无损 |
| **瓦片局部刷新 (Tile 64×64)** | 15.4 KB | 0.02% | **+95.8%** | 6,568 FPS | 100% 无损 |
| **脏矩形局部刷新 (Dirty Rect)** | **6.4 KB** | **0.01%** | **+98.3%** | **7,361 FPS** | **100% 无损** |

> 实测脚本位于 `examples/demo_benchmark.py`，随时可直接复现。

---

## 🏗️ 架构与二进制布局

DyV 文件采用紧凑、流式友好、自包含的二进制结构：

```
┌────────────────────────────────────────────────────────┐
│ Header (26 字节固定头 + TLV 元数据盒，0x00 结尾)          │
├────────────────────────────────────────────────────────┤
│ Frame 0: KEY 关键帧 (全屏基准画布)                      │
│ Frame 1..n: 增量帧                                      │
│   - DELTA_RECTS (0x02): 脏矩形刷新 (x, y, w, h + 局部数据) │
│   - DELTA_TILES (0x01): 瓦片网格局部刷新                 │
│   - EMPTY       (0x03): 零图像空帧 (仅步进 PTS，2字节)   │
├────────────────────────────────────────────────────────┤
│ 0xFF (帧流终止哨兵字节)                                │
│ Trailer 元数据盒 (TLV 结构，包含总时长等信息)           │
├────────────────────────────────────────────────────────┤
│ [可选] 关键帧索引表 (Keyframe Index Table)              │
│ Tail 尾部 (总帧数 u64, 索引偏移 u64, 'DYVE' 4字节魔数)   │
└────────────────────────────────────────────────────────┘
```

完整二进制技术规范详见 [docs/SPEC.md](docs/SPEC.md)。

---

## 🚀 快速上手

### 1. 安装与依赖
- Python 3.8+
- numpy
- Pillow（可选，用于图像导入导出及 JPEG/WEBP 有损瓦片编码）
- zstandard（可选，启用更高效的 zstd 瓦片压缩）

### 2. Python API

#### 编码写入 (DyvWriter)
```python
import numpy as np
from dyv import DyvWriter, TileCodec, Transform

# 创建 Dyv 编码器
with DyvWriter(
    "recording.dyv",
    width=1920,
    height=1080,
    timescale=1000,           # 1 tick = 1 毫秒
    refresh_mode="rect",       # 自动使用脏矩形局部刷新
    tile_codec=TileCodec.ZLIB, # 瓦片/矩形压缩器
    transform=Transform.XOR_DELTA, # 开启 XOR 残差变换进一步压缩
    keyframe_interval=60,     # 每 60 帧写入一个关键帧，便于随机 seek
) as writer:
    # 写入第 1 帧 (关键帧，0.0 秒)
    writer.append_time(first_frame_rgb, seconds=0.0)

    # 写入第 2 帧 (只有输入框局部变动，在 0.25 秒触发)
    # 此时编码器仅提取变动矩形并压缩写入，其余全屏区域不占存储！
    writer.append_time(second_frame_rgb, seconds=0.25)

    # 写入第 3 帧 (经过 2 秒停顿，鼠标移动)
    writer.append_time(third_frame_rgb, seconds=2.25)
```

#### 解码读取 (DyvReader)
```python
from dyv import DyvReader

with DyvReader("recording.dyv") as reader:
    print(f"分辨率: {reader.width}x{reader.height}, 总帧数: {reader.total_frames}")

    # 1. 顺序流式解码
    for frame in reader.iter_frames():
        print(f"帧序号: {frame.ordinal}, 时间戳: {frame.timestamp:.3f}s, 是否关键帧: {frame.is_key}")
        # frame.image 即为合并局部刷新后的完整画面 (H, W, 3)

    # 2. 毫秒级随机精准跳转 (基于尾部关键帧索引)
    target_frame = reader.seek_time(1.50)  # 定位到 1.5 秒
```

### 3. 命令行工具 (CLI)

```bash
# 查看 DyV 视频元数据与结构
python3 -m dyv info recording.dyv

# 测试解码性能与局部刷新统计
python3 -m dyv bench recording.dyv

# 将图片序列编码为 DyV
python3 -m dyv encode "frames/*.png" output.dyv --fps 30 --codec zlib

# 提取各帧保存为图片
python3 -m dyv dump recording.dyv -o output_frames/
```

---

## 🧪 单元测试

项目包含完善的测试套件，覆盖二进制格式协议、编解码、脏矩形检测、瓦片网格、动态帧率与 CLI：
```bash
python3 -m pytest tests -v
```

---

## 📄 开源许可与商业授权条款 (License)

本项目采用 **DyV Source-Available & Commercial License v1.0**（源码可用与商业授权协议）：

1. **非商业与学术用途（免费开源）**：
   - 个人学习、学术科研、非盈利开源项目及技术评估完全免费，允许自由阅读、修改与运行。
2. **商业使用与二次开发商用限制（强制收费与商业授权）**：
   - **直接商用收费**：任何将本项目用于商业产品、收费应用、SaaS 云服务或商业盈利活动，**必须事先向原作者购买正式商业授权并支付授权费用**。
   - **改进后商用必须收费**：任何基于本项目进行的修改、改进、定制、衍生开发或 API 封装后的版本，若用于商业销售、商业部署或提供付费服务，**同样必须获得原作者商业授权并支付商业授权费用**。
   - 商业授权洽谈请联系：[GitHub @l1Ha](https://github.com/l1Ha)。

完整协议细节请参阅项目根目录下的 [LICENSE](LICENSE) 文件。
