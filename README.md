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

# 格式互转：GIF 转换为 DyV (享受局部刷新与体积压缩)
python3 -m dyv convert animation.gif output.dyv

# 格式互转：DyV 导出为兼容 GIF 动图
python3 -m dyv convert input.dyv exported.gif

# 提取各帧保存为图片
python3 -m dyv dump recording.dyv -o output_frames/
```

---

## 🌐 Web 前端极速播放器 (HTML5 Canvas)

DyV 天然契合浏览器渲染引擎。项目内置了零额外转码依赖的 Web 播放器演示：
- 文件路径：`examples/player.html`
- **实现原理**：利用 HTML5 Canvas 2D 渲染上下文的原生 `putImageData(imageData, dx, dy)` API。当收到脏矩形增量帧时，浏览器仅将局部矩形覆盖到画布对应坐标，渲染延迟小于 0.2ms，CPU 占用极低！
- **体验方式**：双击或在本地静态服务器打开 `examples/player.html`，选择任何 `.dyv` 视频文件即可实时流畅播放。

---

## 🧪 单元测试与 CI 持续集成

项目包含完善的自动化测试套件与 GitHub Actions 多版本矩阵持续集成（Python 3.8 ~ 3.12）：
```bash
# 本地运行 17 项全量测试
python3 -m pytest tests -v
```

---

## 📄 开源许可与商业双重授权 (Dual Licensing)

DyV (Dynamic Video) 采用国际工业界成熟的**双重许可模式 (Dual-Licensing Model)**：

### 1. 开源许可：GNU AGPLv3 (免费)
本项目开源版本遵循 [GNU Affero General Public License v3 (AGPLv3)](LICENSE)。
- **适用对象**：个人学习、学术科研、开源项目，以及**愿意同样将自己全部商业/应用源码在 AGPLv3 下公开开源**的使用者。
- **开源传染约束**：AGPLv3 包含严格的网络服务传染条款（第 13 条）。任何组织或个人若在闭源商业产品、收费应用、SaaS 云服务或二次开发项目中集成、修改或通过网络提供 DyV 功能，**其自身软件的全部源代码亦必须依法强制向公众公开**。

### 2. 商业闭源许可：Commercial License (付费商用 & 改进商用免开源)
- **适用对象**：商业公司、闭源软件、收费 SaaS、企业级应用，以及**对本项目进行改进、定制、封装后对外销售或提供付费服务且不愿公开自身源码的商业实体**。
- **核心权益**：
  - ✅ **完全豁免 AGPLv3 开源传染义务**，无需公开任何商业专有代码；
  - ✅ 允许合法闭源商用，允许将改进、优化、定制后的版本直接用于商业营利；
  - ✅ 提供正规商业授权书与合同，符合企业上市法务与合规审计标准。
- **商业授权详情与获取途径**：请参阅详细说明文件 [COMMERCIAL.md](COMMERCIAL.md)，或直接联系版权所有者洽谈：[GitHub @l1Ha](https://github.com/l1Ha)。
