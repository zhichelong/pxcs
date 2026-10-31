# PXC 码流规格（payload 层）

本文描述 PXCS 归档中**每一条 payload 的内部字节布局**，即 `codec` 为 `pxc` /
`pxc_residual` / `zlib_residual` 时那一段字节究竟是什么。

- 配套文档：`FORMAT.md` 描述**容器层**（PXCS header / index / 加密 / 分卷）。
- 对应实现：PXCS 1.3 / PXC 0.2（`pxc_core`）。
- 字节序：**除特别说明外，所有整数均为小端序**。
- 本文的状态：**已冻结**。任何改变字节布局的修改都必须提升 `PXC` 小版本号，
  并同时更新本文与黄金向量集（`tests/fixtures/golden/`）。

编写本文的目的很具体：**让一份全新的实现（例如 C++）只读本文 + 黄金向量
就能写出字节级兼容的编解码器，不需要阅读 Python 源码。**
凡本文没有写清、必须去看 Python 实现才能确定的行为，都属于本文的缺陷。

---

## 0. 术语

| 术语 | 含义 |
|---|---|
| 样本 / sample | 一个通道上的一个数值，如 `uint8` 的 0..255 |
| 通道 / channel | 一个像素的分量数：`L=1`、`LA=2`、`RGB=3`、`RGBA=4` |
| 像素 / pixel | 同一位置上的 `C` 个样本 |
| tile | 图像按 `tile_size` 切出的矩形块，`(H,W,C)`，右/下边界可能不足一格 |
| 残差 / residual | 差分编码时，“当前图”与“参考图”逐样本相减（模 2^n）的结果 |
| 存储类型 | 样本在 PXC 中序列化时使用的类型，见 §1.2 |

---

## 1. 数据模型

### 1.1 图像模式

PXC 只支持下列 12 种 Pillow 模式，`mode_id` 为 `u16`：

| id | mode | C | 位深 | 颜色模型 | alpha | numpy dtype |
|---:|---|---:|---:|---|---|---|
| 1 | `1` | 1 | 1 | 灰度 | 无 | `uint8`（值只有 0/1） |
| 2 | `L` | 1 | 8 | 灰度 | 无 | `uint8` |
| 3 | `LA` | 2 | 8 | 灰度+alpha | 直通 | `uint8` |
| 4 | `P` | 1 | 8 | 调色板 | 无 | `uint8` |
| 5 | `RGB` | 3 | 8 | RGB | 无 | `uint8` |
| 6 | `RGBA` | 4 | 8 | RGBA | 直通 | `uint8` |
| 7 | `CMYK` | 4 | 8 | CMYK | 无 | `uint8` |
| 8 | `I` | 1 | 32 | 整数 | 无 | `int32` |
| 9 | `I;16` | 1 | 16 | 灰度 | 无 | `uint16` |
| 10 | `I;16L` | 1 | 16 | 灰度 | 无 | `uint16` |
| 11 | `I;16B` | 1 | 16 | 灰度 | 无 | `uint16` |
| 12 | `F` | 1 | 32 | 浮点 | 无 | `float32` |

颜色模型编号：`1=灰度`、`2=灰度+alpha`、`3=调色板`、`4=RGB`、`5=RGBA`、
`6=CMYK`、`7=整数`、`8=浮点`。

alpha 类型编号：`0=无`、`1=直通`、`2=调色板透明度`、`3=常量 alpha`。

**注意 `I;16B`**：源文件是大端，但解码后样本在内存中是主机字节序的 `uint16`，
序列化时按小端（见 §1.2）。`I;16` 与 `I;16L` 同义。

### 1.2 样本序列化规则（`storage_dtype`）

PXC 内部**多字节样本一律小端**：

| 样本类型 | 存储类型 | 每样本字节 |
|---|---|---:|
| `uint8` | `uint8` | 1 |
| `uint16` | `<u2` | 2 |
| `uint32` | `<u4` | 4 |
| `int32` | `<i4` | 4 |
| `float32` | `<f4` | 4 |

本节规则适用于**所有** tile payload、调色板条目、SOLID 颜色、RUN 颜色。

唯一的例外是像素哈希里的样本序列化，见 §7.1 —— 那里用的是**主机字节序**，
这是一个已识别但为了向后兼容而保留的缺陷，见 §10.2。

---

## 2. PXC 文件布局

一条 `pxc` payload 就是一个完整的 PXC 文件：

```
偏移 0    ┌────────────────────────────┐
          │ Header（128 字节）          │
          ├────────────────────────────┤
          │ Metadata 块                │  长度由 header.metadata_length 给出
          ├────────────────────────────┤
          │ Chunk 目录（32 × N 字节）   │  N = header.chunk_count
          ├────────────────────────────┤
          │ Chunk 0 payload            │  META
          │ Chunk 1 payload            │  DATA
          │ Chunk 2 payload（可选）     │  ORIG
          ├────────────────────────────┤
          │ Footer（16 字节）           │
          └────────────────────────────┘
```

**块之间没有对齐填充。** 这一点与 PXCS 容器层不同（容器层的 payload 起点按
8 字节对齐），PXC 内部是紧凑排列。

`metadata_offset` 恒为 128；`chunk_dir_offset = 128 + metadata_length`；
第一个 chunk payload 的偏移 = `chunk_dir_offset + 32 * chunk_count`。

### 2.1 Header（128 字节）

结构：`<4sBBHIIQQQQIIIHHHHHHIQQ32sI`

| 偏移 | 大小 | 类型 | 字段 | 说明 |
|---:|---:|---|---|---|
| 0 | 4 | `4s` | magic | `PXC1`（`0x31435850`） |
| 4 | 1 | `B` | version_major | 当前为 **0** |
| 5 | 1 | `B` | version_minor | 当前为 **2** |
| 6 | 2 | `H` | flags | 见下 |
| 8 | 4 | `I` | header_size | 恒为 128 |
| 12 | 4 | `I` | header_crc | CRC32，**本字段按 0 计算** |
| 16 | 8 | `Q` | file_size | 整个 PXC 文件字节数 |
| 24 | 8 | `Q` | metadata_offset | 恒为 128 |
| 32 | 8 | `Q` | metadata_length | Metadata 块字节数 |
| 40 | 8 | `Q` | chunk_dir_offset | `128 + metadata_length` |
| 48 | 4 | `I` | chunk_count | 2 或 3 |
| 52 | 4 | `I` | width | 图像宽 |
| 56 | 4 | `I` | height | 图像高 |
| 60 | 2 | `H` | mode_id | 见 §1.1 |
| 62 | 2 | `H` | bit_depth | 见 §1.1 |
| 64 | 2 | `H` | channels | **实际存储**的通道数（可能比 mode 少 1） |
| 66 | 2 | `H` | color_model | 见 §1.1 |
| 68 | 2 | `H` | alpha_kind | 见 §1.1 |
| 70 | 2 | `H` | compression_id | 1=tile 流，2=passthrough |
| 72 | 4 | `I` | tile_size | compression_id=1 时有效；=2 时必须为 0 |
| 76 | 8 | `Q` | original_size | 源文件字节数（0 表示未知） |
| 84 | 8 | `Q` | data_length | DATA chunk 的长度 |
| 92 | 32 | `32s` | pixel_hash | 见 §7.1 |
| 124 | 4 | `I` | reserved | 必须为 0 |

`flags` 位定义：

| 位 | 名称 | 含义 |
|---:|---|---|
| 0 | `FLAG_HAS_ORIG` | 存在 ORIG chunk |
| 1 | `FLAG_HAS_ALPHA` | 图像带 alpha |
| 2 | `FLAG_HAS_PALETTE` | 图像带调色板 |
| 3 | `FLAG_SOURCE_LOSSY` | 源格式是有损的（JPEG / WebP） |
| 4 | `FLAG_PASSTHROUGH` | DATA 是原始文件字节 |

**header_crc 计算方式**：复制前 128 字节，把 `[12:16]` 置零，对其做 CRC32。

**`channels` 与 `mode` 的关系**：正常情况下 `channels == mode 的通道数`。
当 `alpha_kind == 3`（常量 alpha）时，`channels == mode 通道数 - 1`，
并且 Metadata 里必须有 `constant_alpha` 键（§2.2），解码时用该常量把 alpha
平面补回。此机制只对 `RGBA`（存 3 通道）与 `LA`（存 1 通道）有效。

### 2.2 Metadata 块

```
u32 count
重复 count 次（按 key 升序）：
  u16 key
  u8  type
  u8  reserved（必须为 0）
  u32 length
  bytes[length] payload
```

结构：起始 `u32 count`，每条 `u16 key, u8 type, u8 reserved, u32 length`（8 字节头）。

`type` 取值与长度约束：

| type | 名称 | 长度 |
|---:|---|---|
| 1 | bytes | 任意，≤ 64 MiB |
| 2 | utf8 | 任意，≤ 64 MiB |
| 3 | u32 | 恰好 4 |
| 4 | u64 | 恰好 8 |
| 5 | f64×2 | 恰好 16 |
| 6 | f64 | 恰好 8 |

键表：

| key | 名称 | 类型 | 何时写入 |
|---:|---|---|---|
| 1 | `original_filename` | utf8 | 总是 |
| 2 | `source_format` | utf8 | 总是 |
| 3 | `source_ext` | utf8 | 总是 |
| 4 | `mode_name` | utf8 | 总是 |
| 5 | `palette` | bytes | `P` 模式 |
| 6 | `transparency` | bytes | 有透明度信息时 |
| 7 | `icc_profile` | bytes | 有时 |
| 8 | `exif` | bytes | 有时 |
| 9 | `xmp` | bytes | 有时 |
| 10 | `dpi` | f64×2 | 有时 |
| 11 | `entropy_name` | utf8 | 总是，值 `zlib` |
| 12 | `entropy_version` | utf8 | 总是，值为 zlib 运行时版本 |
| 13 | `created_by` | utf8 | 总是 |
| 14 | `source_file_hash` | bytes | 有时（32 字节 sha256） |
| 15 | `source_pixel_hash` | bytes | 有时（32 字节 sha256） |
| 16 | `gamma` | f64 | 有时 |
| 17 | `comment` | bytes | 有时 |
| 18 | `constant_alpha` | u32 | `alpha_kind == 3` 时必填 |

`transparency` 的 payload 自带标签：

| 标签 | 布局 | 含义 |
|---:|---|---|
| `0x00` | `u32` | 单索引透明度（`P` 模式） |
| `0x01` | 其余字节 | `P` 模式 alpha 表 |
| `0x02` | `u32 × 3` | RGB 颜色键 |

**`entropy_version` 会削弱字节确定性**：它记录的是写归档时的 zlib 版本字符串。
两份实现使用不同 zlib 版本时，即使码流完全一致，这个字段也不同。
因此差分比对必须把该字段视为**可忽略差异**（见 §8.3）。

### 2.3 Chunk 目录

每条 32 字节：`<4sIQQII`

| 偏移 | 大小 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 4 | chunk_id | `META` / `DATA` / `ORIG` |
| 4 | 4 | flags | 恒为 0 |
| 8 | 8 | offset | payload 绝对偏移 |
| 16 | 8 | length | payload 字节数 |
| 24 | 4 | crc | CRC32(payload) |
| 28 | 4 | reserved | 必须为 0 |

Chunk 顺序固定为 `META`、`DATA`、可选 `ORIG`。PXCS 只写 `META` 与 `DATA`；
`ORIG` 为历史预留，当前读取实现必须容忍其存在但可以忽略。

### 2.4 Footer（16 字节）

结构：`<8sII`

| 偏移 | 大小 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 8 | magic | `PXCEND\0\0` |
| 8 | 4 | file_crc | CRC32（从文件头到 footer 之前的所有字节） |
| 12 | 4 | reserved | 必须为 0 |

### 2.5 两种 `compression_id`

| 值 | 名称 | DATA chunk 内容 |
|---:|---|---|
| 1 | tile 流 | 按 §3 的结构编码的 tile 序列 |
| 2 | passthrough | **原始图片文件的完整字节**（JPEG/WebP 等） |

passthrough 时 `tile_size` 必须为 0，且解码后必须满足：
`mode`、`width`、`height` 与 header 一致，`pixel_hash` 与 header 一致
（§7.1）。PXCS 的 `raw` codec 在归档层直接保存原始字节，**不**使用 PXC 的
passthrough 封装；这条路径主要为历史兼容保留。

---

## 3. DATA chunk：tile 流

### 3.1 DATA 头（20 字节）

结构：`<IIIII`

| 偏移 | 字段 | 说明 |
|---:|---|---|
| 0 | version | 恒为 **1** |
| 4 | tile_size | 8..1024 |
| 8 | tiles_x | `ceil(width / tile_size)` |
| 12 | tiles_y | `ceil(height / tile_size)` |
| 16 | tile_count | 必须等于 `tiles_x × tiles_y` |

### 3.2 tile 划分与顺序

tile 按**行优先**编号：`index = ty * tiles_x + tx`。

```
x0 = tx * tile_size          y0 = ty * tile_size
x1 = min(x0 + tile_size, width)
y1 = min(y0 + tile_size, height)
tw = x1 - x0                 th = y1 - y0
```

右侧与底部边界的 tile 尺寸可以小于 `tile_size`。每个 tile 独立编码，
**互不参考**（预测与滤波都在 tile 内部闭合，跨 tile 边界不连续）。

### 3.3 tile 记录

紧随 DATA 头之后，连续 `tile_count` 条记录，每条记录 = 20 字节头 + payload。

头结构：`<BBHIIII`

| 偏移 | 大小 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 1 | mode | 见 §4 |
| 1 | 1 | submode | 见 §4 |
| 2 | 2 | flags | bit0 = payload 已 zlib 压缩 |
| 4 | 4 | raw_len | **解压后**的 payload 字节数 |
| 8 | 4 | comp_len | 实际存储的字节数 |
| 12 | 4 | raw_crc | CRC32(解压后的 payload) |
| 16 | 4 | payload_crc | CRC32(实际存储的字节) |

payload 存储规则（编码器）：

```
comp = zlib.compress(raw, level=6)
if len(comp) < len(raw):  stored, flags = comp, 1
else:                     stored, flags = raw,  0
```

`raw_len != comp_len` 并不能推断是否压缩，必须看 `flags`。

### 3.4 编码器候选与选择规则（字节兼容的关键）

**要做到与参考实现字节一致，必须完整复现本节的选择逻辑**，包括候选集合的
准入条件与平局规则。任何一处不同都会导致 tile 模式不同、进而字节不同。

对一个 tile 依次构造候选（顺序即下文的 1→6）：

**候选 1 —— SOLID**：条件 `tile 的所有样本都等于 tile[0,0]`。

**候选 2 —— PALETTE**：条件是
- 样本类型不是浮点，且
- `unique_count(tile[::4, ::4, :]) <= 256`

其中 `tile[::4, ::4, :]` 表示 **行列都每隔 4 个取一个**（即 1/16 采样），
`unique_count` 统计不同的**像素**（不是通道）个数。这个预筛是有意的近似：
真实颜色数 ≤ 256 但采样后 > 256 的 tile 会被跳过 PALETTE，这是参考实现的行为，
**不是**可以“优化”的地方。

若通过预筛且真实调色板 `K` 满足 `1 < K <= 256`，则**同时**产生两个子候选：
`PALETTE(submode=0, PLAIN)` 与 `PALETTE(submode=1, RLE)`。

**候选 3 —— PRED**：条件为非浮点类型。

**候选 4 —— RUN**：条件为非浮点且
```
flat = tile 按行优先展平为 (N, C)
runs = 1 + count(flat[i] != flat[i-1] for i in 1..N-1)
runs <= max(8, N // 4)
```

**候选 5 —— FILTER**：条件为样本类型是 `uint8` 或 `uint16`。

**候选 6 —— RAW**：无条件。

从候选中选成本最小者：

```
cost(rec) = (20 + rec.comp_len, MODE_RANK[rec.mode], rec.submode)
```

其中 `20` 是 tile 头长度，`MODE_RANK` 为固定顺序索引：

```
MODE_RANK = { SOLID:0, PALETTE:1, PRED:2, RUN:3, FILTER:5→4, RAW:4→5 }
```

准确地说，`MODE_RANK` 是列表 `[SOLID, PALETTE, PRED, RUN, FILTER, RAW]` 中的
下标，即 `SOLID=0, PALETTE=1, PRED=2, RUN=3, FILTER=4, RAW=5`。

比较是**三元组字典序**：长度相同时比模式排名，再相同时比 submode。
因此 PALETTE 的两个子候选在成本相同时选 PLAIN（`submode=0`），
而与 RAW 等价成本时优先一切结构化模式。

---

## 4. tile 模式

所有模式的 payload 都是**未压缩**字节；zlib 由 §3.3 统一处理。
下文 `sd` 表示存储类型（§1.2），`itemsize` 为其字节宽度。

### 4.1 RAW（mode = 4，submode = 0）

payload = tile 的 `H × W × C` 个样本，行优先、通道内联、小端。

`raw_len` 必须等于 `H*W*C*itemsize`。

### 4.2 SOLID（mode = 0，submode = 0）

payload = `tile[0,0]` 的 `C` 个样本（小端）。

`raw_len` 必须等于 `C*itemsize`。解码：把该颜色平铺成 `(H,W,C)`。

### 4.3 PALETTE（mode = 1）

payload：

```
u16  K                    # 调色板条目数，1..256
K × C 个样本              # 调色板本体，每行一个像素颜色
索引体                    # 长度取决于 submode
```

**调色板排序规则（必须精确复现）**：

- `C == 1` 时：按**数值升序**（`uint16` 的 1 排在 256 前面，不是按字节序）。
- `C > 1` 时：把每个像素的 `C` 个样本按**小端字节**串接成 `C*itemsize` 字节，
  按**字节字典序**（memcmp 语义、无符号）升序排列。

这两条是**不同的机制**，但在 PXCS 实际支持的模式上结果一致：`C > 1` 的模式
（`LA`/`RGB`/`RGBA`/`CMYK`）全部是 8 位，此时"字节字典序"等价于"按通道顺序
依次比较通道值的无符号大小"。真正需要注意的只有 `C == 1` 的 16 位模式：
必须用**数值序**，不要退化成字节序。

举例（`C == 3`，8 位）：颜色 `(1,2,3)`、`(0,255,0)`、`(255,0,0)`、`(0,0,255)`
的排序结果是 `(0,0,255) < (0,255,0) < (1,2,3) < (255,0,0)`。

**submode = 0（PLAIN）**：索引体 = `H*W` 个 `u8`，即每个像素在调色板中的下标，
行优先。`raw_len = 2 + K*C*itemsize + H*W`。

**submode = 1（RLE）**：索引体 = 若干 `(u32 run_length, u8 palette_index)` 对，
按顺序铺满 `H*W` 个像素。没有结束标记、没有显式条目数；解码时读取直到铺满
`H*W` 个像素，且必须正好消耗完整个索引体。
`run_length` 不得为 0，也不得越过剩余像素数。

解码时若出现 `index >= K`，视为损坏。

### 4.4 PRED（mode = 2，submode = 0）

样本先做 MED（LOCO-I）预测，再对残差做 zigzag，最后以无符号类型存储。

**残差存储类型**（注意：比样本类型宽一倍）

| 样本类型 | 残差存储类型 | 每残差字节 |
|---|---|---:|
| `uint8` | `<u2` | 2 |
| `uint16` | `<u4` | 4 |
| `int32` | `<u8` | 8 |

`raw_len` 必须等于 `H*W*C * 残差宽度`。

**MED 预测**（逐样本、逐通道独立；利用左 `a`、上 `b`、左上 `c`）：

```
if c >= max(a,b): p = min(a,b)
elif c <= min(a,b): p = max(a,b)
else: p = a + b - c
```

**边界规则**（tile 内部闭合，必须严格按此）：

| 位置 | 预测值 |
|---|---|
| `(0,0)` | `0` |
| 第一行 `(0,x>0)` | `a = x(0,x-1)` |
| 第一列 `(y>0,0)` | `b = x(y-1,0)` |
| 其余 | 上式 MED |

**zigzag**：

```
编码：v >= 0 → u = v << 1
      v < 0  → u = ((-v) << 1) - 1
解码：u 偶 → v = u >> 1
      u 奇 → v = -((u >> 1) + 1)
```

若某个残差超出存储类型的表示范围，该 tile 不能使用 PRED（编码器应放弃该候选）。
解码时若重建出的样本超出样本类型范围，视为损坏。

### 4.5 RUN（mode = 3，submode = 0）

payload = 若干 `(u32 run_length, C 个样本)` 对，按行优先铺满 `H*W` 个像素。
没有结束标记；解码读取直到铺满，且必须正好消耗完 payload。
`run_length` 不得为 0，也不得越界。

注意与 PALETTE-RLE 的差异：RUN 的每个 run 携带**完整颜色**（`C*itemsize` 字节），
PALETTE-RLE 只携带**1 字节索引**。

### 4.6 FILTER（mode = 5，submode = 0）

payload：

```
H 个字节的逐行滤波类型      # 每个字节是 0..4
H*W*C 个样本                # 逐行滤波后的样本，小端
```

`raw_len` 必须等于 `H + H*W*C*itemsize`。

**滤波类型**

| 值 | 名称 | 基准值 base |
|---:|---|---|
| 0 | NONE | 0 |
| 1 | SUB | 左样本 `a` |
| 2 | UP | 上样本 `b` |
| 3 | AVG | `(a + b) // 2` |
| 4 | PAETH | Paeth 预测子 |

滤波与逆滤波都在模 `2^bit` 下进行（`uint8` 为 256，`uint16` 为 65536）：

```
滤波：   out = (x - base) mod 2^bit
逆滤波： x   = (f + base) mod 2^bit
```

边界：`a`/`b`/`c` 在越界位置取 0。

Paeth 预测子：

```
p  = a + b - c
pa = |p - a|, pb = |p - b|, pc = |p - c|
if pa <= pb and pa <= pc: base = a
elif pb <= pc:            base = b
else:                     base = c
```

**编码器只使用 0/1/2**（NONE/SUB/UP）；3/4 必须能解码（历史兼容），但不会写出。

**逐行选择规则（必须精确复现）**：对 `ft` 依次取 `0, 1, 2`（**这个顺序**），
计算整 tile 的滤波结果，然后对每一行单独算分数：

```
score(row) = Σ |signed(row[i])|,  signed(v) = v - 2^bit  if v >= 2^bit / 2 else v
```

即把滤波后的值当作有符号数解读后取绝对值求和。每行选择**分数最小**的类型，
比较用严格小于，因此**分数相同时保留先遍历到的类型**（`ft` 顺序靠前者优先，
即 NONE 优先于 SUB，SUB 优先于 UP）。

---

## 5. 残差层（PXCS 差分）

`pxc` 与 `pxc_residual` 的区别只在**语义**：前者是完整图像，后者是残差图像。
两者的字节结构完全相同（都是 §2–§4 描述的一个完整 PXC 文件）。

### 5.1 差分定义（`make_residual`）

要求参考图与当前图 `mode`、`width`、`height` 完全一致。

| mode / 样本类型 | 差分运算 |
|---|---|
| `1`（1 位） | **逐样本 XOR**（不是相减） |
| `uint8` | `(cur - ref) mod 256` |
| `uint16` | `(cur - ref) mod 65536` |
| 其他 | 不支持差分 |

差分结果仍是样本类型；因此 `0` 表示“相同”，`255`（uint8）表示“差 -1”。

### 5.2 居中（centered）

无损路径下，差分会被整体平移，使“无差异”落在 128：

```
centered = (diff + 128) mod 256        # 仅 uint8
```

`uint16` **不做居中**（`center_residual` 对非 uint8 直接返回原值）。

条目的 `centered` 字段记录是否做过这一步，解码时必须据此还原。

### 5.3 量化（quant > 0，有损）

量化与居中是**同一步**完成的：量化 LUT 直接输出“居中后的值”。

对每个 `u8` 值 `b`：

```
signed    = b if b < 128 else b - 256
q         = max(1, quant)
quantized = sign(signed) * (|signed| // q * q)
out       = (quantized + 128) & 0xFF
```

即：先做朝零取整的量化，再加 128 居中。因此 `0 → 128`、`255(-1) → 127`、
`1 → 129`。

`quant = 0` 时不做量化（等价于纯无损）。量化只对 `uint8` 有效；
`uint16` 残差永远是像素无损。

条目里 `quant > 0` 时必须同时满足 `centered == true`，且 `lossy == true`。

### 5.4 `zlib_residual`

payload = `zlib.compress(residual_samples, level)`，其中 `residual_samples` 是
残差图的 `H*W*C` 个 `u8` 样本（行优先、小端/单字节），未做任何额外包装。

解压后的长度必须等于 `width * height * channels`，否则视为损坏。

这条路径只用于 `uint8` 残差，且是 PXCS 对 JPEG 差分的主力路径
（量化为 16 时，残差里大多数字节是 128，zlib 很容易压缩）。

### 5.5 `pxc_residual`

payload 是按 §2 打包的完整 PXC 文件，其 DATA tile 编码的是**残差图**。
残差图允许经历 §5.2 的居中或 §5.3 的量化。

### 5.6 重建

```
residual = decode(pxc_residual | zlib_residual)

if mode == "1":
    pixels = ref XOR residual                       # 逐样本
elif uint8:
    pixels = (ref + residual) mod 256
    if centered: pixels = (pixels - 128) mod 256    # 等价于 ref + (centered-128)
elif uint16:
    pixels = (ref + residual) mod 65536
```

因为差值是在模 `2^bit` 下计算的，重建不需要知道符号，也不会有溢出分支。

---

## 6. 编码决策（不在字节层，但影响产物）

以下不属于“格式”，但决定一份实现能否产出与参考实现相同的归档。列出以供
差分测试定位问题。

1. **关键帧 vs 差分**：组内第一张必为关键帧；之后每张按参考策略比较误差，
   只有差分“划算”时才用差分，否则用 `raw`（直接存原始文件字节）。
2. **回退阈值**：关键帧的像素编码结果若 `>= 源文件大小 × 1.02`，回退 `raw`。
3. **零差分比例保护**：`zlib_residual` 路径下，若量化后 `等于 128 的比例`
   低于配置阈值（默认 0.40），直接放弃差分并回退。
4. **depth 上限**：连续差分深度不得超过关键帧间隔（默认 16）。

5. **raw 直通不解码**：组内只有一张图（不可能做差分）且走 raw 直通时，
   参考实现不再完整解码该图，条目的 `pixel_sha256` 为空、`constant_alpha`
   为空、`reason` 为 `key_lossy_passthrough`。组内多于一张时必须解码出参考图，
   此时 `pixel_sha256` 写进索引，`reason` 可能是 `decode_error: …`。
   这个差异只影响索引里的可选字段，**payload 字节完全相同**。

这些阈值都写进了条目的 `reason` 字段，便于事后解释“为什么这张图没压下去”。

> `reason` 只是诊断信息，不属于格式契约；第二实现可以写不同的值，
> 差分比对时把它当作可忽略差异。

---

## 7. 哈希与校验

### 7.1 `canonical_pixel_hash`

```
h = SHA-256()
h.update(b"PXC-PIXELS\x00")          # 11 字节
h.update(mode.encode("utf-8"))       # ASCII，如 "RGB" / "I;16"
h.update(LE32(width))
h.update(LE32(height))
h.update(samples)                    # H*W*C 个样本，行优先，主机字节序
digest = h.finalize()                # 32 字节
```

`samples` 是 `channels` 个通道的完整像素（常量 alpha 情况下**包含被省略的 alpha
平面**——编码时在丢弃 alpha 之前算，解码时在补回 alpha 之后算）。

**该哈希覆盖的样本使用主机字节序**（`ndarray.tobytes()` 的语义），而不是 §1.2
规定的小端。在 x86-64 上两者一致；在大端主机上 `uint16` 图像的哈希会不同。
这是已知缺陷，见 §10.2。

### 7.2 三层校验

| 层 | 校验对象 | 存放位置 |
|---|---|---|
| PXCS 层 | payload 明文 | index 的 `data_crc32` / `data_sha256` |
| PXC chunk 层 | 每个 chunk | chunk 目录的 `crc` |
| tile 层 | 每 tile 的压缩前后字节 | tile 头 `payload_crc` / `raw_crc` |
| 图像层 | 重建后的像素 | header 的 `pixel_hash` + index 的 `pixel_sha256` |

任何一个实现都必须**先验校验再使用**；tile 层校验尤其重要，因为 tile 解码器
会按 `raw_len` 分配内存。

### 7.3 解码器的资源保护上限

参考实现强制以下上限，新实现建议一致，否则会被恶意文件打爆内存：

| 项 | 上限 |
|---|---|
| `tile_size` | 8..1024 |
| `tile_count` | 10,000,000 |
| `width * height` | 300,000,000 |
| 单 tile `raw_len` | `max(1024, tw*th*C*16 + 1 MiB)` |
| Metadata 条目数 | 64 |
| Metadata 总长 | 64 MiB |
| 单 PXC 文件 | 受 PXCS 条目 `length` 约束 |

---

## 8. 确定性与比对契约

### 8.1 payload 是确定性的

对同一输入图像、同一参数，参考实现产出的 PXC 字节**完全一致**（已实测）。
因此黄金向量可以按字节比对。

### 8.2 整归档不是确定性的

PXCS 每条归档的 `archive_id` 是 `uuid4()` 随机值，且它参与 payload 的 AEAD
AAD 与索引，因此**整文件哈希永远不可复现**。

结论：**不能用“整文件 SHA256 相同”作为重实现的验收标准。**

### 8.3 正确的比对契约

两份实现产出的归档 A 与 B 判定为等价，当且仅当：

1. 版本号（major / minor）相同；
2. 条目数与条目路径集合相同；
3. 每个条目的 `codec`、明文 payload **字节内容**、`size`、`ref`、`depth`、
   `quant`、`centered`、`lossy`、`pixel_sha256`、`raw_crc32` 相同；
4. 允许不同的：`archive_id`、各层偏移量、各层 CRC（因其由偏移或随机值派生）、
   PXC 内部的 `entropy_version` 字段。

第 4 条中的 `entropy_version` 必须放行，否则任何 zlib 版本差异都会让比对失败。
`tools/pxcs_compare.py` 实现了这份契约。

### 8.4 zlib 版本策略

zlib 的压缩输出在不同版本/不同分支（zlib、zlib-ng、Chromium zlib）之间**不保证
字节相同**，即使 level 相同。

因此：

- **语义兼容**是必须的（能互相读、解出的像素一致）；
- **字节一致**只在“使用同一 zlib 实现与版本”时保证。

如果将来需要跨实现字节一致，必须固定 zlib 依赖版本，并在比对工具里体现。
当前策略：**接受语义兼容，黄金向量锁在生成时的 zlib 版本上**，
向量里记录 `zlib` 版本，版本不符时测试给出提示而不是失败。

---

## 9. 新实现的一致性清单

一份新的编解码实现，必须通过下列检查才算等价：

1. 能解所有黄金向量（§11），逐字节重建像素。
2. 能对黄金向量的输入图**重新编码**出完全相同的 payload 字节。
3. 能读 PXCS 1.0 – 1.3 写出的归档，包括 `passthrough`、`webp_lossless`、
   各种残差组合。
4. 遇到损坏数据时返回明确错误，不崩溃、不静默产出错误像素（§7.2/§7.3）。
5. 常量 alpha、`I;16B` 大端输入、`P` 模式调色板这三条边缘路径正确。
6. 非整除 tile 边界（宽高不是 `tile_size` 整数倍）正确。
7. `FILTER` 能解 5 种类型，但只写 3 种。
8. 与参考实现互操作：A 写 → B 读、B 写 → A 读，双向通过
   `tools/pxcs_compare.py` 与 `verify`。

---

## 10. 已识别的歧义与裁定

### 10.1 调色板排序的双重规则

`C == 1` 用数值序，`C > 1` 用字节序（§4.3）。两者机制不同，但因为
`C > 1` 的模式全部是 8 位，实际结果是一致的；`C == 1` 的 16 位模式是唯一
需要小心的地方（必须数值序）。新实现必须按 §4.3 的描述实现，不能"简化成一套"。

### 10.2 `canonical_pixel_hash` 的主机字节序依赖

§7.1 的样本序列化使用主机字节序，在大端主机上与 §1.2 的小端规则冲突。
**裁定**：维持现状（x86 上是小端，实际影响为零），在文档中标注。
如果将来要做跨架构归档互操作，需要提升 PXC 小版本号并把该哈希改为显式小端。

### 10.3 `entropy_version` 的可忽略性

§2.2 的这个字段会随 zlib 版本变化。**裁定**：视为可忽略差异，不参与等价判定。

### 10.4 PALETTE 的 1/16 采样预筛

§3.4 的候选准入用近似采样，会漏掉一些本可以用 PALETTE 的 tile。
**裁定**：属于编码器行为而非格式，但为实现字节一致，必须原样复现。

### 10.5 `MODE_RANK` 与 `TILE_MODE` 数值不一致

tile 模式数值是 `SOLID=0,PALETTE=1,PRED=2,RUN=3,RAW=4,FILTER=5`，
而平局排名顺序是 `SOLID,PALETTE,PRED,RUN,FILTER,RAW`（FILTER 排在 RAW 前）。
两套编号不同，容易看错。**裁定**：维持，排名表在 §3.4 明确定义。

---

## 11. 黄金向量

`tests/fixtures/golden/` 下保存字节级测试向量，格式见该目录的 `README.md`。

每个向量包含：

- `case.json`：描述（模式、尺寸、tile_size、生成参数、zlib 版本）
- `input.png`（或程序内生成的像素数组，见 `manifest.json` 的 `source` 字段）
- `expected.pxc`：期望的 PXC payload 字节

覆盖范围必须包含：7 种 tile 模式、`uint8`/`uint16`/`int32`、灰度/RGB/RGBA/
调色板、常量 alpha、非整除 tile 边界、`tile_size` 边界值。

生成/校验命令：

```powershell
python -m tools.gen_golden          # 重新生成
python -m tools.gen_golden --check  # 校验当前实现是否仍字节一致
```

新增/修改编码行为时，必须同步更新向量，并说明字节差异的原因。

跨实现比对的正确做法是用 `tools/pxcs_compare.py`（按条目明文 payload 比较，
忽略 `archive_id` 及其派生 CRC），而不是比较整个文件的哈希——见 §8.2。
