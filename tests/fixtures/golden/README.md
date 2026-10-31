# PXC 黄金向量

字节级基准：同一份样本数组 + 同一组参数，任何实现都必须编码出**完全相同的
PXC 字节**。这是"第二实现"（例如 C++）等价性的唯一硬证据。

生成：`python -m tools.gen_golden`
校验：`python -m tools.gen_golden --check`

**不要手工编辑本目录下的文件。** 需要新增覆盖时改 `tools/gen_golden.py`
的 `_build_cases()` 再重新生成。

## 目录结构

```
golden/
  manifest.json          全部向量的描述
  source/<name>.bin      样本数组，小端、行优先、通道内联
  expected/<name>.pxc    期望的 PXC 文件字节
```

## manifest.json

| 字段 | 说明 |
|---|---|
| `format` / `version` | 固定 `PXC-GOLDEN` / 1 |
| `zlib_version` | 生成时的 zlib 运行时版本，见下文"zlib 降级" |
| `numpy_version` | 生成时的 numpy 版本（仅供排查） |
| `original_filename` / `source_format` / `source_ext` | 写入 PXC 元数据的固定值 |
| `coverage` | 全部向量累计的 tile 模式计数 |
| `cases[]` | 向量列表 |

每个 `cases[]` 条目：

| 字段 | 说明 |
|---|---|
| `name` | 唯一名字，也是文件名 |
| `note` | 该向量想覆盖什么 |
| `mode` | Pillow 模式名，见 `CODEC.md` §1.1 |
| `width` / `height` / `channels` | 样本数组形状 |
| `dtype` | `source/*.bin` 的 numpy dtype 名（`uint8` / `uint16` / `int32`）。文件里是小端 |
| `tile_size` | 编码时使用的 tile 尺寸 |
| `source` | 样本数组路径 |
| `source_sha256` | 样本数组原始字节的 SHA-256 |
| `expected` | 期望输出路径 |
| `expected_size` / `expected_sha256` | 期望输出的字节数与 SHA-256 |
| `tile_modes` | 该向量里各 tile 模式出现次数，便于定位"哪个模式的行为变了" |
| `palette_hex` | `P` 模式的调色板字节（`PXC_META_PALETTE`） |
| `transparency_hex` | `transparency` 元数据字节 |
| `constant_alpha` / `header_channels` | 常量 alpha 向量专用：alpha 值与 header 里的通道数 |

## 重建输入数组

```python
arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
arr = arr.reshape(height, width, channels).astype(np.dtype(dtype))
```

## zlib 降级规则

tile payload 与 index 都经过 zlib。zlib 的压缩输出在不同版本/分支之间不保证
字节相同（`CODEC.md` §8.4）。因此测试分两级：

1. **总是执行**：解码 `expected/*.pxc`，断言像素与 `source/*.bin` 完全一致
   （覆盖解码器回归）；
2. **当 `zlib_version` 与当前运行时一致时执行**：重新编码，断言与
   `expected/*.pxc` **逐字节相同**（覆盖编码器回归）。

版本不一致时测试不失败，但会打印提示，说明"本机 zlib 与向量生成时不同，
字节级断言已跳过"。要恢复严格的字节断言，用同一 zlib 版本重新生成向量。

## 当前覆盖

7 种 tile 模式（SOLID / PALETTE-PLAIN / PALETTE-RLE / PRED / RUN / FILTER / RAW）
全部覆盖，另含：`uint8`/`uint16`/`int32`、`P` 模式调色板、透明度元数据、
常量 alpha、非整除 tile 边界、`tile_size` 上下界、调色板双重排序规则。
