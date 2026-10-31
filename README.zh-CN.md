<p align="center">
  <img src="docs/assets/icon.png" width="120" alt="PXCS 图标">
</p>

# PXCS — Picture eXtreme Compression Set

[English](README.md) | **简体中文**

PXCS 把一个图片集合打包成单个 `.pxcs` 归档，并且让"近似重复"的图片变得很便宜：
同一作品的所有变体共享一张参考图，其余的只存相对它的差分。

它是按图集真实的样子设计的——一张画配 40 个差分、一个角色 90 套衣服、一堆只差 seed 的
AI 输出。通用压缩工具（7z / zip / tar）把每张 PNG 当成独立数据块，对这类内容基本无能为力；
PXCS 先解码成像素，找到参考图，只存变化的部分。

**命名。** PXCS = *Picture eXtreme Compression Set*，指归档格式（`.pxcs`）；PXC 是格式
内部的关键帧编码器，不是独立产品。

```
图集目录/                        PXCS 归档
  ├─ 001_a.png   ┐
  ├─ 001_b.png   ├─ 组 (lossy, 1200×1800, RGB)
  └─ 001_c.png   ┘   关键帧 = 001_a.png  → webp_lossless / pxc keyframe
                     b, c  → 相对关键帧的差分 → pxc_residual / zlib_residual
  ├─ 002.png         单张成组 → 自己就是关键帧
  └─ notes.txt       非图片文件 → deflate
```

## 效果

15 个真实图集，84.2 GiB，共 14 590 张图，全部归档 verify 通过（0 失败）。
压缩率 = 归档大小 / 源大小，越低越好。

| 图集 | 规模 | 优化前 | 现在 |
|---|---:|---:|---:|
| JPEG 差分集（大） | 3.9 GiB | 96.7% | **22.6%** |
| JPEG 差分集（超大） | 14.3 GiB | — | **30.4%** |
| PNG 差分集 | 3.7 GiB | 39.7% | **33.7%** |
| JPEG 差分集（中等） | 3.5 GiB | 50.2% | **39.0%** |
| JPEG 散图（无差分） | 0.5 GiB | 98.9% | **83.5%** |
| PNG 散图（无差分） | 0.6 GiB | 84.2% | 84.2% |

完整表格、耗时与按 codec 的字节构成见 [docs/BENCHMARKS.zh-CN.md](docs/BENCHMARKS.zh-CN.md)。

收益主要来自两件事：**按内容分组**（同分辨率不等于同一作品，所以用 32×32 缩略图再切一刀），
以及 **JPEG 无损重打包**（没有邻居可差分的 JPEG 用 JPEG XL 转码，通常是原大小的 79%）。

## 功能

- **组内差分**：JPEG / PNG / TIFF / WebP / RGBA 都支持，每组一张参考图。
- **逐条目选编码器**：差分不比原图小就立刻回退 `raw`，分组分错了最多浪费一次尝试，不会变大。
- **JPEG XL 无损重打包**（`jpegxl` codec）：JPEG 转成 JXL，提取时**逐字节还原**原始文件，
  通常小 20–27%。
- **分卷**：拆成 `.001/.002/...`，从任意一卷提取。
- **加密**：AES-256-GCM + Argon2id，可选加密文件名，每条消息独立 nonce 并绑定归档 id。
  见 [SECURITY.zh-CN.md](SECURITY.zh-CN.md)。
- **校验**：`verify` 会真正解码并比对像素/哈希，不只是查 CRC。
- **并行 create / verify / extract**，带内存预算护栏：预算不够就排队，不会 OOM。
- **中英双语界面**：`--language chinese_simp|english`。

## 快速开始

需要 Windows + Python 3.10 以上（其它平台未测试，见下文）。

```powershell
git clone <本仓库> pxcs
cd pxcs
python -m pip install -e ".[all]"

# 手边没有图？先造一组合成样本
python tests\make_samples.py C:\pxcs-demo

# 创建
python pxcs.py create -i C:\pxcs-demo -o C:\pxcs-demo\demo.pxcs

# 查看
python pxcs.py list C:\pxcs-demo\demo.pxcs --long

# 校验
python pxcs.py verify C:\pxcs-demo\demo.pxcs

# 取回
python pxcs.py extract C:\pxcs-demo\demo.pxcs -o C:\pxcs-restored
```

全部在本地完成，这个项目没有任何联网代码。

两点安装说明：界面默认简体中文，加 `--language english`（或在 `config/config.json` 里把
`"language"` 改成 `english`）切到英文；`language/` 和 `config/` 是从源码树读的，所以要
在仓库目录里运行，或者用上面的可编辑安装（`-e`）——普通的 wheel 安装不会带上这两个目录。

## 命令

| 命令 | 作用 |
|---|---|
| `create -i DIR [DIR...] -o OUT.pxcs` | 创建归档 |
| `list ARCHIVE [--long] [--json]` | 列出条目、编码器、大小 |
| `extract ARCHIVE [-o DIR] [--name PATTERN...]` | 提取文件 |
| `verify ARCHIVE [--quick]` | 完整校验或只校验 CRC |
| `info ARCHIVE` | 头部、版本、分卷信息 |

`create` 常用参数：

| 参数 | 作用 |
|---|---|
| `--threads N` | worker 线程数（默认用满所有核） |
| `--parallel-groups N` | 同时在跑的组数（默认自动） |
| `--memory-limit 4G` | 内存硬上限（默认物理内存的 60%） |
| `--fast` | 速度优先：无损残差改用 zlib |
| `--no-cluster` | 关闭按缩略图的组内切分 |
| `--no-jxl` | 关闭 JPEG XL 重打包（写出 1.2 格式，旧程序可读） |
| `--volumes --volume-size 4G` | 分卷 |
| `--ask-password` / `--password-file PATH` | 加密归档 |
| `--group-mode resolution\|regex\|natural` | 分组方式 |

## 格式与规格

| 文档 | 内容 |
|---|---|
| [docs/FORMAT.zh-CN.md](docs/FORMAT.zh-CN.md) | 容器布局、头/尾、索引、分卷、加密封装 |
| [docs/CODEC.zh-CN.md](docs/CODEC.zh-CN.md) | PXC tile 编码器：模式、比特布局、黄金向量 |
| [docs/BENCHMARKS.zh-CN.md](docs/BENCHMARKS.zh-CN.md) | 语料、压缩率、耗时、字节构成 |
| [docs/VALIDATION.zh-CN.md](docs/VALIDATION.zh-CN.md) | 每轮优化的工程记录 |

英文版本：[FORMAT.md](docs/FORMAT.md)、[CODEC.md](docs/CODEC.md)。

`pxcs.py` 是上面文档里描述的归档 CLI；`pxc.py` 是更早的单图工具（裸 `.pxc` codec 流，
已被 `.pxcs` 取代），保留它是为了让旧文件还能读。

格式版本：读取端支持 minor 0–4。默认写出 `1.2`（加密时 `1.3`，含 `jpegxl` 条目时 `1.4`），
所以旧版读取程序仍然能打开今天生成的普通归档。

## 正确性保证

项目的优先级顺序是：**正确性 > 压缩率 > 速度**。

- payload 是确定性的：同样输入和参数产出逐字节相同的 payload。整个归档不是确定的
  （带随机 `archive_id`），所以对比要用 `tools/pxcs_compare.py` 比 payload。
  条目的**物理顺序**取决于并行完成顺序，不属于格式契约。
- 差分失败会回退 `raw` 并记录 `reason`，不会静默丢弃；组提交失败会中止整个归档，
  绝不写出一份少条目的归档。
- `verify` 会解码而不只是算哈希，篡改的 payload 能被发现（`tests/test_pxcs_resilience.py`）。
- 黄金向量锁死 codec 的字节布局，重构不会悄悄改变格式（`tests/test_pxc_golden.py`）。

## 平台

目前只支持 Windows：代码用 `ctypes`/Win32 读峰值工作集，并假设 NTFS 路径语义。
Linux/macOS 上 create/list/extract 大多能跑但没有测试覆盖，`GetProcessMemoryInfo` 还没有回退实现。

在 Windows 11 + Python 3.11.9 上测试。

## 状态

Beta。格式已经稳定到可以存真实档案（作者自己的图集就在里面），但仍在小版本迭代。
向后兼容是硬性要求：读取端必须能打开历史上写出的一切。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方组件及其许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
