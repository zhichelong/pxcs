# 更新日志

格式 minor 0–4 都能读取。本文件记录功能层面的变化；每个 minor 的字节布局见
[docs/FORMAT.zh-CN.md](docs/FORMAT.zh-CN.md)。

## 1.4.0

- **`jpegxl` 编码器**：通过 JPEG XL 对 JPEG 做无损重打包。JPEG 转成 JXL 流，提取时
  **逐字节还原**原始文件，通常小 20–27%。含 `jpegxl` 条目的归档写成格式 minor 4。
- **所有格式都做相似度切分**：组内按 32×32 灰度缩略图距离再切一刀，不再只对 JPEG 生效。
  在差分密集的集合上收益最大（3.9 GiB 的集合：96.7% → 22.6%）。
- **`verify` 与 `extract` 并行化**（按条目区间切片，每个 worker 打开自己的 reader）。
  8 线程实测分别为 2.46× 与 5.76×。
- **组 key 编码移入 worker 线程池**，此前它在主线程串行执行，是 PNG 集合的瓶颈。
- 新增参数：`--fast`、`--no-cluster`、`--no-jxl`、`--verbose`、`--memory-limit`。
- 修复：`create` 在并行窗口恰好排空、而仍有组未提交时会**静默截断归档**。现在会直接
  抛错并中止，不再写出少条目的归档；同时有回归测试锁定条目顺序。
- 修复：翻译文本里出现裸 `%` 时 `--help` 直接崩溃。

## 1.3.0

- **分卷**：`--volumes --volume-size SIZE`，`.001/.002/...`，从任意一卷提取。
- **加密**：AES-256-GCM + Argon2id，可选加密文件名，按消息类别区分 AAD。
  见 [SECURITY.zh-CN.md](SECURITY.zh-CN.md)。
- **多语言**：界面文本移到 `language/<code>.json`，`--language` 切换。
  自带 `chinese_simp` 与 `english`。

## 1.2.0

- 分组归档：图片按 `(lossy, 宽, 高, 模式)` 分桶，每组一个关键帧，其余存残差。
  如果差分后更大，逐条目回退 `raw`。
- 线程数自动、并行组窗口、内存预算：预算不够就排队，而不是失败。

## 1.0.0 – 1.1.0

- 最初的容器与 PXC tile 编码器（minor 0 与 1）。仍然可读。
