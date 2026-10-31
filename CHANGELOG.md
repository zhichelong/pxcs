# Changelog

Format minors 0-4 are all readable. This file records feature-level changes; the byte
layout of each minor is specified in [docs/FORMAT.md](docs/FORMAT.md).

## 1.4.0

- **`jpegxl` codec**: lossless JPEG repacking through JPEG XL. JPEGs are transcoded to a
  JXL stream and restored byte for byte on extraction; typically 20-27% smaller. Archives
  containing `jpegxl` entries are written as format minor 4.
- **Similarity clustering for every format**: groups are re-split by 32x32 grayscale
  thumbnail distance, not only for JPEG. Largest single win on variant-heavy sets
  (one 3.9 GiB collection: 96.7% -> 22.6%).
- **Parallel `verify` and `extract`** (sliced by entry ranges, each worker with its own
  reader). Measured 2.46x and 5.76x respectively on an 8-thread comparison.
- **Group-key encoding moved into the worker pool**; previously it ran on the main thread
  and bottlenecked PNG collections.
- New flags: `--fast`, `--no-cluster`, `--no-jxl`, `--verbose`, `--memory-limit`.
- Fixed: `create` could silently truncate an archive when the parallel window drained
  while groups were still pending. It now raises and aborts instead of writing a short
  archive, and a regression test pins the entry ordering.
- Fixed: `--help` crashed when a translated string contained a bare `%`.

## 1.3.0

- **Volumes**: `--volumes --volume-size SIZE`, `.001/.002/...`, extract from any volume.
- **Encryption**: AES-256-GCM with Argon2id; optional filename encryption; distinct AAD
  per message class. See [SECURITY.md](SECURITY.md).
- **i18n**: user-facing strings moved to `language/<code>.json`; `--language` selector.
  Ships `chinese_simp` and `english`.

## 1.2.0

- Grouped archives: images bucketed by `(lossy, width, height, mode)`, one key per group,
  the rest stored as residuals. Falls back to `raw` per entry when a delta is not smaller.
- Automatic thread count, parallel group window, and a memory budget that throttles
  instead of failing.

## 1.0.0 - 1.1.0

- Initial container and the PXC tile codec (minors 0 and 1). Still readable.
