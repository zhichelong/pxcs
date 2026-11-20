# Changelog

Format minors 0-4 are all readable. This file records feature-level changes; the byte
layout of each minor is specified in [docs/FORMAT.md](docs/FORMAT.md).

## 1.4.1

- **In-group reference tree**: after thumbnail clustering, each member now references the
  closest member planned before it instead of every member pointing at the group key.
  Depth is capped at 3 and every `ref` still points backwards, so entry order is unchanged
  and older readers need no adaptation.
- **Layered encoding**: layer *d* uses the *reconstructed* pixels of layer *d-1* as its
  reference. This is required for lossy residuals — referencing the original pixels makes
  each layer's quantization error independent and `verify` rejects the archive.
- **Depth follows the codec class.** Lossy residuals accumulate one quantization step
  per layer, so the tree stops at 6; lossless residuals are reversible and go to 8.
  Both are module constants (`TREE_MAX_DEPTH`, `TREE_MAX_DEPTH_LOSSLESS`).
- Measured on 1106 JPEGs / 11.9 GiB of artist variant images: **17.29% -> 9.85%** of
  source size (residual bytes 1222.9 -> 338.4 MiB; key frames unchanged at 833.3 MiB).
  Wall time also drops, 680.4s -> 443.2s, because smaller residuals compress faster.
- Measured on 73 PNGs / 803 MiB: **22.37% -> 16.64%**, 531s -> 260s.
- Quality does not regress: deleting the PSNR of 10 deep entries against the depth-3
  build gives **+0.10 dB on average** (max +0.26) — a deeper tree picks a closer
  reference, so its residual carries less quantization error to begin with.
- Cost: a cold single-image extraction is ~2.2x slower at depth 6 than at depth 1
  (1.3s vs 0.6s per 4K entry); sequential extraction reuses the parent cache.
- No format change: the same minors are written.

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
