# Changelog

Format minors 0-4 are all readable. This file records feature-level changes; the byte
layout of each minor is specified in [docs/FORMAT.md](docs/FORMAT.md).

## 1.6.0

**Cross-group key references**: before writing a group key, compare its thumbnail with
already-written keys of the same resolution. If they are close enough and the encoded
delta is clearly smaller than the keyframe, the key is stored as a delta of that earlier key.

- Default scope is `folder`: only keys from the same source directory are considered.
  `--key-ref-scope resolution` widens it to all same-resolution keys, `off` disables it.
- Lossless sources only, and only "same mode" or "RGB key -> RGBA entry with constant
  alpha". Truly transparent RGBA is never cross-referenced.
- `--key-ref-distance` (default 5.0) is the thumbnail distance threshold;
  `--key-ref-min-saving` (default 0.80) requires the delta to be smaller than that
  fraction of the key.
- All three can be set in `config/config.json` under `create`; CLI defaults come from
  the config.

Measured on one 91-image / 1.75 GiB artwork: keys 7 -> 4, archive 138.62 MiB ->
131.19 MiB (**-5.4%**), create time about +14%. Full verify 91/91 and extracted
pixels/alpha match. The gain is below "two keys become one" because the RGB content of
these RGB/RGBA twins still differs, so the delta is not zero.

**Format**: archives containing a cross-mode ref are written as minor 5. Older readers
(<=1.5) reject them at load time; archives without cross-mode refs still use minor 2/3/4
and remain readable by older versions.

**Config**: `config/config.json` gains `key_ref_scope`, `key_ref_distance` and
`key_ref_min_saving` under `create`.

## 1.5.0

No archive-format change (minor is still 4). This release is about the boundary when
reading untrusted archives, plus lower memory use and more visibility before/after create.

**Security and data integrity**

- **Bounded zlib decompression**: index, residual and plain-file decoders now share
  `pxcs_core/zlibsafe.py`. The old code capped only the first `decompress()` call and then
  called `flush()`, which expands the remaining input without a limit - a 1 MB blob could
  demand 1 GiB of RAM. The new path fixes the limit up front, stops on the first byte over
  it, and checks stream end, trailing bytes and the exact expected length.
- **Reference-chain validation**: the index loader now computes the real chain depth
  iteratively and requires `depth == parent.depth + 1`, capped at 255. Previously only
  `ref < id` was checked, so a forged `depth` could disguise a 1200-level chain as one
  level and turn decoding into a `RecursionError`. Image and DUP resolution are iterative.
- **KDF read limits**: Argon2id parameters above 512 MiB / 8 passes / 8 lanes are rejected
  by default (scrypt: `log2N<=18, r<=8, p<=4`), before any key derivation, with the actual
  parameters in the error. `--allow-expensive-kdf` opts in. Unknown crypto-header flag bits
  are rejected instead of ignored.
- **Lossless WebP detection**: RIFF chunks (`VP8L` / `VP8 `) decide whether the source is
  lossless. Lossless WebP is now treated as pixel-exact; previously it was quantized and
  2.4% of pixels changed in our test. Lossy WebP behaves as before.
- **Residual type safety**: `zlib_residual` only accepts 8-bit samples. `--fast` on 16-bit
  PNGs used to produce an archive that could be created but not decoded; encoding now
  rejects it and falls back to PXC residuals.
- **Extraction path conflicts**: all output paths are planned before touching the disk.
  Under `--lossless`, `a.jpg` and `a.png` both map to `a.png`; the old code silently
  skipped the second entry. It now fails by default, with `--on-conflict skip|rename`.

**Performance and memory**

- **Chunked reference-tree planning**: no `(n, n, 32, 32)` distance matrix. That temporary
  was about 2 GB at n=1000 and 32 GB at n=3000; chunked planning peaks at 5.9 MB / 17.7 MB
  with identical parent choices.
- **Fewer full-image copies**: quantization writes back in slabs, centering is in-place,
  zlib reads a `memoryview`, and residual decoding no longer copies.
- **Memory-budget semantics**: an explicit small `--memory-limit` is no longer silently
  raised to 512 MiB (floor 16 MiB). Peak RSS is recorded (Windows `GetProcessMemoryInfo`,
  Linux `VmHWM`) and printed in the create summary. The decode cache is bounded by bytes as
  well as by entry count.
- **Streaming for large files**: non-image files of 64 MiB or more are no longer read whole
  and recompressed. Three 256 KiB samples decide whether streaming zlib is worthwhile;
  otherwise the file is streamed as store. Two passes (hash, then write) keep dedup working.
  Encrypted archives keep the buffered path because AEAD needs whole payloads.

**Usability**

- `--preset lossless|balanced|max|custom`.
- `--dry-run`: scan, exact duplicate count, whole-group sampling through a real encode, and
  a size range (p10/p50/p90) plus suggested threads/memory. Writes nothing.
- `--quant-list 0,4,8,12,16,24`: explicit quantization steps, all evaluated before picking
  the smallest; `--report-quality` adds real max error and PSNR for lossy entries (off by
  default: a full-image comparison is expensive on 4K images).
- `verify --level metadata|quick|full|source`; `quick` no longer reconstructs images, which
  the old implementation did despite the help text.
- Progress ETA, peak RSS and max/mean reference depth in summaries.

**Compatibility**

- Archive format unchanged (`VERSION_MINOR` stays 4); older readers can read new archives.
- Reading is stricter: length mismatches, trailing bytes, forged depth, unknown KDF
  parameters and unknown flag bits now fail clearly - those files were already broken.
- Archives created with custom expensive KDF parameters need `--allow-expensive-kdf`.
- New archives for lossless WebP sources are a bit larger, which is the cost of
  pixel-exactness.

**Verification**

- 267 tests (208 before), including zlib bombs, reference chains, lossless WebP, path
  conflicts, KDF limits, volume rollback and dry-run.
- 60 JPEGs (758 MiB) against v1.4.1: identical archive size (1 byte from the random
  archive_id), create time indistinguishable in an interleaved A/B run,
  `verify --level full` about 10% faster, create peak RSS 4-9% lower.
  `verify --quick` dropped from 62 s to 0.35 s, and its semantics changed: it really does
  not reconstruct images now.

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
  Wall time also drops by roughly a third, because smaller residuals compress faster.
- Measured on 73 PNGs / 803 MiB: **22.37% -> 16.64%**, 531s -> 260s.
- Quality does not regress: deleting the PSNR of 10 deep entries against the depth-3
  build gives **+0.10 dB on average** (max +0.26) — a deeper tree picks a closer
  reference, so its residual carries less quantization error to begin with.
- Cost: a cold single-image extraction is ~2.2x slower at depth 6 than at depth 1
  (1.3s vs 0.6s per 4K entry); sequential extraction reuses the parent cache.
- **`--tree-depth`**: the reference-tree depth is now configurable (1..255); `0` keeps
  the automatic choice (6 lossy / 8 lossless). Lower values make single-image extraction
  cheaper, higher values squeeze the archive further.
- **Faster tile encoding**: `choose_row_filters` scores all rows in one vectorized pass
  instead of a Python loop per row (3xH small-array calls per tile). Roughly 40%
  faster on a PNG-heavy set, with better multi-core utilisation and byte-identical
  output (all 21 golden vectors unchanged).
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
