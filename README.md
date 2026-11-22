<p align="center">
  <img src="docs/assets/icon.png" width="120" alt="PXCS icon">
</p>

# PXCS — Picture eXtreme Compression Set

**English** | [简体中文](README.zh-CN.md)

PXCS packs an image collection into a single `.pxcs` archive, and makes near-duplicate
images cheap: every variant of the same artwork shares one reference image, and the
remaining variants are stored as deltas against it.

It is built for the way image collections actually look — one drawing with 40 expression
variants, a character sheet with 90 outfit swaps, a folder of AI outputs that differ in
the seed. Generic archivers (7z/zip/tar) treat each PNG as an independent blob and give up
on this content. PXCS decodes to pixels, finds the reference, and stores what changed.

**Naming.** PXCS = *Picture eXtreme Compression Set* — the archive format (`.pxcs`). PXC is
the keyframe codec *inside* the format; it is not a separate product.

```
collection/                     PXCS archive
  ├─ 001_a.png   ┐
  ├─ 001_b.png   ├─ group (lossy, 1200×1800, RGB)
  └─ 001_c.png   ┘   key    = 001_a.png  → webp_lossless / pxc keyframe
                     b, c   → delta against key → pxc_residual / zlib_residual
  ├─ 002.png         singleton group → stored as its own key
  └─ notes.txt       non-image files → deflate
```

## Results

Ratio = archive size / source size; lower is better. Nine collections, 2026-11-20,
comparing the released 1.4.0 with the 1.4.1 reference tree:

| Collection | Source | 1.4.0 | 1.4.1 |
|---|---:|---:|---:|
| PNG variants (53.9 GiB, whole collection) | 53.9 GiB | — | **16.3%** |
| PNG variant set (one character) | 1.3 GiB | 26.8% | **10.2%** |
| JPEG variants, medium | 10.6 GiB | 66.7% | **59.3%** |
| PNG variants, mixed | 2.1 GiB | 42.0% | **37.2%** |
| JPEG variants, medium | 2.7 GiB | 16.1% | **11.8%** |
| JPEG variants, large | 13.8 GiB | 31.1% | **28.6%** |
| Loose JPEGs (no variants) | 3.1 GiB | 80.8% | 80.6% |

Loose images barely move — there is no neighbour to delta against. The gain is
concentrated in sets where many variants share one base drawing, and it scales with
variant density.

A larger 15-collection corpus (84.2 GiB, 14 590 images) is benchmarked in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md), together with timings and the per-codec
byte breakdown.

Three things drive most of the gain: content-aware grouping (same resolution does **not**
mean same artwork, so groups are re-split by 32×32 thumbnail distance), a **reference
tree inside each group** (each variant points at the most similar variant, not just one
key), and lossless JPEG repacking via JPEG XL for images that have no neighbour to
delta against.

## Features

- **Delta coding against a per-group reference tree**: each variant points at the most
  similar variant planned before it (depth 6 for lossy, 8 for lossless), not just one
  group key. Works for JPEG, PNG, TIFF, WebP and raw RGBA.
- **Per-image codec choice**: every entry independently falls back to `raw` if the delta
  is not smaller, so a bad grouping costs at most the wasted attempt, never size.
- **Cross-group key references (opt-in)**: a new group key that closely matches an
  already-written key of the same resolution can be stored as its delta, so one artwork
  split by RGB/RGBA or by cluster boundaries does not pay two keyframes. **Off by
  default**; `--key-ref-scope folder` compares within a directory, `resolution` across
  directories.
- **JPEG XL lossless repacking** (`jpegxl` codec): JPEGs are transcoded to JXL and restored
  **byte for byte** on extraction, typically 20–27% smaller.
- **Volumes**: split into `.001/.002/...`, extract from any volume.
- **Encryption**: AES-256-GCM + Argon2id, optional filename encryption, per-message nonces
  bound to the archive id. See [SECURITY.md](SECURITY.md).
- **Presets and dry runs**: `--preset lossless|balanced|max|custom`; `--dry-run` scans,
  counts exact duplicates, samples whole groups through a real encode, and reports a size
  range with suggested threads/memory.
- **Verification levels**: `verify --level metadata|quick|full|source`. `quick` checks
  payload CRC/SHA without reconstructing images; `full` decodes and compares pixel hashes.
- **Hardened reading**: bounded zlib decompression, reference-chain depth validation, KDF
  read limits (high-cost parameters need `--allow-expensive-kdf`). See [SECURITY.md](SECURITY.md).
- **Streaming for large files**: non-image files of 64 MiB or more are read in chunks, with
  sampling to decide whether compression pays off.
- **Parallel create / verify / extract** with a memory budget; the summary reports real peak RSS.
- **Bilingual UI**: `--language chinese_simp|english`.

## Quick start

**Just want the tool?** Grab the prebuilt Windows executable from
[Releases](https://github.com/zhichelong/pxcs/releases) — no Python required.

To run from source you need Python 3.10+ on Windows (other platforms are untested,
see below).

```powershell
git clone <this repo> pxcs
cd pxcs
python -m pip install -e ".[all]"

# no images handy? generate a synthetic sample tree
python tests\make_samples.py C:\pxcs-demo

# create
python pxcs.py create -i C:\pxcs-demo -o C:\pxcs-demo\demo.pxcs

# inspect
python pxcs.py list C:\pxcs-demo\demo.pxcs --long

# check integrity
python pxcs.py verify C:\pxcs-demo\demo.pxcs

# get the files back
python pxcs.py extract C:\pxcs-demo\demo.pxcs -o C:\pxcs-restored
```

Everything is local. There is no network code in this project.

Two notes about installs: the UI defaults to Simplified Chinese — add `--language english`
(or set `"language": "english"` in `config/config.json`) for English. And `language/` plus
`config/` are read from the source tree, so run from the checkout or use the editable
install shown above; a non-editable wheel install does not carry them.

## Commands

| Command | Purpose |
|---|---|
| `create -i DIR [DIR...] -o OUT.pxcs` | build an archive |
| `list ARCHIVE [--long] [--json]` | entries, codecs, sizes |
| `extract ARCHIVE [-o DIR] [--name PATTERN...]` | restore files |
| `verify ARCHIVE [--quick]` | full or CRC-only verification |
| `info ARCHIVE` | header, version, volume map |

Useful `create` flags:

| Flag | Effect |
|---|---|
| `--threads N` | worker threads (default: all cores) |
| `--parallel-groups N` | groups in flight (default: auto) |
| `--memory-limit 4G` | hard memory budget (default: 60% of RAM) |
| `--fast` | prefer speed: use zlib for lossless residuals |
| `--no-cluster` | disable thumbnail-based group splitting |
| `--no-jxl` | disable JPEG XL repacking (writes format 1.2 archives) |
| `--volumes --volume-size 4G` | split the archive |
| `--ask-password` / `--password-file PATH` | encrypt the archive |
| `--group-mode resolution\|regex\|natural` | how images are bucketed |

## Format and specifications

| Document | Contents |
|---|---|
| [docs/FORMAT.md](docs/FORMAT.md) | container layout, header/footer, index, volumes, crypto envelope |
| [docs/CODEC.md](docs/CODEC.md) | the PXC tile codec: tile modes, bit layout, golden vectors |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | corpus, ratios, timings, byte breakdown |
| [docs/VALIDATION.zh-CN.md](docs/VALIDATION.zh-CN.md) | engineering log of the optimization rounds (Chinese) |

Chinese versions of the specifications: [FORMAT.zh-CN.md](docs/FORMAT.zh-CN.md),
[CODEC.zh-CN.md](docs/CODEC.zh-CN.md).

`pxcs.py` is the archive CLI described above. `pxc.py` is the legacy single-image tooling
for the raw `.pxc` codec stream (superseded by `.pxcs`); it stays in the tree so old files
remain readable.

Format versions: a reader accepts minors 0–6. Writes are `1.2` by default (`1.3` with
encryption, `1.4` when `jpegxl` entries are present; `1.5` for cross-mode residual
references and `1.6` when a zlib residual carries `entry.meta`), so an older reader can
still open plain archives produced today.

## Correctness guarantees

The project's priorities, in order: **correctness > compression ratio > speed**.

- Payloads are deterministic: the same input and options produce byte-identical payloads.
  Whole archives are not (they carry a random `archive_id`), so compare payload-level with
  `tools/pxcs_compare.py`. Entry *order* follows completion order under parallel encoding and
  is not part of the contract.
- A failed delta falls back to `raw` and records the reason; nothing is silently dropped.
  A group that cannot be committed aborts the whole archive instead of writing a short one.
- `verify` decodes, not just hashes: tampered payloads are detected (see
  `tests/test_pxcs_resilience.py`).
- Golden vectors pin the codec byte layout across refactors (`tests/test_pxc_golden.py`).

## Platform

Windows is the supported platform today: the code uses `ctypes`/Win32 for peak working-set
measurement and assumes NTFS path semantics. Linux/macOS will mostly work for
create/list/extract but are untested; `GetProcessMemoryInfo` has no fallback yet.

Tested on Windows 11 with Python 3.11.9.

## Status

Beta. The format is stable enough to store real archives (the author's own collection is
in it), but minors are still being added. Backwards compatibility is a hard requirement:
readers must keep opening everything that has ever been written.

## License

MIT — see [LICENSE](LICENSE). Third-party components and their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
