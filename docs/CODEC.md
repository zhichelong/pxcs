# PXC Bitstream Specification (Payload Layer)

This document describes **the internal byte layout of each payload** in a PXCS
archive, i.e. what that run of bytes actually is when `codec` is `pxc` /
`pxc_residual` / `zlib_residual`.

- Companion document: `FORMAT.md` describes the **container layer** (PXCS header / index / encryption / volumes).
- Corresponding implementation: PXCS 1.3 / PXC 0.2 (`pxc_core`).
- Byte order: **unless stated otherwise, all integers are little-endian**.
- Status of this document: **frozen**. Any modification that changes the byte
  layout must bump the `PXC` minor version, and must update this document and
  the golden vector set (`tests/fixtures/golden/`) at the same time.

The purpose of this document is very concrete: **a brand-new implementation
(e.g. C++) must be able to write a byte-compatible codec by reading only this
document plus the golden vectors, without reading the Python source code.**
Any behavior that this document does not state clearly and that one would have
to consult the Python implementation to determine is a defect of this document.

---

## 0. Terminology

| Term | Meaning |
|---|---|
| sample | one value on one channel, e.g. 0..255 for `uint8` |
| channel | the number of components of a pixel: `L=1`, `LA=2`, `RGB=3`, `RGBA=4` |
| pixel | the `C` samples at the same position |
| tile | a rectangular block cut from the image by `tile_size`, `(H,W,C)`; the right/bottom edge may be less than one full tile |
| residual | in differential coding, the result of subtracting (modulo 2^n) the "current image" and the "reference image" sample by sample |
| storage type | the type used when a sample is serialized in PXC, see §1.2 |

---

## 1. Data Model

### 1.1 Image Modes

PXC supports only the following 12 Pillow modes, with `mode_id` being a `u16`:

| id | mode | C | bit depth | color model | alpha | numpy dtype |
|---:|---|---:|---:|---|---|---|
| 1 | `1` | 1 | 1 | grayscale | none | `uint8` (values are only 0/1) |
| 2 | `L` | 1 | 8 | grayscale | none | `uint8` |
| 3 | `LA` | 2 | 8 | grayscale+alpha | passthrough | `uint8` |
| 4 | `P` | 1 | 8 | palette | none | `uint8` |
| 5 | `RGB` | 3 | 8 | RGB | none | `uint8` |
| 6 | `RGBA` | 4 | 8 | RGBA | passthrough | `uint8` |
| 7 | `CMYK` | 4 | 8 | CMYK | none | `uint8` |
| 8 | `I` | 1 | 32 | integer | none | `int32` |
| 9 | `I;16` | 1 | 16 | grayscale | none | `uint16` |
| 10 | `I;16L` | 1 | 16 | grayscale | none | `uint16` |
| 11 | `I;16B` | 1 | 16 | grayscale | none | `uint16` |
| 12 | `F` | 1 | 32 | float | none | `float32` |

Color model numbers: `1=grayscale`, `2=grayscale+alpha`, `3=palette`, `4=RGB`, `5=RGBA`,
`6=CMYK`, `7=integer`, `8=float`.

Alpha kind numbers: `0=none`, `1=passthrough`, `2=palette transparency`, `3=constant alpha`.

**Note on `I;16B`**: the source file is big-endian, but after decoding the samples are host-endian `uint16` in memory and are
serialized little-endian (see §1.2). `I;16` and `I;16L` are synonymous.

### 1.2 Sample Serialization Rules (`storage_dtype`)

Inside PXC, **multi-byte samples are always little-endian**:

| sample type | storage type | bytes per sample |
|---|---|---:|
| `uint8` | `uint8` | 1 |
| `uint16` | `<u2` | 2 |
| `uint32` | `<u4` | 4 |
| `int32` | `<i4` | 4 |
| `float32` | `<f4` | 4 |

The rules of this section apply to **all** tile payloads, palette entries, SOLID colors, and RUN colors.

The only exception is sample serialization in the pixel hash, see §7.1 — that one uses **host byte order**;
this is an identified defect retained for backward compatibility, see §10.2.

---

## 2. PXC File Layout

A `pxc` payload is exactly one complete PXC file:

```
Offset 0  ┌────────────────────────────┐
          │ Header (128 bytes)         │
          ├────────────────────────────┤
          │ Metadata block             │  length given by header.metadata_length
          ├────────────────────────────┤
          │ Chunk directory (32×N B)   │  N = header.chunk_count
          ├────────────────────────────┤
          │ Chunk 0 payload            │  META
          │ Chunk 1 payload            │  DATA
          │ Chunk 2 payload (optional) │  ORIG
          ├────────────────────────────┤
          │ Footer (16 bytes)          │
          └────────────────────────────┘
```

**There is no alignment padding between blocks.** This differs from the PXCS container layer (where payload
starts are aligned to 8 bytes); PXC is packed internally.

`metadata_offset` is always 128; `chunk_dir_offset = 128 + metadata_length`;
the offset of the first chunk payload = `chunk_dir_offset + 32 * chunk_count`.

### 2.1 Header (128 Bytes)

Structure: `<4sBBHIIQQQQIIIHHHHHHIQQ32sI`

| Offset | Size | Type | Field | Description |
|---:|---:|---|---|---|
| 0 | 4 | `4s` | magic | `PXC1` (`0x31435850`) |
| 4 | 1 | `B` | version_major | currently **0** |
| 5 | 1 | `B` | version_minor | currently **2** |
| 6 | 2 | `H` | flags | see below |
| 8 | 4 | `I` | header_size | always 128 |
| 12 | 4 | `I` | header_crc | CRC32, **this field is computed as 0** |
| 16 | 8 | `Q` | file_size | byte count of the entire PXC file |
| 24 | 8 | `Q` | metadata_offset | always 128 |
| 32 | 8 | `Q` | metadata_length | byte count of the Metadata block |
| 40 | 8 | `Q` | chunk_dir_offset | `128 + metadata_length` |
| 48 | 4 | `I` | chunk_count | 2 or 3 |
| 52 | 4 | `I` | width | image width |
| 56 | 4 | `I` | height | image height |
| 60 | 2 | `H` | mode_id | see §1.1 |
| 62 | 2 | `H` | bit_depth | see §1.1 |
| 64 | 2 | `H` | channels | number of channels **actually stored** (may be one less than the mode's) |
| 66 | 2 | `H` | color_model | see §1.1 |
| 68 | 2 | `H` | alpha_kind | see §1.1 |
| 70 | 2 | `H` | compression_id | 1=tile stream, 2=passthrough |
| 72 | 4 | `I` | tile_size | valid when compression_id=1; must be 0 when =2 |
| 76 | 8 | `Q` | original_size | byte count of the source file (0 means unknown) |
| 84 | 8 | `Q` | data_length | length of the DATA chunk |
| 92 | 32 | `32s` | pixel_hash | see §7.1 |
| 124 | 4 | `I` | reserved | must be 0 |

`flags` bit definitions:

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `FLAG_HAS_ORIG` | an ORIG chunk is present |
| 1 | `FLAG_HAS_ALPHA` | the image has alpha |
| 2 | `FLAG_HAS_PALETTE` | the image has a palette |
| 3 | `FLAG_SOURCE_LOSSY` | the source format is lossy (JPEG / WebP) |
| 4 | `FLAG_PASSTHROUGH` | DATA is the raw file bytes |

**How `header_crc` is computed**: copy the first 128 bytes, zero `[12:16]`, and compute CRC32 over the result.

**Relationship between `channels` and `mode`**: normally `channels == the channel count of mode`.
When `alpha_kind == 3` (constant alpha), `channels == mode channel count - 1`,
and the Metadata must contain a `constant_alpha` key (§2.2); decoding uses that constant to
fill the alpha plane back in. This mechanism is valid only for `RGBA` (stores 3 channels) and `LA` (stores 1 channel).

### 2.2 Metadata Block

```
u32 count
repeated count times (in ascending key order):
  u16 key
  u8  type
  u8  reserved (must be 0)
  u32 length
  bytes[length] payload
```

Structure: starts with `u32 count`; each entry is `u16 key, u8 type, u8 reserved, u32 length` (an 8-byte header).

`type` values and length constraints:

| type | name | length |
|---:|---|---|
| 1 | bytes | arbitrary, ≤ 64 MiB |
| 2 | utf8 | arbitrary, ≤ 64 MiB |
| 3 | u32 | exactly 4 |
| 4 | u64 | exactly 8 |
| 5 | f64×2 | exactly 16 |
| 6 | f64 | exactly 8 |

Key table:

| key | name | type | when written |
|---:|---|---|---|
| 1 | `original_filename` | utf8 | always |
| 2 | `source_format` | utf8 | always |
| 3 | `source_ext` | utf8 | always |
| 4 | `mode_name` | utf8 | always |
| 5 | `palette` | bytes | `P` mode |
| 6 | `transparency` | bytes | when transparency information is present |
| 7 | `icc_profile` | bytes | sometimes |
| 8 | `exif` | bytes | sometimes |
| 9 | `xmp` | bytes | sometimes |
| 10 | `dpi` | f64×2 | sometimes |
| 11 | `entropy_name` | utf8 | always, value `zlib` |
| 12 | `entropy_version` | utf8 | always, value is the zlib runtime version |
| 13 | `created_by` | utf8 | always |
| 14 | `source_file_hash` | bytes | sometimes (32-byte sha256) |
| 15 | `source_pixel_hash` | bytes | sometimes (32-byte sha256) |
| 16 | `gamma` | f64 | sometimes |
| 17 | `comment` | bytes | sometimes |
| 18 | `constant_alpha` | u32 | required when `alpha_kind == 3` |

The `transparency` payload carries its own tag:

| tag | layout | meaning |
|---:|---|---|
| `0x00` | `u32` | single-index transparency (`P` mode) |
| `0x01` | remaining bytes | `P` mode alpha table |
| `0x02` | `u32 × 3` | RGB color key |

**`entropy_version` weakens byte determinism**: it records the zlib version string at the time the archive was written.
When two implementations use different zlib versions, this field differs even if the bitstream is completely identical.
Differential comparison must therefore treat this field as an **ignorable difference** (see §8.3).

### 2.3 Chunk Directory

Each entry is 32 bytes: `<4sIQQII`

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 | chunk_id | `META` / `DATA` / `ORIG` |
| 4 | 4 | flags | always 0 |
| 8 | 8 | offset | absolute offset of the payload |
| 16 | 8 | length | payload byte count |
| 24 | 4 | crc | CRC32(payload) |
| 28 | 4 | reserved | must be 0 |

The chunk order is fixed as `META`, `DATA`, optionally `ORIG`. PXCS writes only `META` and `DATA`;
`ORIG` is a historical reservation — current read implementations must tolerate its presence but may ignore it.

### 2.4 Footer (16 Bytes)

Structure: `<8sII`

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 8 | magic | `PXCEND\0\0` |
| 8 | 4 | file_crc | CRC32 (all bytes from the start of the file up to the footer) |
| 12 | 4 | reserved | must be 0 |

### 2.5 The Two `compression_id` Values

| Value | Name | DATA chunk content |
|---:|---|---|
| 1 | tile stream | a tile sequence encoded according to the structure of §3 |
| 2 | passthrough | **the complete bytes of the original image file** (JPEG/WebP, etc.) |

In passthrough, `tile_size` must be 0, and after decoding the following must hold:
`mode`, `width`, `height` agree with the header, and `pixel_hash` agrees with the header
(§7.1). PXCS's `raw` codec stores the raw bytes directly at the archive layer and does **not**
use PXC's passthrough wrapper; this path is retained mainly for historical compatibility.

---

## 3. DATA Chunk: Tile Stream

### 3.1 DATA Header (20 Bytes)

Structure: `<IIIII`

| Offset | Field | Description |
|---:|---|---|
| 0 | version | always **1** |
| 4 | tile_size | 8..1024 |
| 8 | tiles_x | `ceil(width / tile_size)` |
| 12 | tiles_y | `ceil(height / tile_size)` |
| 16 | tile_count | must equal `tiles_x × tiles_y` |

### 3.2 Tile Partitioning and Order

Tiles are numbered in **row-major** order: `index = ty * tiles_x + tx`.

```
x0 = tx * tile_size          y0 = ty * tile_size
x1 = min(x0 + tile_size, width)
y1 = min(y0 + tile_size, height)
tw = x1 - x0                 th = y1 - y0
```

Tiles at the right and bottom edges may be smaller than `tile_size`. Each tile is encoded independently and
**does not reference any other** (prediction and filtering are closed inside the tile; there is no continuity across
tile boundaries).

### 3.3 Tile Record

Immediately after the DATA header there are `tile_count` consecutive records, each record = a 20-byte header + payload.

Header structure: `<BBHIIII`

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 1 | mode | see §4 |
| 1 | 1 | submode | see §4 |
| 2 | 2 | flags | bit0 = payload is zlib-compressed |
| 4 | 4 | raw_len | payload byte count **after decompression** |
| 8 | 4 | comp_len | byte count actually stored |
| 12 | 4 | raw_crc | CRC32(payload after decompression) |
| 16 | 4 | payload_crc | CRC32(bytes actually stored) |

Payload storage rule (encoder):

```
comp = zlib.compress(raw, level=6)
if len(comp) < len(raw):  stored, flags = comp, 1
else:                     stored, flags = raw,  0
```

`raw_len != comp_len` does not allow inferring whether compression was applied; you must look at `flags`.

### 3.4 Encoder Candidates and Selection Rule (Key to Byte Compatibility)

**To be byte-identical with the reference implementation you must reproduce the selection logic of this section
completely**, including the admission conditions of the candidate set and the tie-breaking rules. Any single
difference produces a different tile mode and hence different bytes.

For a tile, construct the candidates in order (the order is 1→6 below):

**Candidate 1 — SOLID**: condition `all samples of the tile equal tile[0,0]`.

**Candidate 2 — PALETTE**: the condition is
- the sample type is not floating point, and
- `unique_count(tile[::4, ::4, :]) <= 256`

where `tile[::4, ::4, :]` means **take one sample every 4 rows and every 4 columns** (i.e. 1/16 sampling),
and `unique_count` counts the number of distinct **pixels** (not channels). This pre-filter is a deliberate
approximation: a tile whose true color count is ≤ 256 but whose sampled count is > 256 is skipped for PALETTE;
this is the reference implementation's behavior and is **not** a place that may be "optimized".

If the pre-filter passes and the true palette `K` satisfies `1 < K <= 256`, then **at the same time** two sub-candidates
are produced: `PALETTE(submode=0, PLAIN)` and `PALETTE(submode=1, RLE)`.

**Candidate 3 — PRED**: the condition is a non-floating-point type.

**Candidate 4 — RUN**: the condition is non-floating-point and
```
flat = tile flattened row-major into (N, C)
runs = 1 + count(flat[i] != flat[i-1] for i in 1..N-1)
runs <= max(8, N // 4)
```

**Candidate 5 — FILTER**: the condition is that the sample type is `uint8` or `uint16`.

**Candidate 6 — RAW**: unconditional.

Select the candidate with the smallest cost:

```
cost(rec) = (20 + rec.comp_len, MODE_RANK[rec.mode], rec.submode)
```

where `20` is the tile header length and `MODE_RANK` is a fixed-order index:

```
MODE_RANK = { SOLID:0, PALETTE:1, PRED:2, RUN:3, FILTER:5→4, RAW:4→5 }
```

To be precise, `MODE_RANK` is the index into the list `[SOLID, PALETTE, PRED, RUN, FILTER, RAW]`,
i.e. `SOLID=0, PALETTE=1, PRED=2, RUN=3, FILTER=4, RAW=5`.

The comparison is a **triple lexicographic** one: equal length → compare mode rank; still equal → compare submode.
Hence PALETTE's two sub-candidates select PLAIN (`submode=0`) at equal cost,
and any structured mode takes precedence over RAW at equal cost.

---

## 4. Tile Modes

The payload of every mode consists of **uncompressed** bytes; zlib is handled uniformly by §3.3.
Below, `sd` denotes the storage type (§1.2) and `itemsize` its byte width.

### 4.1 RAW (mode = 4, submode = 0)

payload = the tile's `H × W × C` samples, row-major, channels inline, little-endian.

`raw_len` must equal `H*W*C*itemsize`.

### 4.2 SOLID (mode = 0, submode = 0)

payload = the `C` samples of `tile[0,0]` (little-endian).

`raw_len` must equal `C*itemsize`. Decoding: fill `(H,W,C)` with that color.

### 4.3 PALETTE (mode = 1)

payload:

```
u16  K                    # palette entry count, 1..256
K × C samples             # the palette body, one pixel color per row
index body                # length depends on submode
```

**Palette ordering rules (must be reproduced exactly)**:

- When `C == 1`: by **ascending numeric value** (as `uint16`, 1 sorts before 256, not by byte order).
- When `C > 1`: concatenate each pixel's `C` samples as **little-endian bytes** into `C*itemsize` bytes,
  and sort ascending by **byte lexicographic order** (memcmp semantics, unsigned).

These two are **different mechanisms**, but on the modes PXCS actually supports the results agree: all `C > 1` modes
(`LA`/`RGB`/`RGBA`/`CMYK`) are 8-bit, in which case "byte lexicographic order" is equivalent to "compare the channel
values in channel order as unsigned magnitudes". The only real thing to watch is the 16-bit `C == 1` modes:
they must use **numeric order** and must not degenerate into byte order.

Example (`C == 3`, 8-bit): the colors `(1,2,3)`, `(0,255,0)`, `(255,0,0)`, `(0,0,255)`
sort to `(0,0,255) < (0,255,0) < (1,2,3) < (255,0,0)`.

**submode = 0 (PLAIN)**: index body = `H*W` `u8` values, i.e. each pixel's index into the palette, row-major.
`raw_len = 2 + K*C*itemsize + H*W`.

**submode = 1 (RLE)**: index body = a number of `(u32 run_length, u8 palette_index)` pairs that fill
`H*W` pixels in order. There is no end marker and no explicit entry count; decoding reads until `H*W`
pixels are filled, and must consume the entire index body exactly.
`run_length` must not be 0 and must not run past the remaining pixel count.

At decode time, `index >= K` is treated as corruption.

### 4.4 PRED (mode = 2, submode = 0)

Samples are first predicted with MED (LOCO-I), then the residuals are zigzagged, and finally stored as an unsigned type.

**Residual storage type** (note: twice as wide as the sample type)

| sample type | residual storage type | bytes per residual |
|---|---|---:|
| `uint8` | `<u2` | 2 |
| `uint16` | `<u4` | 4 |
| `int32` | `<u8` | 8 |

`raw_len` must equal `H*W*C * residual width`.

**MED prediction** (per sample, per channel, independently; using left `a`, up `b`, upper-left `c`):

```
if c >= max(a,b): p = min(a,b)
elif c <= min(a,b): p = max(a,b)
else: p = a + b - c
```

**Boundary rules** (closed inside the tile, must be followed exactly):

| Position | Predicted value |
|---|---|
| `(0,0)` | `0` |
| first row `(0,x>0)` | `a = x(0,x-1)` |
| first column `(y>0,0)` | `b = x(y-1,0)` |
| otherwise | the MED formula above |

**zigzag**:

```
encode: v >= 0 → u = v << 1
        v < 0  → u = ((-v) << 1) - 1
decode: u even → v = u >> 1
        u odd  → v = -((u >> 1) + 1)
```

If a residual exceeds the representable range of the storage type, that tile cannot use PRED (the encoder should
abandon that candidate). At decode time, if a reconstructed sample falls outside the sample type's range, it is
treated as corruption.

### 4.5 RUN (mode = 3, submode = 0)

payload = a number of `(u32 run_length, C samples)` pairs that fill `H*W` pixels in row-major order.
There is no end marker; decoding reads until the pixels are filled, and must consume the payload exactly.
`run_length` must not be 0 and must not overflow.

Note the difference from PALETTE-RLE: each RUN run carries a **full color** (`C*itemsize` bytes),
whereas PALETTE-RLE carries only a **1-byte index**.

### 4.6 FILTER (mode = 5, submode = 0)

payload:

```
H bytes of per-row filter type   # each byte is 0..4
H*W*C samples                    # samples after per-row filtering, little-endian
```

`raw_len` must equal `H + H*W*C*itemsize`.

**Filter types**

| Value | Name | Base value base |
|---:|---|---|
| 0 | NONE | 0 |
| 1 | SUB | left sample `a` |
| 2 | UP | up sample `b` |
| 3 | AVG | `(a + b) // 2` |
| 4 | PAETH | Paeth predictor |

Filtering and inverse filtering are both performed modulo `2^bit` (256 for `uint8`, 65536 for `uint16`):

```
filter:         out = (x - base) mod 2^bit
inverse filter: x   = (f + base) mod 2^bit
```

Boundaries: `a`/`b`/`c` take 0 at out-of-range positions.

Paeth predictor:

```
p  = a + b - c
pa = |p - a|, pb = |p - b|, pc = |p - c|
if pa <= pb and pa <= pc: base = a
elif pb <= pc:            base = b
else:                     base = c
```

**The encoder uses only 0/1/2** (NONE/SUB/UP); 3/4 must be decodable (historical compatibility) but are never written.

**Per-row selection rule (must be reproduced exactly)**: take `ft` as `0, 1, 2` in turn (**in this order**),
compute the filter result for the whole tile, then score each row separately:

```
score(row) = Σ |signed(row[i])|,  signed(v) = v - 2^bit  if v >= 2^bit / 2 else v
```

That is, interpret the filtered values as signed numbers and sum their absolute values. Each row selects the type with
the **smallest score**, and the comparison uses strict less-than, so **on a tie the type visited first is kept**
(the one earlier in the `ft` order wins, i.e. NONE takes precedence over SUB, and SUB over UP).

---

## 5. Residual Layer (PXCS Differencing)

`pxc` and `pxc_residual` differ only in **semantics**: the former is a complete image, the latter a residual image.
Their byte structure is completely identical (both are one complete PXC file as described in §2–§4).

### 5.1 Difference Definition (`make_residual`)

The reference image and the current image are required to have exactly the same `mode`, `width`, `height`.

| mode / sample type | difference operation |
|---|---|
| `1` (1-bit) | **per-sample XOR** (not subtraction) |
| `uint8` | `(cur - ref) mod 256` |
| `uint16` | `(cur - ref) mod 65536` |
| other | differencing not supported |

The difference result is still the sample type; hence `0` means "identical" and `255` (uint8) means "differs by -1".

### 5.2 Centering (centered)

On the lossless path, the difference is shifted as a whole so that "no difference" lands on 128:

```
centered = (diff + 128) mod 256        # uint8 only
```

`uint16` is **not** centered (`center_residual` returns the input unchanged for anything other than uint8).

The entry's `centered` field records whether this step was performed; decoding must undo it accordingly.

### 5.3 Quantization (quant > 0, Lossy)

Quantization and centering are performed in the **same step**: the quantization LUT emits "the centered value" directly.

For each `u8` value `b`:

```
signed    = b if b < 128 else b - 256
q         = max(1, quant)
quantized = sign(signed) * (|signed| // q * q)
out       = (quantized + 128) & 0xFF
```

That is: first quantize with truncation toward zero, then add 128 to center. Hence `0 → 128`, `255(-1) → 127`,
`1 → 129`.

When `quant = 0` no quantization is performed (equivalent to pure lossless). Quantization is valid only for `uint8`;
`uint16` residuals are always pixel-lossless.

When `quant > 0` in an entry, `centered == true` and `lossy == true` must both hold.

### 5.4 `zlib_residual`

payload = `zlib.compress(residual_samples, level)`, where `residual_samples` are the `H*W*C` `u8` samples of the
residual image (row-major, little-endian / single-byte), with no additional wrapping.

The decompressed length must equal `width * height * channels`, otherwise it is treated as corruption.

This path is used only for `uint8` residuals, and it is PXCS's main path for JPEG differencing
(with quantization 16, most bytes in the residual are 128, which zlib compresses easily).

### 5.5 `pxc_residual`

The payload is a complete PXC file packed according to §2, whose DATA tiles encode the **residual image**.
The residual image may have undergone the centering of §5.2 or the quantization of §5.3.

### 5.6 Reconstruction

```
residual = decode(pxc_residual | zlib_residual)

if mode == "1":
    pixels = ref XOR residual                       # per sample
elif uint8:
    pixels = (ref + residual) mod 256
    if centered: pixels = (pixels - 128) mod 256    # equivalent to ref + (centered-128)
elif uint16:
    pixels = (ref + residual) mod 65536
```

Because the difference is computed modulo `2^bit`, reconstruction needs no knowledge of the sign and has no
overflow branch.

---

## 6. Encoding Decisions (Not at the Byte Layer, but They Affect the Output)

The following is not part of "the format", but it determines whether an implementation can produce the same archives
as the reference implementation. It is listed so that differential testing can localize problems.

1. **Keyframe vs difference**: the first image in a group is always a keyframe; each subsequent image compares the
   error against the reference strategy, and uses differencing only when differencing "pays off", otherwise `raw`
   (storing the original file bytes directly).
2. **Fallback threshold**: if the pixel-encoding result of a keyframe is `>= source file size × 1.02`, fall back to `raw`.
3. **Zero-difference ratio guard**: on the `zlib_residual` path, if after quantization the `proportion of values equal to 128`
   is below the configured threshold (default 0.40), abandon differencing outright and fall back.
4. **depth cap**: the consecutive difference depth must not exceed the keyframe interval (default 16).

5. **raw passthrough does not decode**: when a group contains only one image (differencing is impossible) and raw
   passthrough is taken, the reference implementation no longer fully decodes that image; the entry's `pixel_sha256`
   is empty, `constant_alpha` is empty, and `reason` is `key_lossy_passthrough`. When the group contains more than one
   image the reference image must be decoded; then `pixel_sha256` is written into the index, and `reason` may be
   `decode_error: …`.
   This difference affects only optional fields in the index; **the payload bytes are exactly the same**.

All these thresholds are written into the entry's `reason` field, so that it is possible afterwards to explain
"why this image was not compressed down".

> `reason` is diagnostic information only and is not part of the format contract; a second implementation may write
> different values, and differential comparison treats it as an ignorable difference.

---

## 7. Hashes and Checksums

### 7.1 `canonical_pixel_hash`

```
h = SHA-256()
h.update(b"PXC-PIXELS\x00")          # 11 bytes
h.update(mode.encode("utf-8"))       # ASCII, e.g. "RGB" / "I;16"
h.update(LE32(width))
h.update(LE32(height))
h.update(samples)                    # H*W*C samples, row-major, host byte order
digest = h.finalize()                # 32 bytes
```

`samples` are the complete pixels of `channels` channels (in the constant-alpha case **including the omitted alpha
plane** — computed before alpha is dropped when encoding, and after alpha is restored when decoding).

**The samples covered by this hash use host byte order** (the semantics of `ndarray.tobytes()`), not the
little-endian rule specified in §1.2. On x86-64 the two agree; on a big-endian host the hash of `uint16` images
differs. This is a known defect, see §10.2.

### 7.2 Three Layers of Checksums

| Layer | Subject of the checksum | Stored in |
|---|---|---|
| PXCS layer | payload plaintext | the index's `data_crc32` / `data_sha256` |
| PXC chunk layer | each chunk | the chunk directory's `crc` |
| tile layer | each tile's bytes before and after compression | the tile header's `payload_crc` / `raw_crc` |
| image layer | the reconstructed pixels | the header's `pixel_hash` + the index's `pixel_sha256` |

Every implementation must **verify before use**; the tile-layer checksum is especially important because the tile
decoder allocates memory according to `raw_len`.

### 7.3 Decoder Resource-Protection Limits

The reference implementation enforces the following caps; new implementations are advised to match them, otherwise a
malicious file will blow up memory:

| Item | Cap |
|---|---|
| `tile_size` | 8..1024 |
| `tile_count` | 10,000,000 |
| `width * height` | 300,000,000 |
| single tile `raw_len` | `max(1024, tw*th*C*16 + 1 MiB)` |
| Metadata entry count | 64 |
| Metadata total length | 64 MiB |
| single PXC file | constrained by the PXCS entry `length` |

---

## 8. Determinism and the Comparison Contract

### 8.1 payloads Are Deterministic

For the same input image and the same parameters, the reference implementation produces **completely identical** PXC
bytes (measured). Golden vectors can therefore be compared byte by byte.

### 8.2 Whole Archives Are Not Deterministic

Each PXCS archive's `archive_id` is a random `uuid4()` value, and it participates in the payload's AEAD
AAD and in the index, so **the whole-file hash is never reproducible**.

Conclusion: **"identical whole-file SHA256" cannot be used as the acceptance criterion for a reimplementation.**

### 8.3 The Correct Comparison Contract

Archives A and B produced by two implementations are judged equivalent if and only if:

1. the version numbers (major / minor) are the same;
2. the entry count and the set of entry paths are the same;
3. for every entry, `codec`, plaintext payload **byte content**, `size`, `ref`, `depth`,
   `quant`, `centered`, `lossy`, `pixel_sha256`, `raw_crc32` are the same;
4. the following may differ: `archive_id`, the per-layer offsets, the per-layer CRCs (because they derive from offsets
   or random values), and the `entropy_version` field inside PXC.

`entropy_version` in item 4 must be allowed through, otherwise any zlib version difference makes the comparison fail.
`tools/pxcs_compare.py` implements this contract.

### 8.4 zlib Version Policy

zlib's compressed output is **not guaranteed to be byte-identical** across versions/different branches (zlib,
zlib-ng, Chromium zlib), even at the same level.

Therefore:

- **semantic compatibility** is mandatory (can read each other; the decoded pixels agree);
- **byte identity** is guaranteed only when "the same zlib implementation and version" is used.

If cross-implementation byte identity is needed in the future, the zlib dependency version must be pinned and
reflected in the comparison tool.
Current policy: **accept semantic compatibility, pin the golden vectors to the zlib version used at generation time**,
record the `zlib` version in the vectors, and have the tests warn rather than fail when the version does not match.

---

## 9. Consistency Checklist for a New Implementation

A new codec implementation is equivalent only if it passes the following checks:

1. It can decode all golden vectors (§11), reconstructing pixels byte by byte.
2. It can **re-encode** the input images of the golden vectors into exactly the same payload bytes.
3. It can read archives written by PXCS 1.0 – 1.3, including `passthrough`, `webp_lossless`,
   and all residual combinations.
4. On corrupt data it returns a clear error, without crashing and without silently producing wrong pixels (§7.2/§7.3).
5. The three edge paths — constant alpha, `I;16B` big-endian input, and `P` mode palettes — are correct.
6. Non-divisible tile boundaries (width/height not an integer multiple of `tile_size`) are correct.
7. `FILTER` can decode all 5 types, but writes only 3.
8. It interoperates with the reference implementation: A writes → B reads, B writes → A reads, passing in both
   directions with `tools/pxcs_compare.py` and `verify`.

---

## 10. Identified Ambiguities and Rulings

### 10.1 The Dual Palette Ordering Rule

`C == 1` uses numeric order, `C > 1` uses byte order (§4.3). The two mechanisms differ, but because
all `C > 1` modes are 8-bit the actual results are consistent; the 16-bit `C == 1` modes are the only
place that needs care (numeric order is required). A new implementation must implement this as described in §4.3
and must not "simplify it into a single scheme".

### 10.2 The Host-Byte-Order Dependency of `canonical_pixel_hash`

The sample serialization of §7.1 uses host byte order, which conflicts with the little-endian rule of §1.2 on
big-endian hosts.
**Ruling**: keep the status quo (on x86 it is little-endian, so the practical impact is zero), and note it in the
documentation.
If cross-architecture archive interoperability is required in the future, the PXC minor version must be bumped and the
hash changed to explicit little-endian.

### 10.3 The Ignorability of `entropy_version`

This field of §2.2 varies with the zlib version. **Ruling**: treat it as an ignorable difference; it does not
participate in the equivalence judgement.

### 10.4 The 1/16 Sampling Pre-Filter of PALETTE

The candidate admission of §3.4 uses approximate sampling, which misses some tiles that could have used PALETTE.
**Ruling**: this is encoder behavior rather than format, but for byte identity it must be reproduced verbatim.

### 10.5 Numeric Mismatch Between `MODE_RANK` and `TILE_MODE`

The tile mode values are `SOLID=0,PALETTE=1,PRED=2,RUN=3,RAW=4,FILTER=5`,
while the tie-breaking rank order is `SOLID,PALETTE,PRED,RUN,FILTER,RAW` (FILTER before RAW).
The two numbering systems differ and are easy to misread. **Ruling**: keep it as is; the ranking table is defined
explicitly in §3.4.

---

## 11. Golden Vectors

Byte-level test vectors are stored under `tests/fixtures/golden/`; the format is described in that directory's `README.md`.

Each vector contains:

- `case.json`: the description (mode, dimensions, tile_size, generation parameters, zlib version)
- `input.png` (or a pixel array generated in-program, see the `source` field of `manifest.json`)
- `expected.pxc`: the expected PXC payload bytes

The coverage must include: all 7 tile modes, `uint8`/`uint16`/`int32`, grayscale/RGB/RGBA/
palette, constant alpha, non-divisible tile boundaries, and `tile_size` boundary values.

Generation/verification commands:

```powershell
python -m tools.gen_golden          # regenerate
python -m tools.gen_golden --check  # check that the current implementation is still byte-identical
```

When adding or modifying encoding behavior, the vectors must be updated in sync, and the reason for the byte
difference must be explained.

The correct way to compare across implementations is to use `tools/pxcs_compare.py` (compare by entry plaintext
payload, ignoring `archive_id` and the CRCs derived from it), rather than comparing whole-file hashes — see §8.2.
