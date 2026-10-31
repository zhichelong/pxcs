# PXCS Container Format

This document describes the binary layout, version history, and compatibility rules of PXCS. All integers are little-endian.

- Current implementation: 1.4 (supports up to minor=4)
- Unencrypted archives are written with minor=2 (identical to 1.2)
- Archives containing jpegxl entries are written with minor=4 (older programs report an unknown codec)
- Encrypted archives are written with minor=3

## 1. Overall Layout

### 1.1 Unencrypted (1.0 – 1.2, and unencrypted archives of 1.3)

```text
+------------------------------+ 0
| Header           128 B       |
+------------------------------+ 128
| payload 0 (8-byte aligned)   |
| payload 1                    |
| ...                          |
+------------------------------+ index_offset
| zlib(JSON Index)             |
+------------------------------+
| Footer           64 B        |
+------------------------------+ file_size
```

### 1.2 Encrypted (from 1.3)

```text
+------------------------------+ 0
| Header           128 B       |
+------------------------------+ 128
| Crypto Header    128 B       |  plaintext, containing KDF parameters/salt/check block/index MAC
+------------------------------+ 256  = data_offset
| payload 0 (AEAD ciphertext)  |
| ...                          |
+------------------------------+ index_offset
| Index (see below)            |
+------------------------------+
| Footer           64 B        |
+------------------------------+ file_size
```

When encryption is used and filenames are encrypted, the Index is `nonce(12) || AES-256-GCM(zlib(JSON))`;
otherwise it is still `zlib(JSON)` (integrity is then protected by `index_mac` in the Crypto Header).

## 2. Header (128 bytes)

Structure: `<4sHHIHIQQQ16s4s62sI`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 | magic | `PXCS` |
| 4 | 2 | version_major | must be 1 |
| 6 | 2 | version_minor | 0..4; currently written as 2 / 3 / 4 |
| 8 | 4 | flags | see below |
| 12 | 2 | header_size | fixed at 128 |
| 14 | 4 | entry_count | number of entries |
| 18 | 8 | index_offset | start of the Index (logical offset) |
| 26 | 8 | index_length | Index length |
| 34 | 8 | data_offset | 128 (unencrypted) or 256 (encrypted) |
| 42 | 16 | archive_id | random UUID; must be identical in Header/Footer/Index |
| 58 | 4 | reserved4 | must be 0 |
| 62 | 62 | reserved62 | must be 0 |
| 124 | 4 | header_crc | CRC32 of the first 124 bytes (computed with this field set to 0) |

flags:

| Bit | Name | Meaning |
|---|---|---|
| 0 | HAS_DELTA | delta entries present |
| 1 | HAS_FILES | non-image files present |
| 2 | HAS_DUP | duplicate entries present |
| 3 | HAS_RAW_IMAGES | raw bitstream images present |
| 4 | HAS_LOSSY_DELTA | residual is quantized (approximate pixels) |
| 5 | ENCRYPTED | 1.3: payload/index use AEAD |

## 3. Footer (64 bytes)

Structure: `<8sQQQII16s8s`

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic = `PXCSEND\0` |
| 8 | 8 | file_size (total logical length) |
| 16 | 8 | index_offset |
| 24 | 8 | index_length |
| 32 | 4 | index_crc32 (CRC32 of the Index ciphertext/blobs) |
| 36 | 4 | footer_crc32 (computed with this field set to 0) |
| 40 | 16 | archive_id |
| 56 | 8 | reserved (must be 0) |

## 4. Index (JSON + zlib)

```json
{
  "format": "PXCS",
  "version": 1,
  "archive_id": "<32 hex>",
  "created_by": "PXCS Python 1.3",
  "entries": [ ... ]
}
```

Entry fields (defaults apply to older archives for backward compatibility):

| Field | Type | Description | Introduced in |
|---|---|---|---|
| id | int | contiguous, equal to the array index | 1.0 |
| path | str | POSIX relative path | 1.0 |
| kind | str | `image` / `file` | 1.0 |
| codec | str | pxc / pxc_residual / zlib_residual / webp_lossless / raw / jpegxl / store / zlib / dup | 1.0 |
| offset | int | payload start (logical offset) | 1.0 |
| length | int | number of bytes the payload occupies in the archive (includes nonce/tag when encrypted) | 1.0 |
| plain_length | int? | decrypted payload length; **written only for encrypted archives**; the key is absent when unencrypted (keeping the 1.2 index bytes unchanged). Defaults to length when absent | 1.3 |
| size | int | original file size | 1.0 |
| ref | int? | referenced entry id (must be less than its own id) | 1.0 |
| depth | int | reference depth 0..255 | 1.0 |
| width/height/mode | | image properties | 1.0 |
| quant | int | residual quantization step, 0 = pixel-lossless | 1.2 |
| centered | bool | residual is centered on 128 | 1.2 |
| lossy | bool | reconstructed pixels are approximate | 1.2 |
| pixel_sha256 | str | expected pixel hash (for lossy delta, the hash of the approximate image). **raw entries are usually empty**: the payload is the source file bytes, and the byte-level SHA-256 already covers them completely; an empty value means no pixel-level comparison is performed | 1.0 |
| source_sha256 | str | source file byte hash | 1.0 |
| data_crc32 / data_sha256 | | CRC/SHA of the **plaintext payload** | 1.0 |
| raw_crc32 | int | raw file byte CRC (raw/store/zlib) | 1.0 |
| mtime_ns | int | modification time | 1.0 |
| reason | str | encoding decision reason, for troubleshooting | 1.2 |

For encrypted archives, `data_crc32/data_sha256` still record the **plaintext** digest:
on read, AEAD decryption is performed first and verification second, so tampering with the ciphertext is caught by the GCM tag first.

## 5. Crypto Header (128 bytes, 1.3)

Structure: `<8sHHHH16sIII44s32s4sI`

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 8 | magic | `PXCSCRYP` |
| 8 | 2 | version | 1 |
| 10 | 2 | kdf_id | 1=Argon2id, 2=scrypt |
| 12 | 2 | aead_id | 1=AES-256-GCM |
| 14 | 2 | flags | bit0=filenames encrypted |
| 16 | 16 | salt | random salt |
| 32 | 4 | param1 | Argon2id: memory KiB; scrypt: log2(N) |
| 36 | 4 | param2 | Argon2id: iterations; scrypt: r |
| 40 | 4 | param3 | Argon2id: lanes; scrypt: p |
| 44 | 44 | check_blob | `nonce(12) || AEAD(token16)`, used for passphrase verification |
| 88 | 32 | index_mac | HMAC-SHA256(Index plaintext), used when filenames are not encrypted |
| 120 | 4 | reserved | must be 0 |
| 124 | 4 | header_crc | CRC32 (computed with this field set to 0) |

Key derivation:

```text
master     = KDF(password_NFC, salt, params)
payload_key= HKDF-SHA256(master, info="pxcs/payload-key/v1", 32)
index_key  = HKDF-SHA256(master, info="pxcs/index-key/v1",   32)
check_key  = HKDF-SHA256(master, info="pxcs/check-key/v1",   32)
mac_key    = HKDF-SHA256(master, info="pxcs/index-mac/v1",   32)
```

AEAD messages:

```text
payload = nonce(12) || AES-256-GCM(plaintext, aad = "PXCS1:payload" || archive_id)
index   = nonce(12) || AES-256-GCM(zlib(json), aad = "PXCS1:index" || archive_id)
check   = nonce(12) || AES-256-GCM(token16,  aad = "PXCS1:check"  || archive_id)
```

Each nonce is random and independent per message; the entry count is limited to 1,000,000, far below the safety bound for 96-bit random nonces.
`index_mac = HMAC-SHA256(mac_key, json_bytes)` is written only when filenames are not encrypted,
and is used to prevent changes to paths, offsets, and lengths.

## 6. Volume Splitting (1.3)

Volume splitting does not change any of the structures above: the byte stream of 1.1/1.2 is split by capacity, and a 128-byte volume header is prepended to each volume.

Volume header: `<8sHHIIQQ16s72sI` (128 bytes)

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 8 | magic | `PXCSVOL\0` |
| 8 | 2 | version | 1 |
| 10 | 2 | flags | bit0=this volume is the final volume |
| 12 | 4 | volume_index | starts at 1; must match the filename |
| 16 | 4 | volume_count | total number of volumes |
| 20 | 8 | logical_offset | offset of this volume's first byte within the logical stream |
| 28 | 8 | length | number of logical bytes carried by this volume |
| 36 | 16 | archive_id | matches the inner archive |
| 52 | 72 | reserved | must be 0 |
| 124 | 4 | header_crc | CRC32 |

Constraints (each validated on read):

- filenames are `<base>.<3+ digits>`, and volume numbers are consecutive;
- `volume_index` matches the filename, and `1..volume_count` are all present;
- `logical_offset[i] == sum(length[0..i-1])` (no gaps/overlaps are permitted);
- the actual file size == `128 + length`;
- only the last volume carries the FINAL flag;
- all volumes have the same `archive_id`.

On read, `(logical offset, length)` is mapped to `(volume number, offset within volume)`, so **random extraction of a single image is not invalidated by volume splitting**;
a payload spanning volumes is read in segments by the read layer and then concatenated.

## 7. Version History and Compatibility Rules

| Version | Change | Compatibility strategy |
|---|---|---|
| 1.0 | initial format | — |
| 1.1 | delta/keyframe refinement | missing 1.0 fields default on read |
| 1.2 | entries add `quant/centered/lossy`; quantized residual | old entries default to 0/False on read |
| 1.3 | crypto header, AEAD payload, `plain_length`, volume splitting | unencrypted still writes minor=2; encrypted writes minor=3 |
| 1.4 | adds `jpegxl`: JPEG is losslessly repacked as JPEG XL, enabling byte-exact restoration of the original file | old readers report an error on this codec; `--no-jxl` falls back to 1.2 output |

Parsing rules:

1. `magic` must be `PXCS`, and `version_major` must be 1;
2. `version_minor` not in `(0,1,2,3,4)` → `PxcVersionError`;
3. `flags & ENCRYPTED`:
   - permitted only when `minor >= 3`;
   - `data_offset` must equal 256, and the crypto header must be present;
   - no passphrase → `PxcPasswordRequired`; wrong passphrase/tampering → `PxcAuthenticationError`;
4. unencrypted:
   - `data_offset` must equal 128;
   - encryption-related code does not participate at all, and old archive behavior is unchanged;
5. entry validation (offset out of bounds, ref relationships, path normalization, codec validity) is performed uniformly after the Index is decrypted.

Written version selection:

- no passphrase → `minor = 2` (readable by older programs);
- passphrase present → `minor = 3`;
- whether volume splitting is used does not affect the written minor.

## 8. Known Limitations

- volume splitting has no redundancy and cannot recover data from a lost volume;
- encryption does not hide the total archive size, entry count, or the length of each payload (only the content and filenames are protected);
- the Index as a whole remains subject to the 256 MiB limit;
- a single payload must be loadable into memory in its entirety (large images depend on memory, not on volume splitting).
