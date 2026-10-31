"""PXC 0.1 二进制容器：Header、Metadata TLV、Chunk Directory、DATA/ORIG、Footer。

所有整数均为小端序。多字节偏移使用 u64，便于大文件扩展。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import (
    ALPHA_CONSTANT,
    COMPRESSION_PASSTHROUGH,
    FLAG_PASSTHROUGH,
    META_CONSTANT_ALPHA,
    META_TYPE_U32,
    CHUNK_DATA,
    CHUNK_ENTRY_SIZE,
    CHUNK_ENTRY_STRUCT,
    CHUNK_META,
    CHUNK_ORIG,
    COMPRESSION_PXC_TILE_ZLIB,
    CREATED_BY,
    ENTROPY_NAME,
    FLAG_HAS_ALPHA,
    FLAG_HAS_ORIG,
    FLAG_HAS_PALETTE,
    FLAG_SOURCE_LOSSY,
    HEADER_SIZE,
    HEADER_STRUCT,
    LOSSY_SOURCE_FORMATS,
    MAX_CHUNKS,
    MAX_HEIGHT,
    MAX_METADATA_BYTES,
    MAX_METADATA_ENTRIES,
    MAX_PIXELS,
    MAX_TILE_SIZE,
    MAX_WIDTH,
    MIN_TILE_SIZE,
    META_COMMENT,
    META_CREATED_BY,
    META_DPI,
    META_ENTROPY_NAME,
    META_ENTROPY_VERSION,
    META_EXIF,
    META_GAMMA,
    META_ICC_PROFILE,
    META_KEY_NAMES,
    META_MODE_NAME,
    META_ORIGINAL_FILENAME,
    META_PALETTE,
    META_SOURCE_EXT,
    META_SOURCE_FILE_HASH,
    META_SOURCE_FORMAT,
    META_SOURCE_PIXEL_HASH,
    META_TRANSPARENCY,
    META_TYPE_BYTES,
    META_TYPE_F64,
    META_TYPE_F64X2,
    META_TYPE_U32,
    META_TYPE_U64,
    META_TYPE_UTF8,
    META_XMP,
    MODE_ID_TO_NAME,
    MODE_NAME_TO_ID,
    MODE_SPECS,
    PXC_END_MAGIC,
    PXC_MAGIC,
    TILE_MODE_NAMES,
    VERSION_MAJOR,
    VERSION_MINOR,
)
from .errors import (
    PxcCorruptionError,
    PxcFormatError,
    PxcVersionError,
)
from .image_io import ImageData
from .utils import crc32

FOOTER_SIZE = 16
FOOTER_STRUCT = struct.Struct("<8sII")
META_ENTRY_STRUCT = struct.Struct("<HBBI")
META_HEADER_STRUCT = struct.Struct("<I")


@dataclass
class PxcHeader:
    version_major: int
    version_minor: int
    flags: int
    header_size: int
    file_size: int
    metadata_offset: int
    metadata_length: int
    chunk_dir_offset: int
    chunk_count: int
    width: int
    height: int
    mode_id: int
    bit_depth: int
    channels: int
    color_model: int
    alpha_kind: int
    compression_id: int
    tile_size: int
    original_size: int
    data_length: int
    pixel_hash: bytes


@dataclass
class ChunkEntry:
    chunk_id: bytes
    flags: int
    offset: int
    length: int
    crc: int


@dataclass
class PxcFile:
    header: PxcHeader
    metadata: dict[int, tuple[int, bytes]]
    data_payload: bytes
    orig_payload: Optional[bytes]
    chunks: dict[bytes, ChunkEntry]

    def meta_bytes(self, key: int) -> Optional[bytes]:
        entry = self.metadata.get(key)
        if entry is None:
            return None
        typ, payload = entry
        if typ != META_TYPE_BYTES:
            raise PxcFormatError(f"元数据 {META_KEY_NAMES.get(key, key)} 类型不是 bytes")
        return payload

    def meta_text(self, key: int) -> Optional[str]:
        entry = self.metadata.get(key)
        if entry is None:
            return None
        typ, payload = entry
        if typ != META_TYPE_UTF8:
            raise PxcFormatError(f"元数据 {META_KEY_NAMES.get(key, key)} 类型不是 utf8")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PxcFormatError(
                f"元数据 {META_KEY_NAMES.get(key, key)} 不是合法 UTF-8"
            ) from exc


def _add_text(entries: dict[int, tuple[int, bytes]], key: int, value: str) -> None:
    entries[key] = (META_TYPE_UTF8, value.encode("utf-8"))


def _add_bytes(entries: dict[int, tuple[int, bytes]], key: int, value: Optional[bytes]) -> None:
    if value is not None:
        entries[key] = (META_TYPE_BYTES, bytes(value))


def build_metadata(image: ImageData) -> bytes:
    entries: dict[int, tuple[int, bytes]] = {}
    _add_text(entries, META_ORIGINAL_FILENAME, image.original_filename)
    _add_text(entries, META_SOURCE_FORMAT, image.source_format)
    _add_text(entries, META_SOURCE_EXT, image.source_ext)
    _add_text(entries, META_MODE_NAME, image.mode)
    _add_text(entries, META_ENTROPY_NAME, ENTROPY_NAME)
    _add_text(entries, META_ENTROPY_VERSION, zlib.ZLIB_RUNTIME_VERSION)
    _add_text(entries, META_CREATED_BY, CREATED_BY)

    _add_bytes(entries, META_PALETTE, image.palette)
    _add_bytes(entries, META_TRANSPARENCY, image.transparency)
    _add_bytes(entries, META_ICC_PROFILE, image.icc_profile)
    _add_bytes(entries, META_EXIF, image.exif)
    _add_bytes(entries, META_XMP, image.xmp)
    _add_bytes(entries, META_COMMENT, image.comment)
    _add_bytes(entries, META_SOURCE_FILE_HASH, image.source_file_hash)
    _add_bytes(entries, META_SOURCE_PIXEL_HASH, image.source_pixel_hash)

    if image.constant_alpha is not None:
        entries[META_CONSTANT_ALPHA] = (
            META_TYPE_U32,
            struct.pack("<I", int(image.constant_alpha)),
        )
    if image.dpi is not None:
        entries[META_DPI] = (META_TYPE_F64X2, struct.pack("<dd", *image.dpi))
    if image.gamma is not None:
        entries[META_GAMMA] = (META_TYPE_F64, struct.pack("<d", image.gamma))

    return encode_metadata(entries)


def encode_metadata(entries: dict[int, tuple[int, bytes]]) -> bytes:
    if len(entries) > MAX_METADATA_ENTRIES:
        raise PxcFormatError("元数据条目过多")
    parts = [META_HEADER_STRUCT.pack(len(entries))]
    total = META_HEADER_STRUCT.size
    for key in sorted(entries):
        typ, payload = entries[key]
        if not (0 <= key <= 0xFFFF):
            raise PxcFormatError(f"元数据 key 越界: {key}")
        if typ not in (META_TYPE_BYTES, META_TYPE_UTF8, META_TYPE_U32, META_TYPE_U64, META_TYPE_F64X2, META_TYPE_F64):
            raise PxcFormatError(f"元数据 type 非法: {typ}")
        expected = {
            META_TYPE_U32: 4,
            META_TYPE_U64: 8,
            META_TYPE_F64X2: 16,
            META_TYPE_F64: 8,
        }.get(typ)
        if expected is not None and len(payload) != expected:
            raise PxcFormatError(f"元数据 {key} 长度应为 {expected}")
        parts.append(META_ENTRY_STRUCT.pack(key, typ, 0, len(payload)))
        parts.append(payload)
        total += META_ENTRY_STRUCT.size + len(payload)
        if total > MAX_METADATA_BYTES:
            raise PxcFormatError("元数据总长度超出上限")
    return b"".join(parts)


def decode_metadata(blob: bytes) -> dict[int, tuple[int, bytes]]:
    if len(blob) < META_HEADER_STRUCT.size:
        raise PxcFormatError("元数据块过短")
    count = META_HEADER_STRUCT.unpack_from(blob, 0)[0]
    if count > MAX_METADATA_ENTRIES:
        raise PxcFormatError(f"元数据条目过多: {count}")
    offset = META_HEADER_STRUCT.size
    result: dict[int, tuple[int, bytes]] = {}
    for _ in range(count):
        if offset + META_ENTRY_STRUCT.size > len(blob):
            raise PxcFormatError("元数据条目头不完整")
        key, typ, reserved, length = META_ENTRY_STRUCT.unpack_from(blob, offset)
        offset += META_ENTRY_STRUCT.size
        if reserved != 0:
            raise PxcFormatError("元数据 reserved 字段非 0")
        if length > len(blob) - offset:
            raise PxcFormatError("元数据条目长度越界")
        payload = blob[offset : offset + length]
        offset += length
        if key in result:
            raise PxcFormatError(f"元数据 key 重复: {key}")
        expected = {
            META_TYPE_U32: 4,
            META_TYPE_U64: 8,
            META_TYPE_F64X2: 16,
            META_TYPE_F64: 8,
        }.get(typ)
        if expected is not None and length != expected:
            raise PxcFormatError(f"元数据 {key} 长度应为 {expected}")
        result[key] = (typ, payload)
    if offset != len(blob):
        raise PxcFormatError("元数据块存在多余数据")
    return result


def build_pxc(
    image: ImageData,
    data_payload: bytes,
    orig_payload: Optional[bytes] = None,
    compression_id: int = COMPRESSION_PXC_TILE_ZLIB,
    tile_size: Optional[int] = None,
    header_channels: Optional[int] = None,
    header_alpha_kind: Optional[int] = None,
) -> bytes:
    """组装完整 PXC 文件。

    compression_id=1：DATA 是 tile 流；
    compression_id=2：DATA 是原始文件字节（passthrough）。
    """
    if not data_payload:
        raise PxcFormatError("DATA payload 为空")
    if len(image.pixel_hash) != 32:
        raise PxcFormatError("pixel_hash 必须为 32 字节")
    if compression_id not in (COMPRESSION_PXC_TILE_ZLIB, COMPRESSION_PASSTHROUGH):
        raise PxcFormatError(f"不支持的 compression_id: {compression_id}")

    if header_channels is None:
        header_channels = image.stored_channels
    if header_alpha_kind is None:
        header_alpha_kind = image.alpha_kind

    meta = build_metadata(image)
    chunks: list[tuple[bytes, bytes]] = [(CHUNK_META, meta), (CHUNK_DATA, data_payload)]
    if orig_payload is not None:
        chunks.append((CHUNK_ORIG, orig_payload))
    if len(chunks) > MAX_CHUNKS:
        raise PxcFormatError("chunk 数量过多")

    meta_offset = HEADER_SIZE
    dir_offset = meta_offset + len(meta)
    payload_offset = dir_offset + CHUNK_ENTRY_SIZE * len(chunks)
    entries: list[ChunkEntry] = [
        ChunkEntry(
            chunk_id=CHUNK_META,
            flags=0,
            offset=meta_offset,
            length=len(meta),
            crc=crc32(meta),
        )
    ]
    cursor = payload_offset
    for chunk_id, payload in chunks[1:]:
        entries.append(
            ChunkEntry(
                chunk_id=chunk_id,
                flags=0,
                offset=cursor,
                length=len(payload),
                crc=crc32(payload),
            )
        )
        cursor += len(payload)
    file_size = cursor + FOOTER_SIZE

    if tile_size is None:
        if compression_id != COMPRESSION_PXC_TILE_ZLIB:
            tile_size = 0
        else:
            if len(data_payload) < 8:
                raise PxcFormatError("DATA payload 过短")
            tile_size = struct.unpack_from("<I", data_payload, 4)[0]
    if compression_id == COMPRESSION_PASSTHROUGH and tile_size != 0:
        raise PxcFormatError("passthrough 的 tile_size 必须为 0")
    if compression_id == COMPRESSION_PXC_TILE_ZLIB and not (
        MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE
    ):
        raise PxcFormatError(f"tile_size 非法: {tile_size}")

    flags = 0
    if orig_payload is not None:
        flags |= FLAG_HAS_ORIG
    if compression_id == COMPRESSION_PASSTHROUGH:
        flags |= FLAG_PASSTHROUGH
    if image.has_alpha:
        flags |= FLAG_HAS_ALPHA
    if image.has_palette:
        flags |= FLAG_HAS_PALETTE
    if image.source_format.upper() in LOSSY_SOURCE_FORMATS:
        flags |= FLAG_SOURCE_LOSSY

    def pack_header_with_tile(header_crc: int) -> bytes:
        return HEADER_STRUCT.pack(
            PXC_MAGIC,
            VERSION_MAJOR,
            VERSION_MINOR,
            flags,
            HEADER_SIZE,
            header_crc,
            file_size,
            meta_offset,
            len(meta),
            dir_offset,
            len(chunks),
            image.width,
            image.height,
            MODE_NAME_TO_ID[image.mode],
            image.bit_depth,
            header_channels,
            image.color_model,
            header_alpha_kind,
            compression_id,
            tile_size,
            image.source_file_size,
            len(data_payload),
            image.pixel_hash,
            0,
        )

    zero_header = pack_header_with_tile(0)
    header_crc = crc32(zero_header)
    header = pack_header_with_tile(header_crc)

    out = bytearray()
    out.extend(header)
    out.extend(meta)
    for entry in entries:
        out.extend(
            CHUNK_ENTRY_STRUCT.pack(
                entry.chunk_id, entry.flags, entry.offset, entry.length, entry.crc, 0
            )
        )
    for _chunk_id, payload in chunks[1:]:
        out.extend(payload)
    if len(out) != file_size - FOOTER_SIZE:
        raise PxcFormatError("内部错误：布局长度不一致")
    file_crc = crc32(bytes(out))
    out.extend(FOOTER_STRUCT.pack(PXC_END_MAGIC, file_crc, 0))
    if len(out) != file_size:
        raise PxcFormatError("内部错误：文件长度不一致")
    return bytes(out)


def read_pxc(path: Path) -> PxcFile:
    return parse_pxc_bytes(path.read_bytes())


def parse_pxc_bytes(raw: bytes) -> PxcFile:
    """从内存字节解析 PXC；PXCS 归档内嵌 PXC 时使用。"""
    if len(raw) < HEADER_SIZE + FOOTER_SIZE + 2 * CHUNK_ENTRY_SIZE:
        raise PxcFormatError("文件过短，不是有效的 PXC")
    if raw[:4] != PXC_MAGIC:
        raise PxcFormatError("magic 不匹配，不是 PXC 文件")
    try:
        (
            magic,
            major,
            minor,
            flags,
            header_size,
            header_crc,
            file_size,
            meta_offset,
            meta_len,
            dir_offset,
            chunk_count,
            width,
            height,
            mode_id,
            bit_depth,
            channels,
            color_model,
            alpha_kind,
            compression_id,
            tile_size,
            original_size,
            data_length,
            pixel_hash,
            reserved,
        ) = HEADER_STRUCT.unpack_from(raw, 0)
    except struct.error as exc:
        raise PxcFormatError(f"文件头不完整: {exc}") from exc

    if major != VERSION_MAJOR:
        raise PxcVersionError(
            f"不支持的 PXC 主版本 {major}.{minor}（当前 {VERSION_MAJOR}.{VERSION_MINOR}）"
        )
    if minor > VERSION_MINOR:
        raise PxcVersionError(
            f"PXC 文件版本 {major}.{minor} 高于当前支持的 {VERSION_MAJOR}.{VERSION_MINOR}"
        )
    if header_size != HEADER_SIZE:
        raise PxcFormatError(f"header_size 非法: {header_size}")
    if reserved != 0:
        raise PxcFormatError("header reserved 字段非 0")
    if file_size != len(raw):
        raise PxcFormatError(
            f"file_size 不一致：header={file_size}, actual={len(raw)}"
        )
    header_bytes = bytearray(raw[:HEADER_SIZE])
    header_bytes[12:16] = b"\x00\x00\x00\x00"
    if crc32(bytes(header_bytes)) != header_crc:
        raise PxcCorruptionError("文件头 CRC 校验失败")

    footer_offset = len(raw) - FOOTER_SIZE
    if footer_offset < HEADER_SIZE:
        raise PxcFormatError("文件过短，缺少 footer")
    end_magic, file_crc, footer_reserved = FOOTER_STRUCT.unpack_from(raw, footer_offset)
    if end_magic != PXC_END_MAGIC:
        raise PxcFormatError("footer magic 不匹配")
    if footer_reserved != 0:
        raise PxcFormatError("footer reserved 字段非 0")
    if crc32(raw[:footer_offset]) != file_crc:
        raise PxcCorruptionError("文件整体 CRC 校验失败")

    if not (HEADER_SIZE <= meta_offset <= footer_offset):
        raise PxcFormatError("metadata_offset 越界")
    if not (0 <= meta_len <= footer_offset - meta_offset):
        raise PxcFormatError("metadata_length 越界")
    if meta_offset + meta_len > footer_offset:
        raise PxcFormatError("metadata 越界")
    if not (HEADER_SIZE <= dir_offset <= footer_offset):
        raise PxcFormatError("chunk_dir_offset 越界")
    if not (1 <= chunk_count <= MAX_CHUNKS):
        raise PxcFormatError(f"chunk_count 非法: {chunk_count}")
    dir_size = chunk_count * CHUNK_ENTRY_SIZE
    if dir_offset + dir_size > footer_offset:
        raise PxcFormatError("chunk 目录越界")
    if dir_offset < meta_offset + meta_len:
        raise PxcFormatError("chunk 目录与 metadata 重叠")

    entries: list[ChunkEntry] = []
    for i in range(chunk_count):
        pos = dir_offset + i * CHUNK_ENTRY_SIZE
        chunk_id, cflags, offset, length, ccrc, creserved = CHUNK_ENTRY_STRUCT.unpack_from(
            raw, pos
        )
        if creserved != 0:
            raise PxcFormatError("chunk 目录 reserved 字段非 0")
        if not (HEADER_SIZE <= offset <= footer_offset):
            raise PxcFormatError("chunk offset 越界")
        if not (0 <= length <= footer_offset - offset):
            raise PxcFormatError("chunk length 越界")
        if offset + length > footer_offset:
            raise PxcFormatError("chunk 数据越界")
        if crc32(raw[offset : offset + length]) != ccrc:
            raise PxcCorruptionError(f"chunk {chunk_id!r} CRC 校验失败")
        entries.append(ChunkEntry(chunk_id, cflags, offset, length, ccrc))

    chunks: dict[bytes, ChunkEntry] = {}
    for entry in entries:
        if entry.chunk_id in chunks:
            raise PxcFormatError(f"chunk 重复: {entry.chunk_id!r}")
        chunks[entry.chunk_id] = entry

    if CHUNK_META not in chunks:
        raise PxcFormatError("缺少 META chunk")
    if CHUNK_DATA not in chunks:
        raise PxcFormatError("缺少 DATA chunk")
    if chunks[CHUNK_META].offset != meta_offset or chunks[CHUNK_META].length != meta_len:
        raise PxcFormatError("META chunk 与 header 指向不一致")
    if chunks[CHUNK_DATA].length != data_length:
        raise PxcFormatError("DATA chunk 长度与 header 不一致")

    meta_blob = raw[meta_offset : meta_offset + meta_len]
    metadata = decode_metadata(meta_blob)
    data_payload = raw[
        chunks[CHUNK_DATA].offset : chunks[CHUNK_DATA].offset + chunks[CHUNK_DATA].length
    ]
    orig_payload = None
    if CHUNK_ORIG in chunks:
        entry = chunks[CHUNK_ORIG]
        orig_payload = raw[entry.offset : entry.offset + entry.length]

    # header / metadata / 模式一致性检查
    if mode_id not in MODE_ID_TO_NAME:
        raise PxcFormatError(f"未知 mode_id: {mode_id}")
    mode = MODE_ID_TO_NAME[mode_id]
    spec = MODE_SPECS[mode]
    if width <= 0 or height <= 0 or width > MAX_WIDTH or height > MAX_HEIGHT:
        raise PxcFormatError("图像尺寸非法")
    if width * height > MAX_PIXELS:
        raise PxcFormatError("图像像素数超出安全上限")
    if alpha_kind == ALPHA_CONSTANT:
        expected_channels = spec["channels"] - 1
    else:
        expected_channels = spec["channels"]
    if channels != expected_channels:
        raise PxcFormatError("channels 与 mode/alpha 不一致")
    if bit_depth != spec["bit_depth"]:
        raise PxcFormatError("bit_depth 与 mode 不一致")
    if color_model != spec["color_model"]:
        raise PxcFormatError("color_model 与 mode 不一致")
    if alpha_kind not in (0, 1, 2, ALPHA_CONSTANT):
        raise PxcFormatError("alpha_kind 非法")
    if compression_id not in (COMPRESSION_PXC_TILE_ZLIB, COMPRESSION_PASSTHROUGH):
        raise PxcFormatError(f"不支持的 compression_id: {compression_id}")
    if compression_id == COMPRESSION_PASSTHROUGH:
        if tile_size != 0:
            raise PxcFormatError("passthrough 的 tile_size 必须为 0")
    elif not (MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE):
        raise PxcFormatError(f"tile_size 非法: {tile_size}")
    if len(pixel_hash) != 32:
        raise PxcFormatError("pixel_hash 长度错误")

    meta_mode = metadata.get(META_MODE_NAME)
    if meta_mode is not None:
        typ, payload = meta_mode
        try:
            meta_mode_name = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PxcFormatError("mode_name 不是合法 UTF-8") from exc
        if typ != META_TYPE_UTF8 or meta_mode_name != mode:
            raise PxcFormatError("mode_name 与 header 不一致")

    header = PxcHeader(
        version_major=major,
        version_minor=minor,
        flags=flags,
        header_size=header_size,
        file_size=file_size,
        metadata_offset=meta_offset,
        metadata_length=meta_len,
        chunk_dir_offset=dir_offset,
        chunk_count=chunk_count,
        width=width,
        height=height,
        mode_id=mode_id,
        bit_depth=bit_depth,
        channels=channels,
        color_model=color_model,
        alpha_kind=alpha_kind,
        compression_id=compression_id,
        tile_size=tile_size,
        original_size=original_size,
        data_length=data_length,
        pixel_hash=pixel_hash,
    )
    return PxcFile(
        header=header,
        metadata=metadata,
        data_payload=data_payload,
        orig_payload=orig_payload,
        chunks=chunks,
    )
