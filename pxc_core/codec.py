"""PXC tile 编解码与整图编码/解码。

第一版压缩算法：
- SOLID：整块同色。
- PALETTE：局部调色板 + 索引（可选索引 RLE）。
- PRED：MED 预测 + zigzag 残差。
- RUN：像素行程。
- RAW：原始样本兜底。
每个候选都用 zlib 试压，按“tile 头 + 实际存储字节”选择最小者。
"""
from __future__ import annotations

import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image

from .config import (
    ALPHA_CONSTANT,
    COMPRESSION_PASSTHROUGH,
    META_CONSTANT_ALPHA,
    META_TYPE_U32,
    DATA_FORMAT_VERSION,
    DATA_HEADER_SIZE,
    DATA_HEADER_STRUCT,
    FLAG_PAYLOAD_ZLIB,
    MAX_TILE_SIZE,
    MIN_TILE_SIZE,
    MODE_ID_TO_NAME,
    MODE_SPECS,
    META_DPI,
    META_EXIF,
    META_ICC_PROFILE,
    META_MODE_NAME,
    META_ORIGINAL_FILENAME,
    META_PALETTE,
    META_SOURCE_EXT,
    META_SOURCE_FILE_HASH,
    META_SOURCE_FORMAT,
    META_SOURCE_PIXEL_HASH,
    META_TRANSPARENCY,
    META_XMP,
    META_GAMMA,
    META_COMMENT,
    PALETTE_PLAIN,
    PALETTE_RLE,
    TILE_HEADER_SIZE,
    TILE_HEADER_STRUCT,
    TILE_MODE_NAMES,
    TILE_MODE_ORDER,
    TILE_MODE_FILTER,
    TILE_MODE_PALETTE,
    TILE_MODE_PRED,
    TILE_MODE_RAW,
    TILE_MODE_RUN,
    TILE_MODE_SOLID,
    ZLIB_LEVEL,
)
from .filters import FILTER_TYPES, choose_row_filters, unfilter_rows
from .errors import (
    PxcCorruptionError,
    PxcDecodeError,
    PxcFormatError,
    UnsupportedImageError,
)
from .image_io import (
    MODE_DTYPE,
    ImageData,
    canonical_pixel_hash,
    data_to_pillow,
    decode_transparency,
    load_image_bytes,
    storage_dtype,
)
from .palette import (
    build_palette,
    decode_palette_payload,
    decode_run_payload,
    palette_payload,
    run_payload,
    solid_payload,
)
from .predictor import decode_residuals, encode_residuals, residual_storage_dtype
from .utils import crc32, sha256_bytes


@dataclass
class TileRecord:
    mode: int
    submode: int
    flags: int
    raw_len: int
    comp_len: int
    raw_crc: int
    payload_crc: int
    payload: bytes


@dataclass
class TileStats:
    tiles: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)
    payload_bytes: dict[str, int] = field(default_factory=dict)
    raw_bytes: int = 0
    header_bytes: int = 0

    def add(self, rec: "TileRecord") -> None:
        name = TILE_MODE_NAMES.get(rec.mode, str(rec.mode))
        self.tiles += 1
        self.mode_counts[name] = self.mode_counts.get(name, 0) + 1
        self.payload_bytes[name] = self.payload_bytes.get(name, 0) + rec.comp_len
        self.raw_bytes += rec.raw_len
        self.header_bytes += TILE_HEADER_SIZE


def _compress(raw: bytes) -> tuple[bytes, int]:
    comp = zlib.compress(raw, ZLIB_LEVEL)
    if len(comp) < len(raw):
        return comp, FLAG_PAYLOAD_ZLIB
    return raw, 0


def _make_record(mode: int, submode: int, raw: bytes) -> TileRecord:
    stored, flags = _compress(raw)
    return TileRecord(
        mode=mode,
        submode=submode,
        flags=flags,
        raw_len=len(raw),
        comp_len=len(stored),
        raw_crc=crc32(raw),
        payload_crc=crc32(stored),
        payload=stored,
    )


def _candidate_cost(rec: TileRecord) -> tuple[int, int, int]:
    # 同成本时按固定模式顺序选择；RAW 放最后
    return (TILE_HEADER_SIZE + rec.comp_len, TILE_MODE_ORDER.index(rec.mode), rec.submode)


def _sample_unique_count(tile: np.ndarray, max_count: int = 257) -> int:
    """快速估计 tile 的颜色数；采样结果 > max_count 时可直接判定超过上限。"""
    c = tile.shape[2]
    sample = tile[::4, ::4, :]
    flat = np.ascontiguousarray(sample).reshape(-1, c)
    if c == 1:
        return int(np.unique(flat[:, 0]).size)
    sd = tile.dtype.itemsize
    void_dtype = np.dtype((np.void, c * sd))
    packed = flat.view(void_dtype).ravel()
    return int(np.unique(packed).size)



def encode_tile(tile: np.ndarray) -> TileRecord:
    """尝试所有 tile 模式，返回实际存储成本最小的记录。"""
    if tile.ndim != 3:
        raise ValueError("tile 必须是 (H,W,C)")
    h, w, c = tile.shape
    if h == 0 or w == 0:
        raise ValueError("空 tile")
    dtype = tile.dtype
    records: list[TileRecord] = []

    # SOLID
    if h and w:
        first = tile[0, 0]
        if bool(np.all(tile == first)):
            records.append(_make_record(TILE_MODE_SOLID, 0, solid_payload(tile, dtype)))

    # PALETTE（浮点不支持）。先用 1/16 采样排除颜色很多的照片 tile。
    if dtype.kind != "f" and _sample_unique_count(tile) <= 256:
        try:
            palette, inverse = build_palette(tile)
            k = palette.shape[0]
            if 1 < k <= 256:
                sd = storage_dtype(dtype)
                head = struct.pack("<H", k)
                entries = palette.astype(sd).tobytes()
                # 普通索引
                plain = head + entries + inverse.astype(np.uint8).tobytes()
                records.append(_make_record(TILE_MODE_PALETTE, PALETTE_PLAIN, plain))
                # 索引 RLE
                rle_body = _palette_runs_public(inverse)
                rle = head + entries + rle_body
                records.append(_make_record(TILE_MODE_PALETTE, PALETTE_RLE, rle))
        except (UnsupportedImageError, ValueError):
            pass

    # PRED
    if dtype.kind != "f":
        try:
            raw, _rdtype = encode_residuals(tile)
            records.append(_make_record(TILE_MODE_PRED, 0, raw))
        except UnsupportedImageError:
            pass

    # RUN：只有行程数不太多时才值得试；照片通常 runs≈像素数，可直接跳过。
    if dtype.kind != "f":
        flat = np.ascontiguousarray(tile).reshape(-1, c)
        if flat.shape[0] > 1:
            same = np.all(flat[1:] == flat[:-1], axis=1)
            runs = 1 + int(np.count_nonzero(~same))
            if runs <= max(8, flat.shape[0] // 4):
                try:
                    records.append(_make_record(TILE_MODE_RUN, 0, run_payload(tile, dtype)))
                except UnsupportedImageError:
                    pass

    # FILTER：PNG 风格的逐行滤波（行类型 + 滤波数据）
    if dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
        try:
            types, filtered = choose_row_filters(tile)
            sd = storage_dtype(dtype)
            payload = bytes(types.tobytes()) + filtered.astype(sd).tobytes()
            records.append(_make_record(TILE_MODE_FILTER, 0, payload))
        except (UnsupportedImageError, ValueError):
            pass

    # RAW
    sd = storage_dtype(dtype)
    raw = tile.astype(sd).tobytes()
    records.append(_make_record(TILE_MODE_RAW, 0, raw))

    if not records:
        raise UnsupportedImageError("没有可用的 tile 编码模式")
    return min(records, key=_candidate_cost)


def _palette_runs_public(indices: np.ndarray) -> bytes:
    """复用 palette 模块的 RLE 逻辑；输入是索引数组。"""
    from .palette import _palette_runs

    return _palette_runs(indices)


def _tile_shape(
    tile_index: int,
    tiles_x: int,
    tile_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int, int, int]:
    ty, tx = divmod(tile_index, tiles_x)
    x0 = tx * tile_size
    y0 = ty * tile_size
    x1 = min(x0 + tile_size, width)
    y1 = min(y0 + tile_size, height)
    return y0, y1, x0, x1, y1 - y0, x1 - x0


def encode_tiles(
    image: ImageData,
    tile_size: int,
    threads: int = 1,
    stats: "TileStats | None" = None,
) -> bytes:
    """把整图编码为 DATA chunk payload。

    threads > 1 时按 tile 并行；zlib 会释放 GIL，NumPy 也有部分释放。
    小图会自动退回串行，避免线程池开销。
    """
    if not (MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE):
        raise ValueError(f"tile_size 必须在 {MIN_TILE_SIZE}..{MAX_TILE_SIZE}")
    width, height, c = image.width, image.height, image.stored_channels
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    tile_count = tiles_x * tiles_y
    threads = max(1, int(threads))
    if tile_count < 8 or threads <= 1:
        threads = 1

    def encode_one(idx: int) -> TileRecord:
        y0, y1, x0, x1, th, tw = _tile_shape(
            idx, tiles_x, tile_size, width, height
        )
        tile = image.stored_data[y0:y1, x0:x1, :]
        return encode_tile(tile)

    if threads > 1:
        with ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="pxc-tile"
        ) as pool:
            records = list(pool.map(encode_one, range(tile_count)))
    else:
        records = [encode_one(idx) for idx in range(tile_count)]

    out = bytearray()
    out.extend(
        DATA_HEADER_STRUCT.pack(
            DATA_FORMAT_VERSION, tile_size, tiles_x, tiles_y, tile_count
        )
    )
    for rec in records:
        if stats is not None:
            stats.add(rec)
        out.extend(
            TILE_HEADER_STRUCT.pack(
                rec.mode,
                rec.submode,
                rec.flags,
                rec.raw_len,
                rec.comp_len,
                rec.raw_crc,
                rec.payload_crc,
            )
        )
        out.extend(rec.payload)
    return bytes(out)


def _decode_tile(rec: TileRecord, shape: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    h, w, c = shape
    n = h * w
    sd = storage_dtype(dtype)

    if rec.mode == TILE_MODE_SOLID:
        if rec.raw_len != c * sd.itemsize:
            raise PxcCorruptionError("SOLID payload 长度错误")
        color = np.frombuffer(rec.payload, dtype=sd, count=c).astype(dtype)
        return np.broadcast_to(color, shape).copy()

    if rec.mode == TILE_MODE_PALETTE:
        return decode_palette_payload(rec.payload, shape, dtype, rec.submode)

    if rec.mode == TILE_MODE_PRED:
        return decode_residuals(rec.payload, shape, dtype)

    if rec.mode == TILE_MODE_RUN:
        return decode_run_payload(rec.payload, shape, dtype)

    if rec.mode == TILE_MODE_FILTER:
        if rec.submode != 0:
            raise PxcCorruptionError(f"未知 FILTER 子模式: {rec.submode}")
        data_bytes = n * c * sd.itemsize
        expected = h + data_bytes
        if rec.raw_len != expected:
            raise PxcCorruptionError(
                f"FILTER payload 长度错误：期望 {expected}，实际 {rec.raw_len}"
            )
        if len(rec.payload) < h:
            raise PxcCorruptionError("FILTER 行类型数据不完整")
        types = np.frombuffer(rec.payload, dtype=np.uint8, count=h)
        arr = np.frombuffer(
            rec.payload, dtype=sd, count=n * c, offset=h
        ).reshape(shape)
        return unfilter_rows(arr.astype(dtype), types)

    if rec.mode == TILE_MODE_RAW:
        expected = n * c * sd.itemsize
        if rec.raw_len != expected:
            raise PxcCorruptionError(
                f"RAW payload 长度错误：期望 {expected}，实际 {rec.raw_len}"
            )
        arr = np.frombuffer(rec.payload, dtype=sd, count=n * c).reshape(shape)
        return arr.astype(dtype)

    raise PxcCorruptionError(f"未知 tile 模式: {rec.mode}")


def decode_tiles(
    payload: bytes,
    width: int,
    height: int,
    channels: int,
    dtype: np.dtype,
    expected_tile_size: int,
) -> np.ndarray:
    """解析 DATA payload 并重建整图样本数组。"""
    if len(payload) < DATA_HEADER_SIZE:
        raise PxcCorruptionError("DATA payload 过短")
    version, tile_size, tiles_x, tiles_y, tile_count = DATA_HEADER_STRUCT.unpack_from(
        payload, 0
    )
    if version != DATA_FORMAT_VERSION:
        raise PxcFormatError(f"DATA 版本不支持: {version}")
    if tile_size != expected_tile_size:
        raise PxcCorruptionError(
            f"DATA tile_size 不一致：header={expected_tile_size}, data={tile_size}"
        )
    if tiles_x <= 0 or tiles_y <= 0:
        raise PxcCorruptionError("DATA tile 网格非法")
    if tiles_x * tiles_y != tile_count:
        raise PxcCorruptionError("DATA tile 数量与网格不一致")
    expected_x = (width + tile_size - 1) // tile_size
    expected_y = (height + tile_size - 1) // tile_size
    if tiles_x != expected_x or tiles_y != expected_y:
        raise PxcCorruptionError("DATA tile 网格与图像尺寸不一致")
    if tile_count > 10_000_000:
        raise PxcCorruptionError("DATA tile 数量异常")

    data = np.zeros((height, width, channels), dtype=dtype)
    offset = DATA_HEADER_SIZE
    for idx in range(tile_count):
        y0, y1, x0, x1, th, tw = _tile_shape(idx, tiles_x, tile_size, width, height)
        if offset + TILE_HEADER_SIZE > len(payload):
            raise PxcCorruptionError("tile 记录头不完整")
        mode, submode, flags, raw_len, comp_len, raw_crc, payload_crc = (
            TILE_HEADER_STRUCT.unpack_from(payload, offset)
        )
        offset += TILE_HEADER_SIZE
        if comp_len > len(payload) - offset:
            raise PxcCorruptionError("tile payload 长度越界")
        # 防止损坏/恶意文件声明超大解压长度
        max_raw = max(1024, tw * th * channels * 16 + 1024 * 1024)
        if raw_len > max_raw:
            raise PxcCorruptionError(
                f"tile raw_len 过大: {raw_len} > {max_raw}"
            )
        stored = payload[offset : offset + comp_len]
        offset += comp_len
        if crc32(stored) != payload_crc:
            raise PxcCorruptionError("tile payload CRC 校验失败")
        if flags & FLAG_PAYLOAD_ZLIB:
            try:
                raw = zlib.decompress(stored)
            except zlib.error as exc:
                raise PxcCorruptionError(f"tile zlib 解压失败: {exc}") from exc
        else:
            raw = stored
        if len(raw) != raw_len:
            raise PxcCorruptionError(
                f"tile 解压长度错误：期望 {raw_len}，实际 {len(raw)}"
            )
        if crc32(raw) != raw_crc:
            raise PxcCorruptionError("tile 原始数据 CRC 校验失败")
        if mode not in TILE_MODE_NAMES:
            raise PxcCorruptionError(f"未知 tile 模式 {mode}")
        rec = TileRecord(
            mode=mode,
            submode=submode,
            flags=flags,
            raw_len=raw_len,
            comp_len=comp_len,
            raw_crc=raw_crc,
            payload_crc=payload_crc,
            payload=raw,
        )
        try:
            tile = _decode_tile(rec, (th, tw, channels), dtype)
        except (ValueError, IndexError, struct.error) as exc:
            raise PxcCorruptionError(f"tile 解码失败: {exc}") from exc
        data[y0:y1, x0:x1, :] = tile

    if offset != len(payload):
        raise PxcCorruptionError(
            f"DATA payload 存在多余数据：offset={offset}, len={len(payload)}"
        )
    return data


def _meta_bytes(pxc, key: int) -> bytes | None:
    entry = pxc.metadata.get(key)
    if entry is None:
        return None
    typ, payload = entry
    if typ != 1:
        raise PxcFormatError(f"元数据 {key} 类型错误")
    return payload


def _meta_text(pxc, key: int) -> str | None:
    entry = pxc.metadata.get(key)
    if entry is None:
        return None
    typ, payload = entry
    if typ != 2:
        raise PxcFormatError(f"元数据 {key} 类型错误")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PxcFormatError(f"元数据 {key} 不是合法 UTF-8") from exc


def _decode_passthrough(pxc) -> ImageData:
    header = pxc.header
    mode = MODE_ID_TO_NAME[header.mode_id]
    original_filename = _meta_text(pxc, META_ORIGINAL_FILENAME) or "restored"
    source_format = _meta_text(pxc, META_SOURCE_FORMAT) or ""
    image = load_image_bytes(
        pxc.data_payload,
        original_filename,
        source_file_hash=_meta_bytes(pxc, META_SOURCE_FILE_HASH),
        source_file_size=header.original_size,
        source_format=source_format or None,
    )
    if image.mode != mode:
        raise PxcCorruptionError(
            f"passthrough 解码模式不一致：header={mode}, decoded={image.mode}"
        )
    if image.width != header.width or image.height != header.height:
        raise PxcCorruptionError("passthrough 解码尺寸与 header 不一致")
    if image.pixel_hash != header.pixel_hash:
        raise PxcCorruptionError("passthrough 像素哈希校验失败")
    expected = _meta_bytes(pxc, META_SOURCE_PIXEL_HASH)
    if expected is not None and image.source_pixel_hash != expected:
        raise PxcCorruptionError("passthrough 原始像素字节哈希校验失败")
    return image


def decode_image_data(pxc) -> ImageData:
    """从已解析的 PxcFile 重建 ImageData 并校验哈希。"""
    header = pxc.header
    if header.mode_id not in MODE_ID_TO_NAME:
        raise PxcFormatError(f"未知 mode_id: {header.mode_id}")
    if header.compression_id == COMPRESSION_PASSTHROUGH:
        return _decode_passthrough(pxc)
    mode = MODE_ID_TO_NAME[header.mode_id]
    dtype = MODE_DTYPE[mode]
    full_channels = int(MODE_SPECS[mode]["channels"])
    channels = header.channels
    constant_alpha = None
    if header.alpha_kind == ALPHA_CONSTANT:
        if mode not in ("RGBA", "LA") or channels != full_channels - 1:
            raise PxcFormatError("constant alpha 通道数不一致")
        entry = pxc.metadata.get(META_CONSTANT_ALPHA)
        if entry is None or entry[0] != META_TYPE_U32 or len(entry[1]) != 4:
            raise PxcFormatError("缺少 constant alpha 值")
        constant_alpha = struct.unpack("<I", entry[1])[0]
        if constant_alpha > 255:
            raise PxcFormatError("constant alpha 越界")
    elif channels != full_channels:
        raise PxcFormatError(
            f"通道数不一致：header={header.channels}, mode={mode}"
        )
    if header.width <= 0 or header.height <= 0:
        raise PxcFormatError("图像尺寸非法")
    if header.width * header.height > 300_000_000:
        raise PxcFormatError("图像像素数超出安全上限")

    data = decode_tiles(
        pxc.data_payload,
        header.width,
        header.height,
        channels,
        dtype,
        header.tile_size,
    )
    if constant_alpha is not None:
        alpha = np.full((header.height, header.width, 1), constant_alpha, dtype=dtype)
        data = np.concatenate([data, alpha], axis=2)
        data = np.ascontiguousarray(data)

    palette = _meta_bytes(pxc, META_PALETTE)
    if mode == "P" and palette is None:
        raise PxcFormatError("P 模式缺少调色板")
    transparency = _meta_bytes(pxc, META_TRANSPARENCY)
    icc = _meta_bytes(pxc, META_ICC_PROFILE)
    exif = _meta_bytes(pxc, META_EXIF)
    xmp = _meta_bytes(pxc, META_XMP)

    dpi = None
    dpi_entry = pxc.metadata.get(META_DPI)
    if dpi_entry is not None:
        typ, payload = dpi_entry
        if typ != 5 or len(payload) != 16:
            raise PxcFormatError("DPI 元数据格式错误")
        dpi = struct.unpack("<dd", payload)

    gamma = None
    gamma_entry = pxc.metadata.get(META_GAMMA)
    if gamma_entry is not None:
        typ, payload = gamma_entry
        if typ != 6 or len(payload) != 8:
            raise PxcFormatError("gamma 元数据格式错误")
        gamma = struct.unpack("<d", payload)[0]

    comment = _meta_bytes(pxc, META_COMMENT)
    source_format = _meta_text(pxc, META_SOURCE_FORMAT) or ""
    source_ext = _meta_text(pxc, META_SOURCE_EXT) or ""
    original_filename = _meta_text(pxc, META_ORIGINAL_FILENAME) or "restored"
    mode_name = _meta_text(pxc, META_MODE_NAME)
    if mode_name is not None and mode_name != mode:
        raise PxcFormatError(f"模式名不一致：{mode_name} vs {mode}")

    image = ImageData(
        mode=mode,
        width=header.width,
        height=header.height,
        data=data,
        palette=palette,
        transparency=transparency,
        icc_profile=icc,
        exif=exif,
        xmp=xmp,
        dpi=dpi,
        gamma=gamma,
        comment=comment,
        original_filename=original_filename,
        source_format=source_format,
        source_ext=source_ext,
        source_file_size=header.original_size,
        source_file_hash=_meta_bytes(pxc, META_SOURCE_FILE_HASH),
        source_pixel_hash=_meta_bytes(pxc, META_SOURCE_PIXEL_HASH),
        pixel_hash=canonical_pixel_hash(mode, header.width, header.height, data),
        constant_alpha=constant_alpha,
    )

    if image.pixel_hash != header.pixel_hash:
        raise PxcCorruptionError("整图像素哈希校验失败")

    if image.source_pixel_hash:
        expected = image.source_pixel_hash
        try:
            actual = sha256_bytes(data_to_pillow(image).tobytes())
        except Exception as exc:
            raise PxcDecodeError(f"无法重建图像以校验像素: {exc}") from exc
        if actual != expected:
            raise PxcCorruptionError("原始像素字节哈希校验失败")

    return image
