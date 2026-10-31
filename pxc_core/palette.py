"""调色板、行程（RUN）与 SOLID 的 tile 级 payload 编解码。

所有 payload 都是“未压缩字节串”；zlib 由 codec.py 统一处理。
"""
from __future__ import annotations

import struct

import numpy as np

from .config import PALETTE_PLAIN, PALETTE_RLE
from .errors import PxcCorruptionError, UnsupportedImageError
from .image_io import storage_dtype


def _flat(tile: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(tile).reshape(-1, tile.shape[2])


def solid_payload(tile: np.ndarray, dtype: np.dtype) -> bytes:
    """返回单色 tile 的颜色样本字节。"""
    sd = storage_dtype(dtype)
    return tile[0, 0, :].astype(sd).tobytes()


def build_palette(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (palette[K,C], inverse[N])，按字典序排序保证确定性。"""
    flat = _flat(tile)
    dt = tile.dtype
    c = tile.shape[2]
    if dt.kind == "f":
        raise UnsupportedImageError("浮点图像不支持 PALETTE 模式")
    if c == 1:
        uniq, inv = np.unique(flat[:, 0], return_inverse=True)
        return uniq.reshape(-1, 1).astype(dt), inv.astype(np.int64)
    void_dtype = np.dtype((np.void, c * dt.itemsize))
    packed = flat.view(void_dtype).ravel()
    uniq_packed, inv = np.unique(packed, return_inverse=True)
    uniq = uniq_packed.view(dt).reshape(-1, c)
    return uniq.astype(dt), inv.astype(np.int64)


def _palette_runs(indices: np.ndarray) -> bytes:
    n = indices.shape[0]
    if n == 0:
        return b""
    same = indices[1:] == indices[:-1]
    starts = np.r_[0, np.nonzero(~same)[0] + 1]
    ends = np.r_[starts[1:], n]
    parts: list[bytes] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        parts.append(struct.pack("<I", end - start))
        parts.append(bytes((int(indices[start]),)))
    return b"".join(parts)


def palette_payload(
    tile: np.ndarray,
    dtype: np.dtype,
    rle: bool,
) -> tuple[bytes, int]:
    """返回 (payload, submode)。"""
    palette, inverse = build_palette(tile)
    k = palette.shape[0]
    if k < 1 or k > 256:
        raise UnsupportedImageError("调色板条目数超出 1..256")
    sd = storage_dtype(dtype)
    head = struct.pack("<H", k)
    entries = palette.astype(sd).tobytes()
    if rle:
        body = _palette_runs(inverse)
        return head + entries + body, PALETTE_RLE
    body = inverse.astype(np.uint8).tobytes()
    return head + entries + body, PALETTE_PLAIN


def decode_palette_payload(
    raw: bytes,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    submode: int,
) -> np.ndarray:
    h, w, c = shape
    n = h * w
    if len(raw) < 2:
        raise PxcCorruptionError("PALETTE payload 过短")
    k = struct.unpack_from("<H", raw, 0)[0]
    if not (1 <= k <= 256):
        raise PxcCorruptionError(f"PALETTE 条目数非法: {k}")
    sd = storage_dtype(dtype)
    entry_bytes = c * sd.itemsize
    entries_end = 2 + k * entry_bytes
    if entries_end > len(raw):
        raise PxcCorruptionError("PALETTE 调色板数据不完整")
    palette = np.frombuffer(raw, dtype=sd, count=k * c, offset=2).reshape(k, c).astype(dtype)
    body = raw[entries_end:]

    if submode == PALETTE_PLAIN:
        if len(body) != n:
            raise PxcCorruptionError(
                f"PALETTE 索引长度错误：期望 {n}，实际 {len(body)}"
            )
        indices = np.frombuffer(body, dtype=np.uint8, count=n).astype(np.int64)
    elif submode == PALETTE_RLE:
        indices = np.empty(n, dtype=np.int64)
        offset = 0
        pos = 0
        while pos < n:
            if offset + 5 > len(body):
                raise PxcCorruptionError("PALETTE RLE 数据不完整")
            run_len = struct.unpack_from("<I", body, offset)[0]
            idx = body[offset + 4]
            offset += 5
            if run_len == 0 or pos + run_len > n:
                raise PxcCorruptionError("PALETTE RLE 行程长度非法")
            indices[pos : pos + run_len] = idx
            pos += run_len
        if offset != len(body):
            raise PxcCorruptionError("PALETTE RLE 存在多余数据")
    else:
        raise PxcCorruptionError(f"未知 PALETTE 子模式: {submode}")

    if indices.size and int(indices.max()) >= k:
        raise PxcCorruptionError("PALETTE 索引越界")
    return palette[indices].reshape(h, w, c)


def run_payload(tile: np.ndarray, dtype: np.dtype) -> bytes:
    """按行优先顺序，把 tile 编码为 (u32 长度, 颜色) 序列。"""
    if tile.dtype.kind == "f":
        raise UnsupportedImageError("浮点图像不支持 RUN 模式")
    flat = _flat(tile)
    sd = storage_dtype(dtype)
    n = flat.shape[0]
    if n == 0:
        return b""
    same = np.all(flat[1:] == flat[:-1], axis=1)
    starts = np.r_[0, np.nonzero(~same)[0] + 1]
    ends = np.r_[starts[1:], n]
    parts: list[bytes] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        parts.append(struct.pack("<I", end - start))
        parts.append(flat[start].astype(sd).tobytes())
    return b"".join(parts)


def decode_run_payload(
    raw: bytes,
    shape: tuple[int, int, int],
    dtype: np.dtype,
) -> np.ndarray:
    h, w, c = shape
    n = h * w
    sd = storage_dtype(dtype)
    color_bytes = c * sd.itemsize
    out = np.empty((n, c), dtype=dtype)
    offset = 0
    pos = 0
    while pos < n:
        if offset + 4 + color_bytes > len(raw):
            raise PxcCorruptionError("RUN 数据不完整")
        run_len = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if run_len == 0 or pos + run_len > n:
            raise PxcCorruptionError("RUN 行程长度非法")
        color = np.frombuffer(raw, dtype=sd, count=c, offset=offset).astype(dtype)
        offset += color_bytes
        out[pos : pos + run_len, :] = color
        pos += run_len
    if offset != len(raw):
        raise PxcCorruptionError("RUN 存在多余数据")
    return out.reshape(h, w, c)
