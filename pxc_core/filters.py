"""PNG 风格的行滤波（None/Sub/Up/Average/Paeth）。

用于给 8/16-bit 整数样本提供一个“横向/纵向差分”候选模式。
滤波与逆滤波都按模 2^bit 进行，保证严格可逆。
"""
from __future__ import annotations

import numpy as np

from .errors import UnsupportedImageError

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVG = 3
FILTER_PAETH = 4
FILTER_NAMES = {
    FILTER_NONE: "NONE",
    FILTER_SUB: "SUB",
    FILTER_UP: "UP",
    FILTER_AVG: "AVG",
    FILTER_PAETH: "PAETH",
}
FILTER_TYPES = [FILTER_NONE, FILTER_SUB, FILTER_UP, FILTER_AVG, FILTER_PAETH]
# 新文件只使用可完全向量化逆变换的三种；AVG/PAETH 仅用于兼容旧文件。
FILTER_TYPES_FAST = [FILTER_NONE, FILTER_SUB, FILTER_UP]


def _modulus(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint8):
        return 256
    if dt == np.dtype(np.uint16):
        return 65536
    raise UnsupportedImageError(f"滤波模式不支持样本类型 {dtype}")


def _neighbors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, c = x.shape
    left = np.zeros((h, w, c), dtype=np.int64)
    up = np.zeros((h, w, c), dtype=np.int64)
    ul = np.zeros((h, w, c), dtype=np.int64)
    if w > 1:
        left[:, 1:, :] = x[:, :-1, :]
    if h > 1:
        up[1:, :, :] = x[:-1, :, :]
        if w > 1:
            ul[1:, 1:, :] = x[:-1, :-1, :]
    return left, up, ul


def _paeth(left: np.ndarray, up: np.ndarray, ul: np.ndarray) -> np.ndarray:
    p = left + up - ul
    pa = np.abs(p - left)
    pb = np.abs(p - up)
    pc = np.abs(p - ul)
    return np.where(
        (pa <= pb) & (pa <= pc),
        left,
        np.where(pb <= pc, up, ul),
    )


def filter_tile(tile: np.ndarray, filter_type: int) -> np.ndarray:
    """返回滤波后的样本数组，dtype 与输入相同。"""
    if tile.dtype.kind not in ("u", "i"):
        raise UnsupportedImageError("滤波模式只支持整数样本")
    mod = _modulus(tile.dtype)
    x = tile.astype(np.int64)
    left, up, ul = _neighbors(x)

    if filter_type == FILTER_NONE:
        base = np.zeros_like(x)
    elif filter_type == FILTER_SUB:
        base = left
    elif filter_type == FILTER_UP:
        base = up
    elif filter_type == FILTER_AVG:
        base = (left + up) // 2
    elif filter_type == FILTER_PAETH:
        base = _paeth(left, up, ul)
    else:
        raise UnsupportedImageError(f"未知滤波类型: {filter_type}")

    out = (x - base) % mod
    return out.astype(tile.dtype)


def unfilter_tile(filtered: np.ndarray, filter_type: int) -> np.ndarray:
    """逆滤波；返回 dtype 与输入相同。"""
    if filtered.dtype.kind not in ("u", "i"):
        raise UnsupportedImageError("滤波模式只支持整数样本")
    mod = _modulus(filtered.dtype)
    f = filtered.astype(np.int64)
    h, w, c = f.shape

    if filter_type == FILTER_NONE:
        return filtered.astype(filtered.dtype)
    if filter_type == FILTER_SUB:
        out = np.cumsum(f, axis=1) % mod
        return out.astype(filtered.dtype)
    if filter_type == FILTER_UP:
        out = np.cumsum(f, axis=0) % mod
        return out.astype(filtered.dtype)

    out = np.zeros((h, w, c), dtype=np.int64)
    for y in range(h):
        row = out[y]
        prev = out[y - 1] if y > 0 else None
        for x in range(w):
            left = row[x - 1] if x > 0 else np.zeros(c, dtype=np.int64)
            up = prev[x] if prev is not None else np.zeros(c, dtype=np.int64)
            ul = prev[x - 1] if (prev is not None and x > 0) else np.zeros(c, dtype=np.int64)
            if filter_type == FILTER_AVG:
                base = (left + up) // 2
            elif filter_type == FILTER_PAETH:
                base = _paeth(left, up, ul)
            else:
                raise UnsupportedImageError(f"未知滤波类型: {filter_type}")
            row[x] = (f[y, x] + base) % mod
    return out.astype(filtered.dtype)


def _row_score(row: np.ndarray, mod: int) -> int:
    """PNG 的启发式：把滤波值按有符号数解释，取绝对值之和。"""
    v = row.astype(np.int64)
    half = mod // 2
    signed = np.where(v >= half, v - mod, v)
    return int(np.abs(signed).sum())


def _row_scores(block: np.ndarray, mod: int) -> np.ndarray:
    """一次算出所有行的 score，返回 shape (h,)。

    与逐行调 ``_row_score`` 等价：只是把 (h,w,c) 的求和改成 ``axis=(1,2)``。
    整数加法可交换，逐行求和与整块求和结果完全相同。
    """
    v = block.astype(np.int64)
    half = mod // 2
    signed = np.where(v >= half, v - mod, v)
    return np.abs(signed).sum(axis=(1, 2))


def choose_row_filters(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐行选择滤波类型，返回 (types[H], filtered[H,W,C])。

    选择指标与 PNG 类似：最小化滤波后按有符号数解释的绝对值之和。

    打分是整块向量化的（每 tile 只有 3 次大数组运算），不再逐行循环。
    原先 3×H 次 Python 小数组操作在 128×128 tile 上会频繁争抢 GIL，
    是 PXC 编码只有 4 线程扩展性的主因。
    """
    mod = _modulus(tile.dtype)
    h, w, c = tile.shape
    types = np.zeros(h, dtype=np.uint8)
    out = np.empty_like(tile)
    best_scores = np.full(h, np.iinfo(np.int64).max, dtype=np.int64)
    # 一次只保留一个候选数组，避免多份 tile 同时驻留内存。
    for ft in FILTER_TYPES_FAST:
        f = filter_tile(tile, ft)
        scores = _row_scores(f, mod)
        better = scores < best_scores
        if better.any():
            best_scores[better] = scores[better]
            types[better] = ft
            out[better] = f[better]
    return types, out


def _unfilter_row(
    frow: np.ndarray,
    filter_type: int,
    prev: np.ndarray | None,
    mod: int,
) -> np.ndarray:
    h, w, c = 1, frow.shape[0], frow.shape[1]
    f = frow.astype(np.int64)
    if filter_type == FILTER_NONE:
        return frow.astype(frow.dtype)
    if filter_type == FILTER_SUB:
        return (np.cumsum(f, axis=0) % mod).astype(frow.dtype)
    if filter_type == FILTER_UP:
        if prev is None:
            return frow.astype(frow.dtype)
        return ((f + prev.astype(np.int64)) % mod).astype(frow.dtype)
    if filter_type not in (FILTER_AVG, FILTER_PAETH):
        raise UnsupportedImageError(f"未知滤波类型: {filter_type}")

    out = np.zeros((w, c), dtype=np.int64)
    for x in range(w):
        left = out[x - 1] if x > 0 else np.zeros(c, dtype=np.int64)
        up = prev[x].astype(np.int64) if prev is not None else np.zeros(c, dtype=np.int64)
        ul = (
            prev[x - 1].astype(np.int64)
            if (prev is not None and x > 0)
            else np.zeros(c, dtype=np.int64)
        )
        if filter_type == FILTER_AVG:
            base = (left + up) // 2
        else:
            base = _paeth(left, up, ul)
        out[x] = (f[x] + base) % mod
    return out.astype(frow.dtype)


def unfilter_rows(filtered: np.ndarray, types: np.ndarray) -> np.ndarray:
    """按每行滤波类型逆滤波。"""
    mod = _modulus(filtered.dtype)
    h, w, c = filtered.shape
    if len(types) != h:
        raise UnsupportedImageError("滤波类型数量与行数不一致")
    out = np.empty_like(filtered)
    for y in range(h):
        ft = int(types[y])
        if ft not in FILTER_TYPES:
            raise UnsupportedImageError(f"未知滤波类型: {ft}")
        prev = out[y - 1] if y > 0 else None
        out[y] = _unfilter_row(filtered[y], ft, prev, mod)
    return out
