"""MED 预测与有符号残差的编码/解码。

编码侧可以完全向量化；解码侧存在“左像素”依赖，只能按像素顺序恢复。
第一版按 tile 独立预测，tile 边界使用可确定的 fallback：
- 左上角：预测 0
- 第一行：用左像素
- 第一列：用上像素
- 其余：MED
"""
from __future__ import annotations

import numpy as np

from .errors import PxcCorruptionError, UnsupportedImageError


def residual_storage_dtype(sample_dtype: np.dtype) -> np.dtype:
    """zigzag 残差的无符号存储类型。"""
    dt = np.dtype(sample_dtype)
    if dt == np.dtype(np.uint8):
        return np.dtype("<u2")
    if dt == np.dtype(np.uint16):
        return np.dtype("<u4")
    if dt == np.dtype(np.int32):
        return np.dtype("<u8")
    raise UnsupportedImageError(f"预测模式不支持样本类型 {sample_dtype}")


def zigzag_encode(values: np.ndarray) -> np.ndarray:
    """有符号 -> 无符号：0,-1,1,-2,2 -> 0,1,2,3,4。"""
    v = values.astype(np.int64, copy=False)
    return np.where(v >= 0, v << 1, ((-v) << 1) - 1)


def zigzag_decode(values: np.ndarray) -> np.ndarray:
    """无符号 -> 有符号，与 zigzag_encode 严格互逆。"""
    u = values.astype(np.uint64, copy=False)
    half = (u >> 1).astype(np.int64)
    return np.where((u & 1) != 0, -(half + 1), half)


def med_predict(tile: np.ndarray) -> np.ndarray:
    """对形状 (H,W,C) 的整数 tile 计算 MED 预测值，返回 int64 数组。

    MED (LOCO-I)：
        a = W, b = N, c = NW
        if c >= max(a,b): p = min(a,b)
        elif c <= min(a,b): p = max(a,b)
        else: p = a + b - c
    """
    if tile.ndim != 3:
        raise ValueError("tile 必须是 (H,W,C)")
    h, w, c = tile.shape
    x = tile.astype(np.int64, copy=False)

    a = np.zeros((h, w, c), dtype=np.int64)
    b = np.zeros((h, w, c), dtype=np.int64)
    cc = np.zeros((h, w, c), dtype=np.int64)
    if w > 1:
        a[:, 1:, :] = x[:, :-1, :]
    if h > 1:
        b[1:, :, :] = x[:-1, :, :]
        if w > 1:
            cc[1:, 1:, :] = x[:-1, :-1, :]

    p = np.where(
        cc >= np.maximum(a, b),
        np.minimum(a, b),
        np.where(cc <= np.minimum(a, b), np.maximum(a, b), a + b - cc),
    )

    # 边界 fallback，保证编解码完全一致
    p[0, 0, :] = 0
    if w > 1:
        p[0, 1:, :] = x[0, :-1, :]
    if h > 1:
        p[1:, 0, :] = x[:-1, 0, :]
    return p


def encode_residuals(tile: np.ndarray) -> tuple[bytes, np.dtype]:
    """返回 zigzag 残差的字节串和所用存储 dtype。"""
    if tile.dtype.kind == "f":
        raise UnsupportedImageError("浮点图像不支持 PRED 模式")
    x = tile.astype(np.int64, copy=False)
    p = med_predict(x)
    e = x - p
    u = zigzag_encode(e)
    rdtype = residual_storage_dtype(tile.dtype)
    limit = np.iinfo(rdtype).max
    if u.size and int(u.max()) > limit:
        raise UnsupportedImageError("残差超出存储类型范围")
    return u.astype(rdtype).tobytes(), rdtype


def decode_residuals(
    raw: bytes,
    shape: tuple[int, int, int],
    sample_dtype: np.dtype,
) -> np.ndarray:
    """从 zigzag 残差字节串恢复 tile。"""
    h, w, c = shape
    rdtype = residual_storage_dtype(sample_dtype)
    expected = h * w * c * rdtype.itemsize
    if len(raw) != expected:
        raise PxcCorruptionError(
            f"预测残差长度错误：期望 {expected}，实际 {len(raw)}"
        )
    u = np.frombuffer(raw, dtype=rdtype, count=h * w * c).reshape(h, w, c)
    e = zigzag_decode(u)

    out = np.zeros((h, w, c), dtype=np.int64)
    for y in range(h):
        row = out[y]
        prev = out[y - 1] if y > 0 else None
        for x in range(w):
            if y == 0 and x == 0:
                pred = np.zeros(c, dtype=np.int64)
            elif y == 0:
                pred = row[x - 1]
            elif x == 0:
                pred = prev[x]
            else:
                a = row[x - 1]
                b = prev[x]
                cc = prev[x - 1]
                pred = np.where(
                    cc >= np.maximum(a, b),
                    np.minimum(a, b),
                    np.where(cc <= np.minimum(a, b), np.maximum(a, b), a + b - cc),
                )
            row[x] = pred + e[y, x]

    # 检查结果是否在样本类型范围内，避免损坏文件造成静默溢出
    info = np.iinfo(sample_dtype)
    if out.size and (int(out.min()) < info.min or int(out.max()) > info.max):
        raise PxcCorruptionError("预测重建结果超出样本类型范围")
    return out.astype(sample_dtype)
