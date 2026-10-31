"""JPEG XL 无损重打包：JPEG → JXL → 原始 JPEG 字节。

JPEG 本身已经压过一遍，zlib 对它没有任何办法，所以 PXCS 长期把 JPEG 关键帧
按原字节直通保存（100%）。JPEG XL 的「无损 JPEG 转码」保留的是 DCT 系数和
熵编码状态，解出来能**逐字节还原原始 JPEG**，体积通常小 20%–27%。

依赖 ``imagecodecs``（内置 libjxl）。没装这个依赖时 :func:`available` 返回
False，调用方自动回退 raw，其它功能完全不受影响。
"""
from __future__ import annotations

from typing import Optional

# 只有省到 2% 以上才值得多一个依赖和一层解码。
_MIN_GAIN = 0.98
# 小于这个体积的 JPEG 不值得转（头开销占比高）。
_MIN_BYTES = 4096

_enabled = True
_available: Optional[bool] = None


def available() -> bool:
    """当前环境是否有可用的 JPEG XL 实现。"""
    global _available
    if _available is None:
        try:
            import imagecodecs  # noqa: F401

            _available = hasattr(imagecodecs, "jpegxl_encode_jpeg")
        except Exception:  # noqa: BLE001 - 缺依赖是正常情况，不是错误
            _available = False
    return bool(_available)


def set_enabled(flag: bool) -> None:
    """create 开始前调用：``False`` 时完全不使用 jpegxl 编码。"""
    global _enabled
    _enabled = bool(flag)


def enabled() -> bool:
    return _enabled and available()


def pack(data: bytes) -> Optional[bytes]:
    """把 JPEG 字节重打包成 JXL；不划算或失败时返回 None。"""
    if not enabled():
        return None
    if len(data) < _MIN_BYTES or data[:3] != b"\xff\xd8\xff":
        return None
    try:
        import imagecodecs

        blob = imagecodecs.jpegxl_encode_jpeg(data)
    except Exception:  # noqa: BLE001 - 转码失败就回退 raw
        return None
    if not blob or len(blob) >= int(len(data) * _MIN_GAIN):
        return None
    return bytes(blob)


def unpack(blob: bytes) -> bytes:
    """把 JXL 还原成原始 JPEG 字节；失败抛 ValueError 交给调用方转格式错误。"""
    try:
        import imagecodecs

        return bytes(imagecodecs.jpegxl_decode_jpeg(blob))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"JPEG XL 还原失败: {exc}") from exc
