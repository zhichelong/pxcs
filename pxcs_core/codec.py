"""PXCS 图片编解码。

关键帧：直接复用 PXC 的 tile 编码，把完整 PXC 文件作为 payload。
差分帧：把 (当前像素 - 参考像素) mod 2^n 当成一张“残差图片”，
再交给 PXC 编码。解码时反向相加即可。

这样不需要为 PXCS 重写一套 tile/调色板/16-bit 编解码器，
同时残差图仍然能享受 PXC 的 SOLID/PALETTE/PRED/FILTER 选择。
"""
from __future__ import annotations

import io
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from pxc_core import codec as pxc_codec
from pxc_core.container import build_pxc, parse_pxc_bytes
from pxc_core.codec import decode_image_data, encode_tiles
from pxc_core.i18n import t
from pxc_core.errors import PxcError
from pxc_core.image_io import (
    ImageData,
    canonical_pixel_hash,
    load_image_bytes,
)

from .container import ArchiveEntry


class PxcsImageError(PxcError):
    """图片差分编解码失败。"""


# 这些模式可以做整数模差分；P/I/F 不做（调色板/有符号/浮点语义不适合）。
_DELTA_MODES = {
    "1", "L", "LA", "RGBA", "RGB", "CMYK",
    "I;16", "I;16L", "I;16B",
}


def is_delta_capable(image: ImageData) -> bool:
    return (
        image.mode in _DELTA_MODES
        and image.width > 0
        and image.height > 0
    )


def delta_modulus(mode: str, dtype: np.dtype) -> int:
    if mode == "1":
        return 2
    if dtype == np.dtype(np.uint8):
        return 256
    if dtype == np.dtype(np.uint16):
        return 65536
    raise PxcsImageError(t("codec.mode_no_delta", mode=mode))


def encode_keyframe(image: ImageData, tile_size: int, threads: int = 1) -> bytes:
    """编码为完整 PXC 文件（作为 PXCS 的关键帧 payload）。"""
    payload = encode_tiles(image, tile_size, threads=threads)
    return build_pxc(image, payload, None, tile_size=tile_size)


def encode_webp_lossless(image: ImageData, method: int = 4) -> bytes:
    """把 8-bit RGB/RGBA 图像编码为 WebP lossless（像素精确）。"""
    if image.mode not in {"RGB", "RGBA"}:
        raise PxcsImageError(t("codec.webp_mode", mode=image.mode))
    method = max(0, min(6, int(method)))
    # 不能用 data_to_pillow：它会省略 constant alpha 通道，
    # 导致解出的 RGB/RGBA 模式与原始像素哈希不一致。
    buf = io.BytesIO()
    kwargs = {
        "format": "WEBP",
        "lossless": True,
        "exact": True,
        "method": method,
    }
    if image.icc_profile:
        kwargs["icc_profile"] = image.icc_profile
    if image.exif:
        kwargs["exif"] = image.exif
    if image.xmp:
        kwargs["xmp"] = image.xmp
    Image.fromarray(np.ascontiguousarray(image.data), mode=image.mode).save(
        buf, **kwargs
    )
    return buf.getvalue()


def decode_keyframe(payload: bytes) -> ImageData:
    pxc = parse_pxc_bytes(payload)
    return decode_image_data(pxc)


def make_residual(ref: ImageData, current: ImageData) -> ImageData:
    """生成可逆的残差 ImageData；元数据跟随 current。"""
    if ref.mode != current.mode:
        raise PxcsImageError(
            t("codec.mode_mismatch", ref=ref.mode, current=current.mode)
        )
    if ref.width != current.width or ref.height != current.height:
        raise PxcsImageError(t("codec.size_mismatch"))
    if not is_delta_capable(current):
        raise PxcsImageError(t("codec.mode_no_delta", mode=current.mode))

    mod = delta_modulus(current.mode, current.data.dtype)
    if current.mode == "1":
        diff = np.bitwise_xor(current.data, ref.data)
    elif current.data.dtype == np.dtype(np.uint8):
        diff = np.subtract(current.data, ref.data, dtype=np.uint8)
    elif current.data.dtype == np.dtype(np.uint16):
        diff = np.subtract(current.data, ref.data, dtype=np.uint16)
    else:
        # 保留通用路径；当前 _DELTA_MODES 不会走到这里。
        diff = (current.data.astype(np.int64) - ref.data.astype(np.int64)) % mod
        diff = diff.astype(current.data.dtype)
    diff = np.ascontiguousarray(diff)

    return ImageData(
        mode=current.mode,
        width=current.width,
        height=current.height,
        data=diff,
        palette=None,
        transparency=current.transparency,
        icc_profile=current.icc_profile,
        exif=current.exif,
        xmp=current.xmp,
        dpi=current.dpi,
        gamma=current.gamma,
        comment=current.comment,
        original_filename=current.original_filename,
        source_format=current.source_format,
        source_ext=current.source_ext,
        source_file_size=current.source_file_size,
        source_file_hash=None,
        source_pixel_hash=None,
        pixel_hash=canonical_pixel_hash(
            current.mode, current.width, current.height, diff
        ),
        constant_alpha=None,
    )


def _quant_lut(quant: int) -> np.ndarray:
    """把 mod-256 差分映射为“居中 + 量化”的 uint8 表示。

    0 差分 -> 128；-1 -> 127；+1 -> 129。这样 JPEG 解码噪声
    不会在 0/255 两侧跳变，PXC 的 tile/zlib 更容易压缩。
    """
    q = max(1, int(quant))
    values = np.empty(256, dtype=np.uint8)
    for b in range(256):
        signed = b if b < 128 else b - 256
        quantized = (1 if signed >= 0 else -1) * (abs(signed) // q * q)
        values[b] = (quantized + 128) & 0xFF
    return values


def center_residual(residual: ImageData) -> ImageData:
    """无损地把差分平移到以 128 为中心；8-bit 模式专用。"""
    if residual.data.dtype != np.dtype(np.uint8):
        return residual
    data = np.ascontiguousarray(np.add(residual.data, 128, dtype=np.uint8))
    return replace(
        residual,
        data=data,
        pixel_hash=canonical_pixel_hash(
            residual.mode, residual.width, residual.height, data
        ),
    )


def quantize_residual(residual: ImageData, quant: int) -> ImageData:
    """有损量化“居中差分”；quant=0 时不改变像素。"""
    if quant <= 0 or residual.data.dtype != np.dtype(np.uint8):
        return residual
    lut = _quant_lut(quant)
    data = np.ascontiguousarray(lut[residual.data])
    return replace(
        residual,
        data=data,
        pixel_hash=canonical_pixel_hash(
            residual.mode, residual.width, residual.height, data
        ),
    )


def apply_residual(ref: ImageData, residual: ImageData, centered: bool) -> np.ndarray:
    """把差分样本加回参考图；centered 表示残差以 128 为中心。"""
    if residual.mode == "1":
        return np.bitwise_xor(ref.data, residual.data)
    if residual.data.dtype == np.dtype(np.uint8):
        data = np.add(ref.data, residual.data, dtype=np.uint8)
        if centered:
            # 等价于 ref + (centered - 128)，在 mod 256 下。
            np.subtract(data, 128, out=data, dtype=np.uint8)
        return data
    if residual.data.dtype == np.dtype(np.uint16):
        return np.add(ref.data, residual.data, dtype=np.uint16)
    mod = delta_modulus(residual.mode, residual.data.dtype)
    data = (
        ref.data.astype(np.int64) + residual.data.astype(np.int64)
    ) % mod
    return data.astype(residual.data.dtype)


def encode_residual(
    ref: ImageData,
    current: ImageData,
    tile_size: int,
    threads: int = 1,
    *,
    quant: int = 0,
    centered: bool = False,
) -> bytes:
    residual = make_residual(ref, current)
    if quant > 0:
        residual = quantize_residual(residual, quant)  # LUT 已包含居中
    elif centered:
        residual = center_residual(residual)
    return encode_keyframe(residual, tile_size, threads=threads)


def encode_residual_with_preview(
    ref: ImageData,
    current: ImageData,
    tile_size: int,
    threads: int = 1,
    *,
    quant: int = 0,
    centered: bool = True,
) -> tuple[bytes, bytes]:
    """编码残差，并返回重建近似图的 pixel_sha256。

    有损量化时，归档能重建的是 quantize 后的近似像素；把该近似图的
    哈希写进 index，verify 才能校验“归档内部自洽”。原始像素哈希不再
    可用（这是 JPEG 差分压缩换取体积的代价）。
    """
    residual = make_residual(ref, current)
    if quant > 0:
        residual = quantize_residual(residual, quant)  # LUT 已包含居中
    elif centered:
        residual = center_residual(residual)
    payload = encode_keyframe(residual, tile_size, threads=threads)

    approx_data = apply_residual(ref, residual, centered)
    approx = replace(
        current,
        data=np.ascontiguousarray(approx_data),
        pixel_hash=canonical_pixel_hash(
            current.mode, current.width, current.height, approx_data
        ),
    )
    return payload, approx.pixel_hash


def encode_zlib_residual(
    ref: ImageData,
    current: ImageData,
    *,
    quant: int = 16,
    level: int = 6,
    centered: bool = True,
    min_zero_fraction: float = 0.0,
) -> tuple[bytes, bytes]:
    """快速残差路径：居中/量化后直接 zlib 压缩整张残差数组。

    q=16 的残差大部分是 128，zlib 通常比 PXC tile 快 10 倍以上，
    体积只大 20–40%。JPEG 差分优先用它；像素无损残差仍走 PXC。
    """
    residual = make_residual(ref, current)
    if quant > 0:
        residual = quantize_residual(residual, quant)
    elif centered:
        residual = center_residual(residual)
    if min_zero_fraction > 0 and quant > 0:
        # 残差太“密”时 zlib 会非常慢，而且体积往往大于源文件；
        # 直接拒绝，让调用方回退 raw。测试本数据：稀疏组零比例 0.8–0.99，
        # 内容差异大的组会低很多。
        zero = float(np.mean(residual.data == 128))
        if zero < float(min_zero_fraction):
            raise PxcsImageError(
                t(
                    "codec.zero_fraction",
                    zero=f"{zero:.3f}",
                    minimum=f"{float(min_zero_fraction):.3f}",
                )
            )
    payload = zlib.compress(residual.data.tobytes(), max(1, min(9, int(level))))
    approx_data = apply_residual(ref, residual, centered)
    approx = replace(
        current,
        data=np.ascontiguousarray(approx_data),
        pixel_hash=canonical_pixel_hash(
            current.mode, current.width, current.height, approx_data
        ),
    )
    return payload, approx.pixel_hash


def decode_zlib_residual(
    payload: bytes,
    entry: ArchiveEntry,
    ref: ImageData,
) -> ImageData:
    raw = zlib.decompress(payload)
    channels = ref.channels
    expected = entry.width * entry.height * channels
    if len(raw) != expected:
        raise PxcsImageError(
            t("codec.zlib_length", expected=expected, actual=len(raw))
        )
    data = np.frombuffer(raw, dtype=np.uint8).copy().reshape(
        entry.height, entry.width, channels
    )
    residual = ImageData(
        mode=entry.mode,
        width=entry.width,
        height=entry.height,
        data=data,
    )
    return reconstruct_image(ref, residual, entry)


def reconstruct_image(
    ref: ImageData,
    residual: ImageData,
    entry: ArchiveEntry,
) -> ImageData:
    """把残差图加回参考图，恢复目标 ImageData。"""
    if ref.mode != residual.mode or ref.data.shape != residual.data.shape:
        raise PxcsImageError(t("codec.shape_mismatch"))
    data = apply_residual(ref, residual, bool(entry.centered))
    data = np.ascontiguousarray(data)
    actual_hash = canonical_pixel_hash(
        residual.mode, residual.width, residual.height, data
    )
    if entry.pixel_sha256 and actual_hash.hex() != entry.pixel_sha256:
        raise PxcsImageError(t("codec.restore_hash", path=entry.path))

    target = ImageData(
        mode=residual.mode,
        width=residual.width,
        height=residual.height,
        data=data,
        palette=residual.palette,
        transparency=residual.transparency,
        icc_profile=residual.icc_profile,
        exif=residual.exif,
        xmp=residual.xmp,
        dpi=residual.dpi,
        gamma=residual.gamma,
        comment=residual.comment,
        original_filename=Path(entry.path).name,
        source_format=residual.source_format or entry.format,
        source_ext=residual.source_ext or entry.ext,
        source_file_size=entry.size,
        source_file_hash=(
            bytes.fromhex(entry.source_sha256) if entry.source_sha256 else None
        ),
        source_pixel_hash=None,
        pixel_hash=actual_hash,
        constant_alpha=entry.constant_alpha,
    )
    return target


def load_raw_image(payload: bytes, entry: ArchiveEntry) -> ImageData:
    return load_image_bytes(
        payload,
        Path(entry.path).name,
        source_file_hash=(
            bytes.fromhex(entry.source_sha256) if entry.source_sha256 else None
        ),
        source_file_size=entry.size,
        source_format=entry.format or None,
    )


def sample_score(ref: ImageData, current: ImageData) -> tuple[float, float]:
    """快速估计相似度，返回 (变化通道比例, 平均绝对差)。"""
    if ref.data.shape != current.data.shape or ref.data.dtype != current.data.dtype:
        return 1.0, 1e9
    h, w = current.data.shape[:2]
    step = max(1, min(h, w) // 64)
    a = ref.data[::step, ::step].astype(np.int32)
    b = current.data[::step, ::step].astype(np.int32)
    if a.size == 0:
        return 1.0, 1e9
    d = b - a
    changed = float(np.mean(np.any(d != 0, axis=2)))
    mean_abs = float(np.mean(np.abs(d)))
    return changed, mean_abs


def make_thumbnail(image: ImageData, size: int = 32) -> np.ndarray:
    """生成用于跨图候选搜索的小缩略图（RGB uint8）。

    只用于选参考，不参与最终像素；P 模式不参与差分，因此这里不追求
    调色板颜色的精确性。
    """
    h, w = image.data.shape[:2]
    step = max(1, min(h, w) // max(1, size))
    a = image.data[::step, ::step]
    channels = a.shape[2]
    if channels == 1:
        a = np.repeat(a, 3, axis=2)
    elif channels == 2:
        a = np.repeat(a[..., :1], 3, axis=2)
    else:
        a = a[..., :3]
    if a.dtype == np.dtype(np.uint16):
        a = (a.astype(np.uint32) >> 8).clip(0, 255)
    elif a.dtype == np.dtype(np.int32):
        a = np.clip(a, 0, 255)
    elif a.dtype == np.dtype(np.float32):
        a = np.clip(a, 0, 255)
    return np.ascontiguousarray(a.astype(np.uint8))


def thumbnail_score(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return 1.0, 1e9
    d = b.astype(np.int32) - a.astype(np.int32)
    changed = float(np.mean(np.any(d != 0, axis=2)))
    mean_abs = float(np.mean(np.abs(d)))
    return changed, mean_abs


def set_zlib_level(level: int) -> None:
    pxc_codec.ZLIB_LEVEL = max(1, min(9, int(level)))
