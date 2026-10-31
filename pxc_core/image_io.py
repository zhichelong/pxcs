"""图像读写与 Pillow 模式保持。

设计要点：
- 压缩的是 Pillow 解码后的“样本数组”，而不是把图片统一转成 RGB。
- 对 16-bit 模式保留字节序信息，重建时按原 mode 生成字节。
- 调色板、transparency、ICC/EXIF/XMP/DPI 等元数据单独保存。
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, PngImagePlugin

from .config import (
    ALPHA_PALETTE_TRANSPARENCY,
    MODE_SPECS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from .errors import ImageDecodeError, UnsupportedImageError
from .utils import sha256_bytes, sha256_file


# 每种模式的数据在内存中的 dtype 与存储字节序
MODE_DTYPE: dict[str, np.dtype] = {
    "1": np.dtype(np.uint8),
    "L": np.dtype(np.uint8),
    "LA": np.dtype(np.uint8),
    "P": np.dtype(np.uint8),
    "RGB": np.dtype(np.uint8),
    "RGBA": np.dtype(np.uint8),
    "CMYK": np.dtype(np.uint8),
    "I": np.dtype(np.int32),
    "I;16": np.dtype(np.uint16),
    "I;16L": np.dtype(np.uint16),
    "I;16B": np.dtype(np.uint16),
    "F": np.dtype(np.float32),
}

MODE_BYTE_ORDER: dict[str, str] = {
    "I;16B": "big",
}


@dataclass
class ImageData:
    """解码后的像素数据 + 恢复所需的元数据。"""

    mode: str
    width: int
    height: int
    data: np.ndarray
    palette: Optional[bytes] = None
    transparency: Optional[bytes] = None  # 编码后的 transparency，见 encode_transparency
    icc_profile: Optional[bytes] = None
    exif: Optional[bytes] = None
    xmp: Optional[bytes] = None
    dpi: Optional[tuple[float, float]] = None
    gamma: Optional[float] = None
    comment: Optional[bytes] = None
    original_filename: str = ""
    source_format: str = ""
    source_ext: str = ""
    source_file_size: int = 0
    source_file_hash: Optional[bytes] = None
    source_pixel_hash: Optional[bytes] = None
    pixel_hash: bytes = b""
    constant_alpha: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODE_SPECS:
            raise UnsupportedImageError(f"不支持的图像模式: {self.mode}")
        spec = MODE_SPECS[self.mode]
        self.data = np.ascontiguousarray(self.data)
        if self.data.dtype != MODE_DTYPE[self.mode]:
            self.data = self.data.astype(MODE_DTYPE[self.mode])
        expected_shape = (self.height, self.width, spec["channels"])
        if self.data.shape != expected_shape:
            raise UnsupportedImageError(
                f"模式 {self.mode} 的数据形状应为 {expected_shape}，实际 {self.data.shape}"
            )
        if self.width <= 0 or self.height <= 0:
            raise UnsupportedImageError("图像尺寸必须大于 0")
        if self.constant_alpha is not None:
            if self.mode not in ("RGBA", "LA"):
                raise UnsupportedImageError("constant_alpha 只适用于 RGBA/LA")
            if not (0 <= int(self.constant_alpha) <= 255):
                raise UnsupportedImageError("constant_alpha 必须在 0..255")

    @property
    def channels(self) -> int:
        return int(MODE_SPECS[self.mode]["channels"])

    @property
    def bit_depth(self) -> int:
        return int(MODE_SPECS[self.mode]["bit_depth"])

    @property
    def color_model(self) -> int:
        return int(MODE_SPECS[self.mode]["color_model"])

    @property
    def alpha_kind(self) -> int:
        if self.mode == "P" and self.transparency is not None:
            return ALPHA_PALETTE_TRANSPARENCY
        if self.constant_alpha is not None:
            return 3  # ALPHA_CONSTANT
        return int(MODE_SPECS[self.mode]["alpha"])

    @property
    def stored_channels(self) -> int:
        """实际写入 PXC 的通道数；纯色 alpha 可以被省略。"""
        if self.constant_alpha is not None and self.mode in ("RGBA", "LA"):
            return self.channels - 1
        return self.channels

    @property
    def stored_data(self) -> np.ndarray:
        return self.data[:, :, : self.stored_channels]


    @property
    def dtype(self) -> np.dtype:
        return MODE_DTYPE[self.mode]

    @property
    def sample_bytes(self) -> int:
        return self.dtype.itemsize

    @property
    def has_palette(self) -> bool:
        return self.mode == "P" and self.palette is not None

    @property
    def has_alpha(self) -> bool:
        return self.alpha_kind != 0


def storage_dtype(dtype: np.dtype) -> np.dtype:
    """PXC 内部统一使用小端序存储多字节样本。"""
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint8):
        return np.dtype(np.uint8)
    if dt == np.dtype(np.uint16):
        return np.dtype("<u2")
    if dt == np.dtype(np.uint32):
        return np.dtype("<u4")
    if dt == np.dtype(np.int32):
        return np.dtype("<i4")
    if dt == np.dtype(np.float32):
        return np.dtype("<f4")
    raise UnsupportedImageError(f"不支持的数据类型: {dtype}")


def encode_transparency(value: Any) -> Optional[bytes]:
    """把 Pillow 的 transparency 统一编码成可逆的字节串。

    0x00 + u32：单个索引
    0x01 + bytes：P 模式的 alpha 表
    0x02 + u32*3：RGB 的 tRNS 颜色键
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return b"\x00" + struct.pack("<I", int(value))
    if isinstance(value, (bytes, bytearray)):
        return b"\x01" + bytes(value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return b"\x02" + struct.pack("<III", *(int(x) for x in value))
    raise UnsupportedImageError(f"不支持的 transparency 类型: {type(value)!r}")


def decode_transparency(blob: Optional[bytes]) -> Any:
    if blob is None:
        return None
    if len(blob) < 1:
        raise UnsupportedImageError("transparency 数据为空")
    tag = blob[0]
    if tag == 0:
        if len(blob) != 5:
            raise UnsupportedImageError("transparency int 数据长度错误")
        return struct.unpack("<I", blob[1:])[0]
    if tag == 1:
        return blob[1:]
    if tag == 2:
        if len(blob) != 13:
            raise UnsupportedImageError("transparency tuple 数据长度错误")
        return struct.unpack("<III", blob[1:])
    raise UnsupportedImageError(f"未知的 transparency 标签: {tag}")


def canonical_pixel_hash(mode: str, width: int, height: int, data: np.ndarray) -> bytes:
    """对样本数组做规范化哈希，用于验证解码结果。"""
    h = hashlib.sha256()
    h.update(b"PXC-PIXELS\x00")
    h.update(mode.encode("utf-8"))
    h.update(struct.pack("<II", width, height))
    h.update(np.ascontiguousarray(data).tobytes())
    return h.digest()


def load_image(
    path: Path,
    compute_file_hash: bool = True,
    compute_source_pixel_hash: bool = True,
) -> ImageData:
    """用 Pillow 读取图片，保持 mode，不自动转换颜色空间。"""
    try:
        with Image.open(path) as img:
            img.load()
            n_frames = int(getattr(img, "n_frames", 1) or 1)
            if n_frames > 1 or bool(getattr(img, "is_animated", False)):
                raise UnsupportedImageError(
                    f"暂不支持动画或多帧图片（{n_frames} 帧）: {path.name}"
                )
            if img.mode not in MODE_SPECS:
                raise UnsupportedImageError(
                    f"暂不支持 Pillow 模式 {img.mode!r}: {path.name}"
                )
            return _image_to_data(
                img, path, compute_file_hash, compute_source_pixel_hash
            )
    except UnsupportedImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"无法读取图片 {path}: {exc}") from exc


def load_image_bytes(
    data: bytes,
    filename: str,
    source_file_hash: Optional[bytes] = None,
    source_file_size: int = 0,
    source_format: Optional[str] = None,
) -> ImageData:
    """从内存中的原始图片字节解码（passthrough 模式使用）。"""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            n_frames = int(getattr(img, "n_frames", 1) or 1)
            if n_frames > 1 or bool(getattr(img, "is_animated", False)):
                raise UnsupportedImageError(
                    f"暂不支持动画或多帧图片（{n_frames} 帧）: {filename}"
                )
            if img.mode not in MODE_SPECS:
                raise UnsupportedImageError(
                    f"暂不支持 Pillow 模式 {img.mode!r}: {filename}"
                )
            image = _image_to_data(img, Path(filename), compute_file_hash=False)
    except UnsupportedImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"无法从 PXC 原始字节解码 {filename}: {exc}") from exc

    image.original_filename = Path(filename).name
    image.source_file_size = source_file_size
    image.source_file_hash = source_file_hash
    if source_format:
        image.source_format = source_format.upper()
    return image


def _image_to_data(
    img: Image.Image,
    path: Path,
    compute_file_hash: bool,
    compute_source_pixel_hash: bool = True,
) -> ImageData:
    mode = img.mode
    width, height = img.size
    channels = int(MODE_SPECS[mode]["channels"])

    if mode == "1":
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim != 2:
            raise UnsupportedImageError("模式 1 的数组维度异常")
        arr = arr[:, :, None]
        # 规范化为 0/1，避免 Pillow 某些版本返回 0/255
        arr = (arr != 0).astype(np.uint8)
    elif mode in ("I;16", "I;16L", "I;16B"):
        raw = img.tobytes()
        if mode == "I;16B":
            dt = np.dtype(">u2")
        elif mode == "I;16L":
            dt = np.dtype("<u2")
        else:
            dt = np.dtype("=u2")
        arr = np.frombuffer(raw, dtype=dt).reshape(height, width)
        arr = arr.astype(np.uint16)[:, :, None]
    elif mode == "I":
        arr = np.array(img, dtype=np.int32)[:, :, None]
    elif mode == "F":
        arr = np.array(img, dtype=np.float32)[:, :, None]
    elif mode == "P":
        arr = np.array(img, dtype=np.uint8)[:, :, None]
    else:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

    arr = np.ascontiguousarray(arr)
    expected = (height, width, channels)
    if arr.shape != expected:
        raise UnsupportedImageError(
            f"模式 {mode} 数组形状应为 {expected}，实际 {arr.shape}"
        )

    palette: Optional[bytes] = None
    if mode == "P":
        pal = img.getpalette()
        if not pal:
            raise UnsupportedImageError("P 模式图片缺少调色板")
        try:
            palette = bytes(int(v) & 0xFF for v in pal)
        except (TypeError, ValueError) as exc:
            raise UnsupportedImageError(f"调色板数据异常: {exc}") from exc

    info = img.info
    transparency = encode_transparency(info.get("transparency"))
    icc = info.get("icc_profile")
    if isinstance(icc, memoryview):
        icc = bytes(icc)
    if icc is not None and not isinstance(icc, bytes):
        icc = bytes(icc)

    exif_value = info.get("exif")
    if isinstance(exif_value, Image.Exif):
        exif_value = exif_value.tobytes()
    if exif_value is not None and not isinstance(exif_value, bytes):
        try:
            exif_value = bytes(exif_value)
        except (TypeError, ValueError):
            exif_value = None

    xmp_value = info.get("xmp") or info.get("XML:com.adobe.xmp")
    if isinstance(xmp_value, str):
        xmp_value = xmp_value.encode("utf-8")
    if xmp_value is not None and not isinstance(xmp_value, bytes):
        try:
            xmp_value = bytes(xmp_value)
        except (TypeError, ValueError):
            xmp_value = None

    dpi_value = info.get("dpi")
    dpi: Optional[tuple[float, float]] = None
    if isinstance(dpi_value, (tuple, list)) and len(dpi_value) == 2:
        try:
            dpi = (float(dpi_value[0]), float(dpi_value[1]))
        except (TypeError, ValueError):
            dpi = None

    gamma_value = info.get("gamma")
    gamma: Optional[float] = None
    if isinstance(gamma_value, (int, float)):
        gamma = float(gamma_value)

    comment_value = info.get("comment") or info.get("Comment")
    if isinstance(comment_value, str):
        comment_value = comment_value.encode("utf-8")
    elif isinstance(comment_value, bytearray):
        comment_value = bytes(comment_value)
    if comment_value is not None and not isinstance(comment_value, bytes):
        comment_value = None

    source_format = (img.format or path.suffix.lstrip(".")).upper()
    src_hash = sha256_file(path) if compute_file_hash else None
    src_size = path.stat().st_size if compute_file_hash else 0
    source_pixel_hash = (
        sha256_bytes(img.tobytes()) if compute_source_pixel_hash else None
    )
    pixel_hash = canonical_pixel_hash(mode, width, height, arr)

    # 纯色 alpha 可以省略整个 alpha 平面；解码时再补回。
    constant_alpha = None
    if mode in ("RGBA", "LA"):
        alpha = arr[..., -1]
        if alpha.size and bool(np.all(alpha == alpha.flat[0])):
            constant_alpha = int(alpha.flat[0])

    return ImageData(
        mode=mode,
        width=width,
        height=height,
        data=arr,
        palette=palette,
        transparency=transparency,
        icc_profile=icc,
        exif=exif_value,
        xmp=xmp_value,
        dpi=dpi,
        gamma=gamma,
        comment=comment_value,
        original_filename=path.name,
        source_format=source_format,
        source_ext=path.suffix.lower(),
        source_file_size=src_size,
        source_file_hash=src_hash,
        source_pixel_hash=source_pixel_hash,
        pixel_hash=pixel_hash,
        constant_alpha=constant_alpha,
    )


def data_to_pillow(image: ImageData) -> Image.Image:
    """把样本数组恢复成 Pillow 图像（不写盘）。"""
    mode = image.mode
    width, height = image.width, image.height
    data = np.ascontiguousarray(image.data)

    if mode == "1":
        values = data[:, :, 0]
        if values.size and (values.min() < 0 or values.max() > 1):
            raise UnsupportedImageError("模式 1 的像素值必须是 0 或 1")
        packed = np.packbits(values.astype(bool), axis=1, bitorder="big")
        img = Image.frombytes("1", (width, height), packed.tobytes())
    elif mode == "I;16":
        img = Image.fromarray(np.ascontiguousarray(data[:, :, 0].astype(np.uint16)))
    elif mode == "I;16L":
        raw = data[:, :, 0].astype("<u2").tobytes()
        img = Image.frombytes("I;16L", (width, height), raw)
    elif mode == "I;16B":
        raw = data[:, :, 0].astype(">u2").tobytes()
        img = Image.frombytes("I;16B", (width, height), raw)
    elif mode == "I":
        img = Image.fromarray(np.ascontiguousarray(data[:, :, 0].astype(np.int32)), "I")
    elif mode == "F":
        img = Image.fromarray(np.ascontiguousarray(data[:, :, 0].astype(np.float32)), "F")
    elif mode == "P":
        img = Image.fromarray(np.ascontiguousarray(data[:, :, 0].astype(np.uint8)), "P")
        if image.palette:
            if len(image.palette) % 3 != 0:
                raise UnsupportedImageError("调色板长度必须是 3 的倍数")
            img.putpalette(list(image.palette))
        transparency = decode_transparency(image.transparency)
        if transparency is not None:
            img.info["transparency"] = transparency
    elif mode == "L":
        img = Image.fromarray(np.ascontiguousarray(data[:, :, 0].astype(np.uint8)), "L")
        transparency = decode_transparency(image.transparency)
        if transparency is not None:
            img.info["transparency"] = transparency
    elif mode in ("LA", "RGB", "RGBA", "CMYK"):
        img = Image.fromarray(data, mode)
        transparency = decode_transparency(image.transparency)
        if transparency is not None and mode == "RGB":
            img.info["transparency"] = transparency
    else:
        raise UnsupportedImageError(f"无法重建模式 {mode}")

    return img


def _pnginfo_with_xmp(xmp: Optional[bytes]) -> Optional[PngImagePlugin.PngInfo]:
    if not xmp:
        return None
    try:
        text = xmp.decode("utf-8")
    except UnicodeDecodeError:
        return None
    info = PngImagePlugin.PngInfo()
    info.add_text("XML:com.adobe.xmp", text, zip=False)
    return info


def save_image(
    image: ImageData,
    path: Path,
    format_name: str,
    quality: Optional[int] = None,
) -> None:
    """按指定格式保存图片，尽可能写回 ICC/EXIF/XMP/DPI。

    quality 仅对 JPEG/WebP 生效；None 表示沿用历史默认值：
    JPEG=95、WebP=lossless。
    """
    img = data_to_pillow(image)
    fmt = format_name.upper()
    kwargs: dict[str, Any] = {"format": fmt}
    if quality is not None:
        quality = max(1, min(100, int(quality)))

    if fmt == "PNG":
        pnginfo = _pnginfo_with_xmp(image.xmp)
        if pnginfo is not None:
            kwargs["pnginfo"] = pnginfo
        if image.icc_profile:
            kwargs["icc_profile"] = image.icc_profile
        if image.exif:
            kwargs["exif"] = image.exif
        if image.dpi:
            kwargs["dpi"] = image.dpi
        if image.gamma is not None:
            kwargs["gamma"] = image.gamma
        if image.comment:
            try:
                pnginfo = kwargs.get("pnginfo")
                if pnginfo is None:
                    pnginfo = PngImagePlugin.PngInfo()
                    kwargs["pnginfo"] = pnginfo
                pnginfo.add_text("Comment", image.comment.decode("utf-8"), zip=False)
            except (UnicodeDecodeError, AttributeError):
                pass
    elif fmt == "JPEG":
        if image.icc_profile:
            kwargs["icc_profile"] = image.icc_profile
        if image.exif:
            kwargs["exif"] = image.exif
        if image.xmp:
            kwargs["xmp"] = image.xmp
        if image.dpi:
            kwargs["dpi"] = image.dpi
        kwargs["quality"] = 95 if quality is None else quality
        kwargs["subsampling"] = 0
    elif fmt == "WEBP":
        if image.icc_profile:
            kwargs["icc_profile"] = image.icc_profile
        if image.exif:
            kwargs["exif"] = image.exif
        if image.xmp:
            kwargs["xmp"] = image.xmp
        if quality is None:
            kwargs["lossless"] = True
        else:
            kwargs["quality"] = quality
            kwargs["lossless"] = False
    elif fmt == "TIFF":
        if image.icc_profile:
            kwargs["icc_profile"] = image.icc_profile
        if image.exif:
            kwargs["exif"] = image.exif
        if image.dpi:
            kwargs["dpi"] = image.dpi
        kwargs["compression"] = "tiff_deflate"
    elif fmt == "GIF":
        kwargs["optimize"] = False
    elif fmt == "BMP":
        if image.dpi:
            kwargs["dpi"] = image.dpi
    else:
        raise UnsupportedImageError(f"不支持的输出格式: {fmt}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img.save(path, **kwargs)
    except (TypeError, ValueError, OSError) as exc:
        # 某些 Pillow 版本/格式不支持某个元数据参数，降级重试，避免丢像素。
        retry_kwargs = dict(kwargs)
        for key in ("xmp", "exif", "icc_profile", "dpi", "gamma", "pnginfo"):
            retry_kwargs.pop(key, None)
        try:
            img.save(path, **retry_kwargs)
        except Exception:
            raise exc
        print(f"[WARN] {path.name}: 部分元数据无法写入 {fmt}，已保留像素数据（{exc}）")


def verify_saved_image(path: Path, expected: ImageData) -> None:
    """重新打开刚保存的图片，比较样本数组，防止输出格式改变像素。"""
    with Image.open(path) as img:
        img.load()
        actual = _image_to_data(img, path, compute_file_hash=False)
    if actual.mode != expected.mode:
        # 某些格式会规范化 mode（如 BMP 的 1-bit），像素仍可能一致。
        if actual.data.shape != expected.data.shape or not np.array_equal(actual.data, expected.data):
            raise ImageDecodeError(
                f"恢复图片校验失败：mode {expected.mode} -> {actual.mode}，像素不一致"
            )
        print(f"[WARN] {path.name}: 输出格式将 mode {expected.mode} 规范化为 {actual.mode}")
        return
    if not np.array_equal(actual.data, expected.data):
        raise ImageDecodeError(f"恢复图片校验失败：像素数据不一致 {path.name}")
    if expected.mode == "P":
        if (actual.palette or b"")[: len(expected.palette or b"")] != (expected.palette or b""):
            print(f"[WARN] {path.name}: 调色板被输出格式规范化，索引数据仍一致")


def is_supported_image_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
