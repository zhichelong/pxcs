"""PXC 错误类型。"""
from __future__ import annotations


class PxcError(Exception):
    """PXC 基础异常。"""


class UnsupportedImageError(PxcError):
    """图像格式、模式或特性不受支持。"""


class ImageDecodeError(PxcError):
    """源图片无法读取或解码。"""


class PxcFormatError(PxcError):
    """PXC 文件格式错误（魔数、字段、长度、校验等）。"""


class PxcVersionError(PxcFormatError):
    """PXC 版本不受支持。"""


class PxcCorruptionError(PxcFormatError):
    """PXC 数据损坏（CRC、zlib、像素哈希等）。"""


class PxcEncodeError(PxcError):
    """编码过程失败。"""


class PxcDecodeError(PxcError):
    """解码过程失败。"""
