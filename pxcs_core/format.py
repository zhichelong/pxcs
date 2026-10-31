"""PXCS 归档格式常量。

PXCS 是一个面向图片集合的无损/近无损归档格式：
- 图片先解码为像素，在集合内部寻找参考图并做差分；
- 非图片文件按普通文件压缩存放；
- 归档可以随机提取单个条目，但差分链长度由关键帧限制。

所有整数均为小端序。
"""
from __future__ import annotations

import struct

PXCS_MAGIC = b"PXCS"
PXCS_END_MAGIC = b"PXCSEND\x00"
VERSION_MAJOR = 1
# 当前最高支持的小版本：1.3 引入加密（分卷不改变格式本身）。
VERSION_MINOR = 4
VERSION_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}"
# 未加密归档仍按 1.2 写出，保证旧程序还能读新生成的普通归档。
WRITE_MINOR_PLAIN = 2
WRITE_MINOR_CRYPTO = 3
# 1.4：新增 jpegxl 编码（JPEG 无损重打包成 JPEG XL），旧读者会拒绝。
WRITE_MINOR_JPEGXL = 4
SUPPORTED_MINORS = (0, 1, 2, 3, 4)

# 128 字节固定头。
# magic, major, minor, flags, header_size, entry_count,
# index_offset, index_length, data_offset, archive_id,
# reserved4, reserved62, header_crc
HEADER_STRUCT = struct.Struct("<4sHHIHIQQQ16s4s62sI")
HEADER_SIZE = HEADER_STRUCT.size
assert HEADER_SIZE == 128, HEADER_SIZE

# 64 字节固定尾。
# magic, file_size, index_offset, index_length, index_crc,
# footer_crc, archive_id, reserved8
FOOTER_STRUCT = struct.Struct("<8sQQQII16s8s")
FOOTER_SIZE = FOOTER_STRUCT.size
assert FOOTER_SIZE == 64, FOOTER_SIZE

# Header flags
FLAG_HAS_DELTA = 1 << 0
FLAG_HAS_FILES = 1 << 1
FLAG_HAS_DUP = 1 << 2
FLAG_HAS_RAW_IMAGES = 1 << 3
FLAG_HAS_LOSSY_DELTA = 1 << 4  # 残差含量化，重建像素是近似值
FLAG_ENCRYPTED = 1 << 5        # 1.3：payload 与索引使用 AEAD 加密

# Entry codec
CODEC_PXC = "pxc"                  # 完整 PXC 文件（关键帧）
CODEC_PXC_RESIDUAL = "pxc_residual"  # 完整 PXC 文件，内容为像素残差
CODEC_ZLIB_RESIDUAL = "zlib_residual"  # zlib(居中/量化残差数组)，适合 JPEG 噪声
CODEC_WEBP_LOSSLESS = "webp_lossless"  # WebP 无损关键帧（8-bit RGB/RGBA）
CODEC_RAW = "raw"                  # 原始文件字节（JPEG/WebP/回退）
CODEC_JPEGXL = "jpegxl"            # JPEG 无损重打包（1.4，可字节级还原）
CODEC_STORE = "store"              # 非图片文件，原样
CODEC_ZLIB = "zlib"                # 非图片文件，zlib
CODEC_DUP = "dup"                  # 与前面的条目完全相同
IMAGE_CODECS = {
    CODEC_PXC, CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL,
    CODEC_WEBP_LOSSLESS, CODEC_RAW, CODEC_JPEGXL, CODEC_DUP,
}
RESIDUAL_CODECS = {CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL}
FILE_CODECS = {CODEC_STORE, CODEC_ZLIB, CODEC_DUP}

# Index JSON schema
INDEX_FORMAT = "PXCS"
INDEX_VERSION = 1

# 默认值
DEFAULT_TILE_SIZE = 128
DEFAULT_ZLIB_LEVEL = 6
DEFAULT_KEYFRAME_INTERVAL = 16
MAX_ENTRIES = 1_000_000
MAX_DEPTH = 255
MAX_PATH_BYTES = 4096
