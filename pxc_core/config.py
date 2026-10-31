"""PXC 0.2 全局常量与默认参数。

所有格式相关的魔数、结构体布局和默认值集中在这里，便于后续版本升级。
"""
from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# 版本与文件标识
# ---------------------------------------------------------------------------
PXC_MAGIC = b"PXC1"
PXC_END_MAGIC = b"PXCEND\x00\x00"
VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}"

# ---------------------------------------------------------------------------
# 固定文件头：128 字节，小端序
# 字段偏移见 README。
# ---------------------------------------------------------------------------
HEADER_SIZE = 128
HEADER_STRUCT = struct.Struct(
    "<4sBBHIIQQQQIIIHHHHHHIQQ32sI"
)
assert HEADER_SIZE == HEADER_STRUCT.size, HEADER_STRUCT.size

# 文件头 flags
FLAG_HAS_ORIG = 1 << 0
FLAG_HAS_ALPHA = 1 << 1
FLAG_HAS_PALETTE = 1 << 2
FLAG_SOURCE_LOSSY = 1 << 3
FLAG_PASSTHROUGH = 1 << 4

# ---------------------------------------------------------------------------
# Chunk 目录
# ---------------------------------------------------------------------------
CHUNK_ENTRY_STRUCT = struct.Struct("<4sIQQII")
CHUNK_ENTRY_SIZE = CHUNK_ENTRY_STRUCT.size
MAX_CHUNKS = 16
CHUNK_META = b"META"
CHUNK_DATA = b"DATA"
CHUNK_ORIG = b"ORIG"

# ---------------------------------------------------------------------------
# 分块与压缩
# ---------------------------------------------------------------------------
DEFAULT_TILE_SIZE = 256
MIN_TILE_SIZE = 8
MAX_TILE_SIZE = 1024
ZLIB_LEVEL = 6

DATA_HEADER_STRUCT = struct.Struct("<IIIII")
DATA_HEADER_SIZE = DATA_HEADER_STRUCT.size
DATA_FORMAT_VERSION = 1

# 单个 tile 记录头
TILE_HEADER_STRUCT = struct.Struct("<BBHIIII")
TILE_HEADER_SIZE = TILE_HEADER_STRUCT.size

# tile 模式
TILE_MODE_SOLID = 0
TILE_MODE_PALETTE = 1
TILE_MODE_PRED = 2
TILE_MODE_RUN = 3
TILE_MODE_RAW = 4
TILE_MODE_FILTER = 5
TILE_MODE_NAMES = {
    TILE_MODE_SOLID: "SOLID",
    TILE_MODE_PALETTE: "PALETTE",
    TILE_MODE_PRED: "PRED",
    TILE_MODE_RUN: "RUN",
    TILE_MODE_RAW: "RAW",
    TILE_MODE_FILTER: "FILTER",
}
TILE_MODE_ORDER = [
    TILE_MODE_SOLID,
    TILE_MODE_PALETTE,
    TILE_MODE_PRED,
    TILE_MODE_RUN,
    TILE_MODE_FILTER,
    TILE_MODE_RAW,
]

# PALETTE 子模式
PALETTE_PLAIN = 0
PALETTE_RLE = 1

# 压缩算法 ID
COMPRESSION_PXC_TILE_ZLIB = 1
COMPRESSION_PASSTHROUGH = 2
COMPRESSION_NAMES = {
    COMPRESSION_PXC_TILE_ZLIB: "PXC_TILE_ZLIB",
    COMPRESSION_PASSTHROUGH: "PASSTHROUGH",
}
ENTROPY_NAME = "zlib"

# 每个 tile 的 payload 如果 zlib 后不更小，就原样存储
FLAG_PAYLOAD_ZLIB = 1 << 0

# ---------------------------------------------------------------------------
# 图像模式
# ---------------------------------------------------------------------------
MODE_ID_TO_NAME = {
    1: "1",
    2: "L",
    3: "LA",
    4: "P",
    5: "RGB",
    6: "RGBA",
    7: "CMYK",
    8: "I",
    9: "I;16",
    10: "I;16L",
    11: "I;16B",
    12: "F",
}
MODE_NAME_TO_ID = {v: k for k, v in MODE_ID_TO_NAME.items()}

# 颜色模型
COLOR_MODEL_GRAY = 1
COLOR_MODEL_GRAY_ALPHA = 2
COLOR_MODEL_PALETTE = 3
COLOR_MODEL_RGB = 4
COLOR_MODEL_RGBA = 5
COLOR_MODEL_CMYK = 6
COLOR_MODEL_INTEGER = 7
COLOR_MODEL_FLOAT = 8

# Alpha 类型
ALPHA_NONE = 0
ALPHA_STRAIGHT = 1
ALPHA_PALETTE_TRANSPARENCY = 2
ALPHA_CONSTANT = 3

# 每种 Pillow mode 的通道数、位深、颜色模型、Alpha 类型
MODE_SPECS = {
    "1": {"channels": 1, "bit_depth": 1, "color_model": COLOR_MODEL_GRAY, "alpha": ALPHA_NONE},
    "L": {"channels": 1, "bit_depth": 8, "color_model": COLOR_MODEL_GRAY, "alpha": ALPHA_NONE},
    "LA": {"channels": 2, "bit_depth": 8, "color_model": COLOR_MODEL_GRAY_ALPHA, "alpha": ALPHA_STRAIGHT},
    "P": {"channels": 1, "bit_depth": 8, "color_model": COLOR_MODEL_PALETTE, "alpha": ALPHA_NONE},
    "RGB": {"channels": 3, "bit_depth": 8, "color_model": COLOR_MODEL_RGB, "alpha": ALPHA_NONE},
    "RGBA": {"channels": 4, "bit_depth": 8, "color_model": COLOR_MODEL_RGBA, "alpha": ALPHA_STRAIGHT},
    "CMYK": {"channels": 4, "bit_depth": 8, "color_model": COLOR_MODEL_CMYK, "alpha": ALPHA_NONE},
    "I": {"channels": 1, "bit_depth": 32, "color_model": COLOR_MODEL_INTEGER, "alpha": ALPHA_NONE},
    "I;16": {"channels": 1, "bit_depth": 16, "color_model": COLOR_MODEL_GRAY, "alpha": ALPHA_NONE},
    "I;16L": {"channels": 1, "bit_depth": 16, "color_model": COLOR_MODEL_GRAY, "alpha": ALPHA_NONE},
    "I;16B": {"channels": 1, "bit_depth": 16, "color_model": COLOR_MODEL_GRAY, "alpha": ALPHA_NONE},
    "F": {"channels": 1, "bit_depth": 32, "color_model": COLOR_MODEL_FLOAT, "alpha": ALPHA_NONE},
}

# 支持读取的扩展名（大小写不敏感）
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

# 源格式中视为“有损”的格式（用于 header flag 与说明）
LOSSY_SOURCE_FORMATS = {"JPEG", "JPG", "WEBP"}

# 单个图像最大像素数，防止恶意文件导致巨量内存分配
MAX_PIXELS = 300_000_000
MAX_WIDTH = 1_000_000
MAX_HEIGHT = 1_000_000
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_METADATA_ENTRIES = 64

# 元数据 TLV key
META_ORIGINAL_FILENAME = 1
META_SOURCE_FORMAT = 2
META_SOURCE_EXT = 3
META_MODE_NAME = 4
META_PALETTE = 5
META_TRANSPARENCY = 6
META_ICC_PROFILE = 7
META_EXIF = 8
META_XMP = 9
META_DPI = 10
META_ENTROPY_NAME = 11
META_ENTROPY_VERSION = 12
META_CREATED_BY = 13
META_SOURCE_FILE_HASH = 14
META_SOURCE_PIXEL_HASH = 15
META_GAMMA = 16
META_COMMENT = 17
META_CONSTANT_ALPHA = 18

META_KEY_NAMES = {
    META_ORIGINAL_FILENAME: "original_filename",
    META_SOURCE_FORMAT: "source_format",
    META_SOURCE_EXT: "source_ext",
    META_MODE_NAME: "mode_name",
    META_PALETTE: "palette",
    META_TRANSPARENCY: "transparency",
    META_ICC_PROFILE: "icc_profile",
    META_EXIF: "exif",
    META_XMP: "xmp",
    META_DPI: "dpi",
    META_ENTROPY_NAME: "entropy_name",
    META_ENTROPY_VERSION: "entropy_version",
    META_CREATED_BY: "created_by",
    META_SOURCE_FILE_HASH: "source_file_hash",
    META_SOURCE_PIXEL_HASH: "source_pixel_hash",
    META_GAMMA: "gamma",
    META_COMMENT: "comment",
    META_CONSTANT_ALPHA: "constant_alpha",
}

# 元数据类型
META_TYPE_BYTES = 1
META_TYPE_UTF8 = 2
META_TYPE_U32 = 3
META_TYPE_U64 = 4
META_TYPE_F64X2 = 5
META_TYPE_F64 = 6

CREATED_BY = "PXC Python prototype 0.1"
