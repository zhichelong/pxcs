"""判断 WebP 是有损还是无损。

Pillow 的 ``Image.format`` 只告诉你是 WEBP，不告诉你编码方式；而 PXCS 的
量化残差只对有损源安全。之前按扩展名一刀切，把无损 WebP 也当有损处理，
量化后像素会变。

WebP 是 RIFF 容器，判据很简单：

- 首个图像块是 ``VP8L`` → 无损；
- 首个图像块是 ``VP8 `` → 有损；
- ``VP8X``（扩展格式）后面还会跟真正的图像块，继续往后找；
- ``ANIM``/``ANMF``（动图）不在这里处理，返回 ``None`` 让调用方保守处理。

只读文件头，不解码像素。
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Union

_IMAGE_CHUNKS = {b"VP8L": True, b"VP8 ": False}
_MAX_HEADER_SCAN = 1 << 20  # 只扫前 1 MiB，避免异常文件把整份读进来


def _parse(buf: bytes) -> Optional[bool]:
    if len(buf) < 12 or buf[0:4] != b"RIFF" or buf[8:12] != b"WEBP":
        return None
    offset = 12
    size = len(buf)
    while offset + 8 <= size:
        fourcc = buf[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", buf, offset + 4)[0]
        if fourcc in _IMAGE_CHUNKS:
            return _IMAGE_CHUNKS[fourcc]
        if fourcc in (b"ANIM", b"ANMF"):
            return None
        # 块体按偶数字节对齐
        offset += 8 + chunk_size + (chunk_size & 1)
        if offset > size:
            return None
    return None


def is_lossless_webp(source: Union[bytes, str, Path]) -> Optional[bool]:
    """``True``=无损，``False``=有损，``None``=不是 WebP 或无法判断。"""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _parse(bytes(source[:_MAX_HEADER_SCAN]))
    path = Path(source)
    try:
        with path.open("rb") as fh:
            head = fh.read(12)
            if len(head) < 12 or head[8:12] != b"WEBP":
                return None
            fh.seek(0)
            return _parse(fh.read(_MAX_HEADER_SCAN))
    except OSError:
        return None
