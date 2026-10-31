"""JPEG XL 无损重打包的单元测试。

契约：pack() 出来的东西 unpack() 必须逐字节还原原始 JPEG；开关关掉时完全
不参与；没有依赖时 available() 为 False 且不抛异常。
"""
from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from pxcs_core import jxl


def _sample_jpeg() -> bytes:
    """造一张足够大的渐变 JPEG（太小会被 _MIN_BYTES 直接拒绝）。"""
    width = height = 512
    x = np.arange(width, dtype=np.uint8)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = x
    arr[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    rows = np.arange(height, dtype=np.int16)[:, None]
    arr[..., 2] = ((x[None, :].astype(np.int16) + rows) % 256).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


class JpegXLTests(unittest.TestCase):
    @unittest.skipUnless(jxl.available(), "未安装 imagecodecs")
    def test_roundtrip_is_byte_exact(self) -> None:
        data = _sample_jpeg()
        packed = jxl.pack(data)
        self.assertIsNotNone(packed, "样本 JPEG 应该能转成 jxl")
        self.assertLess(len(packed or b""), len(data), "转码后必须更小")
        self.assertEqual(jxl.unpack(packed or b""), data, "必须逐字节还原")

    @unittest.skipUnless(jxl.available(), "未安装 imagecodecs")
    def test_disabled_never_packs(self) -> None:
        data = _sample_jpeg()
        jxl.set_enabled(False)
        try:
            self.assertIsNone(jxl.pack(data))
        finally:
            jxl.set_enabled(True)

    def test_rejects_non_jpeg(self) -> None:
        self.assertIsNone(jxl.pack(b"not a jpeg at all" * 1000))

    def test_available_is_bool(self) -> None:
        self.assertIsInstance(jxl.available(), bool)


if __name__ == "__main__":
    unittest.main()
