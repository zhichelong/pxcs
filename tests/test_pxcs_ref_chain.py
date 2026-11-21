"""引用图（ref/depth）的加载校验与迭代解码回归。

两条性质：

1. 伪造 ``depth`` 不能绕过链长上限——1200 条 DUP 每条写 depth=1 的归档，
   必须在加载阶段被拒绝，而不是等到解码时抛 RecursionError。
2. 合法链（depth 逐条 +1）按任意长度都能解开，DUP 的文件/图片两条路径都不再递归。
"""
from __future__ import annotations

import io
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from pxc_core.errors import PxcFormatError
from pxcs_core.archive import _ImageResolver, _original_bytes
from pxcs_core.container import ArchiveReader
from tests.test_pxcs_zlib_bounds import _build_archive, _crc


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), "L").save(buf, format="PNG")
    return buf.getvalue()


def _image_entry(eid: int, name: str, codec: str, **kw) -> dict:
    entry = {
        "id": eid, "path": name, "kind": "image", "codec": codec,
        "offset": 0, "length": 0, "size": kw.pop("size", 1),
        "ref": kw.pop("ref", None), "depth": kw.pop("depth", 0),
        "width": 2, "height": 2, "mode": "L", "format": "PNG", "ext": ".png",
        "constant_alpha": None, "quant": 0, "centered": True, "lossy": False,
        "pixel_sha256": "", "source_sha256": "", "data_crc32": 0,
        "raw_crc32": 0, "data_sha256": "", "mtime_ns": 0, "flags": 0, "reason": "",
    }
    entry.update(kw)
    return entry


def _file_entry(eid: int, name: str, codec: str, size: int, **kw) -> dict:
    entry = {
        "id": eid, "path": name, "kind": "file", "codec": codec,
        "offset": 0, "length": 0, "size": size, "ref": kw.pop("ref", None),
        "depth": kw.pop("depth", 0), "width": 0, "height": 0, "mode": "",
        "format": "", "ext": "", "quant": 0, "centered": False, "lossy": False,
        "pixel_sha256": "", "source_sha256": "", "data_crc32": 0,
        "raw_crc32": 0, "data_sha256": "", "mtime_ns": 0, "flags": 0, "reason": "",
    }
    entry.update(kw)
    return entry


class RefChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_forged_depth_is_rejected_at_load(self) -> None:
        png = _png_bytes()
        base = _image_entry(0, "base.png", "raw", size=len(png))
        entries = [base]
        payloads: list[bytes | None] = [png]
        for i in range(1, 1200):
            # 真实链长 1199，声明 depth=1。
            entries.append(
                _image_entry(i, f"d{i}.png", "dup", ref=i - 1, depth=1)
            )
            payloads.append(None)
        path = self.tmp / "forged.pxcs"
        _build_archive(path, entries, payloads)

        with self.assertRaises(PxcFormatError) as ctx:
            with ArchiveReader(path):
                pass
        self.assertIn("depth", str(ctx.exception).lower())

    def test_too_deep_chain_is_rejected(self) -> None:
        png = _png_bytes()
        base = _image_entry(0, "base.png", "raw", size=len(png))
        entries = [base]
        payloads: list[bytes | None] = [png]
        for i in range(1, 300):  # 真实深度 299 > MAX_DEPTH(255)
            entries.append(
                _image_entry(i, f"d{i}.png", "dup", ref=i - 1, depth=i)
            )
            payloads.append(None)
        path = self.tmp / "deep.pxcs"
        _build_archive(path, entries, payloads)
        with self.assertRaises(PxcFormatError):
            with ArchiveReader(path):
                pass

    def test_valid_image_chain_resolves_iteratively(self) -> None:
        png = _png_bytes()
        base = _image_entry(0, "base.png", "raw", size=len(png))
        entries = [base]
        payloads: list[bytes | None] = [png]
        for i in range(1, 65):
            entries.append(_image_entry(i, f"d{i}.png", "dup", ref=i - 1, depth=i))
            payloads.append(None)
        path = self.tmp / "ok.pxcs"
        _build_archive(path, entries, payloads)
        with ArchiveReader(path) as reader:
            image = _ImageResolver(reader).get(64)
        expected = np.asarray(Image.open(io.BytesIO(png)))
        self.assertTrue(np.array_equal(image.data[:, :, 0], expected))

    def test_valid_file_dup_chain_resolves_iteratively(self) -> None:
        raw = b"payload-bytes"
        entries = [_file_entry(0, "a.bin", "store", len(raw))]
        payloads: list[bytes | None] = [raw]
        for i in range(1, 65):
            entries.append(_file_entry(i, f"c{i}.bin", "dup", len(raw), ref=i - 1, depth=i))
            payloads.append(None)
        path = self.tmp / "file.pxcs"
        _build_archive(path, entries, payloads)
        with ArchiveReader(path) as reader:
            self.assertEqual(_original_bytes(reader, reader.get_entry(64)), raw)


if __name__ == "__main__":
    unittest.main()
