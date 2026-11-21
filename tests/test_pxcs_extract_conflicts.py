"""提取输出路径冲突的回归。

``--lossless`` 会把图片统一写成 PNG/TIFF，``same.jpg`` 与 ``same.png`` 会落到
同一个文件。旧实现第二条被 ``target.exists()`` 静默跳过，统计里只算 skipped。
现在默认报错，也可以显式 skip/rename。
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import _planned_target, extract_archive
from pxcs_core.container import PxcsFormatError
from tests.test_pxcs_zlib_bounds import _build_archive
from tests.test_pxcs_ref_chain import _file_entry, _image_entry


def _jpeg(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8), value, np.uint8), "L").save(
        buf, format="JPEG", quality=90
    )
    return buf.getvalue()


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8), value, np.uint8), "L").save(buf, format="PNG")
    return buf.getvalue()


class ExtractConflictTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _archive(self) -> Path:
        jpg, png = _jpeg(200), _png(7)
        entries = [
            _image_entry(0, "same.jpg", "raw", size=len(jpg),
                         format="JPEG", ext=".jpg"),
            _image_entry(1, "same.png", "raw", size=len(png),
                         format="PNG", ext=".png"),
        ]
        path = self.root / "conflict.pxcs"
        _build_archive(path, entries, [jpg, png])
        return path

    def test_default_error_mentions_both_entries(self) -> None:
        archive = self._archive()
        with self.assertRaises(PxcsFormatError) as ctx:
            extract_archive(archive, self.root / "out", lossless=True, quiet=True)
        message = str(ctx.exception)
        self.assertIn("same.jpg", message)
        self.assertIn("same.png", message)

    def test_skip_keeps_first_only(self) -> None:
        archive = self._archive()
        stats = extract_archive(
            archive, self.root / "out", lossless=True, quiet=True, on_conflict="skip"
        )
        self.assertEqual(stats.extracted, 1)
        self.assertEqual(stats.skipped, 1)
        files = list((self.root / "out").rglob("*.png"))
        self.assertEqual(len(files), 1)
        got = np.asarray(Image.open(files[0]))
        self.assertEqual(int(got.flat[0]), 200)  # 第一条（JPEG 的内容）

    def test_rename_keeps_both(self) -> None:
        archive = self._archive()
        stats = extract_archive(
            archive, self.root / "out", lossless=True, quiet=True, on_conflict="rename"
        )
        self.assertEqual(stats.extracted, 2)
        values = sorted(
            int(np.asarray(Image.open(p)).flat[0])
            for p in (self.root / "out").rglob("*.png")
        )
        self.assertEqual(values, [7, 200])

    def test_file_directory_prefix_conflict(self) -> None:
        payload = b"plain"
        entries = [
            _file_entry(0, "a", "store", len(payload)),
            _image_entry(1, "a/b.png", "raw", size=len(_png(1))),
        ]
        path = self.root / "prefix.pxcs"
        _build_archive(path, entries, [payload, _png(1)])
        with self.assertRaises(PxcsFormatError):
            extract_archive(path, self.root / "out2", quiet=True)

    def test_planned_target_matches_lossless_suffix(self) -> None:
        from pxcs_core.container import ArchiveEntry

        entry = ArchiveEntry.from_dict(
            _image_entry(0, "x.jpg", "raw", size=1, format="JPEG", ext=".jpg")
        )
        target = _planned_target(entry, self.root, lossless=True)
        self.assertEqual(target.suffix, ".png")


if __name__ == "__main__":
    unittest.main()
