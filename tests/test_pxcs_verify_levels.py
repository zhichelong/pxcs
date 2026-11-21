"""校验等级的回归。

关键语义：
- metadata 不读 payload；
- quick 只验 CRC/SHA，**绝不重建图片**（旧实现对残差条目仍整链解码）；
- full 才做完整解码 + 像素哈希。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from pxcs_core.archive import (
    CreateOptions,
    _ImageResolver,
    create_archive,
    verify_archive,
)
from pxcs_core.container import PxcsFormatError, ArchiveReader


def _pair(root: Path) -> None:
    base = np.random.default_rng(2).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    other = base.copy()
    other[:24, :24] = (other[:24, :24] // 2).astype(np.uint8)
    Image.fromarray(base).save(root / "a.png")
    Image.fromarray(other).save(root / "b.png")


class VerifyLevelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        src = self.root / "src"
        src.mkdir()
        _pair(src)
        self.archive = self.root / "a.pxcs"
        create_archive(
            self.archive, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_quiet_alias_still_means_quick(self) -> None:
        with mock.patch.object(
            _ImageResolver, "get", side_effect=AssertionError("不应解码")
        ):
            stats = verify_archive(self.archive, quick=True, threads=1)
        self.assertEqual(stats.failed, 0)

    def test_quick_never_decodes_images(self) -> None:
        with mock.patch.object(
            _ImageResolver, "get", side_effect=AssertionError("quick 不应解码")
        ):
            stats = verify_archive(self.archive, level="quick", threads=1)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(stats.checked, 2)

    def test_metadata_never_reads_payload(self) -> None:
        with mock.patch.object(
            ArchiveReader, "read_entry_data", side_effect=AssertionError("不应读 payload")
        ):
            stats = verify_archive(self.archive, level="metadata", threads=1)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(stats.checked, 2)

    def test_full_detects_payload_damage(self) -> None:
        with ArchiveReader(self.archive) as reader:
            entry = next(e for e in reader.entries if e.length > 8 and e.offset > 0)
            offset = entry.offset + 4
        raw = bytearray(self.archive.read_bytes())
        raw[offset] ^= 0xFF
        self.archive.write_bytes(bytes(raw))

        self.assertEqual(verify_archive(self.archive, level="metadata", threads=1).failed, 0)
        self.assertGreaterEqual(verify_archive(self.archive, level="quick", threads=1).failed, 1)

    def test_unknown_level_rejected(self) -> None:
        with self.assertRaises(PxcsFormatError):
            verify_archive(self.archive, level="paranoid", threads=1)

    def test_source_level_still_passes(self) -> None:
        stats = verify_archive(self.archive, level="source", threads=1)
        self.assertEqual(stats.failed, 0)


if __name__ == "__main__":
    unittest.main()
