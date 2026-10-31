"""verify 并行化的等价性测试。

契约：并行校验的结论必须和串行完全一致（通过数/失败数/能否发现损坏），
并行只影响速度。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive, verify_archive
from pxcs_core.container import ArchiveReader


def _make_images(root: Path, count: int) -> None:
    """造 count 张互不相同的小图，保证条目数超过并行阈值。"""
    for index in range(count):
        width = 32 + (index % 5) * 8
        height = 24 + (index % 3) * 8
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = (np.arange(width) + index * 7) % 256
        arr[..., 1] = ((np.arange(height) + index * 3) % 256)[:, None]
        arr[..., 2] = (index * 11) % 256
        Image.fromarray(arr, mode="RGB").save(
            root / f"img_{index:03d}.png", format="PNG"
        )


class VerifyParallelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        _make_images(self.src, 70)
        self.archive = self.tmp / "a.pxcs"
        create_archive(
            self.archive,
            [str(self.src)],
            CreateOptions(quiet=True, progress=False, overwrite=True),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parallel_matches_serial(self) -> None:
        serial = verify_archive(self.archive, threads=1)
        parallel = verify_archive(self.archive, threads=4)
        self.assertEqual(serial.checked, parallel.checked)
        self.assertEqual(serial.failed, parallel.failed)
        self.assertEqual(serial.checked, 70)
        self.assertEqual(serial.failed, 0)

    def test_parallel_detects_corruption(self) -> None:
        with ArchiveReader(self.archive) as reader:
            entry = reader.entries[0]
            offset = entry.offset
        data = bytearray(self.archive.read_bytes())
        # 翻掉 payload 的第一个字节，CRC/SHA 必然对不上
        data[offset] ^= 0xFF
        self.archive.write_bytes(bytes(data))

        parallel = verify_archive(self.archive, threads=4)
        self.assertGreaterEqual(parallel.failed, 1)
        self.assertTrue(any("img_" in message for message in parallel.errors))


if __name__ == "__main__":
    unittest.main()
