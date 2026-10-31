"""extract 并行化的等价性测试。

契约：并行提取的结果必须和串行逐字节一致；"目标已存在就跳过"的语义不能变。
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive, extract_archive


def _make_images(root: Path, count: int) -> None:
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


class ExtractParallelTests(unittest.TestCase):
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

    @staticmethod
    def _digest(root: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_parallel_matches_serial(self) -> None:
        serial_dir = self.tmp / "out_serial"
        parallel_dir = self.tmp / "out_parallel"
        serial = extract_archive(
            self.archive, serial_dir, overwrite=True, quiet=True, threads=1
        )
        parallel = extract_archive(
            self.archive, parallel_dir, overwrite=True, quiet=True, threads=4
        )
        self.assertEqual(serial.extracted, 70)
        self.assertEqual(parallel.extracted, 70)
        self.assertEqual(serial.failed, 0)
        self.assertEqual(parallel.failed, 0)
        self.assertEqual(self._digest(serial_dir), self._digest(parallel_dir))

    def test_skip_existing_in_parallel(self) -> None:
        out = self.tmp / "out"
        extract_archive(self.archive, out, overwrite=True, quiet=True, threads=4)
        again = extract_archive(self.archive, out, overwrite=False, quiet=True, threads=4)
        self.assertEqual(again.extracted, 0)
        self.assertEqual(again.skipped, 70)


if __name__ == "__main__":
    unittest.main()
