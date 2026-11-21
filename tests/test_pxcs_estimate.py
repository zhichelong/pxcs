"""创建前估算（--dry-run）的回归。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, estimate_archive

REPO = Path(__file__).resolve().parents[1]


def _make_set(root: Path) -> None:
    """一张底图 + 三张变体 + 一个重复文件 + 一个非图片文件。"""
    rng = np.random.default_rng(8)
    base = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    for i in range(3):
        arr = base.copy()
        arr[: 8 + i * 4, : 8 + i * 4] = (arr[: 8 + i * 4, : 8 + i * 4] // 2).astype(np.uint8)
        Image.fromarray(arr).save(root / f"v{i}.png")
    shutil.copyfile(root / "v0.png", root / "dup.png")
    (root / "notes.txt").write_text("hello\n" * 100, encoding="utf-8")


class EstimateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "src"
        self.src.mkdir()
        _make_set(self.src)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_and_duplicates(self) -> None:
        stats = estimate_archive(self.root / "out.pxcs", [str(self.src)],
                                 CreateOptions(quiet=True, progress=False, threads=2))
        self.assertEqual(stats.files, 5)
        self.assertEqual(stats.images, 4)
        self.assertEqual(stats.others, 1)
        self.assertGreaterEqual(stats.groups, 1)
        self.assertEqual(stats.duplicates, 1)
        self.assertGreater(stats.duplicate_bytes, 0)

    def test_estimated_range_is_sane(self) -> None:
        stats = estimate_archive(self.root / "out.pxcs", [str(self.src)],
                                 CreateOptions(quiet=True, progress=False, threads=2))
        self.assertGreater(stats.sampled_bytes, 0)
        self.assertLessEqual(stats.estimated_low, stats.estimated_bytes)
        self.assertLessEqual(stats.estimated_bytes, stats.estimated_high)

    def test_dry_run_writes_no_archive(self) -> None:
        out = self.root / "dry" / "out.pxcs"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(REPO / "pxcs.py"), "create",
             "-i", str(self.src), "-o", str(out), "--dry-run", "--threads", "1"],
            capture_output=True, text=True, cwd=str(REPO), timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY-RUN", result.stdout)
        self.assertFalse(out.exists())
        self.assertFalse(out.parent.exists())


if __name__ == "__main__":
    unittest.main()
