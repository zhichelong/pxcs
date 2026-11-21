"""量化档位（--quant-list）与实际误差统计的回归。

- 显式档位只对有损 8-bit 源生效，无损源必须保持 q=0；
- 有损条目要报告 max|diff| 与 PSNR，且数值来自原始像素 vs 归档近似像素。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.container import ArchiveReader


def _gradient(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    base = np.stack(
        [(x * 4 + seed) % 256, (y * 4) % 256, ((x + y) * 2) % 256], axis=-1
    ).astype(np.uint8)
    noise = rng.integers(0, 6, base.shape, dtype=np.uint8)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class QuantListTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _jpeg_pair(self) -> Path:
        src = self.root / "jpg"
        src.mkdir()
        base = _gradient(3)
        other = base.copy()
        other[0:16, 0:16] = (other[0:16, 0:16] // 2).astype(np.uint8)
        Image.fromarray(base, "RGB").save(src / "a.jpg", quality=92)
        Image.fromarray(other, "RGB").save(src / "b.jpg", quality=92)
        return src

    def _png_pair(self) -> Path:
        src = self.root / "png"
        src.mkdir()
        a = _gradient(4)
        b = a.copy()
        b[0:16, 0:16] = (b[0:16, 0:16] // 2).astype(np.uint8)
        Image.fromarray(a, "RGB").save(src / "a.png")
        Image.fromarray(b, "RGB").save(src / "b.png")
        return src

    def test_quant_list_restricts_choices(self) -> None:
        src = self._jpeg_pair()
        out = self.root / "q.pxcs"
        stats = create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          quant_list=(0, 24)),
        )
        self.assertGreaterEqual(stats.lossy_deltas, 1)
        with ArchiveReader(out) as reader:
            quants = {e.quant for e in reader.entries if e.lossy}
        self.assertTrue(quants <= {0, 24}, quants)

    def test_lossless_source_never_quantized(self) -> None:
        src = self._png_pair()
        out = self.root / "png.pxcs"
        stats = create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          quant_list=(0, 8, 24)),
        )
        self.assertEqual(stats.lossy_deltas, 0)
        with ArchiveReader(out) as reader:
            self.assertFalse(any(e.quant for e in reader.entries))

    def test_lossy_stats_are_reported(self) -> None:
        src = self._jpeg_pair()
        out = self.root / "s.pxcs"
        stats = create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          quant_list=(12,)),
        )
        self.assertGreaterEqual(stats.lossy_deltas, 1)
        self.assertGreater(stats.max_pixel_error, 0)
        self.assertGreater(stats.psnr_sum, 0)

    def test_quant_list_rejects_out_of_range(self) -> None:
        from pxcs_core.archive import PxcsFormatError

        src = self._jpeg_pair()
        with self.assertRaises(PxcsFormatError):
            create_archive(
                self.root / "bad.pxcs", [str(src)],
                CreateOptions(quiet=True, progress=False, quant_list=(0, 300)),
            )

    def test_cli_parser(self) -> None:
        from pxcs import _parse_quant_list

        self.assertEqual(_parse_quant_list("0,4,8"), (0, 4, 8))
        self.assertEqual(_parse_quant_list("0,0,4"), (0, 4))
        with self.assertRaises(Exception):
            _parse_quant_list("a,b")
        with self.assertRaises(Exception):
            _parse_quant_list("0,1,2,3,4,5,6,7,8,9")


if __name__ == "__main__":
    unittest.main()
