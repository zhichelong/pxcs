"""WebP 有损/无损识别回归。

无损 WebP 源必须逐像素恢复；有损 WebP 仍按原来的量化差分处理。
之前按扩展名一刀切，把所有 WebP 都当有损，量化会改动无损源的像素。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive, extract_archive, verify_archive
from pxcs_core.container import ArchiveReader
from pxcs_core.webpinfo import is_lossless_webp


def _pattern(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)


class WebpInfoTest(unittest.TestCase):
    def test_lossless_and_lossy_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = _pattern()
            lossless = root / "l.webp"
            lossy = root / "y.webp"
            Image.fromarray(arr, "RGB").save(
                lossless, format="WEBP", lossless=True, exact=True, method=2
            )
            Image.fromarray(arr, "RGB").save(lossy, format="WEBP", quality=85)
            self.assertTrue(is_lossless_webp(lossless))
            self.assertFalse(is_lossless_webp(lossy))
            self.assertIsNone(is_lossless_webp(b"not a webp"))
            self.assertIsNone(is_lossless_webp(lossless.read_bytes()[:4]))

    def test_alpha_webp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.webp"
            arr = np.zeros((16, 16, 4), dtype=np.uint8)
            arr[:, :, :3] = 128
            arr[:, :, 3] = 255
            Image.fromarray(arr, "RGBA").save(
                path, format="WEBP", lossless=True, exact=True, method=1
            )
            self.assertTrue(is_lossless_webp(path))


class WebpArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_pair(self, lossless: bool) -> Path:
        src = self.root / ("lossless" if lossless else "lossy")
        src.mkdir()
        base = _pattern()
        other = base.copy()
        other[0:6, 0:6] = (other[0:6, 0:6] // 3).astype(np.uint8)
        for name, arr in (("a.webp", base), ("b.webp", other)):
            kwargs = (
                {"lossless": True, "exact": True, "method": 2}
                if lossless
                else {"quality": 88}
            )
            Image.fromarray(arr, "RGB").save(src / name, format="WEBP", **kwargs)
        return src

    def test_lossless_webp_is_pixel_exact(self) -> None:
        src = self._make_pair(lossless=True)
        out = self.root / "lossless.pxcs"
        create_archive(
            out, [str(src)], CreateOptions(quiet=True, progress=False, threads=2)
        )
        self.assertEqual(verify_archive(out).failed, 0)
        with ArchiveReader(out) as reader:
            self.assertFalse(
                any(e.lossy for e in reader.entries),
                "无损 WebP 不应产生有损残差",
            )
        dest = self.root / "out"
        extract_archive(out, dest, lossless=True, quiet=True, threads=1)
        for name in ("a", "b"):
            got = np.asarray(Image.open(dest / f"{name}.png").convert("RGB"))
            want = np.asarray(Image.open(src / f"{name}.webp").convert("RGB"))
            self.assertTrue(np.array_equal(got, want), f"{name} 像素不一致")

    def test_lossy_webp_still_quantized(self) -> None:
        src = self._make_pair(lossless=False)
        out = self.root / "lossy.pxcs"
        create_archive(
            out, [str(src)], CreateOptions(quiet=True, progress=False, threads=2)
        )
        with ArchiveReader(out) as reader:
            lossy = [e for e in reader.entries if e.lossy]
        self.assertTrue(lossy, "有损 WebP 仍应按量化差分处理")


if __name__ == "__main__":
    unittest.main()
