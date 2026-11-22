"""跨组 key 引用（RGB -> 常数 alpha RGBA）的回归。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import (
    CreateOptions,
    PxcsFormatError,
    create_archive,
    extract_archive,
    verify_archive,
)
from pxcs_core.container import ArchiveReader


def _base(seed: int = 5, size=(96, 72)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, "RGB").save(path)


def _save_rgba(path: Path, arr: np.ndarray, alpha: int | None = 255) -> None:
    if alpha is None:
        a = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        a[:16, :16] = 0
    else:
        a = np.full((arr.shape[0], arr.shape[1], 1), alpha, dtype=np.uint8)
    Image.fromarray(np.concatenate([arr, a], axis=2), "RGBA").save(path)


class CrossKeyRefTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create(self, src: Path, **kw) -> Path:
        out = self.root / f"{kw.get('key_ref_scope', 'default')}.pxcs"
        create_archive(
            out,
            [str(src)],
            CreateOptions(
                quiet=True, progress=False, threads=2, overwrite=True, **kw
            ),
        )
        return out

    def test_opaque_rgba_references_rgb_key(self) -> None:
        src = self.root / "one"
        src.mkdir()
        arr = _base()
        _save_rgb(src / "a.png", arr)
        _save_rgba(src / "b.png", arr)
        out = self._create(src, key_ref_scope="folder")
        with ArchiveReader(out) as reader:
            rgb = next(e for e in reader.entries if e.path.endswith("a.png"))
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertIn(rgba.codec, {"pxc_residual", "zlib_residual"})
            self.assertEqual(rgba.ref, rgb.id)
            self.assertEqual(rgba.depth, 1)
            self.assertEqual(rgba.mode, "RGBA")
            self.assertEqual(rgba.constant_alpha, 255)
            self.assertEqual(reader.header.version_minor, 5)
        self.assertEqual(verify_archive(out).failed, 0)
        dest = self.root / "out"
        extract_archive(out, dest, lossless=True, quiet=True, threads=1)
        got = np.asarray(Image.open(dest / "b.png"))
        want = np.asarray(Image.open(src / "b.png"))
        self.assertTrue(np.array_equal(got, want))

    def test_scope_off_keeps_separate_key(self) -> None:
        src = self.root / "two"
        src.mkdir()
        arr = _base(6)
        _save_rgb(src / "a.png", arr)
        _save_rgba(src / "b.png", arr)
        out = self._create(src, key_ref_scope="off")
        with ArchiveReader(out) as reader:
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertNotIn(rgba.codec, {"pxc_residual", "zlib_residual"})
            self.assertEqual(rgba.depth, 0)
            self.assertLessEqual(reader.header.version_minor, 4)

    def test_scope_folder_does_not_cross_folders(self) -> None:
        parent = self.root / "parent"
        d1 = parent / "d1"
        d2 = parent / "d2"
        d1.mkdir(parents=True)
        d2.mkdir()
        arr = _base(7)
        _save_rgb(d1 / "a.png", arr)
        _save_rgba(d2 / "b.png", arr)
        out = self._create(parent, key_ref_scope="folder")
        with ArchiveReader(out) as reader:
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertNotIn(rgba.codec, {"pxc_residual", "zlib_residual"})

    def test_scope_resolution_crosses_folders(self) -> None:
        parent = self.root / "parent2"
        d1 = parent / "d1"
        d2 = parent / "d2"
        d1.mkdir(parents=True)
        d2.mkdir()
        arr = _base(8)
        _save_rgb(d1 / "a.png", arr)
        _save_rgba(d2 / "b.png", arr)
        out = self._create(parent, key_ref_scope="resolution")
        with ArchiveReader(out) as reader:
            rgb = next(e for e in reader.entries if e.path.endswith("a.png"))
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertIn(rgba.codec, {"pxc_residual", "zlib_residual"})
            self.assertEqual(rgba.ref, rgb.id)
        self.assertEqual(verify_archive(out).failed, 0)

    def test_transparent_rgba_is_not_cross_referenced(self) -> None:
        src = self.root / "three"
        src.mkdir()
        arr = _base(9)
        _save_rgb(src / "a.png", arr)
        _save_rgba(src / "b.png", arr, alpha=None)
        out = self._create(src, key_ref_scope="folder")
        with ArchiveReader(out) as reader:
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertNotIn(rgba.codec, {"pxc_residual", "zlib_residual"})

    def test_default_scope_is_off(self) -> None:
        src = self.root / "default_off"
        src.mkdir()
        arr = _base(11)
        _save_rgb(src / "a.png", arr)
        _save_rgba(src / "b.png", arr)
        out = self.root / "default.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2),
        )
        with ArchiveReader(out) as reader:
            rgba = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertNotIn(rgba.codec, {"pxc_residual", "zlib_residual"})
            self.assertLessEqual(reader.header.version_minor, 5)

    def test_invalid_scope_rejected(self) -> None:

        src = self.root / "four"
        src.mkdir()
        _save_rgb(src / "a.png", _base(10))
        with self.assertRaises(PxcsFormatError):
            create_archive(
                self.root / "bad.pxcs", [str(src)],
                CreateOptions(quiet=True, progress=False, key_ref_scope="everywhere"),
            )


if __name__ == "__main__":
    unittest.main()
