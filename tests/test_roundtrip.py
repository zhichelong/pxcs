"""PXC 0.1 自动化测试：像素无损、批量、目录、冲突、损坏、ORIG。"""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pxc_core import container
from pxc_core.batch import run_decode, run_encode
from pxc_core.config import CHUNK_DATA
from pxc_core.errors import PxcError
from pxc_core.image_io import data_to_pillow, decode_transparency, load_image

from tests.make_samples import create_all


@contextlib.contextmanager
def silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class PxcRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_root = ROOT / "tests" / "_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        cls._tmp = tempfile.TemporaryDirectory(prefix="pxc_tests_", dir=test_root)
        cls.tmp_root = Path(cls._tmp.name)
        cls.samples = create_all(cls.tmp_root / "samples", include_large=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def new_dir(self, name: str) -> Path:
        p = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=self.tmp_root))
        return p

    def encode(self, inputs, out: Path, **kwargs):
        with silent():
            return run_encode([str(x) for x in inputs], out, **kwargs)

    def decode(self, inputs, out: Path, **kwargs):
        with silent():
            return run_decode([str(x) for x in inputs], out, **kwargs)

    def one_pxc(self, out: Path) -> Path:
        files = sorted(out.rglob("*.pxc"))
        self.assertEqual(len(files), 1, f"期望 1 个 PXC，实际 {files}")
        return files[0]

    def one_image(self, out: Path) -> Path:
        exts = {".png", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        self.assertEqual(len(files), 1, f"期望 1 个恢复图片，实际 {files}")
        return files[0]

    def assert_image_data_equal(self, expected, actual) -> None:
        # Pillow 的 TIFF/PNG 输出可能把 I;16B 规范化为 I;16（字节序标签变化，
        # 像素值仍一致）。这不影响位深和像素无损，单独放行并比较数据。
        sixteen = {"I;16", "I;16L", "I;16B"}
        mode_equivalent = (
            expected.mode == actual.mode
            or (expected.mode in sixteen and actual.mode in sixteen)
        )
        self.assertTrue(
            mode_equivalent,
            f"mode 不一致: {expected.mode!r} != {actual.mode!r}",
        )
        self.assertEqual(expected.width, actual.width)
        self.assertEqual(expected.height, actual.height)
        np.testing.assert_array_equal(expected.data, actual.data)
        if expected.mode == "P":
            self.assertEqual(expected.palette, actual.palette)
            ea = np.array(data_to_pillow(expected).convert("RGBA"))
            aa = np.array(data_to_pillow(actual).convert("RGBA"))
            np.testing.assert_array_equal(ea, aa)

    def roundtrip(self, key: str, tile_size: int = 64):
        src = self.samples[key]
        td = self.new_dir(key)
        enc = td / "encoded"
        dec = td / "decoded"
        stats = self.encode([src], enc, tile_size=tile_size)
        self.assertEqual(stats.failed, 0, stats.errors)
        self.assertEqual(stats.success, 1)
        pxc = self.one_pxc(enc)
        stats2 = self.decode([pxc], dec)
        self.assertEqual(stats2.failed, 0, stats2.errors)
        self.assertEqual(stats2.success, 1)
        restored = self.one_image(dec)
        return src, pxc, restored, stats, stats2

    # ------------------------------------------------------------------
    # pixel-exact roundtrip
    # ------------------------------------------------------------------
    def test_single_modes(self) -> None:
        keys = [
            "solid_rgb",
            "gradient_h",
            "gradient_v",
            "palette",
            "rgba",
            "rgba_invisible",
            "gray",
            "gray_alpha",
            "mono",
            "noise",
            "gray16",
            "gray16be",
            "cmyk",
            "int32",
            "webp",
            "bmp",
            "tiff",
            "meta",
            "photo_jpeg",
        ]
        for key in keys:
            with self.subTest(key=key):
                src, pxc, restored, _stats, _stats2 = self.roundtrip(key)
                expected = load_image(src, compute_file_hash=False)
                actual = load_image(restored, compute_file_hash=False)
                self.assert_image_data_equal(expected, actual)

    def test_large_tiles(self) -> None:
        src, pxc, restored, _stats, _stats2 = self.roundtrip("large", tile_size=256)
        self.assertEqual(len(list(Path(pxc).parent.rglob("*.pxc"))), 1)
        expected = load_image(src, compute_file_hash=False)
        actual = load_image(restored, compute_file_hash=False)
        self.assert_image_data_equal(expected, actual)

    def test_metadata_preserved(self) -> None:
        src, pxc, restored, _stats, _stats2 = self.roundtrip("meta")
        a = load_image(src, compute_file_hash=False)
        b = load_image(restored, compute_file_hash=False)
        self.assertEqual(a.icc_profile, b.icc_profile)
        self.assertTrue(b.exif)
        exif = Image.Exif()
        exif.load(b.exif)
        self.assertEqual(exif.get(0x010E), "PXC test image")
        self.assertTrue(b.xmp and b"PXC test" in b.xmp)
        self.assertIsNotNone(b.dpi)
        assert a.dpi is not None and b.dpi is not None
        self.assertAlmostEqual(a.dpi[0], b.dpi[0], places=1)
        self.assertAlmostEqual(a.dpi[1], b.dpi[1], places=1)

    def test_palette_transparency(self) -> None:
        src, pxc, restored, _stats, _stats2 = self.roundtrip("palette")
        a = load_image(src, compute_file_hash=False)
        b = load_image(restored, compute_file_hash=False)
        self.assertEqual(a.mode, "P")
        self.assertEqual(a.data.tolist(), b.data.tolist())
        self.assertEqual(a.palette, b.palette)
        npa = np.array(data_to_pillow(a).convert("RGBA"))
        npb = np.array(data_to_pillow(b).convert("RGBA"))
        np.testing.assert_array_equal(npa, npb)

    # ------------------------------------------------------------------
    # batch / directory / naming
    # ------------------------------------------------------------------
    def test_constant_alpha_plane_is_dropped(self) -> None:
        src = self.samples["rgba_invisible"]
        td = self.new_dir("const_alpha")
        stats = self.encode([src], td / "enc", tile_size=32)
        self.assertEqual(stats.success, 1, stats.errors)
        pxc = self.one_pxc(td / "enc")
        pf = container.read_pxc(pxc)
        self.assertEqual(pf.header.alpha_kind, 3)
        self.assertEqual(pf.header.channels, 3)
        dec = self.decode([pxc], td / "dec")
        self.assertEqual(dec.success, 1, dec.errors)
        a = load_image(src, compute_file_hash=False)
        b = load_image(self.one_image(td / "dec"), compute_file_hash=False)
        self.assert_image_data_equal(a, b)

    def test_batch_and_stats(self) -> None:
        td = self.new_dir("batch")
        inputs = [self.samples["solid_rgb"], self.samples["gradient_h"], self.samples["gray"]]
        stats = self.encode(inputs, td / "enc", tile_size=32)
        self.assertEqual(stats.failed, 0, stats.errors)
        self.assertEqual(stats.success, 3)
        self.assertEqual(len(list((td / "enc").glob("*.pxc"))), 3)
        self.assertGreater(stats.original_bytes, 0)
        self.assertGreater(stats.pxc_bytes, 0)
        stats2 = self.decode([td / "enc"], td / "dec")
        self.assertEqual(stats2.failed, 0, stats2.errors)
        self.assertEqual(stats2.success, 3)

    def test_recursive_and_structure(self) -> None:
        nested = self.samples["nested_dir"]
        td = self.new_dir("nested")
        flat = self.encode([nested], td / "flat", recursive=False)
        self.assertEqual(flat.success, 1)
        self.assertEqual(flat.failed, 0)
        deep = self.encode([nested], td / "deep", recursive=True)
        self.assertEqual(deep.success, 2)
        self.assertTrue((td / "deep" / "nested" / "a.pxc").exists())
        self.assertTrue((td / "deep" / "nested" / "sub" / "b.pxc").exists())
        dec = self.decode([td / "deep"], td / "restored", recursive=True)
        self.assertEqual(dec.success, 2)
        self.assertTrue((td / "restored" / "deep" / "nested" / "a.png").exists())
        self.assertTrue((td / "restored" / "deep" / "nested" / "sub" / "b.png").exists())

    def test_same_filename_conflict(self) -> None:
        td = self.new_dir("conflict")
        stats = self.encode([self.samples["dir1"], self.samples["dir2"]], td / "enc")
        self.assertEqual(stats.success, 2, stats.errors)
        p1 = td / "enc" / "dir1" / "same.pxc"
        p2 = td / "enc" / "dir2" / "same.pxc"
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        dec = self.decode([p1, p2], td / "dec")
        self.assertEqual(dec.success, 2, dec.errors)
        files = sorted(p.name for p in (td / "dec").glob("*.png"))
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].startswith("same"))
        self.assertTrue(files[1].startswith("same"))
        self.assertNotEqual(files[0], files[1])

    def test_duplicate_input(self) -> None:
        td = self.new_dir("dup")
        src = self.samples["solid_rgb"]
        stats = self.encode([src, src], td / "enc")
        self.assertEqual(stats.success, 1)
        self.assertGreaterEqual(stats.skipped, 1)
        self.assertEqual(len(list((td / "enc").glob("*.pxc"))), 1)

    def test_non_image_ignored(self) -> None:
        td = self.new_dir("ignore")
        inp = td / "in"
        inp.mkdir()
        shutil.copy2(self.samples["solid_rgb"], inp / "a.png")
        (inp / "notes.txt").write_text("hello", encoding="utf-8")
        stats = self.encode([inp], td / "enc")
        self.assertEqual(stats.success, 1)
        self.assertGreaterEqual(stats.skipped, 1)
        self.assertFalse((td / "enc" / "notes.pxc").exists())

    def test_overwrite(self) -> None:
        td = self.new_dir("overwrite")
        src = self.samples["solid_rgb"]
        out1 = td / "no_overwrite"
        self.encode([src], out1, overwrite=False)
        self.encode([src], out1, overwrite=False)
        self.assertTrue((out1 / "solid_rgb.pxc").exists())
        self.assertTrue((out1 / "solid_rgb__2.pxc").exists())

        out2 = td / "with_overwrite"
        self.encode([src], out2, overwrite=False)
        first = (out2 / "solid_rgb.pxc").stat().st_size
        self.encode([src], out2, overwrite=True)
        self.assertTrue((out2 / "solid_rgb.pxc").exists())
        self.assertFalse((out2 / "solid_rgb__2.pxc").exists())
        self.assertGreater(first, 0)

    # ------------------------------------------------------------------
    # ORIG / JPEG
    # ------------------------------------------------------------------
    def test_orig_bytes_restore(self) -> None:
        src = self.samples["photo_jpeg"]
        td = self.new_dir("orig")
        stats = self.encode([src], td / "enc", keep_original=True)
        self.assertEqual(stats.success, 1, stats.errors)
        pxc = self.one_pxc(td / "enc")
        pf = container.read_pxc(pxc)
        self.assertTrue(pf.orig_payload is not None or pf.header.compression_id == 2)
        stats2 = self.decode([pxc], td / "orig_out", restore_original_file=True)
        self.assertEqual(stats2.success, 1, stats2.errors)
        restored = td / "orig_out" / src.name
        self.assertTrue(restored.exists())
        self.assertEqual(restored.read_bytes(), src.read_bytes())

    def test_restore_original_without_orig_fails(self) -> None:
        src = self.samples["solid_rgb"]
        td = self.new_dir("noorig")
        stats = self.encode([src], td / "enc", keep_original=False)
        self.assertEqual(stats.success, 1)
        pxc = self.one_pxc(td / "enc")
        stats2 = self.decode([pxc], td / "out", restore_original_file=True)
        self.assertEqual(stats2.failed, 1)
        self.assertEqual(stats2.success, 0)
        self.assertTrue(any("原始文件" in e for e in stats2.errors))

    def test_jpeg_defaults_to_png(self) -> None:
        src = self.samples["photo_jpeg"]
        td = self.new_dir("jpeg")
        stats = self.encode([src], td / "enc")
        self.assertEqual(stats.success, 1)
        pxc = self.one_pxc(td / "enc")
        dec = self.decode([pxc], td / "dec")
        self.assertEqual(dec.success, 1, dec.errors)
        restored = self.one_image(td / "dec")
        self.assertEqual(restored.suffix.lower(), ".png")
        a = load_image(src, compute_file_hash=False)
        b = load_image(restored, compute_file_hash=False)
        self.assert_image_data_equal(a, b)

    # ------------------------------------------------------------------
    # errors / corruption
    # ------------------------------------------------------------------
    def test_corruption_detected(self) -> None:
        td = self.new_dir("corrupt")
        stats = self.encode([self.samples["solid_rgb"]], td / "enc")
        self.assertEqual(stats.success, 1)
        pxc = self.one_pxc(td / "enc")
        raw = bytearray(pxc.read_bytes())
        pf = container.read_pxc(pxc)
        data_offset = pf.chunks[CHUNK_DATA].offset
        raw[data_offset + 32] ^= 0xFF
        bad = td / "bad.pxc"
        bad.write_bytes(bytes(raw))
        with self.assertRaises(PxcError):
            container.read_pxc(bad)
        stats2 = self.decode([bad], td / "out")
        self.assertEqual(stats2.failed, 1)

        truncated = td / "trunc.pxc"
        truncated.write_bytes(bytes(raw[: len(raw) - 8]))
        with self.assertRaises(PxcError):
            container.read_pxc(truncated)

    def test_animation_rejected(self) -> None:
        td = self.new_dir("anim")
        stats = self.encode([self.samples["animated_gif"]], td / "enc")
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.success, 0)
        self.assertTrue(any(("动画" in e) or ("多帧" in e) for e in stats.errors))

    def test_multipage_tiff_rejected(self) -> None:
        td = self.new_dir("multi")
        stats = self.encode([self.samples["multipage_tif"]], td / "enc")
        self.assertEqual(stats.failed, 1)
        self.assertTrue(any(("动画" in e) or ("多帧" in e) for e in stats.errors))

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    def test_cli_encode_decode(self) -> None:
        td = self.new_dir("cli")
        src = self.samples["gradient_h"]
        env = dict(**__import__("os").environ)
        env["PYTHONIOENCODING"] = "utf-8"
        enc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "pxc.py"),
                "encode",
                "-i",
                str(src),
                "-o",
                str(td / "enc"),
                "--tile-size",
                "32",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self.assertEqual(enc.returncode, 0, enc.stdout + enc.stderr)
        pxc = self.one_pxc(td / "enc")
        dec = subprocess.run(
            [
                sys.executable,
                str(ROOT / "pxc.py"),
                "decode",
                "-i",
                str(pxc),
                "-o",
                str(td / "dec"),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self.assertEqual(dec.returncode, 0, dec.stdout + dec.stderr)
        restored = self.one_image(td / "dec")
        a = load_image(src, compute_file_hash=False)
        b = load_image(restored, compute_file_hash=False)
        self.assert_image_data_equal(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
