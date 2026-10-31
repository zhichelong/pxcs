"""PXCS 归档测试：混合文件、跨图差分、恢复质量、模式与安全。"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pxcs_core.archive import (
    _variant_key,
    CreateOptions,
    archive_info,
    create_archive,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxc_core.codec import TileStats, encode_tiles
from pxc_core.image_io import ImageData
from pxcs_core.container import ArchiveReader, PxcsCorruptionError
from pxcs_core.format import (
    CODEC_DUP, CODEC_JPEGXL, CODEC_PXC, CODEC_PXC_RESIDUAL, CODEC_RAW,
    CODEC_WEBP_LOSSLESS, CODEC_ZLIB_RESIDUAL,
)
from pxcs_core.paths import PxcsPathError, normalize_rel_path, safe_join
from pxcs import _normal_archive_name


def _gradient(w: int = 256, h: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = xx * 255 // max(1, w - 1)
    arr[..., 1] = yy * 255 // max(1, h - 1)
    arr[..., 2] = (xx + yy) * 255 // max(1, w + h - 2)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 90, 90], fill=(240, 30, 60))
    d.ellipse([120, 100, 220, 220], fill=(20, 200, 90))
    return img


class PxcsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-test-"))
        self.src = self.tmp / "src"
        self.src.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_rgb_set(self, count: int = 3) -> None:
        base = _gradient()
        base.save(self.src / "001.png")
        for i in range(1, count):
            img = base.copy()
            d = ImageDraw.Draw(img)
            d.rectangle(
                [10 + i * 8, 10 + i * 4, 50 + i * 8, 50 + i * 4],
                fill=((i * 37) % 256, (i * 71) % 256, (i * 13) % 256),
            )
            img.save(self.src / f"{i + 1:03d}.png")

    def test_mixed_roundtrip_and_delta(self) -> None:
        self._write_rgb_set(3)
        _gradient(128, 128).save(self.src / "004.jpg", quality=90)
        (self.src / "note.txt").write_text("hello pxcs\n" * 50, encoding="utf-8")
        (self.src / "note_copy.txt").write_text("hello pxcs\n" * 50, encoding="utf-8")
        (self.src / "sub").mkdir()
        (self.src / "sub" / "data.bin").write_bytes(bytes(range(256)) * 8)

        archive = self.tmp / "out.pxcs"
        stats = create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True)
        )
        self.assertEqual(stats.images, 4)
        self.assertEqual(stats.files, 3)
        self.assertGreaterEqual(stats.deltas, 1)
        self.assertGreaterEqual(stats.dups, 1)

        with ArchiveReader(archive) as reader:
            codecs = {e.codec for e in reader.entries}
            self.assertIn(CODEC_PXC_RESIDUAL, codecs)
            self.assertIn(CODEC_RAW, codecs)
            self.assertIn(CODEC_DUP, codecs)
            self.assertTrue({CODEC_PXC, CODEC_RAW} & codecs)
            for e in reader.entries:
                self.assertGreaterEqual(e.depth, 0)

        text = list_entries(archive, as_json=True)
        self.assertIn("PXCS", text)
        self.assertIn("001.png", text)
        info = archive_info(archive)
        # info 输出已本地化（原来写死英文标签），这里断言中文标签与数字。
        self.assertIn("图片 4", info)
        self.assertIn("文件 3", info)
        self.assertNotIn("entries:", info)

        verify = verify_archive(archive)
        self.assertEqual(verify.failed, 0)
        self.assertEqual(verify.checked, 7)

        out = self.tmp / "out"
        extract = extract_archive(archive, out, quiet=True)
        self.assertEqual(extract.failed, 0)
        self.assertEqual(extract.extracted, 7)

        base = _gradient()
        for name in ("001.png", "002.png", "003.png"):
            with Image.open(self.src / name) as a, Image.open(out / name) as b:
                a.load()
                b.load()
                self.assertEqual(a.mode, b.mode)
                self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        with Image.open(self.src / "004.jpg") as a, Image.open(out / "004.jpg") as b:
            a.load()
            b.load()
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        self.assertEqual((self.src / "note.txt").read_bytes(), (out / "note.txt").read_bytes())
        self.assertEqual(
            (self.src / "note_copy.txt").read_bytes(),
            (out / "note_copy.txt").read_bytes(),
        )
        self.assertEqual(
            (self.src / "sub" / "data.bin").read_bytes(),
            (out / "sub" / "data.bin").read_bytes(),
        )

    def test_raw_jpeg_quality_restore(self) -> None:
        img = _gradient(200, 160)
        jpg = self.src / "photo.jpg"
        img.save(jpg, quality=92)
        archive = self.tmp / "photo.pxcs"
        create_archive(archive, [str(jpg)], CreateOptions(quiet=True))

        exact_dir = self.tmp / "exact"
        extract_archive(archive, exact_dir, quiet=True)
        self.assertEqual(jpg.read_bytes(), (exact_dir / "photo.jpg").read_bytes())

        q_dir = self.tmp / "q50"
        extract_archive(archive, q_dir, quality=50, quiet=True)
        q_file = q_dir / "photo.jpg"
        self.assertTrue(q_file.exists())
        self.assertLess(q_file.stat().st_size, jpg.stat().st_size)
        with Image.open(q_file) as im:
            im.load()
            self.assertEqual(im.size, (200, 160))

    def test_non_adjacent_group_search(self) -> None:
        w = h = 256
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.zeros((h, w, 3), np.uint8)
        a[..., 0] = xx * 255 // (w - 1)
        a[..., 1] = yy * 255 // (h - 1)
        a[..., 2] = 60
        b = np.full((h, w, 3), 40, np.uint8)
        b[..., 2] = 220
        b[::7, ::7] = 250
        for name, arr in [("01_a1.png", a), ("02_b1.png", b)]:
            Image.fromarray(arr, "RGB").save(self.src / name)
        a2 = a.copy()
        a2[20:120, 20:120] = (250, 20, 20)
        b2 = b.copy()
        b2[80:200, 80:220] = (250, 200, 20)
        Image.fromarray(a2, "RGB").save(self.src / "03_a2.png")
        Image.fromarray(b2, "RGB").save(self.src / "04_b2.png")

        archive = self.tmp / "search.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(quiet=True, search_window=8, group_mode="natural"),
        )
        with ArchiveReader(archive) as reader:
            refs = {e.path: e.ref for e in reader.entries}
        self.assertEqual(refs["03_a2.png"], 0)
        self.assertEqual(refs["04_b2.png"], 1)
        self.assertEqual(verify_archive(archive).failed, 0)

    def test_lossless_grouped_before_lossy(self) -> None:
        img = _gradient(256, 256)
        img.save(self.src / "a.jpg", quality=90)
        img.save(self.src / "b.png")
        archive = self.tmp / "order.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        with ArchiveReader(archive) as reader:
            self.assertEqual(reader.entries[0].path, "b.png")
            self.assertEqual(reader.entries[1].path, "a.jpg")
        self.assertEqual(verify_archive(archive).failed, 0)

    def test_same_size_jpeg_delta(self) -> None:
        base = _gradient(1024, 1024)
        base.save(self.src / "p0.jpg", quality=95, subsampling=0)
        variant = base.copy()
        d = ImageDraw.Draw(variant)
        d.rectangle([120, 160, 480, 520], fill=(20, 220, 90))
        variant.save(self.src / "p1.jpg", quality=95, subsampling=0)

        archive = self.tmp / "jpeg_delta.pxcs"
        stats = create_archive(
            archive, [str(self.src)],
            CreateOptions(quiet=True, delta_quant=0),
        )
        self.assertEqual(stats.raw_images, 1)
        self.assertEqual(stats.deltas, 1)
        with ArchiveReader(archive) as reader:
            self.assertIn(reader.entries[0].codec, {CODEC_RAW, CODEC_JPEGXL})
            self.assertEqual(reader.entries[1].codec, CODEC_PXC_RESIDUAL)
            self.assertEqual(reader.entries[1].ref, 0)
        self.assertEqual(verify_archive(archive).failed, 0)

        out = self.tmp / "jpeg_delta_out"
        extract_archive(archive, out, quiet=True)
        with Image.open(out / "p1.jpg") as im:
            im.load()
            self.assertEqual(im.size, (1024, 1024))

        # --lossless：残差帧落盘必须与源像素一致，而不是 q95 重编码。
        out_l = self.tmp / "jpeg_delta_lossless"
        extract_archive(archive, out_l, quiet=True, lossless=True)
        with Image.open(self.src / "p1.jpg") as a, Image.open(out_l / "p1.png") as b:
            a.load()
            b.load()
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))

    def test_webp_lossless_keyframe(self) -> None:
        self._write_rgb_set(2)
        archive = self.tmp / "webp_key.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(quiet=True, keyframe_codec="webp"),
        )
        with ArchiveReader(archive) as reader:
            self.assertIn(
                CODEC_WEBP_LOSSLESS, {e.codec for e in reader.entries}
            )
        self.assertEqual(verify_archive(archive).failed, 0)
        out = self.tmp / "webp_key_out"
        stats = extract_archive(archive, out, quiet=True, lossless=True)
        self.assertEqual(stats.failed, 0)
        for name in ("001.png", "002.png"):
            with Image.open(self.src / name) as a, Image.open(out / name) as b:
                a.load()
                b.load()
                self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))

    def test_bmp_tiff_gif_restore(self) -> None:
        img = _gradient(300, 200)
        img.save(self.src / "a.bmp")
        img.save(self.src / "a.tiff")
        img.quantize(colors=64).save(self.src / "a.gif")
        archive = self.tmp / "formats.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        self.assertEqual(verify_archive(archive).failed, 0)
        out = self.tmp / "formats-out"
        stats = extract_archive(archive, out, quiet=True)
        self.assertEqual(stats.failed, 0)
        for name in ("a.bmp", "a.tiff", "a.gif"):
            with Image.open(self.src / name) as a, Image.open(out / name) as b:
                a.load()
                b.load()
                self.assertEqual(a.mode, b.mode, name)
                self.assertTrue(
                    np.array_equal(np.asarray(a), np.asarray(b)), name
                )

    def test_duplicate_image_and_empty_file(self) -> None:
        img = _gradient(160, 120)
        img.save(self.src / "a.jpg", quality=90)
        shutil.copyfile(self.src / "a.jpg", self.src / "b.jpg")
        (self.src / "empty.bin").write_bytes(b"")
        archive = self.tmp / "dup.pxcs"
        stats = create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        self.assertEqual(stats.dups, 1)
        self.assertEqual(stats.images, 2)
        self.assertEqual(stats.files, 1)
        self.assertEqual(verify_archive(archive).failed, 0)
        out = self.tmp / "dup-out"
        extract_archive(archive, out, quiet=True)
        self.assertEqual((out / "a.jpg").read_bytes(), (out / "b.jpg").read_bytes())
        self.assertEqual((out / "empty.bin").read_bytes(), b"")

    def test_modes_roundtrip(self) -> None:
        w = h = 128
        yy, xx = np.mgrid[0:h, 0:w]
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[..., 0] = xx * 2
        rgb[..., 1] = yy * 2
        rgb[..., 2] = xx + yy

        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
        for i in range(2):
            arr = rgba.copy()
            arr[10 + i:30 + i, 20:50, :3] = 200
            Image.fromarray(arr, "RGBA").save(self.src / f"rgba_{i}.png")

        la = np.dstack([rgb[..., 0], np.full((h, w), 128, np.uint8)])
        for i in range(2):
            arr = la.copy()
            arr[10 + i:30 + i, 20:50, 0] = 255
            Image.fromarray(arr, "LA").save(self.src / f"la_{i}.png")

        l_arr = rgb[..., 0]
        for i in range(2):
            arr = l_arr.copy()
            arr[10 + i:30 + i, 20:50] = 200
            Image.fromarray(arr, "L").save(self.src / f"l_{i}.png")

        p_img = Image.fromarray(rgb, "RGB").quantize(colors=64)
        for i in range(2):
            arr = np.asarray(p_img).copy()
            arr[10 + i:30 + i, 20:50] = (i * 7) % 64
            Image.fromarray(arr, "P").save(self.src / f"p_{i}.png")

        i16 = (xx.astype(np.uint16) * 257 + yy.astype(np.uint16) * 3)
        for i in range(2):
            arr = i16.copy()
            arr[10 + i:30 + i, 20:50] = 40000
            Image.fromarray(arr).save(self.src / f"i16_{i}.png")

        archive = self.tmp / "modes.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        self.assertEqual(verify_archive(archive).failed, 0)
        out = self.tmp / "modes-out"
        stats = extract_archive(archive, out, quiet=True)
        self.assertEqual(stats.failed, 0)
        for src_file in sorted(self.src.glob("*.png")):
            with Image.open(src_file) as a, Image.open(out / src_file.name) as b:
                a.load()
                b.load()
                self.assertEqual(a.mode, b.mode, src_file.name)
                self.assertTrue(
                    np.array_equal(np.asarray(a), np.asarray(b)), src_file.name
                )

    def test_tiled_encoding_threads_match(self) -> None:
        img = _gradient(256, 256)
        data = np.asarray(img, dtype=np.uint8)
        image = ImageData(mode="RGB", width=256, height=256, data=data)
        serial = encode_tiles(image, 64, threads=1)
        stats = TileStats()
        parallel = encode_tiles(image, 64, threads=4, stats=stats)
        self.assertEqual(serial, parallel)
        self.assertGreater(stats.tiles, 0)
        self.assertEqual(sum(stats.mode_counts.values()), stats.tiles)

    def test_variant_key(self) -> None:
        self.assertEqual(_variant_key("111919913_p1.png"), "111919913")
        self.assertEqual(_variant_key("111919913-p2.jpg"), "111919913")
        self.assertEqual(_variant_key("random_p1_v2.png"), "")
        self.assertEqual(_variant_key("no_suffix.png"), "")

    def test_path_safety(self) -> None:
        for bad in ("../evil", "a/../../b", "C:/evil", "/abs", "a\\..\\b", "CON.txt"):
            with self.assertRaises(PxcsPathError, msg=bad):
                normalize_rel_path(bad)
        with self.assertRaises(PxcsPathError):
            safe_join(self.tmp, "../evil")
        self.assertEqual(normalize_rel_path("a\\b\\c.png"), "a/b/c.png")

    def test_corruption_detection(self) -> None:
        self._write_rgb_set(2)
        archive = self.tmp / "corrupt.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        with ArchiveReader(archive) as reader:
            entry = reader.entries[0]
            payload_offset = entry.offset
        raw = bytearray(archive.read_bytes())
        raw[payload_offset + 3] ^= 0xFF
        archive.write_bytes(bytes(raw))
        stats = verify_archive(archive)
        self.assertGreater(stats.failed, 0)

    def test_keyframe_interval_and_none(self) -> None:
        self._write_rgb_set(6)
        archive = self.tmp / "kf.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(
                quiet=True, keyframe_interval=2, reference="chain",
                group_mode="natural",
            ),
        )
        with ArchiveReader(archive) as reader:
            self.assertTrue(any(e.codec == CODEC_PXC_RESIDUAL for e in reader.entries))
            for e in reader.entries:
                self.assertLessEqual(e.depth, 2)

        no_delta = self.tmp / "none.pxcs"
        create_archive(
            no_delta,
            [str(self.src)],
            CreateOptions(quiet=True, reference="none", group_mode="natural"),
        )
        with ArchiveReader(no_delta) as reader:
            self.assertFalse(any(e.codec == CODEC_PXC_RESIDUAL for e in reader.entries))

    def test_normal_archive_name(self) -> None:
        folder = Path(r"C:\data\summer\main\2021")
        self.assertEqual(
            _normal_archive_name(folder, 3), "summer_main_2021.pxcs"
        )
        self.assertEqual(
            _normal_archive_name(folder, 2), "main_2021.pxcs"
        )

    def test_filename_hash_helpers(self) -> None:
        # 防止测试文件本身被误当作图片；同时验证文件哈希路径。
        text = "abc" * 100
        p = self.src / "x.txt"
        p.write_text(text, encoding="utf-8")
        archive = self.tmp / "file.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        out = self.tmp / "file-out"
        extract_archive(archive, out, quiet=True)
        self.assertEqual((out / "x.txt").read_bytes(), text.encode("utf-8"))

    def test_resolution_one_key_per_group(self) -> None:
        base = _gradient(256, 256)
        for i in range(3):
            img = base.copy()
            ImageDraw.Draw(img).rectangle([5 + i, 5, 20 + i, 20], fill=(i * 40, 90, 200))
            img.save(self.src / f"a{i}.png")
        base2 = _gradient(320, 240)
        for i in range(2):
            img = base2.copy()
            ImageDraw.Draw(img).rectangle([3 + i, 3, 18 + i, 18], fill=(200, i * 50, 10))
            img.save(self.src / f"b{i}.png")
        archive = self.tmp / "resolution.pxcs"
        stats = create_archive(
            archive, [str(self.src)],
            CreateOptions(quiet=True, threads=4, parallel_groups=3, delta_quant=0),
        )
        self.assertEqual(stats.groups, 2)
        self.assertEqual(stats.keys, 2)
        with ArchiveReader(archive) as reader:
            groups = {}
            for e in reader.entries:
                groups.setdefault((e.width, e.height, e.mode), []).append(e)
            self.assertEqual(len(groups), 2)
            for members in groups.values():
                key = members[0]
                self.assertIsNone(key.ref)
                for member in members[1:]:
                    self.assertEqual(member.ref, key.id)
                    self.assertEqual(member.depth, 1)
        self.assertEqual(verify_archive(archive).failed, 0)

    def test_regex_mode_and_fallback(self) -> None:
        base = _gradient(48, 48)
        names = ["1234_p0.png", "1234_p1.png", "5678_p2.png", "5678_p3.png"]
        for i, name in enumerate(names):
            img = base.copy()
            ImageDraw.Draw(img).rectangle([2 + i, 2, 12 + i, 12], fill=(i * 40, 30, 220))
            img.save(self.src / name)
        archive = self.tmp / "regex.pxcs"
        stats = create_archive(
            archive, [str(self.src)],
            CreateOptions(
                quiet=True, group_mode="regex", group_regex=r"\d+_p",
                group_regex_fallback="resolution", delta_quant=0,
                threads=4, parallel_groups=2,
            ),
        )
        self.assertEqual(stats.groups, 2)
        self.assertEqual(stats.keys, 2)
        self.assertEqual(verify_archive(archive).failed, 0)

        # 2023 风格的 no-match 名称：回退到分辨率，而不是逐张 1205 个 key。
        other = self.tmp / "other"
        other.mkdir()
        for i in range(2):
            img = base.copy()
            ImageDraw.Draw(img).rectangle([5 + i, 5, 20, 20], fill=(i * 50, 100, 0))
            img.save(other / f"01_319{i}.jpg", quality=90)
        archive2 = self.tmp / "regex_fallback.pxcs"
        stats2 = create_archive(
            archive2, [str(other)],
            CreateOptions(
                quiet=True, group_mode="regex", group_regex=r"\d+_p",
                group_regex_fallback="resolution", delta_quant=0,
                threads=4, parallel_groups=2,
            ),
        )
        self.assertEqual(stats2.groups, 1)
        self.assertEqual(verify_archive(archive2).failed, 0)

    def test_parallel_and_serial_groups_same_layout(self) -> None:
        base = _gradient(64, 64)
        for i in range(4):
            img = base.copy()
            ImageDraw.Draw(img).rectangle([3 + i, 3, 15 + i, 15], fill=(i * 60, 0, 0))
            img.save(self.src / f"{i:02d}.png")
        outputs = []
        for pg, th in ((1, 1), (3, 4)):
            archive = self.tmp / f"p{pg}.pxcs"
            create_archive(
                archive, [str(self.src)],
                CreateOptions(
                    quiet=True, group_mode="resolution", delta_quant=0,
                    threads=th, parallel_groups=pg,
                ),
            )
            with ArchiveReader(archive) as reader:
                outputs.append([
                    (e.path, e.codec, e.ref, e.quant)
                    for e in reader.entries
                ])
        self.assertEqual(outputs[0], outputs[1])

    def test_lossy_zlib_residual_verifies(self) -> None:
        base = _gradient(512, 512)
        base.save(self.src / "p0.jpg", quality=95, subsampling=0)
        variant = base.copy()
        ImageDraw.Draw(variant).rectangle([80, 80, 300, 300], fill=(20, 220, 90))
        variant.save(self.src / "p1.jpg", quality=95, subsampling=0)
        archive = self.tmp / "lossy.pxcs"
        stats = create_archive(
            archive, [str(self.src)],
            CreateOptions(
                quiet=True, delta_quant=12, delta_quant_max=16,
                threads=4, parallel_groups=2,
            ),
        )
        self.assertGreaterEqual(stats.deltas, 1)
        self.assertGreaterEqual(stats.lossy_deltas, 1)
        with ArchiveReader(archive) as reader:
            self.assertTrue(any(e.codec == CODEC_ZLIB_RESIDUAL for e in reader.entries))
            self.assertTrue(any(e.lossy and e.quant for e in reader.entries))
        self.assertEqual(verify_archive(archive).failed, 0)
        out = self.tmp / "lossy_out"
        extract = extract_archive(archive, out, quiet=True)
        self.assertEqual(extract.failed, 0)
        self.assertTrue((out / "p1.jpg").exists())

    def test_duplicate_png_not_lossy_dup(self) -> None:
        img = _gradient(96, 96)
        img.save(self.src / "a.png")
        shutil.copyfile(self.src / "a.png", self.src / "b.png")
        archive = self.tmp / "pngdup.pxcs"
        stats = create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        self.assertEqual(stats.dups, 0)
        out = self.tmp / "pngdup_out"
        extract_archive(archive, out, quiet=True)
        self.assertEqual((self.src / "b.png").read_bytes(), (out / "b.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
