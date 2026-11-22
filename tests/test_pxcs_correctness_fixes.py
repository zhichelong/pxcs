"""复审报告指出的正确性问题的回归测试。

覆盖：
1. 父节点编码失败/回退后，子节点不能继续引用它；
2. key worker 异常时整组成员都必须记账，不能静默漏掉；
3. 带元数据的图不能走 zlib 残差（会丢 ICC/EXIF/XMP）；
4. minor 与特性的组合校验；
5. 组内深度预算；
6. PXC tile 的有界解压。
"""
from __future__ import annotations

import tempfile
import struct
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageCms

from pxc_core.zlibsafe import decompress_bounded
from pxc_core.codec import decode_tiles
from pxc_core.config import (
    DATA_FORMAT_VERSION, DATA_HEADER_STRUCT, FLAG_PAYLOAD_ZLIB,
    TILE_HEADER_STRUCT,
)
from pxcs_core import grouped_create
from pxcs_core.archive import (
    CreateOptions,
    PxcsFormatError,
    create_archive,
    verify_archive,
)
from pxcs_core.container import ArchiveReader
from pxcs_core.format import HEADER_STRUCT
from tests.test_pxcs_zlib_bounds import _build_archive
from pxcs_core.grouped_create import _local_plan_depth


def _base(seed: int = 1, size=(96, 72)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _variant(arr: np.ndarray, offset: int, shift: int) -> np.ndarray:
    out = arr.copy()
    x = 8 + offset * 12
    out[x:x + 12, x:x + 12] = np.clip(
        out[x:x + 12, x:x + 12].astype(np.int16) + shift, 0, 255
    ).astype(np.uint8)
    return out


def _save_jpeg(path: Path, arr: np.ndarray, exif: bool = False) -> None:
    kwargs = {}
    if exif:
        meta = Image.Exif()
        meta[0x010E] = "pxcs test"
        kwargs["exif"] = meta
    Image.fromarray(arr, "RGB").save(path, format="JPEG", quality=92, **kwargs)


class RefConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parent_write_failure_forces_child_raw(self) -> None:
        """A 编码时用了 B 的重建图，A 最后回退 raw；B 必须一起回退。"""
        src = self.root / "src"
        src.mkdir()
        base = _base(2)
        _save_jpeg(src / "0_key.jpg", base)
        a = _variant(base, 0, 6)
        _save_jpeg(src / "1_a.jpg", a)
        _save_jpeg(src / "2_b.jpg", _variant(a, 1, 6))

        real = grouped_create._write_result
        fired = {"done": False}

        def flaky(writer, entries, result, raw_hash_index, **kwargs):
            if result.item.source.path.name == "1_a.jpg" and not fired["done"]:
                fired["done"] = True
                raise PxcsFormatError("injected write failure")
            return real(writer, entries, result, raw_hash_index, **kwargs)

        out = self.root / "a.pxcs"
        with mock.patch.object(grouped_create, "_write_result", side_effect=flaky):
            create_archive(
                out, [str(src)],
                CreateOptions(quiet=True, progress=False, threads=2,
                              self_check=True),
            )
        self.assertTrue(fired["done"])
        stats = verify_archive(out)
        self.assertEqual(stats.failed, 0, stats.errors)
        with ArchiveReader(out) as reader:
            a_entry = next(e for e in reader.entries if e.path.endswith("1_a.jpg"))
            b_entry = next(e for e in reader.entries if e.path.endswith("2_b.jpg"))
            # 回退目标可能是 raw 或 jpegxl（JPEG 的字节级直通），
            # 关键是不能再是依赖 A 重建图的残差。
            self.assertNotIn(a_entry.codec, {"pxc_residual", "zlib_residual"})
            self.assertNotIn(b_entry.codec, {"pxc_residual", "zlib_residual"})


class KeyWorkerExceptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_all_members_are_recorded_when_key_worker_raises(self) -> None:
        src = self.root / "src"
        src.mkdir()
        for i in range(10):
            Image.fromarray(_base(10 + i), "RGB").save(src / f"{i}.png")
        out = self.root / "a.pxcs"
        with mock.patch.object(
            grouped_create, "_encode_key", side_effect=RuntimeError("boom")
        ):
            stats = create_archive(
                out, [str(src)],
                CreateOptions(quiet=True, progress=False, threads=2,
                              self_check=False),
            )
        self.assertEqual(stats.files_total, 10)
        self.assertEqual(stats.skipped, 10, stats.errors)
        self.assertEqual(len(stats.errors), 10)


class MetadataResidualTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make(self, name: str, exif: bool) -> Path:
        src = self.root / name
        src.mkdir()
        base = _base(20)
        _save_jpeg(src / "a.jpg", base, exif=exif)
        _save_jpeg(src / "b.jpg", _variant(base, 0, 4), exif=exif)
        return src

    def test_metadata_survives_zlib_residual(self) -> None:
        src = self._make("with_exif", True)
        out = self.root / "with.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          residual_codec="zlib", self_check=True),
        )
        with ArchiveReader(out) as reader:
            codecs = [e.codec for e in reader.entries]
            meta_entries = [e for e in reader.entries if e.meta]
            self.assertEqual(reader.header.version_minor, 6)
        self.assertIn("zlib_residual", codecs)
        self.assertTrue(meta_entries)
        self.assertEqual(verify_archive(out).failed, 0)
        dest = self.root / "with_out"
        from pxcs_core.archive import extract_archive

        extract_archive(out, dest, quiet=True, threads=1)
        extracted = dest / "b.jpg"
        self.assertTrue(extracted.exists())
        exif = Image.open(extracted).getexif()
        self.assertEqual(exif.get(0x010E), "pxcs test")

    def test_cross_key_zlib_residual_keeps_meta(self) -> None:
        src = self.root / "cross_meta"
        src.mkdir()
        arr = _base(40)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.fromarray(arr, "RGB").save(src / "a.png")
        rgba = np.concatenate(
            [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2
        )
        Image.fromarray(rgba, "RGBA").save(src / "b.png", icc_profile=icc)
        out = self.root / "cross_meta.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          residual_codec="zlib"),
        )
        with ArchiveReader(out) as reader:
            rgba_entry = next(e for e in reader.entries if e.path.endswith("b.png"))
            self.assertEqual(rgba_entry.codec, "zlib_residual")
            self.assertTrue(rgba_entry.meta and rgba_entry.meta.get("icc"))
            self.assertEqual(reader.header.version_minor, 6)
        self.assertEqual(verify_archive(out).failed, 0)

    def test_without_metadata_still_uses_zlib(self) -> None:
        src = self._make("no_exif", False)
        out = self.root / "without.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          residual_codec="zlib", self_check=True),
        )
        with ArchiveReader(out) as reader:
            codecs = [e.codec for e in reader.entries]
        self.assertIn("zlib_residual", codecs)


class MinorGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _patch_minor(self, path: Path, minor: int) -> None:
        raw = bytearray(path.read_bytes())
        fields = list(HEADER_STRUCT.unpack_from(raw, 0))
        fields[2] = minor
        zero = HEADER_STRUCT.pack(*fields[:12], 0)
        fields[12] = zlib.crc32(zero) & 0xFFFFFFFF
        HEADER_STRUCT.pack_into(raw, 0, *fields)
        path.write_bytes(bytes(raw))

    def _craft_invalid_cross_mode(self, path: Path) -> None:
        png = Path(self.root / "tiny.png")
        Image.fromarray(_base(50, (8, 8)), "RGB").save(png)
        payload = png.read_bytes()
        entries = [
            {
                "id": 0, "path": "a.png", "kind": "image", "codec": "raw",
                "offset": 0, "length": 0, "size": len(payload), "ref": None,
                "depth": 0, "width": 8, "height": 8, "mode": "RGB",
                "format": "PNG", "ext": ".png", "constant_alpha": None,
                "quant": 0, "centered": True, "lossy": False,
                "pixel_sha256": "", "source_sha256": "", "data_crc32": 0,
                "raw_crc32": 0, "data_sha256": "", "mtime_ns": 0,
                "flags": 0, "reason": "",
            },
            {
                "id": 1, "path": "b.png", "kind": "image",
                "codec": "pxc_residual", "offset": 0, "length": 0, "size": 8,
                "ref": 0, "depth": 1, "width": 8, "height": 8, "mode": "RGBA",
                "format": "PNG", "ext": ".png", "constant_alpha": None,
                "quant": 0, "centered": True, "lossy": False,
                "pixel_sha256": "", "source_sha256": "", "data_crc32": 0,
                "raw_crc32": 0, "data_sha256": "", "mtime_ns": 0,
                "flags": 0, "reason": "",
            },
        ]
        _build_archive(path, entries, [payload, b"x" * 8])
        raw = bytearray(path.read_bytes())
        fields = list(HEADER_STRUCT.unpack_from(raw, 0))
        fields[2] = 5
        zero = HEADER_STRUCT.pack(*fields[:12], 0)
        fields[12] = zlib.crc32(zero) & 0xFFFFFFFF
        HEADER_STRUCT.pack_into(raw, 0, *fields)
        path.write_bytes(bytes(raw))

    def test_invalid_cross_mode_semantics_rejected(self) -> None:
        path = self.root / "bad_cross.pxcs"
        self._craft_invalid_cross_mode(path)
        with self.assertRaises(PxcsFormatError):
            with ArchiveReader(path):
                pass

    def test_cross_mode_ref_requires_minor_5(self) -> None:
        src = self.root / "src"
        src.mkdir()
        arr = _base(30)
        Image.fromarray(arr, "RGB").save(src / "a.png")
        Image.fromarray(
            np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], 2),
            "RGBA",
        ).save(src / "b.png")
        out = self.root / "a.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2),
        )
        with ArchiveReader(out) as reader:
            self.assertEqual(reader.header.version_minor, 5)
        self._patch_minor(out, 4)
        with self.assertRaises(PxcsFormatError):
            with ArchiveReader(out):
                pass

    def test_entry_meta_requires_minor_6(self) -> None:
        src = self.root / "meta"
        src.mkdir()
        base = _base(31)
        _save_jpeg(src / "a.jpg", base, exif=True)
        _save_jpeg(src / "b.jpg", _variant(base, 0, 4), exif=True)
        out = self.root / "meta.pxcs"
        create_archive(
            out, [str(src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          residual_codec="zlib"),
        )
        with ArchiveReader(out) as reader:
            self.assertEqual(reader.header.version_minor, 6)
        self._patch_minor(out, 5)
        with self.assertRaises(PxcsFormatError):
            with ArchiveReader(out):
                pass


class DepthBudgetTest(unittest.TestCase):
    def test_local_plan_depth_leaves_room(self) -> None:
        self.assertEqual(_local_plan_depth(8, 0), 8)
        self.assertEqual(_local_plan_depth(8, 250), 5)
        self.assertEqual(_local_plan_depth(8, 254), 1)
        self.assertEqual(_local_plan_depth(8, 255), 0)


class PxcBoundedZlibTest(unittest.TestCase):
    def test_tile_bomb_is_bounded(self) -> None:
        bomb = zlib.compress(b"\x00" * (1 << 20), 9)
        with self.assertRaises(Exception):
            decompress_bounded(bomb, limit=1024, expected=1024, label="tile")

    def test_decode_tiles_rejects_declared_small_bomb(self) -> None:
        bomb = zlib.compress(b"\x00" * (1 << 20), 9)
        head = TILE_HEADER_STRUCT.pack(
            0, 0, FLAG_PAYLOAD_ZLIB, 16, len(bomb), 0, zlib.crc32(bomb) & 0xFFFFFFFF
        )
        payload = DATA_HEADER_STRUCT.pack(DATA_FORMAT_VERSION, 4, 1, 1, 1) + head + bomb
        with self.assertRaises(Exception):
            decode_tiles(payload, 4, 4, 3, np.dtype(np.uint8), 4)

    def test_unknown_tile_flags_rejected(self) -> None:
        raw = b"\x00" * (4 * 4 * 3)
        head = TILE_HEADER_STRUCT.pack(
            0, 0, 0x02, len(raw), len(raw), 0, zlib.crc32(raw) & 0xFFFFFFFF
        )
        payload = DATA_HEADER_STRUCT.pack(DATA_FORMAT_VERSION, 4, 1, 1, 1) + head + raw
        with self.assertRaises(Exception):
            decode_tiles(payload, 4, 4, 3, np.dtype(np.uint8), 4)


if __name__ == "__main__":
    unittest.main()
