"""PXCS 分卷归档测试：布局、随机提取、缺卷/损坏/错序、兼容性。"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pxcs_core.archive import (
    CreateOptions,
    archive_info,
    create_archive,
    default_output_dir,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxcs_core.container import ArchiveReader, PxcsFormatError
from pxcs_core.volumes import (
    VOLUME_HEADER_SIZE,
    PxcsVolumeError,
    find_volume_set,
    volume_path,
)


def _write_sources(root: Path, count: int = 4) -> None:
    """生成足够大的样本，保证默认分卷尺寸下能切出多卷。"""
    rng = np.random.default_rng(1234)
    for i in range(count):
        arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        ImageDraw.Draw(img).rectangle([4 + i, 4, 40, 40], fill=(200, i * 40, 20))
        img.save(root / f"{i:03d}.png")
    (root / "notes").mkdir()
    (root / "notes" / "readme.txt").write_text("pxcs volumes\n" * 400, encoding="utf-8")


class VolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-vol-"))
        self.src = self.tmp / "src"
        self.src.mkdir()
        _write_sources(self.src)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------ 基础
    def test_single_file_unchanged_without_volumes(self) -> None:
        archive = self.tmp / "plain.pxcs"
        stats = create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        self.assertTrue(archive.is_file())
        self.assertEqual(stats.volumes, 1)
        self.assertEqual(list(self.tmp.glob("plain.pxcs.*")), [])
        # 未启用分卷时仍然是 1.2 格式，旧程序可读。
        with ArchiveReader(archive) as reader:
            self.assertEqual(reader.header.version_minor, 2)
            self.assertEqual(reader.volume_count, 0)

    def test_two_volume_sizes(self) -> None:
        for size in (4096, 16384):
            archive = self.tmp / f"v{size}.pxcs"
            stats = create_archive(
                archive, [str(self.src)],
                CreateOptions(quiet=True, volume_size=size),
            )
            parts = sorted(self.tmp.glob(f"v{size}.pxcs.*"))
            self.assertGreaterEqual(len(parts), 2, f"size={size}")
            self.assertEqual(stats.volumes, len(parts))
            for index, part in enumerate(parts, 1):
                self.assertEqual(part.name, f"v{size}.pxcs.{index:03d}")
                self.assertLessEqual(part.stat().st_size, size)
                self.assertGreater(part.stat().st_size, VOLUME_HEADER_SIZE)
            self.assertEqual(verify_archive(archive).failed, 0)
            out = self.tmp / f"out{size}"
            result = extract_archive(archive, out, quiet=True)
            self.assertEqual(result.failed, 0)
            for name in ("000.png", "003.png", "notes/readme.txt"):
                self.assertEqual(
                    (self.src / name).read_bytes(), (out / name).read_bytes(), name
                )

    def test_list_and_info_report_volumes(self) -> None:
        archive = self.tmp / "listed.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=4096)
        )
        text = list_entries(archive)
        self.assertIn("分卷", text)
        self.assertIn("notes/readme.txt", text)
        info = archive_info(archive)
        self.assertIn("volumes:", info)
        json_text = list_entries(archive, as_json=True)
        self.assertIn('"volumes"', json_text)

    def test_single_entry_extract_across_volumes(self) -> None:
        archive = self.tmp / "one.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=4096)
        )
        out = self.tmp / "one-out"
        stats = extract_archive(archive, out, names=["002.png"], quiet=True)
        self.assertEqual(stats.extracted, 1)
        self.assertEqual(
            (self.src / "002.png").read_bytes(), (out / "002.png").read_bytes()
        )
        self.assertFalse((out / "000.png").exists())

    def test_any_volume_path_opens_the_set(self) -> None:
        archive = self.tmp / "any.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=4096)
        )
        second = volume_path(archive, 2)
        self.assertTrue(second.is_file())
        # 只给第 2 卷也能定位整组
        with ArchiveReader(second) as reader:
            self.assertEqual(len(reader.entries), 5)
            self.assertEqual(reader.volume_count, len(find_volume_set(second)))
        # 不存在的 base 路径会回退到 <base>.001
        with ArchiveReader(archive) as reader:
            self.assertEqual(len(reader.entries), 5)

    def test_default_output_dir_for_volume(self) -> None:
        self.assertEqual(
            default_output_dir(Path("a.pxcs.003")), Path("a")
        )
        self.assertEqual(default_output_dir(Path("a.pxcs")), Path("a"))

    # ------------------------------------------------------------ 错误
    def test_missing_volume(self) -> None:
        archive = self.tmp / "miss.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        victim = volume_path(archive, 2)
        victim.unlink()
        with self.assertRaises(PxcsVolumeError) as ctx:
            list_entries(archive)
        message = str(ctx.exception)
        self.assertIn("002", message)
        self.assertIn("缺少分卷", message)

    def test_wrong_volume_order(self) -> None:
        archive = self.tmp / "order.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        first = volume_path(archive, 1)
        second = volume_path(archive, 2)
        a, b = first.read_bytes(), second.read_bytes()
        first.write_bytes(b)
        second.write_bytes(a)
        with self.assertRaises(PxcsVolumeError) as ctx:
            list_entries(archive)
        self.assertIn("顺序", str(ctx.exception))

    def test_corrupted_volume_header(self) -> None:
        archive = self.tmp / "hdr.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        part = volume_path(archive, 1)
        raw = bytearray(part.read_bytes())
        raw[20] ^= 0xFF  # 卷头内字段
        part.write_bytes(bytes(raw))
        with self.assertRaises(PxcsVolumeError):
            list_entries(archive)

    def test_corrupted_volume_payload(self) -> None:
        archive = self.tmp / "pay.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        part = volume_path(archive, 2)
        raw = bytearray(part.read_bytes())
        raw[VOLUME_HEADER_SIZE + 4] ^= 0xFF
        part.write_bytes(bytes(raw))
        stats = verify_archive(archive)
        self.assertGreater(stats.failed, 0)

    def test_truncated_volume(self) -> None:
        archive = self.tmp / "trunc.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        part = volume_path(archive, 1)
        part.write_bytes(part.read_bytes()[:-16])
        with self.assertRaises(PxcsVolumeError):
            list_entries(archive)

    def test_stale_volume_cleanup_on_overwrite(self) -> None:
        archive = self.tmp / "stale.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, volume_size=2048)
        )
        count = len(list(self.tmp.glob("stale.pxcs.*")))
        self.assertGreaterEqual(count, 3)
        stale = volume_path(archive, count + 2)
        stale.write_bytes(b"stale")
        # 不覆盖时报错
        with self.assertRaises(PxcsFormatError):
            create_archive(
                archive, [str(self.src)],
                CreateOptions(quiet=True, volume_size=2048),
            )
        create_archive(
            archive, [str(self.src)],
            CreateOptions(quiet=True, volume_size=2048, overwrite=True),
        )
        self.assertFalse(stale.exists())
        self.assertEqual(verify_archive(archive).failed, 0)

    def test_volume_size_too_small(self) -> None:
        archive = self.tmp / "small.pxcs"
        with self.assertRaises(PxcsFormatError):
            create_archive(
                archive, [str(self.src)],
                CreateOptions(quiet=True, volume_size=128),
            )
        self.assertFalse(archive.exists())

    def test_aborted_create_leaves_no_volume_files(self) -> None:
        archive = self.tmp / "abort.pxcs"

        class Boom(Exception):
            pass

        opts = CreateOptions(quiet=True, volume_size=2048)
        original = opts.progress
        # 用不存在的输入触发失败，确认没有留下 .001/.tmp
        with self.assertRaises(Exception):
            create_archive(archive, [str(self.tmp / "missing")], opts)
        self.assertEqual(list(self.tmp.glob("abort.pxcs*")), [])
        self.assertTrue(original)


if __name__ == "__main__":
    unittest.main()
