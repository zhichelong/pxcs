"""raw 直通跳过解码（延迟解码）的行为测试。

背景：`load_image()` 会把整张图解码成像素。对以 raw 直通保存的 JPEG/WebP 来说，
存的就是源文件字节；解码这一步只为确认"能解出像素"并取到 `constant_alpha`，
而这两项对 raw 条目的重建没有任何作用——`pixel_sha256` 从来不会为 raw 条目写入
（实测旧归档 107 个 raw 条目，0 个带像素哈希）。但它是这类归档创建耗时的主要部分。

优化后：
- 组内只有一张（不会做差分）且走 raw 直通时，完全不解码；
- 组内多于一张时仍需参考图，行为与之前完全一致；
- `--decode-raw` / `CreateOptions.decode_raw` 可恢复"总是解码"的旧行为，
  用于确认图片确实可解码。

这些测试锁定上面三条，避免将来"顺手优化"把差分组也一起改掉。
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from pxcs_core import grouped_create
from pxcs_core.archive import (
    CreateOptions,
    create_archive,
    extract_archive,
    verify_archive,
)
from pxcs_core.container import ArchiveReader
from pxcs_core.format import CODEC_RAW


def _make_jpeg(path: Path, size: tuple[int, int], tint: int) -> None:
    """生成一张有损 JPEG，内容带渐变噪声，避免退化成一两字节。"""
    width, height = size
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (tint + x * 3) % 256,
                (tint + y * 5) % 256,
                (tint + (x ^ y)) % 256,
            )
    img.save(path, format="JPEG", quality=92)


class LazyRawDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _entries(self, archive: Path) -> list:
        with ArchiveReader(archive) as reader:
            return list(reader.entries)

    def test_singleton_group_does_not_decode(self) -> None:
        """单张成组：创建过程中不应调用 load_image。"""
        src = self.tmp / "src1"
        src.mkdir()
        _make_jpeg(src / "one.jpg", (64, 48), 10)

        calls = []
        real = grouped_create.load_image

        def counting(*args, **kwargs):
            calls.append(args[0] if args else kwargs.get("path"))
            return real(*args, **kwargs)

        with mock.patch.object(grouped_create, "load_image", counting):
            archive = self.tmp / "single.pxcs"
            create_archive(
                archive, [str(src)], CreateOptions(quiet=True, progress=False)
            )

        self.assertEqual(calls, [], "单张 raw 直通不应解码")
        entries = self._entries(archive)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].codec, CODEC_RAW)
        self.assertEqual(entries[0].reason, "key_lossy_passthrough")
        self.assertTrue(entries[0].source_sha256, "源文件字节哈希必须仍然记录")
        self.assertEqual(entries[0].width, 64)
        self.assertEqual(entries[0].height, 48)

    def test_singleton_archive_verifies_and_extracts_exactly(self) -> None:
        """跳过解码后，校验与提取仍必须完全正确（raw 是逐字节直通）。"""
        src = self.tmp / "src2"
        src.mkdir()
        _make_jpeg(src / "one.jpg", (64, 48), 40)
        archive = self.tmp / "single.pxcs"
        create_archive(archive, [str(src)], CreateOptions(quiet=True, progress=False))

        self.assertEqual(verify_archive(archive).failed, 0)

        out = self.tmp / "out"
        extract_archive(archive, out, quiet=True)
        self.assertEqual(
            hashlib.sha256((src / "one.jpg").read_bytes()).hexdigest(),
            hashlib.sha256((out / "one.jpg").read_bytes()).hexdigest(),
            "raw 直通提取必须字节一致",
        )

    def test_multi_image_group_still_decodes_and_deltas(self) -> None:
        """多张成组：必须解码出参考图，差分与像素哈希都保留。"""
        src = self.tmp / "src3"
        src.mkdir()
        for index in range(3):
            _make_jpeg(src / f"v{index}.jpg", (96, 72), 30 + index * 4)

        archive = self.tmp / "multi.pxcs"
        stats = create_archive(
            archive, [str(src)], CreateOptions(quiet=True, progress=False)
        )
        self.assertEqual(stats.groups, 1, "同尺寸应归为一个分辨率组")
        self.assertGreaterEqual(stats.deltas, 1, "组内应至少产生一个差分")

        entries = self._entries(archive)
        residuals = [e for e in entries if e.codec != CODEC_RAW]
        self.assertTrue(residuals, "应当存在差分条目")
        for entry in residuals:
            self.assertTrue(
                entry.pixel_sha256, f"{entry.path} 是差分条目，必须有像素哈希"
            )
            self.assertGreaterEqual(entry.depth, 1)
        self.assertEqual(verify_archive(archive).failed, 0)

    def test_decode_raw_flag_recovers_validation(self) -> None:
        """--decode-raw 会真的解码：被截断的图片会被识别为解码失败。

        默认（跳过解码）下，截断的 JPEG 会作为 raw 原样归档，字节完整保留；
        开启该选项后，创建阶段就能发现"解不出来"。
        """
        src = self.tmp / "src4"
        src.mkdir()
        path = src / "truncated.jpg"
        _make_jpeg(path, (64, 64), 5)
        blob = path.read_bytes()
        path.write_bytes(blob[: int(len(blob) * 0.6)])  # 截掉后半段像素数据

        lazy = self.tmp / "lazy.pxcs"
        create_archive(lazy, [str(src)], CreateOptions(quiet=True, progress=False))
        lazy_reason = self._entries(lazy)[0].reason
        self.assertEqual(
            lazy_reason, "key_lossy_passthrough", "默认路径不解码，因此发现不了"
        )

        strict = self.tmp / "strict.pxcs"
        create_archive(
            strict,
            [str(src)],
            CreateOptions(quiet=True, progress=False, decode_raw=True),
        )
        strict_reason = self._entries(strict)[0].reason
        self.assertTrue(
            strict_reason.startswith("decode_error"),
            f"开启 decode_raw 后应记录解码失败，实际 reason={strict_reason!r}",
        )
        # 两种模式下字节都完整保留
        with ArchiveReader(lazy) as reader:
            self.assertEqual(
                reader.read_entry_data(reader.entries[0]), path.read_bytes()
            )

    def test_payload_corruption_still_detected(self) -> None:
        """raw 条目没有像素哈希，payload 篡改仍必须被字节级校验抓到。"""
        src = self.tmp / "src5"
        src.mkdir()
        _make_jpeg(src / "one.jpg", (48, 48), 7)
        archive = self.tmp / "corrupt.pxcs"
        create_archive(archive, [str(src)], CreateOptions(quiet=True, progress=False))

        with ArchiveReader(archive) as reader:
            entry = reader.entries[0]
            offset, length = entry.offset, entry.length
        self.assertGreater(length, 8)

        blob = bytearray(archive.read_bytes())
        blob[offset + length // 2] ^= 0xFF  # 只翻 payload 中间的一个字节
        archive.write_bytes(bytes(blob))
        self.assertGreater(verify_archive(archive).failed, 0, "篡改必须被检出")

    def test_raw_payload_identical_in_both_modes(self) -> None:
        """是否解码不能改变 raw payload 本身，只影响可选的诊断字段。"""
        src = self.tmp / "src6"
        src.mkdir()
        _make_jpeg(src / "one.jpg", (56, 40), 90)

        lazy = self.tmp / "lazy.pxcs"
        strict = self.tmp / "strict.pxcs"
        create_archive(lazy, [str(src)], CreateOptions(quiet=True, progress=False))
        create_archive(
            strict,
            [str(src)],
            CreateOptions(quiet=True, progress=False, decode_raw=True),
        )

        def payload_of(path: Path) -> bytes:
            with ArchiveReader(path) as reader:
                return reader.read_entry_data(reader.entries[0])

        self.assertEqual(
            hashlib.sha256(payload_of(lazy)).hexdigest(),
            hashlib.sha256(payload_of(strict)).hexdigest(),
            "两种模式的 raw payload 必须完全一致",
        )


if __name__ == "__main__":
    unittest.main()
