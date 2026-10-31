"""写完归档后的自检测试。

`ArchiveWriter.finish()` 只负责落盘与 CRC；磁盘满、杀软改写、网络盘掉线这类
问题要等下次 `verify` 才会暴露。所以创建流程在 finish 之后会立刻用
`ArchiveReader` 重开一次，核对索引长度与条目数。

这里的测试锁定两件事：
1. 自检发现异常时**必须报错**，而且归档文件要保留（便于取证/事后 verify）；
2. `self_check=False`（CLI `--no-self-check`）时确实不做这件事。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from pxcs_core import grouped_create
from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.grouped_create import _self_check_archive
from pxcs_core.archive import PxcsFormatError


def _write_jpeg(path: Path, size: tuple[int, int], tint: int) -> None:
    width, height = size
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((tint + x) % 256, (tint + y) % 256, (tint + x) % 256)
    img.save(path, format="JPEG", quality=86)


class _FakeHeader:
    def __init__(self, index_length: int = 128) -> None:
        self.index_length = index_length


class _FakeReader:
    """最小可用的 ArchiveReader 替身，用来制造自检失败。"""

    def __init__(self, entries: int = 0, index_length: int = 128) -> None:
        self.entries = [object()] * entries
        self.header = _FakeHeader(index_length)

    def __enter__(self) -> "_FakeReader":
        return self

    def __exit__(self, *exc) -> None:
        return None


class SelfCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        _write_jpeg(self.src / "a_p0.jpg", (72, 54), 25)
        _write_jpeg(self.src / "a_p1.jpg", (72, 54), 95)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_self_check_passes_for_normal_archive(self) -> None:
        archive = self.tmp / "ok.pxcs"
        stats = create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
        )
        self.assertEqual(stats.images, 2)
        self.assertTrue(archive.is_file())

    def test_self_check_reports_unreadable_archive(self) -> None:
        """重开归档失败时必须报错，且归档文件保留。"""
        archive = self.tmp / "bad.pxcs"

        def boom(*args, **kwargs):
            raise OSError("模拟磁盘损坏")

        with mock.patch.object(grouped_create, "ArchiveReader", boom):
            with self.assertRaises(PxcsFormatError) as ctx:
                create_archive(
                    archive,
                    [str(self.src)],
                    CreateOptions(quiet=True, progress=False),
                )
        self.assertIn("自检失败", str(ctx.exception))
        self.assertTrue(archive.is_file(), "自检失败也要保留文件，便于事后 verify")

    def test_self_check_detects_entry_count_mismatch(self) -> None:
        archive = self.tmp / "mismatch.pxcs"
        with mock.patch.object(
            grouped_create,
            "ArchiveReader",
            lambda *a, **k: _FakeReader(entries=1),
        ):
            with self.assertRaises(PxcsFormatError) as ctx:
                create_archive(
                    archive,
                    [str(self.src)],
                    CreateOptions(quiet=True, progress=False),
                )
        self.assertIn("entries", str(ctx.exception))

    def test_self_check_detects_empty_index(self) -> None:
        archive = self.tmp / "noindex.pxcs"
        with mock.patch.object(
            grouped_create,
            "ArchiveReader",
            lambda *a, **k: _FakeReader(entries=2, index_length=0),
        ):
            with self.assertRaises(PxcsFormatError):
                create_archive(
                    archive,
                    [str(self.src)],
                    CreateOptions(quiet=True, progress=False),
                )

    def test_no_self_check_skips_verification(self) -> None:
        """--no-self-check：即使重开会失败也不报错。"""
        archive = self.tmp / "skip.pxcs"

        def boom(*args, **kwargs):
            raise OSError("不该被调用")

        with mock.patch.object(grouped_create, "ArchiveReader", boom):
            stats = create_archive(
                archive,
                [str(self.src)],
                CreateOptions(
                    quiet=True, progress=False, self_check=False
                ),
            )
        self.assertEqual(stats.images, 2)
        self.assertTrue(archive.is_file())

    def test_self_check_function_rejects_zero_entries_when_expected(self) -> None:
        """直接调用自检函数时，条目数不符要抛错。"""
        archive = self.tmp / "direct.pxcs"
        create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
        )
        _self_check_archive(archive, 2, None)  # 正确，不抛
        with self.assertRaises(PxcsFormatError):
            _self_check_archive(archive, 99, None)


class SelfCheckExitCodeTests(unittest.TestCase):
    """CLI 层：自检失败必须是退出码 4，而不是混进 1。"""

    def test_cli_returns_four(self) -> None:
        import contextlib
        import io

        import pxcs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            _write_jpeg(src / "a_p0.jpg", (64, 48), 12)

            def boom(*args, **kwargs):
                raise OSError("模拟自检失败")

            buffer = io.StringIO()
            with mock.patch.object(grouped_create, "ArchiveReader", boom):
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(
                    buffer
                ):
                    code = pxcs.main(
                        [
                            "create",
                            "-i",
                            str(src),
                            "-o",
                            str(root / "a.pxcs"),
                            "--overwrite",
                            "--quiet",
                            "--no-progress",
                        ]
                    )
            self.assertEqual(code, 4, buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
