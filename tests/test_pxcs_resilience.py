"""创建过程的容错测试：单个文件读不了不能毁掉整次归档。

这条需求来自真实场景：1290 张、几分钟的任务，只要有一张图被其它程序占用
（索引器、同步盘占位符、权限），旧实现就会让整次创建抛异常作废。

现在的契约：
- 组 key 读不了 → 组内向后最多试 ``MAX_KEY_CANDIDATES`` 个成员当 key；
- 成员读不了 → 记为跳过，其余成员照常；
- 后缀为"跳过数 > 0 时退出码 3"。

测试手法：临时把 ``pathlib.Path.read_bytes`` 换成对指定文件名抛
``PermissionError`` 的版本。raw 直通路径用的正是 ``read_bytes``，
而 Pillow 解码走内建 ``open``，所以这个替换足够精确。
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import pxcs
from pxcs_core.archive import CreateOptions, create_archive, extract_archive
from pxcs_core.container import ArchiveReader


def _write_jpeg(path: Path, size: tuple[int, int], tint: int) -> None:
    width, height = size
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (tint + x) % 256,
                (tint + y * 3) % 256,
                (tint + x * y) % 256,
            )
    img.save(path, format="JPEG", quality=88)


@contextlib.contextmanager
def deny_read(*names: str):
    """让指定的文件名在 ``Path.read_bytes`` 上抛 PermissionError。"""
    real = Path.read_bytes
    denied = set(names)

    def fake(self: Path) -> bytes:
        if self.name in denied:
            raise PermissionError(13, "拒绝访问", str(self))
        return real(self)

    with mock.patch.object(Path, "read_bytes", fake):
        yield


class SkipOnUnreadableTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unreadable_key_falls_back_to_next_member(self) -> None:
        """组 key 读不了时用下一张当 key，组内差分仍然成立。"""
        for index in range(3):
            _write_jpeg(self.src / f"a_p{index}.jpg", (96, 72), 30 + index * 6)

        archive = self.tmp / "out.pxcs"
        with deny_read("a_p0.jpg"):
            stats = create_archive(
                archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
            )

        self.assertEqual(stats.skipped, 1, "被拒绝读的那个文件应记为跳过")
        self.assertEqual(len(stats.errors), 1)
        self.assertIn("a_p0.jpg", stats.errors[0])

        with ArchiveReader(archive) as reader:
            paths = [e.path for e in reader.entries]
            self.assertNotIn("a_p0.jpg", paths)
            self.assertIn("a_p1.jpg", paths, "下一张应顶替成为 key")
            self.assertIn("a_p2.jpg", paths)
            deltas = [e for e in reader.entries if e.codec != "raw"]
            self.assertTrue(deltas, "顶替 key 之后组内仍应产生差分")

    def test_unreadable_singleton_group_produces_empty_archive(self) -> None:
        """唯一一张读不了：归档仍要正常生成（0 条目），而不是崩溃。"""
        _write_jpeg(self.src / "only.jpg", (64, 48), 10)
        archive = self.tmp / "empty.pxcs"
        with deny_read("only.jpg"):
            stats = create_archive(
                archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
            )
        self.assertEqual(stats.skipped, 1)
        self.assertTrue(archive.is_file(), "归档文件仍然要生成")
        with ArchiveReader(archive) as reader:
            self.assertEqual(len(reader.entries), 0)

    def test_unreadable_plain_file_is_skipped(self) -> None:
        """非图片文件读不了不能影响图片条目。"""
        _write_jpeg(self.src / "a_p0.jpg", (64, 48), 20)
        (self.src / "notes.txt").write_text("hello", encoding="utf-8")

        archive = self.tmp / "mixed.pxcs"
        with deny_read("notes.txt"):
            stats = create_archive(
                archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
            )
        self.assertEqual(stats.skipped, 1)
        with ArchiveReader(archive) as reader:
            self.assertEqual([e.path for e in reader.entries], ["a_p0.jpg"])

    def test_normal_run_has_no_skips(self) -> None:
        _write_jpeg(self.src / "a_p0.jpg", (64, 48), 50)
        archive = self.tmp / "ok.pxcs"
        stats = create_archive(
            archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
        )
        self.assertEqual(stats.skipped, 0)
        self.assertEqual(stats.errors, [])

    def test_skipped_entries_are_still_extractable(self) -> None:
        """跳过一个之后，剩下的仍必须能完整提取。

        注意这里必须让被拒的文件走 raw 路径才会真正触发读取失败：
        差分编码走 Pillow 解码，不经过 ``Path.read_bytes``。所以这里拒掉的是
        组 key（key 一定走 raw 直通），而不是随便挑一张成员。
        """
        for index in range(3):
            _write_jpeg(self.src / f"a_p{index}.jpg", (80, 60), 60 + index * 5)
        archive = self.tmp / "rest.pxcs"
        with deny_read("a_p0.jpg"):
            create_archive(
                archive, [str(self.src)], CreateOptions(quiet=True, progress=False)
            )
        out = self.tmp / "out"
        stats = extract_archive(archive, out, quiet=True)
        self.assertEqual(stats.failed, 0)
        extracted = sorted(p.name for p in out.iterdir())
        self.assertEqual(extracted, ["a_p1.jpg", "a_p2.jpg"])


class CreateExitCodeTests(unittest.TestCase):
    """退出码约定：0=完整，3=有文件被跳过。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        _write_jpeg(self.src / "a_p0.jpg", (64, 48), 15)
        _write_jpeg(self.src / "a_p1.jpg", (64, 48), 75)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, output: Path) -> int:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            return pxcs.main(
                [
                    "create",
                    "-i",
                    str(self.src),
                    "-o",
                    str(output),
                    "--overwrite",
                    "--quiet",
                    "--no-progress",
                ]
            )

    def test_exit_zero_when_complete(self) -> None:
        self.assertEqual(self._run(self.tmp / "ok.pxcs"), 0)

    def test_exit_three_when_files_skipped(self) -> None:
        with deny_read("a_p0.jpg"):
            code = self._run(self.tmp / "partial.pxcs")
        self.assertEqual(code, 3, "有跳过时必须是退出码 3")
        self.assertTrue((self.tmp / "partial.pxcs").is_file())


if __name__ == "__main__":
    unittest.main()
