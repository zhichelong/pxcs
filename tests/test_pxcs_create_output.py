"""创建过程的终端输出回归测试。

这一组测试来自一个真实崩溃：

    TypeError: t() got multiple values for argument 'key'
    grouped_create.py ... create_grouped_archive

原因：i18n 的 `t(key, **kwargs)` 把 `key` 当成了正式参数名，而
`create.group_line` 的文案里带 `{key}` 占位符、调用时传了 `key=...`，
两者撞名。它只在**普通模式（拖文件夹）**触发——普通模式默认
`show_groups=True`，而其它测试都用 `show_groups=False`，所以一直没被发现。

修复方式是把 `key` 改成位置限定参数（`key: str, /`）。这里的测试锁定：
1. 任何语言条目都可以用名为 `key` 的格式化占位符；
2. `show_groups=True` + 非 quiet 的创建路径不再抛异常，且真的打印了分组；
3. 普通模式 `pxcs <目录>` 的完整 CLI 路径也能跑通。
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pxc_core.i18n import configure, reset_for_tests, t
from pxcs_core.archive import CreateOptions, create_archive


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


class I18nKeyPlaceholderTests(unittest.TestCase):
    def test_format_parameter_named_key(self) -> None:
        """`t()` 必须允许格式化参数叫 key（历史上正是因为这里撞名）。"""
        reset_for_tests()
        configure(language="chinese_simp")
        text = t(
            "create.group_line",
            index=7,
            width=640,
            height=480,
            mode="RGB",
            count=3,
            key="a_p0.jpg",
        )
        self.assertIn("a_p0.jpg", text)
        self.assertIn("640", text)
        self.assertNotIn("{key}", text)

    def test_group_line_key_missing_keeps_placeholder(self) -> None:
        """缺少 key 参数时保留占位符，不抛异常。"""
        reset_for_tests()
        configure(language="chinese_simp")
        text = t(
            "create.group_line",
            index=1,
            width=2,
            height=3,
            mode="L",
            count=1,
        )
        self.assertIn("{key}", text)


class CreateOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        # 两种分辨率 → 至少两个组，确保分组汇总与逐组输出都会被执行
        _write_jpeg(self.src / "a_p0.jpg", (96, 64), 10)
        _write_jpeg(self.src / "a_p1.jpg", (96, 64), 90)
        _write_jpeg(self.src / "b_p0.jpg", (128, 80), 40)
        _write_jpeg(self.src / "b_p1.jpg", (128, 80), 140)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_show_groups_does_not_crash(self) -> None:
        """show_groups=True 时必须能打印每组一行，且带 key= 文件名。"""
        buffer = io.StringIO()
        archive = self.tmp / "groups.pxcs"
        with contextlib.redirect_stdout(buffer):
            stats = create_archive(
                archive,
                [str(self.src)],
                CreateOptions(
                    show_groups=True,
                    quiet=False,
                    progress=False,
                    overwrite=True,
                ),
            )
        self.assertEqual(stats.groups, 2)
        output = buffer.getvalue()
        self.assertIn("[GROUP", output, "缺少分组行输出")
        self.assertIn("key=", output, "分组行应包含 key= 文件名")
        self.assertIn("a_p0.jpg", output)

    def test_details_does_not_crash(self) -> None:
        """--details 走 progress.detail 路径，也要能正常输出。"""
        buffer = io.StringIO()
        archive = self.tmp / "details.pxcs"
        with contextlib.redirect_stdout(buffer):
            stats = create_archive(
                archive,
                [str(self.src)],
                CreateOptions(
                    details=True,
                    show_groups=True,
                    quiet=False,
                    progress=False,
                    detail_lines=5,
                    overwrite=True,
                ),
            )
        self.assertEqual(stats.images, 4)
        self.assertTrue(archive.is_file())

    def test_ignored_options_warn_once(self) -> None:
        """分组模式下 --search/--keyframe 等不生效，必须明确告警。"""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            create_archive(
                self.tmp / "warn.pxcs",
                [str(self.src)],
                CreateOptions(
                    quiet=False,
                    progress=False,
                    search_window=99,
                    keyframe_interval=3,
                    overwrite=True,
                ),
            )
        output = buffer.getvalue()
        self.assertIn("--search", output)
        self.assertIn("--keyframe", output)
        self.assertIn("忽略", output)
        self.assertEqual(output.count("只在 --group-mode natural"), 1, "只该告警一次")

    def test_default_options_do_not_warn(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            create_archive(
                self.tmp / "nowarn.pxcs",
                [str(self.src)],
                CreateOptions(
                    quiet=False, progress=False, overwrite=True
                ),
            )
        self.assertNotIn("只在 --group-mode natural", buffer.getvalue())

    def test_existing_archive_in_input_is_skipped(self) -> None:
        """输入目录里已有的 .pxcs 不能再被当成输入（否则体积会翻倍）。"""
        # 先造一个"旧归档"放在输入目录里
        old_archive = self.src / "previous.pxcs"
        with contextlib.redirect_stdout(io.StringIO()):
            create_archive(
                old_archive,
                [str(self.src)],
                CreateOptions(quiet=True, progress=False, overwrite=True),
            )
        self.assertTrue(old_archive.is_file())

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stats = create_archive(
                self.tmp / "fresh.pxcs",
                [str(self.src)],
                CreateOptions(quiet=False, progress=False, overwrite=True),
            )
        self.assertEqual(stats.images, 4, "只应包含 4 张图片，不应把旧归档算进去")
        self.assertIn("已跳过 1 个 .pxcs", buffer.getvalue())

    def test_normal_mode_cli(self) -> None:
        """普通模式 `pxcs <目录>` 的完整路径能跑通（历史上崩溃的入口）。"""
        import pxcs

        previous = Path.cwd()
        work = self.tmp / "cwd"
        work.mkdir()
        buffer = io.StringIO()
        try:
            os.chdir(work)
            with contextlib.redirect_stdout(buffer):
                code = pxcs.main([str(self.src)])
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0, buffer.getvalue())
        produced = list(work.glob("*.pxcs"))
        self.assertEqual(len(produced), 1, f"普通模式应产出 1 个归档: {produced}")
        self.assertIn("[GROUP", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
