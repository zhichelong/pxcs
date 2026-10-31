"""CLI 帮助文本必须能正常渲染。

argparse 会对每个 help 字符串再做一次 %-格式化，任何裸 ``%`` 都会让
``--help`` 直接抛 ``ValueError: unsupported format character``。这类错误
只有在真正渲染帮助时才暴露，所以单独钉一个测试（曾经真的踩过：
"体积约大 15%"）。
"""
from __future__ import annotations

import argparse
import unittest

from pxcs import build_parser
from pxcs_core.settings import load_settings


class HelpRendersTests(unittest.TestCase):
    def test_main_help(self) -> None:
        parser = build_parser(load_settings())
        text = parser.format_help()
        self.assertIn("PXCS", text)

    def test_every_subcommand_help(self) -> None:
        parser = build_parser(load_settings())
        count = 0
        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in action.choices.items():
                with self.subTest(command=name):
                    self.assertTrue(sub.format_help().strip())
                    count += 1
        self.assertGreaterEqual(count, 4)


if __name__ == "__main__":
    unittest.main()
