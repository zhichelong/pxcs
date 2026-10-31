"""多语言测试：加载、切换、回退、格式化参数、CLI 与错误信息。"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pxc_core import i18n
from pxc_core.i18n import (
    DEFAULT_LANGUAGE,
    Translator,
    configure,
    reset_for_tests,
    t,
)

import pxcs
from pxcs_core.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_DIR = PROJECT_ROOT / "language"


class TranslatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-i18n-"))
        self.translator = Translator(DEFAULT_LANGUAGE, LANGUAGE_DIR)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        i18n.reset_for_tests()

    # ------------------------------------------------------------ 正常
    def test_default_language_loads(self) -> None:
        self.assertEqual(self.translator.language, "chinese_simp")
        self.assertEqual(self.translator.missing_languages, set())
        self.assertIn("归档", self.translator.t("cli.extract_help"))
        self.assertIn("简体中文", self.translator.info().get("name", ""))

    def test_format_arguments(self) -> None:
        text = self.translator.t(
            "list.header",
            version="1.3",
            entries=3,
            original="1 KiB",
            archive="900 B",
            ratio="87.89%",
        )
        self.assertIn("PXCS 1.3", text)
        self.assertIn("3 条目", text)
        self.assertIn("87.89%", text)

    def test_missing_format_argument_is_kept(self) -> None:
        text = self.translator.t("list.header", version="1.3")
        self.assertIn("{entries}", text)  # 保留占位符而不是抛异常

    def test_unknown_key_returns_key(self) -> None:
        self.assertEqual(self.translator.t("no.such.key"), "no.such.key")
        self.assertIn("no.such.key", self.translator.missing_keys)

    # ------------------------------------------------------------ 回退
    def test_partial_language_falls_back(self) -> None:
        path = self.tmp / "english_test.json"
        path.write_text(
            json.dumps(
                {"_meta": {"name": "Test"}, "cli": {"fail": "[FAIL-EN] {reason}"}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        translator = Translator("english_test", self.tmp)
        self.assertEqual(translator.t("cli.fail", reason="x"), "[FAIL-EN] x")
        # 该语言没翻译的键回退到默认语言
        self.assertIn("归档", translator.t("cli.extract_help"))
        self.assertEqual(translator.missing_languages, set())

    def test_missing_language_file_falls_back_to_default(self) -> None:
        translator = Translator("does_not_exist", LANGUAGE_DIR)
        self.assertIn("does_not_exist", translator.missing_languages)
        self.assertIn("归档", translator.t("cli.extract_help"))

    def test_configure_switches_language(self) -> None:
        path = self.tmp / "fake_lang.json"
        path.write_text(
            json.dumps({"cli": {"create_help": "MAKE"}}, ensure_ascii=False),
            encoding="utf-8",
        )
        i18n.configure("fake_lang", self.tmp)
        self.assertEqual(t("cli.create_help"), "MAKE")
        i18n.configure(DEFAULT_LANGUAGE, LANGUAGE_DIR)
        self.assertNotEqual(t("cli.create_help"), "MAKE")


class CliLanguageTests(unittest.TestCase):
    """通过配置文件切换语言，验证 CLI 帮助与错误信息都走语言文件。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-cli-i18n-"))
        self.lang_dir = self.tmp / "language"
        self.lang_dir.mkdir()
        (self.lang_dir / "english_test.json").write_text(
            json.dumps(
                {
                    "_meta": {"name": "Test English"},
                    "cli": {
                        "fail": "[FAIL-EN] {reason}",
                        "create_help": "create an archive",
                        "no_args_hint": "EN hint {version}",
                    },
                    "volume": {"not_found": "EN not found: {path}"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.config = self.tmp / "config.json"
        self.config.write_text(
            json.dumps({"language": "english_test"}), encoding="utf-8"
        )
        self._env = {
            "PXCS_CONFIG": os.environ.get("PXCS_CONFIG"),
            "PXCS_LANGUAGE_DIR": os.environ.get("PXCS_LANGUAGE_DIR"),
        }
        os.environ["PXCS_CONFIG"] = str(self.config)
        os.environ["PXCS_LANGUAGE_DIR"] = str(self.lang_dir)
        i18n.reset_for_tests()

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        i18n.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_config_switches_language(self) -> None:
        self.assertEqual(load_settings()["language"], "english_test")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(pxcs, "_interactive", return_value=False):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = pxcs.main(["list", str(self.tmp / "missing.pxcs")])
        self.assertEqual(code, 1)
        self.assertIn("EN not found", err.getvalue())
        self.assertIn("[FAIL-EN]", err.getvalue())

    def test_cli_help_uses_language_file(self) -> None:
        i18n.configure("english_test", self.lang_dir)
        parser = pxcs.build_parser(load_settings())
        help_text = parser.format_help()
        self.assertIn("create an archive", help_text)
        self.assertNotIn("cli.create_help", help_text)
        self.assertNotIn("cli.description", help_text)

    def test_help_command_uses_configured_language(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                pxcs.main(["--help"])
        text = out.getvalue()
        self.assertIn("create an archive", text)
        self.assertNotIn("cli.create_help", text)

    def test_language_option_overrides_config(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                pxcs.main(["--language", "chinese_simp", "--help"])
        text = out.getvalue()
        self.assertIn("创建 .pxcs 归档", text)


class LocalizedInfoTests(unittest.TestCase):
    """`info` 的输出必须和其它命令一样走语言文件。"""

    def test_info_output_is_localized(self) -> None:
        from pxcs_core.archive import archive_info

        reset_for_tests()
        configure(language="chinese_simp")
        fixture = Path(__file__).resolve().parent / "fixtures" / "legacy_v1_2.pxcs"
        self.assertTrue(fixture.is_file(), fixture)
        text = archive_info(fixture)
        self.assertIn("条目", text)
        self.assertIn("归档体积", text)
        for english in ("entries:", "keyframes:", "original:", "max_depth:"):
            self.assertNotIn(english, text, f"info 仍输出未本地化的 {english}")


class LanguageFileConsistencyTests(unittest.TestCase):
    """代码里用到的键必须在语言文件里存在，否则界面会出现裸键名。"""

    def test_all_used_keys_defined(self) -> None:
        used = _collect_keys(PROJECT_ROOT)
        data = json.loads(
            (LANGUAGE_DIR / "chinese_simp.json").read_text(encoding="utf-8")
        )
        defined = set(_flatten(data))
        missing = sorted(used - defined)
        self.assertEqual(missing, [], f"语言文件缺少这些键: {missing}")

    def test_all_placeholders_are_supplied(self) -> None:
        """每个 t("键", ...) 调用点都必须提供该文案需要的全部占位符。

        这条检查来自一个真实崩溃：`create.group_line` 的文案里有 `{key}`，
        调用点传了 `key=...`，而 `t(key, **kwargs)` 的形参也叫 `key`，
        于是 `TypeError: t() got multiple values for argument 'key'`。
        修复是把键名改成位置限定参数；本测试确保没有第二个同类疏漏
        （占位符没被提供时不会崩，但界面会直接显示 `{key}` 这种裸占位符）。
        """
        data = json.loads(
            (LANGUAGE_DIR / "chinese_simp.json").read_text(encoding="utf-8")
        )
        strings = _flatten(data)
        placeholder = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")
        problems: list[str] = []
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            if any(
                part in {"tests", "tools", "__pycache__", "build", "dist"}
                for part in path.relative_to(PROJECT_ROOT).parts
            ):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name != "t" or not node.args:
                    continue
                first = node.args[0]
                if not isinstance(first, ast.Constant) or not isinstance(
                    first.value, str
                ):
                    continue
                key = first.value
                if "." not in key or key not in strings:
                    continue
                needed = set(placeholder.findall(strings[key]))
                provided = {kw.arg for kw in node.keywords if kw.arg}
                gap = needed - provided
                if gap:
                    problems.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                        f"{key} 缺少占位符 {sorted(gap)}"
                    )
        self.assertEqual(problems, [], "这些调用点没有提供全部占位符:\n" + "\n".join(problems))

    def test_resilience_keys_defined(self) -> None:
        """容错与自检相关的新键必须存在，否则用户看不到跳过原因。"""
        from pxc_core.i18n import Translator

        translator = Translator(DEFAULT_LANGUAGE, LANGUAGE_DIR)
        for key in (
            "create.skipped_warning",
            "create.skipped_detail",
            "create.skipped_more",
            "create.skipped_line",
            "create.self_check_failed",
            "create.option_ignored",
            "cli.create_verbose_help",
            "cli.no_self_check_help",
        ):
            self.assertIsNotNone(translator.get(key), f"缺少语言键 {key}")

    def test_language_file_has_no_duplicate_semantics(self) -> None:
        data = json.loads(
            (LANGUAGE_DIR / "chinese_simp.json").read_text(encoding="utf-8")
        )
        flat = _flatten(data)
        # 键名必须都是 ASCII 点分形式，不能直接用中文句子当键
        for key in flat:
            self.assertRegex(key, r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
        self.assertTrue(all(isinstance(v, str) and v for v in flat.values()))


def _collect_keys(root: Path) -> set[str]:
    keys: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.add(node.value)
        elif isinstance(node, ast.IfExp):
            collect(node.body)
            collect(node.orelse)

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path) or "tests" in path.parts:
            continue
        if path.name.startswith("_") or path.name.startswith("scan_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "t"
                and node.args
            ):
                collect(node.args[0])
    return keys


def _flatten(data: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in data.items():
        if key == "_meta":
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


if __name__ == "__main__":
    unittest.main()
