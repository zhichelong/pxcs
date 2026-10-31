"""用户配置加载的边界情况。

重点是 Windows 上很容易踩到的一点：记事本 / PowerShell 写出的 UTF-8 文件常常带
BOM，按 ``utf-8`` 读会让 ``json.loads`` 直接失败，配置被静默忽略——用户改了
``config/config.json`` 却毫无效果。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pxcs_core.settings import load_settings


class SettingsFileTests(unittest.TestCase):
    def _write(self, path: Path, payload: dict, *, bom: bool) -> Path:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        if bom:
            raw = b"\xef\xbb\xbf" + raw
        path.write_bytes(raw)
        return path

    def test_plain_config_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._write(Path(tmp) / "config.json", {"language": "english"}, bom=False)
            with mock.patch.dict(os.environ, {"PXCS_CONFIG": str(cfg)}):
                self.assertEqual(load_settings().get("language"), "english")

    def test_config_with_bom_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._write(Path(tmp) / "config.json", {"language": "english"}, bom=True)
            with mock.patch.dict(os.environ, {"PXCS_CONFIG": str(cfg)}):
                self.assertEqual(load_settings().get("language"), "english")

    def test_broken_config_falls_back_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text("{ not json", encoding="utf-8")
            with mock.patch.dict(os.environ, {"PXCS_CONFIG": str(cfg)}):
                with mock.patch("sys.stderr") as err:
                    settings = load_settings()
                self.assertEqual(settings.get("language"), "chinese_simp")
                self.assertTrue(err.write.called or err.flush.called)

    def test_language_file_with_bom_loads(self) -> None:
        from pxc_core import i18n

        with tempfile.TemporaryDirectory() as tmp:
            payload = {"_meta": {"language": "test", "name": "Test", "version": 1},
                       "cli": {"x": "hello {name}"}}
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            (Path(tmp) / "test.json").write_bytes(b"\xef\xbb\xbf" + raw)
            translator = i18n.Translator("test", Path(tmp))
            self.assertEqual(translator.t("cli.x", name="world"), "hello world")


if __name__ == "__main__":
    unittest.main()
