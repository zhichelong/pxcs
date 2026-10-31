"""轻量多语言文本加载与格式化。

设计目标：

- 面向用户的文本全部放在 ``language/<code>.json``；
- 键名使用稳定的点分命名（``errors.path.empty``），不在代码里用中文句子当键；
- 缺少键时逐级回退：当前语言 -> 默认语言 -> 键名本身，任何情况下都不抛异常；
- 支持 ``str.format`` 风格的命名参数（文件名、大小、路径、错误原因等）；
- 新增语言只需要放一个 JSON 文件，不需要改业务代码。

模块本身不依赖 pxc_core 的其他部分，PXCS / PXC 都可以直接使用。
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

DEFAULT_LANGUAGE = "chinese_simp"
LANGUAGE_DIR_ENV = "PXCS_LANGUAGE_DIR"
_LANGUAGE_NAME_RE = r"^[A-Za-z0-9_\-]+$"


class LanguageError(Exception):
    """语言文件缺失或格式错误。"""


class _SafeDict(dict):
    """format_map 用：缺少参数时保留 ``{name}`` 原样，而不是抛 KeyError。"""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return "{" + key + "}"


def _flatten_ok(obj: Any) -> bool:
    return isinstance(obj, dict)


def candidate_language_dirs() -> list[Path]:
    """语言目录查找顺序，与 config 的查找逻辑保持一致。"""
    dirs: list[Path] = []
    env = os.environ.get(LANGUAGE_DIR_ENV)
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path.cwd() / "language")
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent / "language")
    dirs.append(Path(__file__).resolve().parents[1] / "language")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


class Translator:
    """一个语言包的加载、查表和格式化。"""

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        directory: Optional[Path] = None,
    ) -> None:
        self.language = _normalize_language(language)
        self.directory = Path(directory) if directory is not None else None
        self._data: dict[str, Any] = {}
        self._fallback: dict[str, Any] = {}
        self.missing_keys: set[str] = set()
        self.missing_languages: set[str] = set()
        self.load()

    # ---------------------------------------------------------------- loading
    def _find_file(self, language: str) -> Optional[Path]:
        candidates: list[Path] = []
        if self.directory is not None:
            candidates.append(self.directory / f"{language}.json")
        candidates.extend(path / f"{language}.json" for path in candidate_language_dirs())
        for path in candidates:
            if path.is_file():
                return path
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise LanguageError(f"{path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LanguageError(f"{path}: {exc}") from exc
        if not isinstance(data, dict):
            raise LanguageError(f"{path}: 根节点必须是对象")
        return data

    def load(self) -> None:
        self._data = {}
        self._fallback = {}
        path = self._find_file(self.language)
        if path is None:
            self.missing_languages.add(self.language)
        else:
            self._data = self._read_json(path)

        if self.language != DEFAULT_LANGUAGE:
            fallback_path = self._find_file(DEFAULT_LANGUAGE)
            if fallback_path is None:
                self.missing_languages.add(DEFAULT_LANGUAGE)
            else:
                self._fallback = self._read_json(fallback_path)
        elif self._data:
            self._fallback = self._data

    # --------------------------------------------------------------- lookup
    @staticmethod
    def _walk(data: dict[str, Any], key: str) -> Optional[str]:
        node: Any = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, str) else None

    def get(self, key: str) -> Optional[str]:
        text = self._walk(self._data, key)
        if text is not None:
            return text
        text = self._walk(self._fallback, key)
        if text is not None:
            return text
        self.missing_keys.add(key)
        return None

    def t(self, key: str, /, **kwargs: Any) -> str:
        """取文本并格式化；找不到时返回键名，绝不抛异常。

        ``key`` 是**位置限定**参数（``/``）：这样格式化参数里可以出现名为
        ``key`` 的占位符（如 ``create.group_line`` 的 ``key={key}``），
        不会与键名参数撞名。
        """
        text = self.get(key)
        if text is None:
            return key
        if not kwargs:
            return text
        try:
            return text.format_map(_SafeDict(kwargs))
        except (ValueError, IndexError):
            # 模板里有坏花括号时退回原文，避免因为一条文案让整个操作失败。
            return text

    # ---------------------------------------------------------------- misc
    def info(self) -> dict[str, Any]:
        meta = self._data.get("_meta")
        return meta if isinstance(meta, dict) else {}

    def available(self) -> list[str]:
        names: set[str] = set()
        for directory in candidate_language_dirs():
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                names.add(path.stem)
        if self.directory is not None and self.directory.is_dir():
            for path in self.directory.glob("*.json"):
                names.add(path.stem)
        return sorted(names)


def _normalize_language(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_LANGUAGE
    for suffix in (".json",):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    if "/" in text or "\\" in text or text.startswith("."):
        # 只接受语言代码，防止配置里写路径导致任意文件读取。
        return DEFAULT_LANGUAGE
    return text


_TRANSLATOR: Optional[Translator] = None
_LOCK = threading.Lock()


def configure(
    language: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Translator:
    """重新加载语言包；CLI 启动时调用一次。"""
    global _TRANSLATOR
    with _LOCK:
        _TRANSLATOR = Translator(language or DEFAULT_LANGUAGE, directory)
        return _TRANSLATOR


def get_translator() -> Translator:
    global _TRANSLATOR
    with _LOCK:
        if _TRANSLATOR is None:
            _TRANSLATOR = Translator(DEFAULT_LANGUAGE)
        return _TRANSLATOR


def t(key: str, /, **kwargs: Any) -> str:
    """模块级快捷方式，等价于 ``get_translator().t(key, **kwargs)``。

    ``key`` 为位置限定参数，避免与格式化占位符 ``{key}`` 冲突。
    """
    return get_translator().t(key, **kwargs)


def reset_for_tests() -> None:
    """测试用：丢弃当前语言包，下次访问时重新加载。"""
    global _TRANSLATOR
    with _LOCK:
        _TRANSLATOR = None


def ensure_stdio_utf8() -> None:
    """把 stdout/stderr 统一为 UTF-8（errors=replace）。

    Windows 控制台与 CI 的默认编码可能是 cp1252 / cp936，直接 print 中文文案会抛
    ``UnicodeEncodeError``。CLI 入口早就修过这一点，但库被直接调用（测试、脚本）
    时没人负责，于是在非中文的 Windows 上会崩。这里做成幂等的，可以随便调。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
