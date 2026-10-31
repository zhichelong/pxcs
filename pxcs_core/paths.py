"""路径收集、自然排序与安全提取。

归档中的路径永远是 POSIX 风格的相对路径；提取时必须拒绝：
绝对路径、盘符、`..`、NUL/控制字符、Windows 保留名、尾随点/空格。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from pxc_core.errors import PxcError
from pxc_core.i18n import t
from pxc_core.utils import safe_label


class PxcsPathError(PxcError):
    """归档路径不安全或不合法。"""


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(value: str) -> tuple:
    """001 < 002 < 010，而不是字符串排序。"""
    parts = _NATURAL_RE.split(value.casefold())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _looks_absolute(value: str) -> bool:
    if value.startswith(("/", "\\")):
        return True
    if re.match(r"^[A-Za-z]:", value):
        return True
    return False


def normalize_rel_path(value: str) -> str:
    """把任意用户/归档路径规范成安全的 POSIX 相对路径。"""
    if not value or not value.strip():
        raise PxcsPathError(t("paths.empty"))
    if "\x00" in value:
        raise PxcsPathError(t("paths.nul"))
    if len(value.encode("utf-8", "surrogateescape")) > 4096:
        raise PxcsPathError(t("paths.too_long"))

    value = value.replace("\\", "/")
    if _looks_absolute(value):
        raise PxcsPathError(t("paths.absolute", value=repr(value)))

    parts: list[str] = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise PxcsPathError(t("paths.parent", value=repr(value)))
        if any(ord(ch) < 32 for ch in part):
            raise PxcsPathError(t("paths.control", value=repr(value)))
        if any(ch in '<>"|?*' for ch in part):
            raise PxcsPathError(t("paths.windows_chars", value=repr(value)))
        if ":" in part:
            raise PxcsPathError(t("paths.colon", value=repr(value)))
        if part.rstrip(" .") != part:
            raise PxcsPathError(t("paths.trailing", value=repr(value)))
        base = part.split(".", 1)[0].upper()
        if base in _WINDOWS_RESERVED:
            raise PxcsPathError(t("paths.reserved", value=repr(value)))
        if len(part.encode("utf-8", "surrogateescape")) > 255:
            raise PxcsPathError(t("paths.component_long", value=repr(value)))
        parts.append(part)

    if not parts:
        raise PxcsPathError(t("paths.empty_after", value=repr(value)))
    return "/".join(parts)


def safe_join(root: Path, rel: str) -> Path:
    """把归档相对路径安全地拼到输出目录下。"""
    rel_norm = normalize_rel_path(rel)
    root_abs = root.resolve()
    target = (root_abs / Path(*PurePosixPath(rel_norm).parts)).resolve()
    if target != root_abs and root_abs not in target.parents:
        raise PxcsPathError(t("paths.escape", value=repr(rel)))
    return target


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel: str


def _dir_prefix(name: str) -> str:
    return safe_label(name, "root")


def collect_source_files(
    inputs: Iterable[str],
    recursive: bool = True,
    exclude: Optional[Path] = None,
) -> list[SourceFile]:
    """收集输入文件，生成稳定的相对路径。

    - 只有一个目录输入：目录内容放在归档根；
    - 多个输入：目录用目录名做前缀，单文件用文件名；
    - 输入文件/目录不存在、重复相对路径时报错。
    """
    paths = [Path(v).expanduser() for v in inputs]
    if not paths:
        raise PxcsPathError(t("paths.no_input"))

    existing = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise PxcsPathError(t("paths.input_missing", path=missing[0]))

    single_dir = len(existing) == 1 and existing[0].is_dir()
    result: list[SourceFile] = []
    seen_rel: dict[str, Path] = {}
    exclude_norm = os.path.normcase(str(exclude.resolve())) if exclude else None

    def add(path: Path, rel: str) -> None:
        if exclude_norm and os.path.normcase(str(path.resolve())) == exclude_norm:
            return
        rel_norm = normalize_rel_path(rel)
        key = rel_norm.casefold()  # Windows 大小写不敏感，避免解压冲突
        if key in seen_rel:
            raise PxcsPathError(
                t(
                    "paths.conflict",
                    path=repr(rel_norm),
                    first=seen_rel[key],
                    second=path,
                )
            )
        seen_rel[key] = path
        result.append(SourceFile(path=path, rel=rel_norm))

    for input_path in existing:
        if input_path.is_file():
            add(input_path, input_path.name)
            continue
        if not input_path.is_dir():
            raise PxcsPathError(t("paths.not_file_or_dir", path=input_path))

        prefix = "" if single_dir else _dir_prefix(input_path.name) + "/"
        pattern = input_path.rglob("*") if recursive else input_path.glob("*")
        files = [p for p in pattern if p.is_file()]
        for path in sorted(files, key=lambda p: natural_key(str(p.relative_to(input_path)))):
            try:
                rel_local = path.relative_to(input_path).as_posix()
            except ValueError:
                rel_local = path.name
            add(path, prefix + rel_local)

    if not result:
        raise PxcsPathError(t("paths.no_files"))
    result.sort(key=lambda item: natural_key(item.rel))
    return result


def guess_is_windows() -> bool:
    return sys.platform.startswith("win")


def ensure_long_path(path: Path) -> Path:
    """Windows 长路径前缀；失败时原样返回。"""
    if not guess_is_windows():
        return path
    try:
        text = str(path)
        if text.startswith("\\\\?\\"):
            return path
        if len(text) >= 240:
            return Path("\\\\?\\" + os.path.abspath(text))
    except OSError:
        pass
    return path
