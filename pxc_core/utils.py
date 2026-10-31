"""通用工具：哈希、CRC、体积格式化、原子写入、输出命名。"""
from __future__ import annotations

import hashlib
import os
import re
import zlib
from pathlib import Path

from .errors import PxcError


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> bytes:
    """流式计算文件 SHA-256，避免一次读入大文件。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.digest()


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def human_size(num: int) -> str:
    """把字节数格式化为便于阅读的字符串（二进制单位）。"""
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num} B"


def ratio_text(part: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{part / total * 100.0:.2f}%"


def safe_label(name: str, fallback: str = "root") -> str:
    """把目录名转换为安全的输出子目录名。"""
    label = name.strip()
    if not label:
        return fallback
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label)
    label = label.rstrip(" .")
    return label or fallback


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """先写同目录临时文件再 os.replace，避免中途失败留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def unique_target(target: Path, planned: set[Path], overwrite: bool) -> Path:
    """确定性地解决输出名冲突。

    - target 未被规划且（不存在或允许覆盖）时直接使用；
    - 否则依次尝试 name__2.pxc、name__3.pxc ...
    """
    target = target.resolve() if target.is_absolute() else target
    key = _norm_path(target)

    def available(p: Path) -> bool:
        if _norm_path(p) in planned:
            return False
        if p.exists() and not overwrite:
            return False
        return True

    if available(target):
        planned.add(key)
        return target

    stem = target.stem
    suffix = target.suffix
    for i in range(2, 10000):
        candidate = target.with_name(f"{stem}__{i}{suffix}")
        if available(candidate):
            planned.add(_norm_path(candidate))
            return candidate
    raise PxcError(f"无法为 {target} 找到可用的输出文件名")


def _norm_path(path: Path) -> Path:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return Path(os.path.normcase(str(resolved)))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
