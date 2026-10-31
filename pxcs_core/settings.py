"""用户配置加载（config/config.json）。

配置文件按以下顺序查找，后者不会覆盖显式设置的前者：
1. PXCS_CONFIG 环境变量
2. 当前目录/config/config.json
3. exe 同目录/config/config.json（打包版）
4. 项目根目录/config/config.json（源码版）
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    # 界面语言：language/<code>.json；默认简体中文。
    "language": "chinese_simp",
    "progress": {
        "enabled": True,
        "details": False,
        "detail_lines": 3,
        "refresh_seconds": 0.2,
    },
    "create": {
        "group_mode": "resolution",
        "group_regex": r"\d+_p",
        "group_regex_fallback": "resolution",
        "delta_quant": 12,
        "delta_quant_max": 16,
        "residual_min_zero_fraction": 0.40,
        "residual_codec": "auto",
        "threads": 0,           # 0 = 用满所有核心
        "parallel_groups": 0,   # 0 = 与线程数一致
        "memory_limit_mb": 0,   # 0 = 按可用内存自动

        "show_groups": False,
        "volumes": False,
        "volume_size": 4 * 1024 ** 3,
        "decode_raw": False,
    },
    "crypto": {
        # auto = 有 Argon2id 就用 Argon2id，否则 scrypt
        "kdf": "auto",
        "encrypt_filenames": True,
        "kdf_memory_kib": 262144,
        "kdf_iterations": 3,
        "kdf_lanes": 4,
    },
    "normal_mode": {
        "name_parts": 3,
        "pause_on_finish": True,
        "show_groups": True,
    },
}


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("PXCS_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path.cwd() / "config" / "config.json")
    if getattr(sys, "frozen", False):
        paths.append(Path(sys.executable).resolve().parent / "config" / "config.json")
    paths.append(Path(__file__).resolve().parents[1] / "config" / "config.json")
    return paths


def _merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings() -> dict[str, Any]:
    data = copy.deepcopy(DEFAULT_SETTINGS)
    found = [path for path in _candidate_paths() if path.is_file()]
    # 候选顺序 = 优先级从高到低；从低往高合并，保证前面的配置覆盖后面的。
    for path in reversed(found):
        try:
            incoming = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(incoming, dict):
            data = _merge(data, incoming)
    progress = data.setdefault("progress", {})
    progress["enabled"] = bool(progress.get("enabled", True))
    progress["details"] = bool(progress.get("details", False))
    try:
        progress["detail_lines"] = max(0, int(progress.get("detail_lines", 3)))
    except (TypeError, ValueError):
        progress["detail_lines"] = 3
    try:
        progress["refresh_seconds"] = max(
            0.05, float(progress.get("refresh_seconds", 0.2))
        )
    except (TypeError, ValueError):
        progress["refresh_seconds"] = 0.2
    create = data.setdefault("create", {})
    if create.get("group_mode") not in {"resolution", "regex", "natural"}:
        create["group_mode"] = "resolution"
    create["group_regex"] = str(create.get("group_regex", r"\d+_p"))
    if create.get("group_regex_fallback") not in {"resolution", "singleton", "error"}:
        create["group_regex_fallback"] = "resolution"
    if create.get("residual_codec") not in {"auto", "zlib", "pxc"}:
        create["residual_codec"] = "auto"
    for key, default, lo, hi in (
        ("delta_quant", 12, 0, 255),
        ("delta_quant_max", 16, 0, 255),
        ("threads", 0, 0, 256),
        ("parallel_groups", 0, 0, 256),
        ("memory_limit_mb", 0, 0, 1024 * 1024),
    ):
        try:
            create[key] = max(lo, min(hi, int(create.get(key, default))))
        except (TypeError, ValueError):
            create[key] = default
    try:
        create["residual_min_zero_fraction"] = max(
            0.0, min(1.0, float(create.get("residual_min_zero_fraction", 0.40)))
        )
    except (TypeError, ValueError):
        create["residual_min_zero_fraction"] = 0.40
    data["language"] = str(data.get("language", "chinese_simp")).strip() or "chinese_simp"
    create["decode_raw"] = bool(create.get("decode_raw", False))
    create["volumes"] = bool(create.get("volumes", False))
    try:
        create["volume_size"] = max(1024, int(create.get("volume_size", 4 * 1024 ** 3)))
    except (TypeError, ValueError):
        create["volume_size"] = 4 * 1024 ** 3
    crypto = data.setdefault("crypto", {})
    if crypto.get("kdf") not in {"auto", "argon2id", "scrypt"}:
        crypto["kdf"] = "auto"
    crypto["encrypt_filenames"] = bool(crypto.get("encrypt_filenames", True))
    for key, default, lo, hi in (
        ("kdf_memory_kib", 262144, 8192, 2 * 1024 * 1024),
        ("kdf_iterations", 3, 1, 64),
        ("kdf_lanes", 4, 1, 16),
    ):
        try:
            crypto[key] = max(lo, min(hi, int(crypto.get(key, default))))
        except (TypeError, ValueError):
            crypto[key] = default
    normal = data.setdefault("normal_mode", {})
    try:
        normal["name_parts"] = max(1, min(20, int(normal.get("name_parts", 3))))
    except (TypeError, ValueError):
        normal["name_parts"] = 3
    normal["pause_on_finish"] = bool(normal.get("pause_on_finish", True))
    normal["show_groups"] = bool(normal.get("show_groups", True))
    return data
