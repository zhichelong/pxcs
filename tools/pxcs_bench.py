#!/usr/bin/env python3
"""PXCS 性能基线采集工具。

用法::

    python -m tools.pxcs_bench <图片目录或.pxcs> --repeat 1 --json [--password-file PATH]

输入是目录时完整跑一遍 create → list → verify → extract；输入是归档时
只跑 list → verify → extract。每一步单独计时，重复 N 次后取中位数，
并附带环境指纹（Python/numpy/Pillow/zlib/CPU/OS 等），便于跨机器比较。

约定：

- 所有写操作都发生在 ``tempfile.mkdtemp()`` 下，正常结束或异常都会清理；
- 吞吐量按输入原始字节计算（1 MB = 1024 * 1024 字节），与压缩后体积无关；
- 不修改输入归档；口令只从 ``--password-file`` 读取，不写入 argv 或日志。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import PIL

from pxc_core.errors import PxcError
from pxcs_core.archive import (
    CreateOptions,
    CreateStats,
    create_archive,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxcs_core.format import (
    CODEC_DUP,
    CODEC_PXC,
    CODEC_PXC_RESIDUAL,
    CODEC_RAW,
    CODEC_WEBP_LOSSLESS,
    CODEC_ZLIB_RESIDUAL,
    VERSION_STRING,
)

SCHEMA_VERSION = 1
_TEMP_PREFIX = "pxcs-bench-"
_BYTES_PER_MB = 1024 * 1024
_STEPS: tuple[str, ...] = ("create", "list", "verify", "extract")
_KEYFRAME_CODECS = frozenset({CODEC_PXC, CODEC_WEBP_LOSSLESS})
_RESIDUAL_CODECS = frozenset({CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL})


# ---------------------------------------------------------------- 环境指纹

def _distribution_version(name: str) -> Optional[str]:
    """查询已安装发行版的版本号；未安装或元数据损坏时返回 None。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_environment(*, include_hostname: bool = True) -> dict[str, Any]:
    """收集会影响基线数据的运行环境信息。"""
    environment: dict[str, Any] = {
        "pxcs_version": VERSION_STRING,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "zlib_compiled": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "machine": platform.machine(),
    }
    cryptography = _distribution_version("cryptography")
    if cryptography is not None:
        environment["cryptography"] = cryptography
    if include_hostname:
        environment["hostname"] = platform.node() or "unknown"
    return environment


# ---------------------------------------------------------------- 计时与统计

def _timed_call(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[float, Any]:
    """用 ``perf_counter`` 计时调用，返回 (秒, 返回值)。"""
    started = time.perf_counter()
    value = func(*args, **kwargs)
    return time.perf_counter() - started, value


def _quiet_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """吞掉被测函数自己打印的摘要/进度，保证 --json 的 stdout 干净。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _timed_quiet(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[float, Any]:
    """计时并屏蔽被测函数的标准输出。"""
    return _timed_call(_quiet_call, func, *args, **kwargs)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(list(values)))


def _throughput(input_bytes: int, seconds: float) -> float:
    """输入原始字节吞吐量，单位 MB/s（1 MB = 1024 * 1024 字节）。"""
    if seconds <= 0.0 or input_bytes <= 0:
        return 0.0
    return input_bytes / _BYTES_PER_MB / seconds


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _round_throughput(value: float) -> float:
    return round(float(value), 3)


# ---------------------------------------------------------------- 输入扫描

def _scan_directory(root: Path) -> tuple[int, int]:
    """递归统计目录里的文件数与总字节数（与 create 收集的输入口径一致）。"""
    files = 0
    total = 0
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as iterator:
            for entry in iterator:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    total += entry.stat(follow_symlinks=False).st_size
    return files, total


# ---------------------------------------------------------------- 条目统计

def _counts_from_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    """从 ``list --json`` 的 entries 统计各类编码数量（归档模式用）。"""
    counts = {
        "entries": len(entries),
        "images": 0,
        "files": 0,
        "keys": 0,
        "deltas": 0,
        "lossy_deltas": 0,
        "raw_images": 0,
        "dups": 0,
    }
    for entry in entries:
        codec = str(entry.get("codec", ""))
        is_image = entry.get("kind") == "image"
        if is_image:
            counts["images"] += 1
        else:
            counts["files"] += 1
        if codec in _KEYFRAME_CODECS:
            counts["keys"] += 1
        elif codec in _RESIDUAL_CODECS:
            counts["deltas"] += 1
        elif codec == CODEC_RAW and is_image:
            counts["raw_images"] += 1
        if codec == CODEC_DUP:
            counts["dups"] += 1
        if entry.get("lossy"):
            counts["lossy_deltas"] += 1
    return counts


def _counts_from_stats(stats: CreateStats, entries: int) -> dict[str, Optional[int]]:
    """从 ``CreateStats`` 取创建阶段的真实计数（目录模式用）。"""
    return {
        "entries": entries,
        "images": int(stats.images),
        "files": int(stats.files),
        "keys": int(stats.keys),
        "deltas": int(stats.deltas),
        "lossy_deltas": int(stats.lossy_deltas),
        "raw_images": int(stats.raw_images),
        "dups": int(stats.dups),
        "groups": int(stats.groups),
    }


# ---------------------------------------------------------------- 结果结构

@dataclass
class BenchmarkResult:
    """一次性能基线的完整结果；``to_dict()`` 即 ``--json`` 的结构。"""

    mode: str
    target: str
    repeat: int
    environment: dict[str, Any]
    input_bytes: int
    input_files: int
    archive_bytes: int
    archive_path: Optional[str]
    entries: int
    images: int
    files: int
    keys: int
    deltas: int
    lossy_deltas: int
    raw_images: int
    dups: int
    groups: Optional[int]
    timings: dict[str, float]
    runs: list[dict[str, float]]
    verify_failed: int
    extract_failed: int
    temp_root: str

    @property
    def total_seconds(self) -> float:
        """各步骤中位耗时之和；不是墙钟时间，但可比较整体开销。"""
        return float(sum(self.timings.values()))

    def to_dict(self) -> dict[str, Any]:
        ratio = (self.archive_bytes / self.input_bytes) if self.input_bytes else None
        inverse = (self.input_bytes / self.archive_bytes) if self.archive_bytes else None
        return {
            "tool": {"name": "pxcs_bench", "schema": SCHEMA_VERSION},
            "mode": self.mode,
            "target": self.target,
            "repeat": self.repeat,
            "environment": dict(self.environment),
            "input": {
                "bytes": self.input_bytes,
                "files": self.input_files,
            },
            "archive": {
                "path": self.archive_path,
                "bytes": self.archive_bytes,
                "entries": self.entries,
                "images": self.images,
                "files": self.files,
                "keys": self.keys,
                "deltas": self.deltas,
                "lossy_deltas": self.lossy_deltas,
                "raw_images": self.raw_images,
                "dups": self.dups,
                "groups": self.groups,
                "compression_ratio": None if ratio is None else round(ratio, 6),
                "input_to_archive": None if inverse is None else round(inverse, 4),
            },
            "timings": {
                step: _round_seconds(value) for step, value in self.timings.items()
            },
            "throughput_mb_s": {
                step: _round_throughput(_throughput(self.input_bytes, value))
                for step, value in self.timings.items()
            },
            "total_seconds": _round_seconds(self.total_seconds),
            "total_throughput_mb_s": _round_throughput(
                _throughput(self.input_bytes, self.total_seconds)
            ),
            "runs": [
                {step: _round_seconds(value) for step, value in run.items()}
                for run in self.runs
            ],
            "checks": {
                "verify_failed": self.verify_failed,
                "extract_failed": self.extract_failed,
            },
        }


# ---------------------------------------------------------------- 两种模式

def _new_run_dir(temp_root: Path, run_index: int) -> Path:
    run_dir = temp_root / f"run-{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _benchmark_directory(
    target: Path,
    *,
    repeat: int,
    password: Optional[str],
    environment: dict[str, Any],
    temp_root: Path,
) -> BenchmarkResult:
    """目录模式：create → list → verify → extract，重复取中位数。"""
    input_files, input_bytes = _scan_directory(target)
    if input_files == 0:
        raise ValueError(f"输入目录里没有文件: {target}")

    samples: list[dict[str, float]] = []
    counts: dict[str, Optional[int]] = {}
    archive_bytes = 0
    verify_failed = 0
    extract_failed = 0

    for run_index in range(1, repeat + 1):
        run_dir = _new_run_dir(temp_root, run_index)
        archive_path = run_dir / "bench.pxcs"
        options = CreateOptions(
            quiet=True,
            progress=False,
            overwrite=True,
            password=password,
        )
        create_seconds, create_stats = _timed_quiet(
            create_archive, archive_path, [str(target)], options
        )
        list_seconds, list_text = _timed_quiet(
            list_entries, archive_path, as_json=True, password=password
        )
        listing = json.loads(list_text)
        verify_seconds, verify_stats = _timed_quiet(
            verify_archive, archive_path, password=password
        )
        extract_seconds, extract_stats = _timed_quiet(
            extract_archive,
            archive_path,
            run_dir / "extracted",
            quiet=True,
            password=password,
        )

        samples.append(
            {
                "create": create_seconds,
                "list": list_seconds,
                "verify": verify_seconds,
                "extract": extract_seconds,
            }
        )
        counts = _counts_from_stats(
            create_stats, len(listing.get("entries", []))
        )
        archive_bytes = int(create_stats.archive_bytes) or int(
            archive_path.stat().st_size
        )
        verify_failed = int(verify_stats.failed)
        extract_failed = int(extract_stats.failed)

        # 每次迭代后立即删除本轮产物，避免 repeat 较大时把临时盘写满。
        shutil.rmtree(run_dir, ignore_errors=True)

    timings = {
        step: _round_seconds(_median([run[step] for run in samples]))
        for step in _STEPS
    }
    return BenchmarkResult(
        mode="directory",
        target=str(target),
        repeat=repeat,
        environment=environment,
        input_bytes=input_bytes,
        input_files=input_files,
        archive_bytes=archive_bytes,
        archive_path=None,
        entries=int(counts.get("entries") or 0),
        images=int(counts.get("images") or 0),
        files=int(counts.get("files") or 0),
        keys=int(counts.get("keys") or 0),
        deltas=int(counts.get("deltas") or 0),
        lossy_deltas=int(counts.get("lossy_deltas") or 0),
        raw_images=int(counts.get("raw_images") or 0),
        dups=int(counts.get("dups") or 0),
        groups=None if counts.get("groups") is None else int(counts["groups"]),
        timings=timings,
        runs=samples,
        verify_failed=verify_failed,
        extract_failed=extract_failed,
        temp_root=str(temp_root),
    )


def _benchmark_archive(
    target: Path,
    *,
    repeat: int,
    password: Optional[str],
    environment: dict[str, Any],
    temp_root: Path,
) -> BenchmarkResult:
    """归档模式：只读 list / verify / extract，不创建也不改动归档。"""
    samples: list[dict[str, float]] = []
    counts: dict[str, int] = {}
    input_bytes = 0
    archive_bytes = 0
    verify_failed = 0
    extract_failed = 0

    for run_index in range(1, repeat + 1):
        run_dir = _new_run_dir(temp_root, run_index)
        list_seconds, list_text = _timed_quiet(
            list_entries, target, as_json=True, password=password
        )
        listing = json.loads(list_text)
        entries = listing.get("entries", [])
        counts = _counts_from_entries(entries)
        input_bytes = int(listing.get("original_size") or 0) or sum(
            int(entry.get("size", 0)) for entry in entries
        )
        archive_bytes = int(listing.get("physical_size") or 0)
        if archive_bytes <= 0:
            archive_bytes = int(listing.get("archive_size") or 0)
        if archive_bytes <= 0:
            archive_bytes = int(target.stat().st_size)

        verify_seconds, verify_stats = _timed_quiet(
            verify_archive, target, password=password
        )
        extract_seconds, extract_stats = _timed_quiet(
            extract_archive,
            target,
            run_dir / "extracted",
            quiet=True,
            password=password,
        )
        samples.append(
            {
                "list": list_seconds,
                "verify": verify_seconds,
                "extract": extract_seconds,
            }
        )
        verify_failed = int(verify_stats.failed)
        extract_failed = int(extract_stats.failed)
        shutil.rmtree(run_dir, ignore_errors=True)

    timings = {
        step: _round_seconds(_median([run[step] for run in samples]))
        for step in ("list", "verify", "extract")
    }
    return BenchmarkResult(
        mode="archive",
        target=str(target),
        repeat=repeat,
        environment=environment,
        input_bytes=input_bytes,
        input_files=int(counts.get("entries") or 0),
        archive_bytes=archive_bytes,
        archive_path=str(target),
        entries=int(counts.get("entries") or 0),
        images=int(counts.get("images") or 0),
        files=int(counts.get("files") or 0),
        keys=int(counts.get("keys") or 0),
        deltas=int(counts.get("deltas") or 0),
        lossy_deltas=int(counts.get("lossy_deltas") or 0),
        raw_images=int(counts.get("raw_images") or 0),
        dups=int(counts.get("dups") or 0),
        groups=None,
        timings=timings,
        runs=samples,
        verify_failed=verify_failed,
        extract_failed=extract_failed,
        temp_root=str(temp_root),
    )


def run_benchmark(
    target: Path | str,
    *,
    repeat: int = 1,
    password: Optional[str] = None,
    include_hostname: bool = True,
) -> BenchmarkResult:
    """采集一次性能基线。

    所有临时写入都在同一个 ``mkdtemp`` 目录内，函数返回前（含异常）
    统一清理。``password`` 只用于内存中的加密/解密，不落盘、不打印。
    """
    if repeat < 1:
        raise ValueError("repeat 必须 >= 1")
    target_path = Path(target).expanduser()
    if not target_path.exists():
        raise FileNotFoundError(f"输入不存在: {target_path}")

    environment = collect_environment(include_hostname=include_hostname)
    temp_root = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
    try:
        if target_path.is_dir():
            return _benchmark_directory(
                target_path,
                repeat=repeat,
                password=password,
                environment=environment,
                temp_root=temp_root,
            )
        return _benchmark_archive(
            target_path,
            repeat=repeat,
            password=password,
            environment=environment,
            temp_root=temp_root,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ---------------------------------------------------------------- 输出

def _mb_text(value: int) -> str:
    return f"{value / _BYTES_PER_MB:.2f} MiB"


def format_result(result: BenchmarkResult) -> str:
    """把结果渲染成人类可读的表格（时间保留 2 位小数）。"""
    env = result.environment
    if result.mode == "directory":
        mode_text = "目录（create → list → verify → extract）"
    else:
        mode_text = "归档（list → verify → extract）"
    lines: list[str] = [
        "PXCS 性能基线",
        f"目标      : {result.target}",
        f"模式      : {mode_text}，重复 {result.repeat} 次取中位数",
        f"环境      : Python {env.get('python')} ({env.get('python_implementation')})"
        f" | CPU {env.get('cpu_count')} 核 | {env.get('os')}",
        f"库        : numpy {env.get('numpy')} | Pillow {env.get('pillow')}"
        f" | zlib {env.get('zlib_runtime')} | PXCS {env.get('pxcs_version')}",
    ]
    if env.get("cryptography"):
        lines[-1] += f" | cryptography {env.get('cryptography')}"
    if env.get("hostname"):
        lines.append(f"主机      : {env.get('hostname')}")
    lines.append("")
    lines.append(
        f"输入      : {_mb_text(result.input_bytes)} / {result.input_files} 个文件"
    )
    lines.append(
        f"归档      : {_mb_text(result.archive_bytes)} / {result.entries} 条"
        f"（images={result.images}, files={result.files}）"
    )
    ratio = result.archive_bytes / result.input_bytes if result.input_bytes else 0.0
    inverse = result.input_bytes / result.archive_bytes if result.archive_bytes else 0.0
    lines.append(
        f"压缩率    : 归档/输入 = {ratio * 100:.2f}%"
        f"（输入/归档 = {inverse:.2f}x）"
    )
    group_text = f"groups={result.groups}, " if result.groups is not None else ""
    lines.append(
        f"编码分布  : {group_text}key={result.keys}, delta={result.deltas}"
        f"（有损 {result.lossy_deltas}）, raw={result.raw_images}, dup={result.dups}"
    )
    lines.append(
        f"检查      : verify 失败 {result.verify_failed}，"
        f"extract 失败 {result.extract_failed}"
    )
    lines.append("")
    lines.append(f"{'步骤':<8}{'中位耗时(秒)':>16}{'吞吐量(MB/s)':>18}")
    lines.append("-" * 42)
    for step in _STEPS:
        if step not in result.timings:
            continue
        seconds = result.timings[step]
        lines.append(
            f"{step:<8}{seconds:>16.2f}"
            f"{_throughput(result.input_bytes, seconds):>18.2f}"
        )
    lines.append("-" * 42)
    lines.append(
        f"{'合计':<8}{result.total_seconds:>16.2f}"
        f"{_throughput(result.input_bytes, result.total_seconds):>18.2f}"
    )
    lines.append("")
    lines.append("吞吐量 = 输入原始字节 / 耗时（1 MB = 1024 * 1024 字节）。")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if not (1 <= number <= 100):
        raise argparse.ArgumentTypeError("必须在 1..100 之间")
    return number


def _read_password_file(path: str) -> str:
    """读取口令文件第一行（只去掉行尾换行，保留首尾空格）。"""
    file_path = Path(path).expanduser()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PxcError(f"读取口令文件失败: {file_path}: {exc}") from exc
    raw = raw.lstrip("\ufeff")
    lines = raw.splitlines()
    if not lines:
        raise PxcError(f"口令文件为空: {file_path}")
    password = lines[0]
    if not password:
        raise PxcError(f"口令文件第一行为空: {file_path}")
    return password


def _ensure_utf8_stdio() -> None:
    """把控制台输出统一成 UTF-8，避免 Windows 上中文标签乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m tools.pxcs_bench",
        description=(
            "PXCS 性能基线采集：目录模式跑 create/list/verify/extract，"
            "归档模式只跑 list/verify/extract；重复 N 次取中位数。"
        ),
    )
    parser.add_argument("target", metavar="PATH", help="图片目录或 .pxcs 归档")
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        metavar="N",
        help="重复次数，取各步骤中位数（默认 1）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出机器可读 JSON（便于前后对比）",
    )
    parser.add_argument(
        "--password-file",
        default=None,
        metavar="PATH",
        help="口令文件，第一行作为口令；仅内存使用，不写入输出",
    )
    parser.add_argument(
        "--no-hostname",
        action="store_true",
        help="环境指纹中不包含机器名",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口，返回进程退出码。"""
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        password = (
            _read_password_file(args.password_file)
            if args.password_file
            else None
        )
        result = run_benchmark(
            args.target,
            repeat=args.repeat,
            password=password,
            include_hostname=not args.no_hostname,
        )
    except (PxcError, OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
