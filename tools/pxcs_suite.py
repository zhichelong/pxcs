"""基线测试台：对真实数据集跑 create + verify，统计压缩率与按 codec 的体积构成。

存在意义：优化之前先有可复现的数字。没有基线，任何"变快了"都是感觉。

用法：
    python -m tools.pxcs_suite --list
    python -m tools.pxcs_suite --names set-a,set-b --out report.json
    python -m tools.pxcs_suite --names set-a --no-verify --keep

数据集清单放在外部 JSON 里（格式见 datasets.example.json），不进仓库：
    {"name": {"path": "...", "note": "..."}, ...}
查找顺序：--datasets PATH -> 环境变量 PXCS_DATASETS -> 当前目录 datasets.json
        -> 仓库根目录 datasets.json。
输出目录默认是系统临时目录；每个数据集跑完即删归档，除非 --keep。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):  # 允许 python tools/pxcs_suite.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pxcs_core.archive import (  # noqa: E402
    CreateOptions,
    create_archive,
    verify_archive,
)
from pxcs_core.container import ArchiveReader  # noqa: E402

# name -> (path, 备注)：从外部清单读，仓库里只放 datasets.example.json
DATASETS_ENV = "PXCS_DATASETS"
DATASETS_FILE = "datasets.json"


def load_datasets(explicit: str | None = None) -> dict[str, tuple[str, str]]:
    """读取数据集清单；找不到或格式不对时返回空表，不抛异常。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(DATASETS_ENV):
        candidates.append(Path(os.environ[DATASETS_ENV]))
    candidates.append(Path.cwd() / DATASETS_FILE)
    candidates.append(Path(__file__).resolve().parent.parent / DATASETS_FILE)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"数据集清单读取失败 {path}: {exc}", file=sys.stderr)
            return {}
        items = data.get("datasets") if isinstance(data, dict) else None
        if not isinstance(items, dict):
            items = data
        out: dict[str, tuple[str, str]] = {}
        for name, item in items.items():
            if isinstance(item, str):
                out[str(name)] = (item, "")
            elif isinstance(item, dict):
                out[str(name)] = (str(item.get("path", "")), str(item.get("note", "")))
        return out
    return {}


def _codec_breakdown(archive: Path) -> dict[str, dict[str, int]]:
    """按 codec 汇总占用字节。stored 用明文长度，避免受加密头影响。"""
    out: dict[str, dict[str, int]] = {}
    with ArchiveReader(archive) as reader:
        for entry in reader.entries:
            slot = out.setdefault(
                entry.codec, {"count": 0, "stored": 0, "source": 0}
            )
            slot["count"] += 1
            slot["stored"] += int(entry.plain_length or entry.length or 0)
            slot["source"] += int(entry.size or 0)
    return out


def _reason_breakdown(archive: Path) -> dict[str, dict[str, int]]:
    """按 reason 汇总：回答"这张图为什么没被压下去"。"""
    out: dict[str, dict[str, int]] = {}
    with ArchiveReader(archive) as reader:
        for entry in reader.entries:
            reason = (entry.reason or entry.codec or "?").split(":")[0].strip()
            slot = out.setdefault(reason, {"count": 0, "stored": 0, "source": 0})
            slot["count"] += 1
            slot["stored"] += int(entry.plain_length or entry.length or 0)
            slot["source"] += int(entry.size or 0)
    return out


def _peak_working_set_mb() -> float:
    """进程生命周期内的峰值工作集（Windows）。多次运行时只会更大。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        # 必须显式声明 argtypes：HANDLE 是 64 位，按默认 int 传会被截断。
        func = ctypes.windll.psapi.GetProcessMemoryInfo
        func.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PMC),
            wintypes.DWORD,
        ]
        func.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if func(handle, ctypes.byref(pmc), ctypes.sizeof(PMC)):
            return round(pmc.PeakWorkingSetSize / 1048576, 1)
    except Exception:
        pass
    return 0.0

def run_one(
    name: str,
    path: str,
    work: Path,
    *,
    verify: bool = True,
    keep: bool = False,
    options: Optional[CreateOptions] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "path": path, "ok": False}
    src = Path(path)
    if not src.exists():
        record["error"] = "路径不存在"
        return record

    work.mkdir(parents=True, exist_ok=True)
    archive = work / f"{name}.pxcs"
    for stale in work.glob(f"{name}.pxcs*"):
        stale.unlink(missing_ok=True)

    opts = options or CreateOptions(quiet=True, progress=False)
    started = time.perf_counter()
    try:
        stats = create_archive(archive, [str(src)], opts)
        record["create_seconds"] = round(time.perf_counter() - started, 3)
    except Exception as exc:  # noqa: BLE001 - 测试台要记录而不是抛
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
        if not keep:
            for stale in work.glob(f"{name}.pxcs*"):
                stale.unlink(missing_ok=True)
        return record

    physical = sum(p.stat().st_size for p in work.glob(f"{name}.pxcs*"))
    record.update(
        {
            "ok": True,
            "archive_bytes": archive.stat().st_size,
            "physical_bytes": physical,
            "source_bytes": stats.original_bytes,
            "ratio": round(archive.stat().st_size / stats.original_bytes, 6)
            if stats.original_bytes
            else 0.0,
            "images": stats.images,
            "files": stats.files,
            "groups": stats.groups,
            "keys": stats.keys,
            "deltas": stats.deltas,
            "lossy_deltas": stats.lossy_deltas,
            "dups": stats.dups,
            "raw_images": stats.raw_images,
            "skipped": stats.skipped,
            "report_elapsed": stats.elapsed,
            "by_codec": _codec_breakdown(archive),
            "by_reason": _reason_breakdown(archive),
            "peak_ws_mb": _peak_working_set_mb(),
        }
    )

    if verify:
        started = time.perf_counter()
        try:
            vstats = verify_archive(archive)
            record["verify_seconds"] = round(time.perf_counter() - started, 3)
            record["verify"] = {
                "checked": vstats.checked,
                "failed": vstats.failed,
                "lossy": getattr(vstats, "lossy", 0),
            }
        except Exception as exc:  # noqa: BLE001
            record["verify"] = {"error": f"{type(exc).__name__}: {exc}"}

    if not keep:
        for stale in work.glob(f"{name}.pxcs*"):
            stale.unlink(missing_ok=True)
        shutil.rmtree(archive, ignore_errors=True)
    return record


def _fmt_row(rec: dict[str, Any]) -> str:
    if not rec.get("ok"):
        return f"{rec['name']:<12} FAILED  {rec.get('error', '')[:80]}"
    mb = rec["source_bytes"] / (1024 * 1024)
    return (
        f"{rec['name']:<12} {mb:9.1f} MiB -> {rec['ratio'] * 100:6.2f}%  "
        f"{rec['create_seconds']:7.1f}s  "
        f"{rec['images']:5d} 图 {rec['deltas']:5d} 差分 {rec['raw_images']:5d} raw  "
        f"verify={rec.get('verify', {}).get('failed', '?')} 失败"
    )


def _codec_table(records: list[dict[str, Any]]) -> None:
    totals: dict[str, dict[str, int]] = {}
    for rec in records:
        if not rec.get("ok"):
            continue
        for codec, slot in rec["by_codec"].items():
            agg = totals.setdefault(codec, {"count": 0, "stored": 0, "source": 0})
            for key in ("count", "stored", "source"):
                agg[key] += slot[key]
    if not totals:
        return
    grand = sum(v["stored"] for v in totals.values()) or 1
    print("\n按 codec 的占用（stored = 归档中实际字节）")
    print(f"{'codec':<16}{'条目':>8}{'源大小':>12}{'占用':>12}{'占比':>8}{'压缩率':>9}")
    for codec, slot in sorted(totals.items(), key=lambda kv: -kv[1]["stored"]):
        ratio = slot["stored"] / slot["source"] if slot["source"] else 0.0
        print(
            f"{codec:<16}{slot['count']:>8}{slot['source'] / 1048576:>10.0f}M"
            f"{slot['stored'] / 1048576:>10.0f}M{slot['stored'] / grand * 100:>7.1f}%"
            f"{ratio * 100:>8.1f}%"
        )


def _reason_table(records: list[dict[str, Any]]) -> None:
    totals: dict[str, dict[str, int]] = {}
    for rec in records:
        if not rec.get("ok"):
            continue
        for reason, slot in rec.get("by_reason", {}).items():
            agg = totals.setdefault(reason, {"count": 0, "stored": 0, "source": 0})
            for key in ("count", "stored", "source"):
                agg[key] += slot[key]
    if not totals:
        return
    grand = sum(v["stored"] for v in totals.values()) or 1
    print("\n按 reason 的占用（未压缩的大头通常在这里）")
    print(f"{'reason':<32}{'条目':>8}{'源大小':>12}{'占用':>12}{'占比':>8}")
    for reason, slot in sorted(totals.items(), key=lambda kv: -kv[1]["stored"])[:12]:
        print(
            f"{reason:<32}{slot['count']:>8}"
            f"{slot['source'] / 1048576:>10.0f}M{slot['stored'] / 1048576:>10.0f}M"
            f"{slot['stored'] / grand * 100:>7.1f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pxcs-suite", description="PXCS 真实数据集基线测试台"
    )
    parser.add_argument("--list", action="store_true", help="列出数据集")
    parser.add_argument("--names", help="逗号分隔的数据集名，默认内置基线集合")
    parser.add_argument("--datasets", help="数据集清单 JSON 路径（默认自动查找）")
    parser.add_argument("--all", action="store_true", help="跑全部数据集")
    parser.add_argument("--out", help="报告 JSON 输出路径")
    parser.add_argument("--work", default="", help="归档临时目录（默认系统临时目录）")
    parser.add_argument("--keep", action="store_true", help="保留归档")
    parser.add_argument("--no-verify", action="store_true", help="跳过 verify")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--parallel-groups", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=0, help="0=用默认")
    parser.add_argument(
        "--residual-codec", default="", choices=["auto", "zlib", "pxc"],
        help="残差编码器；默认用 auto",
    )
    parser.add_argument("--no-self-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = load_datasets(args.datasets)
    if not datasets:
        print(
            "没有找到数据集清单：用 --datasets 指定，"
            "或把 datasets.json 放到当前目录",
            file=sys.stderr,
        )
        return 2
    if args.list:
        for name, (path, note) in datasets.items():
            exists = "OK " if Path(path).exists() else "缺失"
            print(f"{name:<12} {exists} {note:<18} {path}")
        return 0

    if args.all:
        names = list(datasets)
    elif args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = list(datasets)

    unknown = [n for n in names if n not in datasets]
    if unknown:
        print(f"未知数据集: {', '.join(unknown)}", file=sys.stderr)
        return 2

    extra: dict[str, Any] = {}
    if args.tile_size:
        extra["tile_size"] = args.tile_size
    if args.residual_codec:
        extra["residual_codec"] = args.residual_codec
    if args.no_self_check:
        extra["self_check"] = False
    options = CreateOptions(
        quiet=True, progress=False, threads=args.threads,
        parallel_groups=args.parallel_groups, **extra,
    )

    work = Path(args.work or (Path(tempfile.gettempdir()) / "pxcs-bench"))
    records: list[dict[str, Any]] = []
    for name in names:
        path, note = datasets[name]
        print(f"\n=== {name} ({note}) ===", flush=True)
        rec = run_one(
            name, path, work,
            verify=not args.no_verify, keep=args.keep, options=options,
        )
        rec["note"] = note
        records.append(rec)
        print(_fmt_row(rec), flush=True)

    _codec_table(records)
    _reason_table(records)

    if args.out:
        payload = {"options": {"threads": args.threads,
                               "parallel_groups": args.parallel_groups,
                               "tile_size": args.tile_size or None,
                               "residual_codec": args.residual_codec or "auto"},
                   "records": records}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告: {args.out}")

    failed = [r["name"] for r in records if not r.get("ok")]
    if failed:
        print(f"\n失败: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
