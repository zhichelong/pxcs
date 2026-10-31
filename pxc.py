#!/usr/bin/env python3
"""PXC 0.2 命令行入口。

用法：
    python pxc.py encode -i <文件/目录...> -o <输出目录> [-r] [--overwrite]
    python pxc.py decode -i <PXC/目录...> -o <输出目录> [-r] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pxc_core import VERSION_STRING
from pxc_core.batch import run_decode, run_encode
from pxc_core.config import DEFAULT_TILE_SIZE, MAX_TILE_SIZE, MIN_TILE_SIZE
from pxc_core.errors import PxcError


def _ensure_utf8_stdio() -> None:
    """PyInstaller / 部分 Windows 控制台会回落到 cp936，统一成 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pxc.py",
        description="PXC 0.2 实验性像素无损图片压缩格式",
    )
    parser.add_argument("--version", action="version", version=f"PXC {VERSION_STRING}")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="把图片压缩为 .pxc")
    enc.add_argument("-i", "--input", nargs="+", required=True, help="图片文件或目录，可多个")
    enc.add_argument("-o", "--output", required=True, help="输出目录")
    enc.add_argument("-r", "--recursive", action="store_true", help="递归扫描输入目录")
    enc.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    enc.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"分块尺寸（{MIN_TILE_SIZE}..{MAX_TILE_SIZE}，默认 {DEFAULT_TILE_SIZE}）",
    )
    enc.add_argument(
        "--keep-original",
        action="store_true",
        help="在 PXC 中额外保存原始文件字节（供 --restore-original-file 使用）",
    )
    enc.add_argument(
        "--algorithm",
        choices=("auto", "pixels", "passthrough"),
        default="auto",
        help="auto=JPEG/WebP 直存原始码流，其他走像素压缩；pixels=强制像素模式；passthrough=强制直存",
    )
    enc.add_argument(
        "--zlib-level",
        type=int,
        choices=(1, 3, 6, 9),
        default=6,
        help="zlib 压缩级别：6=默认速度/体积平衡，9=更小但更慢",
    )
    enc.add_argument(
        "--threads",
        type=int,
        default=0,
        help="tile 编码线程数；0=自动（默认，最多 32）",
    )

    dec = sub.add_parser("decode", help="把 .pxc 恢复为图片")
    dec.add_argument("-i", "--input", nargs="+", required=True, help="PXC 文件或目录，可多个")
    dec.add_argument("-o", "--output", required=True, help="输出目录")
    dec.add_argument("-r", "--recursive", action="store_true", help="递归扫描输入目录")
    dec.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    dec.add_argument(
        "--restore-original-file",
        action="store_true",
        help="恢复编码时用 --keep-original 保存的原始文件字节",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "encode":
        if not (MIN_TILE_SIZE <= args.tile_size <= MAX_TILE_SIZE):
            parser.error(
                f"--tile-size 必须在 {MIN_TILE_SIZE}..{MAX_TILE_SIZE} 之间"
            )
        output = Path(args.output).expanduser()
        if output.exists() and not output.is_dir():
            print(f"[FAIL] 输出路径不是目录: {output}", file=sys.stderr)
            return 2
        print(f"PXC {VERSION_STRING} encode | tile_size={args.tile_size} | threads={args.threads}")
        stats = run_encode(
            inputs=args.input,
            output_dir=output,
            recursive=args.recursive,
            overwrite=args.overwrite,
            tile_size=args.tile_size,
            keep_original=args.keep_original,
            algorithm=args.algorithm,
            zlib_level=args.zlib_level,
            threads=args.threads,
        )
        return 1 if stats.failed else 0

    if args.command == "decode":
        output = Path(args.output).expanduser()
        if output.exists() and not output.is_dir():
            print(f"[FAIL] 输出路径不是目录: {output}", file=sys.stderr)
            return 2
        print(f"PXC {VERSION_STRING} decode")
        stats = run_decode(
            inputs=args.input,
            output_dir=output,
            recursive=args.recursive,
            overwrite=args.overwrite,
            restore_original_file=args.restore_original_file,
        )
        return 1 if stats.failed else 0

    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PxcError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(2)
