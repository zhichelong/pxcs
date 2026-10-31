#!/usr/bin/env python3
"""PXCS 1.3 命令行入口。

PXCS 面向“一批相似图片”的归档：图片在集合内部做跨图差分，
非图片文件按普通文件打包；提取时按原始格式恢复，可选择 JPEG/WebP 质量。
1.3 新增分卷与 AES-256-GCM 加密；两者可以同时使用。

所有面向用户的文本都来自 ``language/<code>.json``。
"""
from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from pxc_core.errors import PxcError
from pxc_core.i18n import configure as configure_i18n
from pxc_core.i18n import t
from pxc_core.utils import safe_label

from pxcs_core.archive import (
    CreateOptions,
    archive_info,
    create_archive,
    default_output_dir,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxcs_core.container import PxcsSelfCheckError
from pxcs_core.crypto import PxcAuthenticationError, PxcPasswordRequired
from pxcs_core.format import VERSION_STRING
from pxcs_core.settings import load_settings
from pxcs_core.volumes import DEFAULT_VOLUME_SIZE, MIN_VOLUME_SIZE

# PXCS 自己的上限是 300M 像素；把 Pillow 的默认 89M 警告阈值抬高，
# 避免 90–300MP 的插画每次都在 stderr 打印 DecompressionBombWarning。
from PIL import Image as _PILImage

_PILImage.MAX_IMAGE_PIXELS = 300_000_000

_SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtT]?[bB]?)\s*$")
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024 ** 2,
    "mb": 1024 ** 2,
    "g": 1024 ** 3,
    "gb": 1024 ** 3,
    "t": 1024 ** 4,
    "tb": 1024 ** 4,
}


def _ensure_utf8_stdio() -> None:
    """PyInstaller 在部分 Windows 控制台上会回落到 cp936，统一成 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


# ---------------------------------------------------------------- 参数解析

def _parse_quality(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"auto", "automatic"}:
        return 95
    try:
        q = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(t("cli.quality_invalid")) from exc
    if not (1 <= q <= 100):
        raise argparse.ArgumentTypeError(t("cli.quality_range"))
    return q


def parse_size(value: str) -> int:
    """把 ``2G`` / ``512M`` / ``734003200`` 解析成字节数（二进制单位）。"""
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError(t("cli.size_invalid", value=value))
    match = _SIZE_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(t("cli.size_invalid", value=value))
    unit = match.group("unit").lower()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(t("cli.size_invalid", value=value))
    number = float(match.group("value")) * _SIZE_UNITS[unit]
    size = int(number)
    if size < MIN_VOLUME_SIZE:
        raise argparse.ArgumentTypeError(
            t("volume.too_small", value=size, minimum=MIN_VOLUME_SIZE)
        )
    return size


def _size_text(value: int) -> str:
    for unit, factor in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if value >= factor:
            return f"{value / factor:.2f} {unit}"
    return f"{value} B"


def _add_password_input(
    parser: argparse.ArgumentParser, *, create_mode: bool
) -> None:
    group = parser.add_argument_group(
        t("cli.password_group") if create_mode else t("cli.password_group_read")
    )
    group.add_argument(
        "--password-file",
        default=None,
        metavar="PATH",
        help=t("cli.password_file_help"),
    )
    group.add_argument(
        "--password-stdin",
        action="store_true",
        help=t("cli.password_stdin_help"),
    )
    if create_mode:
        group.add_argument(
            "--ask-password",
            action="store_true",
            help=t("cli.ask_password_help"),
        )


def _add_common_create(parser: argparse.ArgumentParser, settings: dict) -> None:
    parser.add_argument("-i", "--input", nargs="+", required=True, help=t("cli.input_help"))
    parser.add_argument("-o", "--output", required=True, help=t("cli.output_help"))
    parser.add_argument("--no-recursive", action="store_true", help=t("cli.no_recursive_help"))
    parser.add_argument(
        "--algorithm",
        choices=["auto", "pixel", "passthrough"],
        default="auto",
        help=t("cli.algorithm_help"),
    )
    create_cfg = settings.get("create", {})
    parser.add_argument(
        "--group-mode",
        choices=["resolution", "regex", "natural"],
        default=str(create_cfg.get("group_mode", "resolution")),
        help=t("cli.group_mode_help"),
    )
    parser.add_argument(
        "--group-regex",
        default=str(create_cfg.get("group_regex", r"\d+_p")),
        help=t("cli.group_regex_help"),
    )
    parser.add_argument(
        "--group-regex-fallback",
        choices=["resolution", "singleton", "error"],
        default=str(create_cfg.get("group_regex_fallback", "resolution")),
        help=t("cli.group_fallback_help"),
    )
    parser.add_argument(
        "--delta-quant",
        type=int,
        default=int(create_cfg.get("delta_quant", 12)),
        help=t("cli.delta_quant_help"),
    )
    parser.add_argument(
        "--delta-quant-max",
        type=int,
        default=int(create_cfg.get("delta_quant_max", 16)),
        help=t("cli.delta_quant_max_help"),
    )
    parser.add_argument(
        "--residual-min-zero",
        type=float,
        default=float(create_cfg.get("residual_min_zero_fraction", 0.40)),
        help=t("cli.residual_min_zero_help"),
    )
    parser.add_argument(
        "--residual-codec",
        choices=["auto", "zlib", "pxc"],
        default=str(create_cfg.get("residual_codec", "auto")),
        help=t("cli.residual_codec_help"),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=t("cli.fast_help"),
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help=t("cli.no_cluster_help"),
    )
    parser.add_argument(
        "--no-jxl",
        action="store_true",
        help=t("cli.no_jxl_help"),
    )
    parser.add_argument(
        "--parallel-groups",
        type=int,
        default=int(create_cfg.get("parallel_groups", 0)),
        help=t("cli.parallel_groups_help"),
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        metavar="SIZE",
        help=t("cli.memory_limit_help"),
    )
    parser.add_argument(
        "--show-groups",
        action="store_true",
        default=bool(create_cfg.get("show_groups", False)),
        help=t("cli.show_groups_help"),
    )
    parser.add_argument(
        "--decode-raw",
        action="store_true",
        default=bool(create_cfg.get("decode_raw", False)),
        help=t("cli.decode_raw_help"),
    )
    parser.add_argument("--tile-size", type=int, default=128, help=t("cli.tile_size_help"))
    parser.add_argument(
        "--zlib-level",
        type=int,
        choices=[1, 3, 6, 9],
        default=6,
        help=t("cli.zlib_level_help"),
    )
    parser.add_argument(
        "--keyframe", type=int, default=16, help=t("cli.keyframe_help")
    )
    parser.add_argument(
        "--reference",
        choices=["auto", "chain", "anchor", "none"],
        default="auto",
        help=t("cli.reference_help"),
    )
    parser.add_argument("--search", type=int, default=24, help=t("cli.search_help"))
    parser.add_argument(
        "--keyframe-codec",
        choices=["auto", "pxc", "webp"],
        default="auto",
        help=t("cli.keyframe_codec_help"),
    )
    parser.add_argument(
        "--webp-method",
        type=int,
        choices=range(0, 7),
        default=4,
        help=t("cli.webp_method_help"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(create_cfg.get("threads", 0)),
        help=t("cli.threads_help"),
    )
    parser.add_argument("--overwrite", action="store_true", help=t("cli.overwrite_help"))
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=bool(settings["progress"]["enabled"]),
        help=t("cli.progress_help"),
    )
    parser.add_argument(
        "--details",
        action="store_true",
        default=bool(settings["progress"]["details"]),
        help=t("cli.details_help"),
    )
    parser.add_argument(
        "--detail-lines",
        type=int,
        default=int(settings["progress"]["detail_lines"]),
        help=t("cli.detail_lines_help"),
    )
    parser.add_argument("--quiet", action="store_true", help=t("cli.quiet_help"))
    parser.add_argument(
        "--verbose", action="store_true", help=t("cli.create_verbose_help")
    )
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help=t("cli.no_self_check_help"),
    )

    # ---- 分卷
    volume_group = parser.add_argument_group(t("cli.volume_group"))
    volume_group.add_argument(
        "--volumes",
        action=argparse.BooleanOptionalAction,
        default=None,  # None = 用配置；显式 --volumes/--no-volumes 优先
        help=t("cli.volumes_help"),
    )
    volume_group.add_argument(
        "--volume-size",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help=t(
            "cli.volume_size_help",
            default=_size_text(int(create_cfg.get("volume_size", DEFAULT_VOLUME_SIZE))),
        ),
    )

    # ---- 加密
    _add_password_input(parser, create_mode=True)
    parser.add_argument(
        "--encrypt-names",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.get("crypto", {}).get("encrypt_filenames", True)),
        help=t("cli.encrypt_names_help"),
    )
    parser.add_argument(
        "--kdf",
        choices=["auto", "argon2id", "scrypt"],
        default=str(settings.get("crypto", {}).get("kdf", "auto")),
        help=t("cli.kdf_help"),
    )


def _add_password_input_read(parser: argparse.ArgumentParser) -> None:
    _add_password_input(parser, create_mode=False)


def build_parser(settings: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pxcs.py",
        description=t("cli.description", version=VERSION_STRING),
        epilog=t("cli.epilog"),
    )
    parser.add_argument(
        "--version", action="version", version=f"PXCS {VERSION_STRING}"
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help=t("cli.language_help"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help=t("cli.create_help"))
    _add_common_create(create, settings)

    listing = sub.add_parser("list", help=t("cli.list_help"))
    listing.add_argument("archive", help=t("cli.archive_help"))
    listing.add_argument("--json", action="store_true", help=t("cli.json_help"))
    listing.add_argument("--long", action="store_true", help=t("cli.long_help"))
    _add_password_input_read(listing)

    extract = sub.add_parser("extract", help=t("cli.extract_help"))
    extract.add_argument("archive", help=t("cli.archive_help"))
    extract.add_argument(
        "-o", "--output", default=None, help=t("cli.extract_output_help")
    )
    extract.add_argument(
        "--name", nargs="*", default=None, help=t("cli.extract_name_help")
    )
    extract.add_argument(
        "--quality", type=_parse_quality, default=None, help=t("cli.extract_quality_help")
    )
    extract.add_argument(
        "--lossless", action="store_true", help=t("cli.extract_lossless_help")
    )
    extract.add_argument("--overwrite", action="store_true", help=t("cli.overwrite_help"))
    extract.add_argument("--verbose", action="store_true", help=t("cli.verbose_help"))
    extract.add_argument("--quiet", action="store_true", help=t("cli.quiet_help"))
    _add_password_input_read(extract)

    verify = sub.add_parser("verify", help=t("cli.verify_help"))
    verify.add_argument("archive", help=t("cli.archive_help"))
    verify.add_argument("--quick", action="store_true", help=t("cli.verify_quick_help"))
    _add_password_input_read(verify)

    info = sub.add_parser("info", help=t("cli.info_help"))
    info.add_argument("archive", help=t("cli.archive_help"))
    _add_password_input_read(info)

    return parser


# ---------------------------------------------------------------- 口令输入

def _read_password_file(path: str) -> str:
    file_path = Path(path).expanduser()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PxcError(t("cli.password_file_error", path=file_path, reason=exc)) from exc
    if not raw:
        raise PxcError(t("cli.password_empty_file", path=file_path))
    # 只去掉行尾换行，保留首尾空格（空格可以是口令的一部分）。
    return raw.splitlines()[0] if raw.splitlines() else ""


def _password_from_args(args: argparse.Namespace) -> Optional[str]:
    """非交互路径：--password-file / --password-stdin。"""
    if getattr(args, "password_file", None):
        return _read_password_file(args.password_file)
    if getattr(args, "password_stdin", False):
        if sys.stdin is None:
            raise PxcError(t("cli.password_stdin_unavailable"))
        line = sys.stdin.readline()
        if not line:
            raise PxcError(t("cli.password_stdin_empty"))
        return line.rstrip("\r\n")
    return None


def _interactive() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except (OSError, ValueError):
        return False


def _prompt_password(prompt_key: str = "cli.password_prompt") -> Optional[str]:
    if not _interactive():
        return None
    try:
        return getpass.getpass(t(prompt_key))
    except (EOFError, KeyboardInterrupt):
        return None


def _prompt_new_password() -> Optional[str]:
    first = _prompt_password("cli.password_new_prompt")
    if first is None:
        return None
    if len(first) < 1:
        print(t("cli.password_empty"), file=sys.stderr)
        return None
    second = _prompt_password("cli.password_confirm_prompt")
    if second != first:
        print(t("cli.password_mismatch"), file=sys.stderr)
        return None
    if len(first) < 8:
        print(t("crypto.weak_password"), file=sys.stderr)
    return first


def _create_password(args: argparse.Namespace) -> Optional[str]:
    """创建时的口令来源：文件/标准输入/显式交互询问；默认不加密。"""
    password = _password_from_args(args)
    if password is not None:
        if not password:
            raise PxcError(t("cli.password_empty"))
        if len(password) < 8:
            print(t("crypto.weak_password"), file=sys.stderr)
        return password
    if getattr(args, "ask_password", False):
        password = _prompt_new_password()
        if password is None:
            raise PxcError(t("cli.password_aborted"))
        return password
    return None


def _run_with_password(
    fn: Callable[[Optional[str]], int],
    provided: Optional[str],
    *,
    interactive: bool,
    allow_retry: bool = True,
) -> int:
    """打开加密归档时按需询问口令；口令永远不回显、不落盘、不进日志。"""
    password = provided
    attempts = 0
    while True:
        try:
            return fn(password)
        except PxcPasswordRequired:
            if password is not None:
                raise
            if not interactive:
                raise
            password = _prompt_password()
            if password is None:
                raise PxcError(t("cli.password_aborted")) from None
        except PxcAuthenticationError:
            attempts += 1
            if not allow_retry or not interactive or attempts >= 3:
                raise
            print(
                t("cli.password_retry", attempts=attempts, maximum=3),
                file=sys.stderr,
            )
            password = _prompt_password()
            if password is None:
                raise PxcError(t("cli.password_aborted")) from None


# ---------------------------------------------------------------- 普通模式

def _normal_archive_name(folder: Path, name_parts: int) -> str:
    """按最后 N 个目录名生成归档名：.../summer/main/2021 -> summer_main_2021.pxcs"""
    parts: list[str] = []
    for part in folder.resolve().parts:
        if len(part) == 3 and part[1:] == ":\\":
            continue  # Windows 盘符 C:\
        if part in ("\\", "/"):
            continue
        parts.append(part)
    selected = parts[-max(1, name_parts):] or ["archive"]
    labels = [safe_label(part, "part") for part in selected]
    return "_".join(labels) + ".pxcs"


def _run_normal_create(folder: Path, settings: dict) -> int:
    """普通模式：pxcs.exe <一个文件夹>，输出到当前目录。"""
    name_parts = int(settings["normal_mode"]["name_parts"])
    output = Path.cwd() / _normal_archive_name(folder, name_parts)
    create_cfg = settings.get("create", {})
    volume_size = (
        int(create_cfg.get("volume_size", DEFAULT_VOLUME_SIZE))
        if create_cfg.get("volumes", False)
        else None
    )
    opts = CreateOptions(
        progress=bool(settings["progress"]["enabled"]),
        details=bool(settings["progress"]["details"]),
        detail_lines=int(settings["progress"]["detail_lines"]),
        progress_refresh_seconds=float(settings["progress"]["refresh_seconds"]),
        group_mode=str(create_cfg.get("group_mode", "resolution")),
        group_regex=str(create_cfg.get("group_regex", r"\d+_p")),
        group_regex_fallback=str(create_cfg.get("group_regex_fallback", "resolution")),
        delta_quant=int(create_cfg.get("delta_quant", 12)),
        delta_quant_max=int(create_cfg.get("delta_quant_max", 16)),
        residual_codec=str(create_cfg.get("residual_codec", "auto")),
        residual_min_zero_fraction=float(
            create_cfg.get("residual_min_zero_fraction", 0.40)
        ),
        threads=int(create_cfg.get("threads", 0)),
        parallel_groups=int(create_cfg.get("parallel_groups", 3)),
        show_groups=bool(
            settings.get("normal_mode", {}).get(
                "show_groups", create_cfg.get("show_groups", False)
            )
        ),
        volume_size=volume_size,
    )
    code = 1
    try:
        create_archive(output, [str(folder)], opts)
        code = 0
    except (PxcError, OSError, ValueError) as exc:
        print(t("cli.fail", reason=exc), file=sys.stderr)
    _pause_normal_mode(settings)
    return code


def _pause_normal_mode(settings: dict) -> None:
    """普通模式完成后不要立刻关窗；双击运行时尤其重要。"""
    if not bool(settings.get("normal_mode", {}).get("pause_on_finish", True)):
        return
    stdin = sys.stdin
    stdout = sys.stdout
    if stdin is None or stdout is None:
        return
    try:
        # 源码模式只在交互终端等待；打包后的 exe 即使被重定向也等待，
        # 因为用户最常见的入口是双击/拖放。config 可关闭。
        if not stdout.isatty() and not getattr(sys, "frozen", False):
            return
    except (OSError, ValueError):
        return
    try:
        input(t("cli.pause_hint"))
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------- 命令

def _resolve_read_password(args: argparse.Namespace) -> Optional[str]:
    return _password_from_args(args)


def _memory_limit_mb(value: Optional[str]) -> int:
    """--memory-limit 解析成 MiB；空 / 0 表示自动。"""
    if value is None or str(value).strip() == "":
        return 0
    size = parse_size(str(value))
    return max(0, int(size // (1024 * 1024)))


def _command_create(args: argparse.Namespace, settings: dict) -> int:
    output = Path(args.output)
    if output.suffix.lower() != ".pxcs":
        output = output.with_suffix(".pxcs")
    password = _create_password(args)
    config_volumes = bool(settings.get("create", {}).get("volumes", False))
    if args.volumes is None:
        # 只给 --volume-size 也隐含启用分卷；显式 --no-volumes 优先于它。
        use_volumes = config_volumes or args.volume_size is not None
    else:
        use_volumes = bool(args.volumes)
    volume_size = args.volume_size
    if volume_size is None:
        volume_size = int(
            settings.get("create", {}).get("volume_size", DEFAULT_VOLUME_SIZE)
        )
    if not use_volumes:
        volume_size = None
    residual_codec = args.residual_codec
    webp_method = args.webp_method
    if args.fast:
        # 速度优先：无损残差也交给 zlib（比 PXC tile 快一个数量级）。
        # 刻意不动 --webp-method：method 0 会让 PNG 关键帧明显变大，
        # 实测 4 张 PNG 从 90.96% 退化到 99.69%（3 张直接回退 raw）。
        residual_codec = "zlib"
    opts = CreateOptions(
        recursive=not args.no_recursive,
        algorithm=args.algorithm,
        tile_size=args.tile_size,
        zlib_level=args.zlib_level,
        keyframe_interval=args.keyframe,
        reference=args.reference,
        keyframe_codec=args.keyframe_codec,
        webp_method=webp_method,
        search_window=args.search,
        threads=args.threads,
        progress=bool(args.progress and not args.quiet),
        details=bool(args.details and not args.quiet),
        detail_lines=max(0, int(args.detail_lines)),
        overwrite=args.overwrite,
        quiet=args.quiet,
        group_mode=args.group_mode,
        group_regex=args.group_regex,
        group_regex_fallback=args.group_regex_fallback,
        delta_quant=args.delta_quant,
        delta_quant_max=args.delta_quant_max,
        residual_codec=residual_codec,
        residual_min_zero_fraction=args.residual_min_zero,
        cluster_similar=not args.no_cluster,
        jpegxl=not args.no_jxl,
        parallel_groups=args.parallel_groups,
        memory_limit_mb=_memory_limit_mb(args.memory_limit),
        show_groups=args.show_groups,
        decode_raw=args.decode_raw,
        verbose=args.verbose,
        self_check=not args.no_self_check,
        password=password,
        encrypt_filenames=bool(args.encrypt_names),
        kdf=args.kdf,
        volume_size=volume_size,
    )
    stats = create_archive(output, args.input, opts)
    # 3 = 归档已生成，但有文件被跳过；调用方需要据此判断归档是否完整。
    return 3 if getattr(stats, "skipped", 0) else 0


def _command_extract(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    output = Path(args.output) if args.output else default_output_dir(archive)
    provided = _resolve_read_password(args)

    def run(password: Optional[str]) -> int:
        stats = extract_archive(
            archive,
            output,
            names=args.name,
            overwrite=args.overwrite,
            quality=args.quality,
            lossless=args.lossless,
            verbose=args.verbose,
            quiet=args.quiet,
            password=password,
        )
        return 0 if stats.failed == 0 else 1

    return _run_with_password(run, provided, interactive=_interactive())


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    settings = load_settings()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    # 语言要在建 parser 之前加载，--language 优先于配置文件。
    language = None
    for index, value in enumerate(raw_args):
        if value == "--language" and index + 1 < len(raw_args):
            language = raw_args[index + 1]
        elif value.startswith("--language="):
            language = value.split("=", 1)[1]
    try:
        translator = configure_i18n(language or settings.get("language"))
    except Exception:
        translator = configure_i18n(None)
    if translator.missing_languages and language:
        print(
            t("cli.language_missing", code=translator.language),
            file=sys.stderr,
        )

    # 拖放/双击便捷入口：pxcs.exe 合集.pxcs -> 直接解压到同名目录。
    if len(raw_args) == 1:
        one = Path(raw_args[0]).expanduser()
        if one.is_dir():
            return _run_normal_create(one, settings)
        if one.suffix.lower() == ".pxcs" or re.search(r"\.\d{3,}$", one.name):
            raw_args = ["extract", raw_args[0]]
    if not raw_args:
        # 双击 exe 时没有参数：给出提示并等待拖入/输入归档路径。
        print(t("cli.no_args_hint", version=VERSION_STRING))
        try:
            value = input(t("cli.archive_prompt")).strip().strip('"')
        except EOFError:
            return 2
        if not value:
            return 2
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return _run_normal_create(candidate, settings)
        raw_args = ["extract", value]
    parser = build_parser(settings)
    args = parser.parse_args(raw_args)
    if getattr(args, "language", None):
        configure_i18n(args.language)
    try:
        if args.command == "create":
            return _command_create(args, settings)

        if args.command == "list":
            provided = _resolve_read_password(args)

            def run(password: Optional[str]) -> int:
                print(
                    list_entries(
                        Path(args.archive),
                        as_json=args.json,
                        long=args.long,
                        password=password,
                    )
                )
                return 0

            return _run_with_password(run, provided, interactive=_interactive())

        if args.command == "extract":
            return _command_extract(args)

        if args.command == "verify":
            provided = _resolve_read_password(args)

            def run(password: Optional[str]) -> int:
                stats = verify_archive(
                    Path(args.archive), quick=args.quick, password=password
                )
                return 0 if stats.failed == 0 else 1

            return _run_with_password(run, provided, interactive=_interactive())

        if args.command == "info":
            provided = _resolve_read_password(args)

            def run(password: Optional[str]) -> int:
                print(archive_info(Path(args.archive), password=password))
                return 0

            return _run_with_password(run, provided, interactive=_interactive())

        parser.error(t("cli.unknown_command"))
        return 2
    except PxcsSelfCheckError as exc:
        # 4 = 归档写完了，但重开自检失败；文件保留，便于事后 verify。
        print(t("cli.fail", reason=exc), file=sys.stderr)
        return 4
    except (PxcError, OSError, ValueError) as exc:
        print(t("cli.fail", reason=exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
