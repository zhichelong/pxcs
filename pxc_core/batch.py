"""批量处理：输入收集、目录结构保留、输出命名、统计。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import codec, container
from .config import (
    COMPRESSION_PASSTHROUGH,
    DEFAULT_TILE_SIZE,
    META_ORIGINAL_FILENAME,
    META_SOURCE_FILE_HASH,
    MODE_SPECS,
)
from .errors import PxcError, PxcFormatError, UnsupportedImageError
from .image_io import (
    ImageData,
    is_supported_image_path,
    load_image,
    save_image,
    verify_saved_image,
)
from .utils import (
    atomic_write_bytes,
    human_size,
    ratio_text,
    safe_label,
    sha256_bytes,
    unique_target,
)


@dataclass
class InputItem:
    path: Path
    root_label: Optional[str]
    relative: Path


@dataclass
class BatchStats:
    success: int = 0
    failed: int = 0
    skipped: int = 0
    original_bytes: int = 0
    pxc_bytes: int = 0
    restored_bytes: int = 0
    errors: list[str] = field(default_factory=list)


def _norm(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(str(path.absolute()))


def _dir_labels(dirs: list[Path]) -> dict[Path, str]:
    """为每个输入目录分配稳定的输出子目录名。"""
    labels: dict[Path, str] = {}
    used: set[str] = set()
    for directory in dirs:
        base = safe_label(directory.name, "root")
        label = base
        i = 2
        while label.lower() in used:
            label = f"{base}_{i}"
            i += 1
        used.add(label.lower())
        labels[directory] = label
    return labels


def collect_inputs(
    inputs: Iterable[str],
    recursive: bool,
    for_decode: bool,
) -> tuple[list[InputItem], int, list[str]]:
    """收集文件；返回 (items, skipped, errors)。"""
    raw_paths: list[Path] = []
    for value in inputs:
        raw_paths.append(Path(value).expanduser())

    # 先给目录分配 label
    unique_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for p in raw_paths:
        if p.is_dir():
            key = _norm(p)
            if key not in seen_dirs:
                seen_dirs.add(key)
                unique_dirs.append(p)
    labels = _dir_labels(unique_dirs)

    items: list[InputItem] = []
    skipped = 0
    errors: list[str] = []
    seen_files: set[str] = set()

    def want(path: Path) -> bool:
        if for_decode:
            return path.suffix.lower() == ".pxc"
        return is_supported_image_path(path)

    def add_file(path: Path, root_label: Optional[str], relative: Path) -> None:
        nonlocal skipped
        key = _norm(path)
        if key in seen_files:
            skipped += 1
            print(f"[SKIP] 重复输入: {path}")
            return
        seen_files.add(key)
        items.append(InputItem(path=path, root_label=root_label, relative=relative))

    for p in raw_paths:
        if not p.exists():
            errors.append(f"输入不存在: {p}")
            print(f"[FAIL] 输入不存在: {p}")
            continue
        if p.is_file():
            if not want(p):
                skipped += 1
                kind = "PXC 文件" if for_decode else "图片文件"
                print(f"[SKIP] 不是{kind}: {p}")
                continue
            add_file(p, None, Path(p.name))
            continue
        if p.is_dir():
            label = labels[p]
            pattern = p.rglob("*") if recursive else p.glob("*")
            files = sorted((x for x in pattern if x.is_file()), key=lambda x: str(x).lower())
            if not files:
                print(f"[INFO] 目录为空或没有匹配文件: {p}")
                continue
            for f in files:
                if not want(f):
                    skipped += 1
                    print(f"[SKIP] 忽略非目标文件: {f}")
                    continue
                try:
                    rel = f.relative_to(p)
                except ValueError:
                    rel = Path(f.name)
                add_file(f, label, rel)
            continue
        errors.append(f"既不是文件也不是目录: {p}")
        print(f"[FAIL] 既不是文件也不是目录: {p}")

    return items, skipped, errors


def _plan_path(
    output_dir: Path,
    item: InputItem,
    filename: str,
    planned: set[Path],
    overwrite: bool,
) -> Path:
    if item.root_label:
        base = output_dir / item.root_label / item.relative.parent / filename
    else:
        base = output_dir / filename
    return unique_target(base, planned, overwrite)


def choose_output_format(image: ImageData) -> tuple[str, str]:
    """选择像素恢复格式。

    - JPEG/WebP 默认 PNG（避免再次有损编码/格式能力不足）。
    - PNG 保持 PNG。
    - BMP/TIFF 在没有会丢失的元数据且模式兼容时保持原格式。
    - CMYK/I/F 只能 TIFF。
    """
    mode = image.mode
    png_modes = {"1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I;16L", "I;16B"}
    has_meta = bool(
        image.icc_profile
        or image.exif
        or image.xmp
        or image.dpi
        or image.gamma
        or image.comment
    )
    src = (image.source_format or "").upper()

    if mode in ("CMYK", "I", "F"):
        return "TIFF", (".tiff" if image.source_ext == ".tiff" else ".tif")
    if mode not in png_modes:
        raise UnsupportedImageError(f"模式 {mode} 没有安全的像素恢复格式")

    if src == "BMP" and mode in ("1", "L", "RGB") and not has_meta:
        return "BMP", ".bmp"
    if src == "TIFF" and not has_meta and mode in ("1", "L", "RGB", "RGBA", "I;16", "I;16B"):
        return "TIFF", (".tiff" if image.source_ext == ".tiff" else ".tif")
    return "PNG", ".png"


def run_encode(
    inputs: Iterable[str],
    output_dir: Path,
    recursive: bool = False,
    overwrite: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    keep_original: bool = False,
    algorithm: str = "auto",
    zlib_level: int = 6,
    threads: int = 1,
) -> BatchStats:
    stats = BatchStats()
    if threads <= 0:
        threads = min(32, max(1, (os.cpu_count() or 4)))
    previous_zlib_level = codec.ZLIB_LEVEL
    codec.ZLIB_LEVEL = max(1, min(9, int(zlib_level)))
    items, skipped, errors = collect_inputs(inputs, recursive, for_decode=False)
    stats.skipped += skipped
    stats.failed += len(errors)
    stats.errors.extend(errors)
    planned: set[Path] = set()
    total = len(items)
    print(f"[INFO] 待编码文件: {total}")

    for index, item in enumerate(items, 1):
        start = time.perf_counter()
        target = _plan_path(output_dir, item, item.path.stem + ".pxc", planned, overwrite)
        src_size = item.path.stat().st_size
        try:
            image = load_image(item.path, compute_file_hash=True)
            src_format = (image.source_format or "").upper()
            passthrough = algorithm == "passthrough" or (
                algorithm == "auto" and src_format in {"JPEG", "JPG", "WEBP"}
            )
            pass_alpha = (
                2
                if image.mode == "P" and image.transparency is not None
                else int(MODE_SPECS[image.mode]["alpha"])
            )
            if passthrough:
                # 有损/已压缩格式：保存原始码流，像素由解码器重新得到。
                # 这既避免像素域膨胀，也让 --restore-original-file 逐字节恢复。
                orig_bytes = item.path.read_bytes()
                pxc_bytes = container.build_pxc(
                    image,
                    orig_bytes,
                    None,
                    compression_id=COMPRESSION_PASSTHROUGH,
                    tile_size=0,
                    header_channels=image.channels,
                    header_alpha_kind=pass_alpha,
                )
            else:
                data_payload = codec.encode_tiles(image, tile_size, threads=threads)
                orig_bytes = item.path.read_bytes() if keep_original else None
                pxc_bytes = container.build_pxc(image, data_payload, orig_bytes)
                # 自动模式下，如果像素模式反而比源文件大，退回原始码流。
                if (
                    algorithm == "auto"
                    and not keep_original
                    and src_format in {"PNG", "BMP", "TIFF", "TIF"}
                    and len(pxc_bytes) >= src_size
                ):
                    passthrough_bytes = container.build_pxc(
                        image,
                        item.path.read_bytes(),
                        None,
                        compression_id=COMPRESSION_PASSTHROUGH,
                        tile_size=0,
                        header_channels=image.channels,
                        header_alpha_kind=pass_alpha,
                    )
                    if len(passthrough_bytes) < len(pxc_bytes):
                        pxc_bytes = passthrough_bytes
            atomic_write_bytes(target, pxc_bytes)
            elapsed = time.perf_counter() - start
            stats.success += 1
            stats.original_bytes += src_size
            stats.pxc_bytes += len(pxc_bytes)
            print(
                f"[OK] {item.path} -> {target} | "
                f"{human_size(src_size)} -> {human_size(len(pxc_bytes))} "
                f"({ratio_text(len(pxc_bytes), src_size)}) | {elapsed:.2f}s"
            )
        except (PxcError, OSError, ValueError, MemoryError) as exc:
            stats.failed += 1
            msg = f"{item.path}: {type(exc).__name__}: {exc}"
            stats.errors.append(msg)
            print(f"[FAIL] {msg}")
            continue
        except Exception as exc:  # 单个文件失败不终止批次
            stats.failed += 1
            msg = f"{item.path}: 未预期错误 {type(exc).__name__}: {exc}"
            stats.errors.append(msg)
            print(f"[FAIL] {msg}")
            continue

    codec.ZLIB_LEVEL = previous_zlib_level
    _print_encode_summary(stats, total)
    return stats


def _print_encode_summary(stats: BatchStats, total: int) -> None:
    print("-" * 60)
    print(
        f"完成: 成功 {stats.success}, 失败 {stats.failed}, 跳过 {stats.skipped}"
        f"（候选 {total}）"
    )
    if stats.original_bytes > 0:
        print(f"原始总大小: {human_size(stats.original_bytes)}")
        print(f"PXC 总大小: {human_size(stats.pxc_bytes)}")
        print(
            f"压缩比(PXC/原始): {ratio_text(stats.pxc_bytes, stats.original_bytes)}"
            f" | 节省: {ratio_text(stats.original_bytes - stats.pxc_bytes, stats.original_bytes)}"
        )


def run_decode(
    inputs: Iterable[str],
    output_dir: Path,
    recursive: bool = False,
    overwrite: bool = False,
    restore_original_file: bool = False,
) -> BatchStats:
    stats = BatchStats()
    items, skipped, errors = collect_inputs(inputs, recursive, for_decode=True)
    stats.skipped += skipped
    stats.failed += len(errors)
    stats.errors.extend(errors)
    planned: set[Path] = set()
    total = len(items)
    print(f"[INFO] 待恢复文件: {total}")

    for item in items:
        start = time.perf_counter()
        try:
            pxc = container.read_pxc(item.path)
            if restore_original_file:
                _restore_original(pxc, item, output_dir, planned, overwrite, stats, start)
            else:
                image = codec.decode_image_data(pxc)
                fmt, ext = choose_output_format(image)
                stem = Path(image.original_filename).stem or item.path.stem
                filename = stem + ext
                target = _plan_path(output_dir, item, filename, planned, overwrite)
                save_image(image, target, fmt)
                verify_saved_image(target, image)
                elapsed = time.perf_counter() - start
                stats.success += 1
                stats.pxc_bytes += item.path.stat().st_size
                stats.restored_bytes += target.stat().st_size
                print(
                    f"[OK] {item.path} -> {target} | "
                    f"{human_size(item.path.stat().st_size)} -> "
                    f"{human_size(target.stat().st_size)} | {elapsed:.2f}s"
                )
        except (PxcError, OSError, ValueError, MemoryError) as exc:
            stats.failed += 1
            msg = f"{item.path}: {type(exc).__name__}: {exc}"
            stats.errors.append(msg)
            print(f"[FAIL] {msg}")
            continue
        except Exception as exc:
            stats.failed += 1
            msg = f"{item.path}: 未预期错误 {type(exc).__name__}: {exc}"
            stats.errors.append(msg)
            print(f"[FAIL] {msg}")
            continue

    _print_decode_summary(stats, total)
    return stats


def _restore_original(
    pxc: container.PxcFile,
    item: InputItem,
    output_dir: Path,
    planned: set[Path],
    overwrite: bool,
    stats: BatchStats,
    start: float,
) -> None:
    if pxc.header.compression_id == COMPRESSION_PASSTHROUGH:
        payload = pxc.data_payload
    elif pxc.orig_payload is not None:
        payload = pxc.orig_payload
    else:
        raise PxcFormatError(
            "该 PXC 没有保存原始文件字节（像素模式编码时未使用 --keep-original），"
            "不能逐字节恢复原始文件"
        )
    expected_hash = pxc.meta_bytes(META_SOURCE_FILE_HASH)
    if expected_hash is not None:
        actual_hash = sha256_bytes(payload)
        if actual_hash != expected_hash:
            raise PxcFormatError("原始文件哈希校验失败")
    name = pxc.meta_text(META_ORIGINAL_FILENAME) or (item.path.stem + ".bin")
    name = Path(name).name or (item.path.stem + ".bin")
    target = _plan_path(output_dir, item, name, planned, overwrite)
    atomic_write_bytes(target, payload)
    if expected_hash is not None:
        if sha256_bytes(target.read_bytes()) != expected_hash:
            raise PxcFormatError("写出的原始文件哈希校验失败")
    elapsed = time.perf_counter() - start
    stats.success += 1
    stats.pxc_bytes += item.path.stat().st_size
    stats.restored_bytes += target.stat().st_size
    print(
        f"[OK] {item.path} -> {target} | 原始文件字节恢复 | "
        f"{human_size(target.stat().st_size)} | {elapsed:.2f}s"
    )


def _print_decode_summary(stats: BatchStats, total: int) -> None:
    print("-" * 60)
    print(
        f"完成: 成功 {stats.success}, 失败 {stats.failed}, 跳过 {stats.skipped}"
        f"（候选 {total}）"
    )
    if stats.pxc_bytes > 0:
        print(f"PXC 总大小: {human_size(stats.pxc_bytes)}")
        print(f"恢复总大小: {human_size(stats.restored_bytes)}")
