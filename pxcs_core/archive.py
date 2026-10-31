"""PXCS 归档的创建、列出、提取与校验。"""
from __future__ import annotations

import concurrent.futures as cf
import fnmatch
import os
import re
import sys
import time
import zlib

import numpy as np
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from pxc_core.config import MODE_SPECS, SUPPORTED_IMAGE_EXTENSIONS
from pxc_core.errors import PxcError, PxcFormatError
from pxc_core.i18n import t
from PIL import Image

# PXCS 支持到 300M 像素；高于 Pillow 默认 89M 只应是 warning，
# 由 PXCS 自己的尺寸检查决定是否处理。
Image.MAX_IMAGE_PIXELS = 300_000_000

from pxc_core.image_io import (
    ImageData,
    canonical_pixel_hash,
    is_supported_image_path,
    load_image,
    load_image_bytes,
    save_image,
    verify_saved_image,
)
from pxc_core.utils import (
    atomic_write_bytes,
    crc32,
    human_size,
    ratio_text,
    sha256_bytes,
)

from .codec import (
    decode_keyframe,
    decode_zlib_residual,
    encode_keyframe,
    encode_residual,
    encode_residual_with_preview,
    encode_webp_lossless,
    is_delta_capable,
    load_raw_image,
    make_thumbnail,
    reconstruct_image,
    sample_score,
    set_zlib_level,
    thumbnail_score,
)
from . import jxl
from .container import (
    ArchiveEntry,
    ArchiveReader,
    ArchiveWriter,
    PxcsFormatError,
    compute_flags,
)
from .format import (
    CODEC_DUP,
    CODEC_PXC,
    CODEC_PXC_RESIDUAL,
    CODEC_WEBP_LOSSLESS,
    CODEC_RAW,
    CODEC_JPEGXL,
    CODEC_STORE,
    CODEC_ZLIB,
    CODEC_ZLIB_RESIDUAL,
    DEFAULT_KEYFRAME_INTERVAL,
    DEFAULT_TILE_SIZE,
    DEFAULT_ZLIB_LEVEL,
)
from .progress import ProgressDisplay
from .paths import (
    PxcsPathError,
    SourceFile,
    collect_source_files,
    natural_key,
    safe_join,
)

_LOSSLESS_FORMATS = {"PNG", "BMP", "TIFF", "TIF"}
# 缩略图粗排分数超过该值时，认为同尺寸但内容不相近，直接关键帧。
_DELTA_SCORE_CUTOFF = 0.85
# Pixiv 常见的 p0/p1/p2 变体名，末尾的 _pN/-pN 不作为“不同作品”的证据。
_VARIANT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)[_-]p\d+$", re.IGNORECASE)


def _variant_key(path_text: str) -> str:
    stem = Path(path_text).stem
    match = _VARIANT_SUFFIX_RE.match(stem)
    return match.group("base").casefold() if match else ""
_RAW_IMAGE_FORMATS = {"JPEG", "JPG", "WEBP"}
_IMAGE_EXTS = SUPPORTED_IMAGE_EXTENSIONS

# 输出归档自己产生、但可能是上一次运行留下的文件：重新创建时不能当成输入。
_ARCHIVE_ARTIFACT_RE = re.compile(r"^.*\.tmp-[0-9a-zA-Z-]+$|^.*\.\d{3,}$")
# 任何 .pxcs / .pxcs.001：把归档再打包进归档几乎总是误操作，且会让体积翻倍。
_PXCS_ARTIFACT_RE = re.compile(r"^.*\.pxcs(?:\.\d{3,})?$", re.IGNORECASE)


def _is_pxcs_artifact(name: str) -> bool:
    """输入目录里已有的 .pxcs（含分卷）不应再被当成输入。"""
    return bool(_PXCS_ARTIFACT_RE.match(name))


def _is_archive_artifact(name: str, output_name: str) -> bool:
    """判断输入目录里的文件是不是同一个归档的临时文件/旧分卷。"""
    if not name.startswith(output_name):
        return False
    if name == output_name:
        return True
    suffix = name[len(output_name):]
    if not suffix.startswith("."):
        return False
    rest = suffix[1:]
    return rest.isdigit() and len(rest) >= 3 or rest.startswith("tmp-")


def default_output_dir(archive: Path) -> Path:
    """提取默认目录：``a.pxcs`` 与 ``a.pxcs.001`` 都得到 ``a``。"""
    from .volumes import parse_volume_name

    parsed = parse_volume_name(archive.name)
    base = archive.parent / parsed[0] if parsed else archive
    return base.with_suffix("")


def _crypto_params(opts: "CreateOptions"):
    """把 CreateOptions 里的口令/KDF 设置转成 CryptoParams。"""
    if not opts.password:
        return None
    from .crypto import CryptoParams

    return CryptoParams.from_config(
        kdf=opts.kdf,
        argon2_memory_kib=opts.kdf_memory_kib,
        argon2_iterations=opts.kdf_iterations,
        argon2_lanes=opts.kdf_lanes,
        encrypt_filenames=opts.encrypt_filenames,
    )


@dataclass
class CreateOptions:
    recursive: bool = True
    algorithm: str = "auto"  # auto | pixel | passthrough
    tile_size: int = DEFAULT_TILE_SIZE
    zlib_level: int = DEFAULT_ZLIB_LEVEL
    keyframe_interval: int = DEFAULT_KEYFRAME_INTERVAL
    reference: str = "auto"  # auto | chain | anchor | none
    keyframe_codec: str = "auto"  # auto | pxc | webp
    webp_method: int = 4  # 0..6
    search_window: int = 24
    threads: int = 0  # 0=用满所有核心
    # 并发任务的内存预算（MiB）；0=按可用物理内存自动计算（见 limits.py）。
    memory_limit_mb: int = 0
    progress: bool = True
    details: bool = False
    detail_lines: int = 3
    progress_refresh_seconds: float = 0.2
    overwrite: bool = False
    quiet: bool = False
    # resolution=按分辨率一组一个 key（默认）；regex=按文件名正则分组；
    # natural=保留旧的逐张相似度搜索。
    group_mode: str = "resolution"
    group_regex: str = r"\d+_p"
    # regex 一个都不命中时：resolution=回退分辨率分组（推荐）；singleton=逐张 key；error=报错。
    group_regex_fallback: str = "resolution"
    # 有损残差量化步长；0=像素无损。只用于 JPEG/WebP 组。
    delta_quant: int = 12
    delta_quant_max: int = 16
    # 量化残差中 128（零差分）比例低于该值时，直接回退 raw，避免 zlib 处理高熵数据。
    residual_min_zero_fraction: float = 0.40
    # auto=量化残差用 zlib（快），无损残差用 PXC；zlib/pxc 可强制。
    residual_codec: str = "auto"
    # 同一分辨率的大组按缩略图再按内容切分（每个子簇一个 key）。
    # 关掉就是旧行为：整组只对着首图做差分。
    cluster_similar: bool = True
    # JPEG 关键帧/散图是否走 JPEG XL 无损重打包（1.4 格式特性）。
    # 关掉可以保持 1.2 兼容（旧 exe 能读）。
    jpegxl: bool = True
    parallel_groups: int = 0  # 0 = 与线程数一致
    show_groups: bool = False
    # 逐个文件打印被跳过的原因（默认只在摘要里列出前若干项）。
    verbose: bool = False
    # finish() 之后重开归档做一次结构自检；失败说明写入不可信。
    self_check: bool = True
    # raw 直通（JPEG/WebP 关键帧）是否仍完整解码一次。
    # 默认关闭：raw 存的就是源文件字节，字节级 SHA-256 已完整覆盖完整性；
    # 解码只为确认"能解出像素"并取到 constant_alpha，而这两个信息对 raw
    # 条目的重建没有任何作用。实测这一步占这类归档创建耗时四成以上。
    # 组内多于一张或 --decode-raw 时会强制解码（前者需要参考图）。
    decode_raw: bool = False
    # 加密：口令为空表示不加密。口令只保存在内存里，不写日志、不进 argv。
    password: Optional[str] = field(default=None, repr=False)
    encrypt_filenames: bool = True
    kdf: str = "auto"  # auto | argon2id | scrypt
    kdf_memory_kib: int = 262144
    kdf_iterations: int = 3
    kdf_lanes: int = 4
    # 分卷：None/0 表示单文件；否则每卷最大字节数。
    volume_size: Optional[int] = None


@dataclass
class CreateStats:
    files_total: int = 0
    images: int = 0
    files: int = 0
    keyframes: int = 0
    deltas: int = 0
    raw_images: int = 0
    dups: int = 0
    fallbacks: int = 0
    original_bytes: int = 0
    archive_bytes: int = 0
    elapsed: float = 0.0
    groups: int = 0
    keys: int = 0
    lossy_deltas: int = 0
    volumes: int = 0
    # 读不到/写不进而被跳过的文件数；>0 时 create 的退出码是 3。
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # 汇总信息里不含口令
        return (
            f"CreateStats(files_total={self.files_total}, images={self.images}, "
            f"archive_bytes={self.archive_bytes}, volumes={self.volumes})"
        )


def summary_text(stats: "CreateStats", path: Path, extra: str = "") -> str:
    """创建完成后的单行摘要，自然模式与分组模式共用。"""
    return t(
        "create.summary",
        path=path,
        original=human_size(stats.original_bytes),
        archive=human_size(stats.archive_bytes),
        ratio=ratio_text(stats.archive_bytes, stats.original_bytes),
        groups=stats.groups,
        keys=stats.keys,
        images=stats.images,
        deltas=stats.deltas,
        lossy=stats.lossy_deltas,
        raw=stats.raw_images,
        files=stats.files,
        dups=stats.dups,
        elapsed=f"{stats.elapsed:.2f}",
        extra=extra,
    )


@dataclass
class ImageProbe:
    format: str
    width: int
    height: int
    mode: str
    lossy: bool


@dataclass
class WorkItem:
    source: SourceFile
    group_key: tuple
    probe: Optional[ImageProbe] = None


def _probe_image(path: Path) -> Optional[ImageProbe]:
    """只读图片头，用于按尺寸/模式分组，不加载整张像素。"""
    if not is_supported_image_path(path):
        return None
    try:
        with Image.open(path) as img:
            if int(getattr(img, "n_frames", 1) or 1) > 1:
                return None
            if bool(getattr(img, "is_animated", False)):
                return None
            if img.mode not in MODE_SPECS:
                return None
            fmt = (img.format or path.suffix.lstrip(".")).upper()
            if fmt == "JPG":
                fmt = "JPEG"
            if fmt == "TIF":
                fmt = "TIFF"
            return ImageProbe(
                format=fmt,
                width=int(img.width),
                height=int(img.height),
                mode=str(img.mode),
                lossy=fmt in _RAW_IMAGE_FORMATS,
            )
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def _order_work_items(
    sources: list[SourceFile],
    *,
    group_mode: str = "resolution",
    group_regex: str = r"\d+_p",
    group_regex_fallback: str = "resolution",
    scan_callback: Optional[Callable[[int, int], None]] = None,
) -> list[WorkItem]:
    """按所选模式分组并自然排序。

    resolution：精确 (w,h,mode,lossy) 一组，组内自然序，默认模式。
    regex：文件名命中正则的捕获/匹配串 + 分辨率一组；未命中的每张单独一组。
    natural：保留旧行为，仍按分辨率聚在一起，后续做跨图相似度搜索。
    """
    if group_mode not in {"resolution", "regex", "natural"}:
        raise PxcsFormatError(t("archive.group_mode_unknown", mode=group_mode))
    regex = None
    if group_mode == "regex":
        try:
            regex = re.compile(group_regex)
        except re.error as exc:
            raise PxcsFormatError(
                t("archive.group_regex_invalid", regex=group_regex, reason=exc)
            ) from exc
        if regex.search("") is not None and regex.pattern == "":
            raise PxcsFormatError(t("archive.group_regex_empty"))

    groups: dict[tuple, list[tuple[SourceFile, ImageProbe]]] = defaultdict(list)
    files: list[WorkItem] = []
    regex_hits = 0
    for scan_index, source in enumerate(sources, 1):
        if scan_callback is not None and (
            scan_index == 1 or scan_index % 100 == 0 or scan_index == len(sources)
        ):
            try:
                scan_callback(scan_index, len(sources))
            except Exception:
                pass
        probe = _probe_image(source.path)
        if probe is None:
            files.append(WorkItem(source=source, group_key=("file",), probe=None))
            continue
        if regex is not None:
            match = regex.search(Path(source.rel).stem)
            if match:
                regex_hits += 1
                token = (
                    match.group(1)
                    if regex.groups and match.lastindex
                    else match.group(0)
                )
            else:
                token = None
            if token is None:
                # 未命中的图片不强行合并，避免把不同作品塞进一个 key。
                key = ("regex_none", source.rel, probe.width, probe.height, probe.mode)
            else:
                key = (token, probe.width, probe.height, probe.mode, probe.lossy)
        else:
            key = (probe.lossy, probe.width, probe.height, probe.mode)
        groups[key].append((source, probe))

    if regex is not None and regex_hits == 0:
        if group_regex_fallback == "error":
            raise PxcsFormatError(t("archive.regex_no_match", regex=group_regex))
        if group_regex_fallback == "resolution":
            # 每个文件一个 key 会把 1200 张图变成 1200 组；回退分辨率更安全。
            return _order_work_items(
                sources,
                group_mode="resolution",
                group_regex_fallback=group_regex_fallback,
                scan_callback=scan_callback,
            )

    ordered: list[WorkItem] = []
    for key, members in sorted(
        groups.items(),
        key=lambda kv: (kv[0], -len(kv[1])),
    ):
        for source, probe in sorted(members, key=lambda item: natural_key(item[0].rel)):
            ordered.append(WorkItem(source=source, group_key=key, probe=probe))
    ordered.extend(sorted(files, key=lambda item: natural_key(item.source.rel)))
    return ordered


@dataclass
class ExtractStats:
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    bytes: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class VerifyStats:
    checked: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _fmt_image(entry: ArchiveEntry) -> str:
    return f"{entry.width}x{entry.height} {entry.mode}"


def _source_ext(path: Path) -> str:
    return path.suffix.lower()


def _source_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".tif":
        return "TIFF"
    return ext.lstrip(".").upper()


def _make_entry(
    entry_id: int,
    path: str,
    kind: str,
    codec: str,
    *,
    size: int,
    mtime_ns: int,
    width: int = 0,
    height: int = 0,
    mode: str = "",
    source_sha256: str = "",
    payload_info: Optional[tuple[int, int, int, str]] = None,
    raw_crc32: int = 0,
    image: Optional[ImageData] = None,
    ref: Optional[int] = None,
    depth: int = 0,
    format_name: str = "",
    ext: str = "",
    constant_alpha: Optional[int] = None,
    pixel_sha256: str = "",
    quant: int = 0,
    centered: bool = False,
    lossy: bool = False,
    reason: str = "",
) -> ArchiveEntry:
    if payload_info:
        offset, length, plain_length, data_crc, data_sha = payload_info
    else:
        offset, length, plain_length, data_crc, data_sha = 0, 0, None, 0, ""
    return ArchiveEntry(
        id=entry_id,
        path=path,
        kind=kind,
        codec=codec,
        offset=offset,
        length=length,
        plain_length=plain_length,
        size=size,
        ref=ref,
        depth=depth,
        width=image.width if image is not None else width,
        height=image.height if image is not None else height,
        mode=image.mode if image is not None else mode,
        format=format_name or (image.source_format if image is not None else ""),
        ext=ext,
        constant_alpha=constant_alpha,
        quant=quant,
        centered=centered,
        lossy=lossy,
        pixel_sha256=pixel_sha256 or (
            image.pixel_hash.hex() if image is not None else ""
        ),
        source_sha256=source_sha256,
        data_crc32=data_crc,
        raw_crc32=raw_crc32,
        data_sha256=data_sha,
        mtime_ns=mtime_ns,
        reason=reason,
    )


def _format_supports(image: ImageData, fmt: str) -> bool:
    mode = image.mode
    if fmt == "JPEG":
        return mode in {"L", "RGB", "CMYK"}
    if fmt == "WEBP":
        return mode in {"RGB", "RGBA"}
    if fmt == "BMP":
        return mode in {"1", "L", "P", "RGB"}
    if fmt == "GIF":
        return mode in {"L", "P"}
    if fmt == "PNG":
        return mode in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I;16L", "I;16B"}
    if fmt == "TIFF":
        return True
    return False


def _choose_format(entry: ArchiveEntry, image: Optional[ImageData] = None) -> str:
    fmt = (entry.format or "").upper()
    if fmt in {"JPG", "JPEG"}:
        fmt = "JPEG"
    elif fmt in {"PNG", "WEBP", "BMP", "TIFF", "TIF", "GIF"}:
        fmt = "TIFF" if fmt == "TIF" else fmt
    else:
        ext = (entry.ext or "").lower()
        if ext in (".jpg", ".jpeg"):
            fmt = "JPEG"
        elif ext == ".webp":
            fmt = "WEBP"
        elif ext == ".gif":
            fmt = "GIF"
        elif ext == ".bmp":
            fmt = "BMP"
        elif ext in (".tif", ".tiff"):
            fmt = "TIFF"
        elif ext == ".png":
            fmt = "PNG"
        else:
            fmt = ""
    if image is not None:
        if fmt and _format_supports(image, fmt):
            return fmt
        if image.mode in {"CMYK", "I", "F"}:
            return "TIFF"
        if image.mode in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I;16L", "I;16B"}:
            return "PNG"
        return "TIFF"
    if fmt:
        return fmt
    mode = entry.mode
    if mode in {"CMYK", "I", "F"}:
        return "TIFF"
    return "PNG"


def _safe_set_mtime(path: Path, mtime_ns: int) -> None:
    if not mtime_ns:
        return
    try:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    except OSError:
        pass


def _compress_file(raw: bytes, level: int) -> tuple[str, bytes, int]:
    raw_crc = crc32(raw)
    if not raw:
        return CODEC_STORE, b"", raw_crc
    comp = zlib.compress(raw, level)
    if len(comp) + 16 < len(raw):
        return CODEC_ZLIB, comp, raw_crc
    return CODEC_STORE, raw, raw_crc


def _atomic_save_image(
    image: ImageData,
    target: Path,
    format_name: str,
    quality: Optional[int],
    overwrite: bool,
) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return False
    tmp = target.with_name(target.name + f".pxtmp-{os.getpid()}-{time.time_ns()}")
    try:
        save_image(image, tmp, format_name, quality=quality)
        fmt = format_name.upper()
        if fmt in {"PNG", "BMP", "TIFF", "GIF"} and quality is None:
            # PNG/TIFF/BMP/GIF 必须像素无损；WebP lossless 也检查。
            verify_saved_image(tmp, image)
        os.replace(tmp, target)
        return True
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


class _ImageResolver:
    """按需解码图片，并缓存少量最近图像供差分链使用。"""

    def __init__(self, reader: ArchiveReader, cache_size: int = 4) -> None:
        self.reader = reader
        self.cache: OrderedDict[int, ImageData] = OrderedDict()
        self.cache_size = max(1, cache_size)
        self.visiting: set[int] = set()

    def _remember(self, entry_id: int, image: ImageData) -> ImageData:
        self.cache[entry_id] = image
        self.cache.move_to_end(entry_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return image

    def get(self, entry_id: int) -> ImageData:
        if entry_id in self.cache:
            self.cache.move_to_end(entry_id)
            return self.cache[entry_id]
        if entry_id in self.visiting:
            raise PxcsFormatError(t("archive.delta_cycle"))
        entry = self.reader.get_entry(entry_id)
        if entry.kind != "image":
            raise PxcsFormatError(t("archive.not_image", path=entry.path))
        self.visiting.add(entry_id)
        try:
            if entry.codec == CODEC_PXC:
                image = decode_keyframe(self.reader.read_entry_data(entry))
                if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
                    raise PxcsFormatError(t("archive.keyframe_pixel_hash", path=entry.path))
            elif entry.codec == CODEC_WEBP_LOSSLESS:
                payload = self.reader.read_entry_data(entry)
                if len(payload) != entry.stored_length:
                    raise PxcsFormatError(t("archive.webp_key_length", path=entry.path))
                image = load_image_bytes(
                    payload,
                    Path(entry.path).name,
                    source_file_size=entry.size,
                    source_format=entry.format or "WEBP",
                )
                # WebP 会把“全透明/全不透明”的 alpha 平面省略。
                # 关键帧条目记录了 constant_alpha，解码后补回完整通道。
                if (
                    entry.constant_alpha is not None
                    and entry.mode == "RGBA"
                    and image.mode == "RGB"
                ):
                    alpha = np.full(
                        (image.height, image.width, 1),
                        int(entry.constant_alpha),
                        dtype=image.data.dtype,
                    )
                    data = np.ascontiguousarray(
                        np.concatenate([image.data, alpha], axis=2)
                    )
                    image = replace(
                        image,
                        mode="RGBA",
                        data=data,
                        constant_alpha=int(entry.constant_alpha),
                        pixel_hash=canonical_pixel_hash(
                            "RGBA", image.width, image.height, data
                        ),
                    )
                if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
                    raise PxcsFormatError(t("archive.webp_key_hash", path=entry.path))
            elif entry.codec == CODEC_PXC_RESIDUAL:
                if entry.ref is None:
                    raise PxcsFormatError(t("archive.residual_need_ref"))
                ref = self.get(entry.ref)
                residual = decode_keyframe(self.reader.read_entry_data(entry))
                image = reconstruct_image(ref, residual, entry)
            elif entry.codec == CODEC_ZLIB_RESIDUAL:
                if entry.ref is None:
                    raise PxcsFormatError(t("archive.zlib_residual_need_ref"))
                ref = self.get(entry.ref)
                payload = self.reader.read_entry_data(entry)
                image = decode_zlib_residual(payload, entry, ref)
            elif entry.codec in {CODEC_RAW, CODEC_JPEGXL}:
                payload = self.reader.read_entry_data(entry)
                if entry.codec == CODEC_JPEGXL:
                    payload = _unpack_jpeg(entry, payload)
                if len(payload) != entry.size:
                    raise PxcsFormatError(t("archive.raw_image_length", path=entry.path))
                if entry.source_sha256 and sha256_bytes(payload).hex() != entry.source_sha256:
                    raise PxcsFormatError(t("archive.raw_image_sha", path=entry.path))
                image = load_raw_image(payload, entry)
                if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
                    raise PxcsFormatError(t("archive.raw_image_pixel_hash", path=entry.path))
            elif entry.codec == CODEC_DUP:
                if entry.ref is None:
                    raise PxcsFormatError(t("archive.dup_image_need_ref"))
                image = self.get(entry.ref)
            else:
                raise PxcsFormatError(t("archive.unknown_image_codec", codec=entry.codec))
            return self._remember(entry_id, image)
        finally:
            self.visiting.discard(entry_id)


def _original_bytes(reader: ArchiveReader, entry: ArchiveEntry) -> bytes:
    """取回文件原始字节；DUP 递归解析。"""
    if entry.codec == CODEC_DUP:
        if entry.ref is None:
            raise PxcsFormatError(t("archive.dup_need_ref"))
        return _original_bytes(reader, reader.get_entry(entry.ref))
    payload = reader.read_entry_data(entry)
    if entry.codec == CODEC_JPEGXL:
        return _unpack_jpeg(entry, payload)
    if entry.codec == CODEC_STORE:
        return payload
    if entry.codec == CODEC_ZLIB:
        try:
            raw = zlib.decompress(payload)
        except zlib.error as exc:
            raise PxcsFormatError(
                t("archive.zlib_failed", path=entry.path, reason=exc)
            ) from exc
        if len(raw) != entry.size:
            raise PxcsFormatError(t("archive.unzip_length", path=entry.path))
        if crc32(raw) != entry.raw_crc32:
            raise PxcsFormatError(t("archive.raw_crc", path=entry.path))
        if entry.source_sha256 and sha256_bytes(raw).hex() != entry.source_sha256:
            raise PxcsFormatError(t("archive.raw_sha", path=entry.path))
        return raw
    if entry.codec == CODEC_RAW:
        return payload
    raise PxcsFormatError(
            t("archive.no_original_bytes", path=entry.path, codec=entry.codec)
        )


def _unpack_jpeg(entry: ArchiveEntry, blob: bytes) -> bytes:
    """jpegxl 还原成原始 JPEG 字节；失败转成格式错误。"""
    try:
        return jxl.unpack(blob)
    except ValueError as exc:
        raise PxcsFormatError(
            t("archive.jxl_decode", path=entry.path, reason=exc)
        ) from exc


def _raw_payload_for_image(reader: ArchiveReader, entry: ArchiveEntry) -> bytes:
    if entry.codec in {CODEC_RAW, CODEC_JPEGXL}:
        payload = reader.read_entry_data(entry)
        if entry.codec == CODEC_JPEGXL:
            return _unpack_jpeg(entry, payload)
        return payload
    if entry.codec == CODEC_DUP:
        if entry.ref is None:
            raise PxcsFormatError(t("archive.dup_need_ref"))
        return _raw_payload_for_image(reader, reader.get_entry(entry.ref))
    raise PxcsFormatError(
            t("archive.not_raw_image", path=entry.path, codec=entry.codec)
        )


def _match_names(path: str, patterns: Optional[Iterable[str]]) -> bool:
    if not patterns:
        return True
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatchcase(normalized, p) for p in patterns)


def _extract_file(
    reader: ArchiveReader,
    entry: ArchiveEntry,
    target: Path,
    overwrite: bool,
) -> bool:
    if target.exists() and not overwrite:
        return False
    raw = _original_bytes(reader, entry)
    if len(raw) != entry.size:
        raise PxcsFormatError(t("archive.file_length", path=entry.path))
    if entry.raw_crc32 and crc32(raw) != entry.raw_crc32:
        raise PxcsFormatError(t("archive.file_crc", path=entry.path))
    if entry.source_sha256 and sha256_bytes(raw).hex() != entry.source_sha256:
        raise PxcsFormatError(t("archive.file_sha", path=entry.path))
    atomic_write_bytes(target, raw)
    _safe_set_mtime(target, entry.mtime_ns)
    return True


def _extract_image(
    reader: ArchiveReader,
    resolver: _ImageResolver,
    entry: ArchiveEntry,
    target: Path,
    overwrite: bool,
    quality: Optional[int],
    lossless: bool = False,
) -> bool:
    if target.exists() and not overwrite:
        return False
    if lossless:
        # 强制输出像素无损格式：JPEG/WebP 解码后写 PNG/TIFF，
        # 不保留原始字节，但保证像素与归档内部重建一致。
        image = resolver.get(entry.id)
        if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
            raise PxcsFormatError(t("archive.image_pixel_hash", path=entry.path))
        fmt = "TIFF" if image.mode in {"CMYK", "I", "F"} else "PNG"
        if not _atomic_save_image(image, target, fmt, None, overwrite):
            return False
        _safe_set_mtime(target, entry.mtime_ns)
        return True
    fmt = _choose_format(entry)
    use_quality = quality is not None and fmt in {"JPEG", "WEBP"}

    # 原始 JPEG/WebP 在未要求重编码时直接落盘，保持字节和元数据。
    if entry.codec in {CODEC_RAW, CODEC_JPEGXL, CODEC_DUP} and not use_quality:
        try:
            payload = _raw_payload_for_image(reader, entry)
        except PxcsFormatError:
            payload = None
        if payload is not None:
            if len(payload) != entry.size:
                raise PxcsFormatError(t("archive.raw_image_length", path=entry.path))
            if entry.source_sha256 and sha256_bytes(payload).hex() != entry.source_sha256:
                raise PxcsFormatError(t("archive.raw_image_sha", path=entry.path))
            atomic_write_bytes(target, payload)
            _safe_set_mtime(target, entry.mtime_ns)
            return True

    image = resolver.get(entry.id)
    if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
        raise PxcsFormatError(t("archive.image_pixel_hash", path=entry.path))
    fmt = _choose_format(entry, image)
    use_quality = quality is not None and fmt in {"JPEG", "WEBP"}
    # 有损差分/JPEG 重编码的提示在 extract_archive 里汇总一次，
    # 避免 1000+ 张差分图刷屏。
    if not _atomic_save_image(
        image, target, fmt, quality if use_quality else None, overwrite
    ):
        return False
    _safe_set_mtime(target, entry.mtime_ns)
    return True


def create_archive(
    output: Path,
    inputs: Iterable[str],
    options: Optional[CreateOptions] = None,
) -> CreateStats:
    opts = options or CreateOptions()
    if opts.algorithm not in {"auto", "pixel", "passthrough"}:
        raise PxcsFormatError(t("archive.option_algorithm", value=opts.algorithm))
    if opts.reference not in {"auto", "chain", "anchor", "none"}:
        raise PxcsFormatError(t("archive.option_reference", value=opts.reference))
    if opts.group_mode not in {"resolution", "regex", "natural"}:
        raise PxcsFormatError(t("archive.option_group_mode", value=opts.group_mode))
    if not (0 <= opts.delta_quant <= 255):
        raise PxcsFormatError(t("archive.option_delta_quant"))
    if not (0 <= opts.delta_quant_max <= 255):
        raise PxcsFormatError(t("archive.option_delta_quant_max"))
    if opts.residual_codec not in {"auto", "zlib", "pxc"}:
        raise PxcsFormatError(t("archive.option_residual_codec"))
    if not (0.0 <= opts.residual_min_zero_fraction <= 1.0):
        raise PxcsFormatError(t("archive.option_residual_min_zero"))
    if opts.group_mode == "regex":
        try:
            re.compile(opts.group_regex)
        except re.error as exc:
            raise PxcsFormatError(
                t("archive.group_regex_invalid", regex=opts.group_regex, reason=exc)
            ) from exc
        if opts.group_regex_fallback not in {"resolution", "singleton", "error"}:
            raise PxcsFormatError(t("archive.option_group_fallback"))
    if opts.parallel_groups < 0:
        raise PxcsFormatError(t("archive.option_parallel_groups"))
    if opts.keyframe_codec not in {"auto", "pxc", "webp"}:
        raise PxcsFormatError(t("archive.option_keyframe_codec", value=opts.keyframe_codec))
    if not (0 <= opts.webp_method <= 6):
        raise PxcsFormatError(t("archive.option_webp_method"))
    if opts.tile_size < 8 or opts.tile_size > 1024:
        raise PxcsFormatError(t("archive.option_tile_size"))
    if opts.keyframe_interval < 0:
        raise PxcsFormatError(t("archive.option_keyframe_interval"))
    if opts.search_window < 0:
        raise PxcsFormatError(t("archive.option_search_window"))
    if opts.volume_size is not None:
        from .volumes import MIN_VOLUME_SIZE

        if int(opts.volume_size) < MIN_VOLUME_SIZE:
            raise PxcsFormatError(
                t("volume.too_small", value=int(opts.volume_size), minimum=MIN_VOLUME_SIZE)
            )
    if opts.password is not None and not str(opts.password):
        raise PxcsFormatError(t("crypto.password_empty"))
    if opts.kdf not in {"auto", "argon2id", "scrypt"}:
        raise PxcsFormatError(t("crypto.unknown_kdf_name", name=opts.kdf))
    from . import jxl as _jxl
    from .limits import resolve_threads

    _jxl.set_enabled(bool(getattr(opts, "jpegxl", True)))
    threads = resolve_threads(opts.threads)

    if opts.group_mode in {"resolution", "regex"}:
        # 分组模式：按组并行；natural 模式继续使用下面的逐张相似度搜索。
        from .grouped_create import create_grouped_archive
        return create_grouped_archive(output, inputs, opts)

    set_zlib_level(opts.zlib_level)
    started = time.perf_counter()
    input_paths = list(inputs)
    sources = collect_source_files(
        input_paths, recursive=opts.recursive, exclude=output
    )
    sources = [
        item for item in sources
        if not _is_archive_artifact(item.path.name, output.name)
        and not _is_pxcs_artifact(item.path.name)
    ]
    work_items = _order_work_items(
        sources,
        group_mode=opts.group_mode,
        group_regex=opts.group_regex,
        group_regex_fallback=opts.group_regex_fallback,
    )
    image_group_keys = {
        item.group_key for item in work_items if item.probe is not None
    }
    stats = CreateStats(files_total=len(sources), groups=len(image_group_keys))
    strict_group = opts.group_mode in {"resolution", "regex"}
    total_source = sum(s.path.stat().st_size for s in sources)
    stats.original_bytes = total_source
    title = str(input_paths[0]) if input_paths else str(output)
    progress = ProgressDisplay(
        title,
        len(sources),
        total_source,
        enabled=bool(opts.progress and not opts.quiet),
        details=bool(opts.details and not opts.quiet),
        detail_lines=opts.detail_lines,
        refresh_seconds=opts.progress_refresh_seconds,
    )
    progress.start()

    writer = ArchiveWriter(
        output,
        overwrite=opts.overwrite,
        zlib_level=opts.zlib_level,
        password=opts.password,
        encrypt_filenames=opts.encrypt_filenames,
        crypto_params=_crypto_params(opts),
        volume_size=opts.volume_size,
    )
    entries: list[ArchiveEntry] = []
    raw_hash_index: dict[str, int] = {}
    last_pixel: Optional[ArchiveEntry] = None
    last_image: Optional[ImageData] = None
    anchor_id: Optional[int] = None
    anchor_image: Optional[ImageData] = None
    # 只保存小缩略图，不保存全部原图；用于跨图候选搜索。
    history: list[tuple[ArchiveEntry, np.ndarray, SourceFile]] = []
    current_group_key: Optional[tuple] = None

    def remember(entry: ArchiveEntry, image: ImageData, source: SourceFile) -> None:
        if opts.search_window <= 0 or not is_delta_capable(image):
            return
        history.append((entry, make_thumbnail(image), source))
        keep = max(1, opts.search_window)
        if len(history) > keep:
            del history[:-keep]

    if not opts.quiet and not opts.progress:
        print(
            t(
                "archive.create_info",
                count=len(sources),
                size=human_size(total_source),
                group_mode=opts.group_mode,
                groups=len(image_group_keys),
                tile=opts.tile_size,
                keyframe=opts.keyframe_interval,
                reference=opts.reference,
                workers=threads,
            )
        )
    if opts.show_groups and not opts.quiet:
        counts: dict[tuple, int] = defaultdict(int)
        for item in work_items:
            if item.probe is not None:
                counts[item.group_key] += 1
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"[GROUP] {key} x{count}")

    processed_so_far = 0
    prev_size = 0
    prev_rel = ""
    group_index = 0
    try:
        for index, item in enumerate(work_items, 1):
            source = item.source
            src_size = source.path.stat().st_size
            if index > 1:
                processed_so_far += prev_size
            group_text = ""
            if item.probe is not None:
                group_text = t(
                    "archive.group_progress",
                    index=max(1, group_index),
                    total=len(image_group_keys),
                )
            progress.update(
                index - 1,
                source.rel + group_text,
                processed_so_far,
                writer.position(),
            )
            prev_size = src_size
            prev_rel = source.rel
            if item.group_key != current_group_key:
                # 换尺寸/模式/有损-无损分组时，清空参考状态，避免旧组锚点污染。
                last_pixel = None
                last_image = None
                anchor_id = None
                anchor_image = None
                history.clear()
                current_group_key = item.group_key
                if item.probe is not None:
                    group_index += 1
            rel = source.rel
            src_path = source.path
            st = src_path.stat()
            mtime_ns = st.st_mtime_ns
            src_size = st.st_size
            image: Optional[ImageData] = None
            image_error: Optional[str] = None

            if is_supported_image_path(src_path):
                try:
                    image = load_image(src_path, compute_file_hash=True)
                except PxcError as exc:
                    image_error = str(exc)

            source_sha = ""
            raw_bytes: Optional[bytes] = None
            fallback_reason = ""
            if image is not None:
                source_sha = (
                    image.source_file_hash.hex() if image.source_file_hash else ""
                )
            else:
                raw_bytes = src_path.read_bytes()
                source_sha = sha256_bytes(raw_bytes).hex()

            if image_error and not opts.quiet:
                print(
                    t("archive.decode_failed", path=rel, reason=image_error)
                )

            # ---- 完全相同文件：只对可还原原始字节的条目去重 ----
            if (
                source_sha
                and source_sha in raw_hash_index
                and not (strict_group and item.probe is not None and anchor_id is None)
            ):
                ref_id = raw_hash_index[source_sha]
                ref_entry = entries[ref_id]
                if ref_entry.kind == ("image" if image is not None else "file"):
                    entry = _make_entry(
                        len(entries),
                        rel,
                        ref_entry.kind,
                        CODEC_DUP,
                        size=src_size,
                        mtime_ns=mtime_ns,
                        source_sha256=source_sha,
                        image=image,
                        ref=ref_id,
                        depth=1,
                        format_name=(
                            image.source_format if image is not None else ""
                        ),
                        ext=_source_ext(src_path),
                        constant_alpha=(
                            image.constant_alpha if image is not None else None
                        ),
                        reason="duplicate",
                    )
                    entries.append(entry)
                    stats.dups += 1
                    if ref_entry.kind == "image":
                        stats.images += 1
                    else:
                        stats.files += 1
                    progress.detail(f"[DUP ] {rel} -> #{ref_id}")
                    continue

            if image is not None:
                src_format = (image.source_format or _source_format(src_path)).upper()
                is_raw = (
                    opts.algorithm == "passthrough"
                    or (
                        opts.algorithm == "auto"
                        and src_format in _RAW_IMAGE_FORMATS
                    )
                )
                prefer_raw = is_raw
                delta_allowed = is_delta_capable(image) and (
                    not prefer_raw
                    or (
                        opts.algorithm == "auto"
                        and opts.reference != "none"
                    )
                )

                # ---- 选择参考图 ----
                ref_entry: Optional[ArchiveEntry] = None
                ref_image: Optional[ImageData] = None
                if delta_allowed and strict_group and opts.reference != "none":
                    # 分辨率/正则模式：每组只有一个 key，所有后续图片都贴这个 key。
                    # 不做“同尺寸但不同作品”的提前淘汰，交给残差体积决定是否回退。
                    if anchor_id is not None and anchor_image is not None:
                        ref_entry = entries[anchor_id]
                        ref_image = anchor_image
                elif delta_allowed:
                    current_thumb = make_thumbnail(image)
                    seen_candidates: set[int] = set()
                    candidates: list[
                        tuple[float, ArchiveEntry, Optional[ImageData], Optional[SourceFile]]
                    ] = []

                    def add_candidate(
                        candidate: ArchiveEntry,
                        candidate_image: Optional[ImageData],
                        candidate_source: Optional[SourceFile],
                        candidate_thumb: Optional[np.ndarray],
                    ) -> None:
                        if candidate.id in seen_candidates:
                            return
                        if candidate.mode != image.mode:
                            return
                        if candidate.width != image.width or candidate.height != image.height:
                            return
                        if candidate_thumb is None:
                            if candidate_image is None:
                                return
                            candidate_thumb = make_thumbnail(candidate_image)
                        changed, mean_abs = thumbnail_score(
                            current_thumb, candidate_thumb
                        )
                        seen_candidates.add(candidate.id)
                        candidates.append(
                            (
                                changed + mean_abs / 255.0,
                                candidate,
                                candidate_image,
                                candidate_source,
                            )
                        )

                    if (
                        opts.reference in {"auto", "chain"}
                        and last_pixel is not None
                        and last_image is not None
                        and (
                            opts.keyframe_interval == 0
                            or last_pixel.depth < opts.keyframe_interval
                        )
                    ):
                        add_candidate(last_pixel, last_image, None, None)
                    if (
                        opts.reference in {"auto", "anchor"}
                        and anchor_id is not None
                        and anchor_image is not None
                    ):
                        add_candidate(entries[anchor_id], anchor_image, None, None)
                    if opts.reference == "auto" and opts.search_window > 0:
                        for h_entry, h_thumb, h_source in history[-opts.search_window:]:
                            if (
                                opts.keyframe_interval == 0
                                or h_entry.depth < opts.keyframe_interval
                            ):
                                add_candidate(h_entry, None, h_source, h_thumb)

                    candidates.sort(key=lambda item: item[0])
                    if candidates and candidates[0][0] >= _DELTA_SCORE_CUTOFF:
                        current_variant = _variant_key(rel)
                        # 旧逻辑的严重 bug：文件名不匹配 pN 时把候选清空，
                        # 导致所有同尺寸差分图都变成 raw。只在确实有
                        # variant 且找到同 variant 候选时才过滤。
                        if current_variant:
                            filtered = [
                                item for item in candidates
                                if _variant_key(item[1].path) == current_variant
                            ]
                            if filtered:
                                candidates = filtered

                    for _score, candidate, candidate_image, candidate_source in candidates:
                        if candidate_image is None:
                            if candidate_source is None:
                                continue
                            try:
                                loaded = load_image(
                                    candidate_source.path, compute_file_hash=True
                                )
                            except PxcError:
                                continue
                            if (
                                loaded.mode != image.mode
                                or loaded.width != image.width
                                or loaded.height != image.height
                            ):
                                continue
                            if (
                                candidate.source_sha256
                                and loaded.source_file_hash
                                and loaded.source_file_hash.hex()
                                != candidate.source_sha256
                            ):
                                continue
                            candidate_image = loaded
                        ref_entry = candidate
                        ref_image = candidate_image
                        break

                    if ref_entry is not None and ref_image is not None:
                        changed, mean_abs = sample_score(ref_image, image)
                        threshold = 20.0 if prefer_raw else 64.0
                        if changed >= 0.985 and mean_abs >= threshold:
                            ref_entry = None
                            ref_image = None
                            fallback_reason = "dissimilar"

                if ref_entry is not None and ref_image is not None:
                        changed, mean_abs = sample_score(ref_image, image)
                        # 有损源先用更严格的阈值排除“同尺寸但不同作品”，
                        # 避免为明显不同的 JPEG 白算一次残差。
                        threshold = 20.0 if prefer_raw else 64.0
                        if changed >= 0.985 and mean_abs >= threshold:
                            ref_entry = None
                            ref_image = None
                            fallback_reason = "dissimilar"

                if ref_entry is not None and ref_image is not None:
                    # JPEG/WebP 的像素差分会被解码噪声淹没。用“居中 + 量化”
                    # 把 ±quant 内的噪声归零；q=16 在实测 2023 组里通常能把
                    # 6MB JPEG 的差分压到 0.3–4MB。q=0 保留像素级差分。
                    is_lossy_image = image.source_format.upper() in _RAW_IMAGE_FORMATS
                    quant_base = (
                        int(opts.delta_quant)
                        if strict_group and is_lossy_image
                        and image.data.dtype == np.dtype(np.uint8)
                        else 0
                    )
                    quant_values: list[int] = [quant_base]
                    if (
                        strict_group
                        and quant_base > 0
                        and opts.delta_quant_max > quant_base
                    ):
                        quant_values.append(int(opts.delta_quant_max))

                    chosen_payload: Optional[bytes] = None
                    chosen_hash = ""
                    chosen_quant = 0
                    for quant in quant_values:
                        if strict_group:
                            payload, approx_hash = encode_residual_with_preview(
                                ref_image,
                                image,
                                opts.tile_size,
                                threads=threads,
                                quant=quant,
                                centered=True,
                            )
                            approx_hex = approx_hash.hex()
                        else:
                            payload = encode_residual(
                                ref_image, image, opts.tile_size, threads=threads
                            )
                            approx_hex = image.pixel_hash.hex()
                        if (
                            chosen_payload is None
                            or len(payload) < len(chosen_payload)
                        ):
                            chosen_payload = payload
                            chosen_hash = approx_hex
                            chosen_quant = quant
                        # 只有严格模式下才为了争取 < 源文件而尝试更大量化；
                        # natural 模式仍保持原来的无损残差行为。
                        if (
                            not strict_group
                            or src_size <= 0
                            or len(payload) < src_size * 0.98
                        ):
                            break

                    residual_payload = chosen_payload or b""
                    # 残差若接近/超过原文件，回退原始码流。
                    if (
                        src_size > 0
                        and len(residual_payload) >= src_size * 0.98
                    ):
                        ref_entry = None
                        ref_image = None
                        fallback_reason = "residual_larger_than_source"
                        stats.fallbacks += 1
                    else:
                        info = writer.add_payload(residual_payload)
                        entry = _make_entry(
                            len(entries),
                            rel,
                            "image",
                            CODEC_PXC_RESIDUAL,
                            size=src_size,
                            mtime_ns=mtime_ns,
                            source_sha256=source_sha,
                            payload_info=info,
                            image=image,
                            ref=ref_entry.id,
                            depth=ref_entry.depth + 1,
                            ext=_source_ext(src_path),
                            constant_alpha=image.constant_alpha,
                            pixel_sha256=chosen_hash,
                            quant=chosen_quant,
                            centered=bool(strict_group),
                            lossy=bool(chosen_quant > 0),
                            reason=(
                                "residual_lossy" if chosen_quant > 0 else "residual"
                            ),
                        )
                        entries.append(entry)
                        stats.images += 1
                        stats.deltas += 1
                        if chosen_quant > 0:
                            stats.lossy_deltas += 1
                        last_pixel = entry
                        last_image = image
                        remember(entry, image, source)
                        progress.detail(
                            f"[DELT] {rel} <- #{ref_entry.id} | "
                            f"{human_size(src_size)} -> "
                            f"{human_size(len(residual_payload))}"
                            + (f" q={chosen_quant}" if chosen_quant else "")
                        )
                        continue

                # ---- 关键帧 ----
                if prefer_raw:
                    if raw_bytes is None:
                        raw_bytes = src_path.read_bytes()
                    info = writer.add_payload(raw_bytes)
                    entry = _make_entry(
                        len(entries),
                        rel,
                        "image",
                        CODEC_RAW,
                        size=src_size,
                        mtime_ns=mtime_ns,
                        source_sha256=source_sha,
                        payload_info=info,
                        raw_crc32=crc32(raw_bytes),
                        image=image,
                        ext=_source_ext(src_path),
                        constant_alpha=image.constant_alpha,
                        reason=fallback_reason or "lossy_passthrough",
                    )
                    entries.append(entry)
                    raw_hash_index.setdefault(source_sha, entry.id)
                    stats.images += 1
                    stats.raw_images += 1
                    if anchor_id is None:
                        stats.keys += 1
                    last_pixel = entry
                    last_image = image
                    remember(entry, image, source)
                    if anchor_id is None:
                        anchor_id = entry.id
                        anchor_image = image
                    progress.detail(
                        t(
                            "progress.detail_raw_source",
                            path=rel,
                            size=human_size(src_size),
                        )
                    )
                    continue

                if (
                    opts.keyframe_codec in {"auto", "webp"}
                    and image.mode in {"RGB", "RGBA"}
                    and src_format in _LOSSLESS_FORMATS
                ):
                    payload = encode_webp_lossless(image, method=opts.webp_method)
                    keyframe_codec = CODEC_WEBP_LOSSLESS
                else:
                    payload = encode_keyframe(image, opts.tile_size, threads=threads)
                    keyframe_codec = CODEC_PXC
                # 像素模式比源文件还大时，允许回退到原始字节（仍可作为后续参考）。
                if (
                    opts.algorithm == "auto"
                    and src_format in _LOSSLESS_FORMATS
                    and src_size > 0
                    and len(payload) >= src_size * 1.02
                ):
                    if raw_bytes is None:
                        raw_bytes = src_path.read_bytes()
                    info = writer.add_payload(raw_bytes)
                    entry = _make_entry(
                        len(entries),
                        rel,
                        "image",
                        CODEC_RAW,
                        size=src_size,
                        mtime_ns=mtime_ns,
                        source_sha256=source_sha,
                        payload_info=info,
                        raw_crc32=crc32(raw_bytes),
                        image=image,
                        ext=_source_ext(src_path),
                        constant_alpha=image.constant_alpha,
                        reason="pixel_pxc_larger",
                    )
                    entries.append(entry)
                    raw_hash_index.setdefault(source_sha, entry.id)
                    stats.images += 1
                    stats.raw_images += 1
                    stats.fallbacks += 1
                    last_pixel = entry
                    last_image = image
                    remember(entry, image, source)
                    if anchor_id is None:
                        anchor_id = entry.id
                        anchor_image = image
                    progress.detail(
                        t("progress.detail_raw_fallback", path=rel)
                    )
                    continue

                info = writer.add_payload(payload)
                entry = _make_entry(
                    len(entries),
                    rel,
                    "image",
                    keyframe_codec,
                    size=src_size,
                    mtime_ns=mtime_ns,
                    source_sha256=source_sha,
                    payload_info=info,
                    image=image,
                    ext=_source_ext(src_path),
                    constant_alpha=image.constant_alpha,
                    reason=fallback_reason or ("first" if last_pixel is None else "keyframe"),
                )
                entries.append(entry)
                stats.images += 1
                stats.keyframes += 1
                if anchor_id is None:
                    stats.keys += 1
                last_pixel = entry
                last_image = image
                remember(entry, image, source)
                if anchor_id is None:
                    anchor_id = entry.id
                    anchor_image = image
                progress.detail(
                    f"[KEY ] {rel} | {human_size(src_size)} -> "
                    f"{human_size(len(payload))}"
                )
                continue

            # ---- 非图片文件 ----
            assert raw_bytes is not None
            codec, payload, raw_crc = _compress_file(raw_bytes, opts.zlib_level)
            info = writer.add_payload(payload)
            entry = _make_entry(
                len(entries),
                rel,
                "file",
                codec,
                size=src_size,
                mtime_ns=mtime_ns,
                source_sha256=source_sha,
                payload_info=info,
                raw_crc32=raw_crc,
                ext=_source_ext(src_path),
                format_name=_source_format(src_path),
                reason=codec,
            )
            entries.append(entry)
            raw_hash_index.setdefault(source_sha, entry.id)
            stats.files += 1
            progress.detail(
                f"[FILE] {rel} | {human_size(src_size)} -> "
                f"{human_size(len(payload))} ({codec})"
            )

        flags = compute_flags(entries)
        final_path, archive_size = writer.finish(entries, flags)
        stats.archive_bytes = archive_size
        stats.elapsed = time.perf_counter() - started
        processed_so_far += prev_size
        progress.update(
            len(work_items), prev_rel, processed_so_far, archive_size
        )
        progress.finish()
    except BaseException:
        writer.abort()
        raise

    if not opts.quiet:
        print("-" * 72)
        print(summary_text(stats, final_path))
    return stats


def archive_banner(reader: ArchiveReader) -> str:
    """list/info 共用的一行摘要：版本、条目数、体积、分卷与加密状态。"""
    total_source = sum(e.size for e in reader.entries)
    text = t(
        "list.header",
        version=f"{reader.header.version_major}.{reader.header.version_minor}",
        entries=len(reader.entries),
        original=human_size(total_source),
        archive=human_size(reader.logical_size),
        ratio=ratio_text(reader.logical_size, total_source),
    )
    if reader.volume_count > 1:
        text += t("list.volumes", count=reader.volume_count)
    if reader.is_encrypted:
        text += t(
            "list.encrypted_names" if reader.names_encrypted else "list.encrypted_plain"
        )
    return text


def list_entries(
    archive: Path,
    *,
    as_json: bool = False,
    long: bool = False,
    password: Optional[str] = None,
) -> str:
    import json

    with ArchiveReader(archive, password=password) as reader:
        total_source = sum(e.size for e in reader.entries)
        if as_json:
            data = {
                "format": "PXCS",
                "version": 1,
                "archive": str(archive),
                "archive_size": reader.logical_size,
                "physical_size": reader.physical_size,
                "volumes": reader.volume_count,
                "encrypted": reader.is_encrypted,
                "names_encrypted": reader.names_encrypted,
                "original_size": total_source,
                "entries": [e.to_dict() for e in reader.entries],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        lines: list[str] = [archive_banner(reader)]
        for e in reader.entries:
            ref = f" <-#{e.ref}" if e.ref is not None else ""
            if e.codec == CODEC_DUP:
                detail = "DUP"
            elif e.stored_length:
                detail = (
                    f"{human_size(e.stored_length)} "
                    f"({ratio_text(e.stored_length, e.size)})"
                )
            else:
                detail = "0 B"
            kind = "IMG" if e.kind == "image" else "FILE"
            if e.kind == "image":
                detail = f"{_fmt_image(e)} {detail}"
                if e.lossy:
                    detail += f" [lossy q={e.quant}]"
            if long:
                lines.append(
                    f"{e.id:5d} {kind:4s} {e.codec:13s} {detail:32s} "
                    f"{e.path}{ref}"
                )
            else:
                lines.append(f"{e.id:5d} {kind:4s} {e.codec:13s} {e.path}{ref}")
        return "\n".join(lines)


def _print_progress(done: int, total: int) -> None:
    """默认提取进度：TTY 单行刷新；重定向时每 5% 打一行。"""
    if total <= 0:
        return
    pct = done * 100.0 / total
    filled = int(pct // 5)
    bar = "#" * filled + "-" * (20 - filled)
    line = t("progress.extract", bar=bar, done=done, total=total, pct=pct)
    if sys.stdout.isatty():
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        return
    step = max(1, total // 20)
    if done == 1 or done == total or done % step == 0:
        print(line)


def _extract_one(
    reader: ArchiveReader,
    resolver: "_ImageResolver",
    entry: ArchiveEntry,
    output_dir: Path,
    overwrite: bool,
    quality: Optional[int],
    lossless: bool,
) -> bool:
    """提取单个条目，返回是否真的写了盘（False = 目标已存在被跳过）。"""
    target = safe_join(output_dir, entry.path)
    if entry.kind == "image":
        if lossless:
            fmt = "TIFF" if entry.mode in {"CMYK", "I", "F"} else "PNG"
            target = target.with_suffix(".tif" if fmt == "TIFF" else ".png")
        return _extract_image(
            reader, resolver, entry, target, overwrite, quality, lossless=lossless
        )
    return _extract_file(reader, entry, target, overwrite)


def _extract_range(
    archive: Path,
    password: Optional[str],
    output_dir: Path,
    entry_ids: list[int],
    overwrite: bool,
    quality: Optional[int],
    lossless: bool,
) -> tuple[int, int, int, list[str], int]:
    """提取一段条目，返回 (成功, 跳过, 失败, 错误, 字节数)。"""
    extracted = skipped = failed = 0
    total_bytes = 0
    errors: list[str] = []
    with ArchiveReader(archive, password=password) as reader:
        resolver = _ImageResolver(reader)
        for entry_id in entry_ids:
            entry = reader.get_entry(entry_id)
            try:
                if _extract_one(
                    reader, resolver, entry, output_dir, overwrite, quality, lossless
                ):
                    extracted += 1
                    total_bytes += entry.size
                else:
                    skipped += 1
            except (PxcError, OSError, ValueError) as exc:
                failed += 1
                errors.append(f"{entry.path}: {exc}")
    return extracted, skipped, failed, errors, total_bytes


# 少于这么多条目时，开线程池的开销比收益还大。
_EXTRACT_PARALLEL_MIN = 32


def extract_archive(
    archive: Path,
    output_dir: Path,
    *,
    names: Optional[list[str]] = None,
    overwrite: bool = False,
    quality: Optional[int] = None,
    lossless: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    password: Optional[str] = None,
    threads: int = 0,
) -> ExtractStats:
    from .limits import resolve_threads

    workers = resolve_threads(threads)
    stats = ExtractStats()
    output_dir.mkdir(parents=True, exist_ok=True)
    with ArchiveReader(archive, password=password) as reader:
        selected = [e for e in reader.entries if _match_names(e.path, names)]
        encrypted = reader.is_encrypted
    if not selected:
        raise PxcsFormatError(t("extract.no_match"))
    if not quiet:
        lossy_count = sum(1 for e in selected if e.lossy)
        print(
            t("extract.info", count=len(selected), output=output_dir)
            + (
                t("extract.quality_suffix", quality=quality)
                if quality is not None
                else ""
            )
        )
        if lossy_count:
            print(t("extract.lossy_warning", count=lossy_count))

    # verbose 要逐条打印，走并行会串行化输出，所以那种情况保持顺序。
    parallel = (
        workers > 1
        and len(selected) >= _EXTRACT_PARALLEL_MIN
        and not encrypted
        and not verbose
    )
    if parallel:
        chunk = (len(selected) + workers - 1) // workers
        slices = [
            [e.id for e in selected[start:start + chunk]]
            for start in range(0, len(selected), chunk)
        ]
        done = 0
        with cf.ThreadPoolExecutor(max_workers=len(slices)) as pool:
            futures = [
                pool.submit(
                    _extract_range,
                    archive,
                    password,
                    output_dir,
                    ids,
                    overwrite,
                    quality,
                    lossless,
                )
                for ids in slices
            ]
            for future in cf.as_completed(futures):
                extracted, skipped, failed, errors, total_bytes = future.result()
                stats.extracted += extracted
                stats.skipped += skipped
                stats.failed += failed
                stats.bytes += total_bytes
                stats.errors.extend(errors)
                done += extracted + skipped + failed
                if not quiet:
                    _print_progress(done, len(selected))
        for message in stats.errors:
            print(f"[FAIL] {message}")
        if not quiet and sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        with ArchiveReader(archive, password=password) as reader:
            resolver = _ImageResolver(reader)
            for index, entry in enumerate(selected, 1):
                try:
                    if _extract_one(
                        reader,
                        resolver,
                        entry,
                        output_dir,
                        overwrite,
                        quality,
                        lossless,
                    ):
                        stats.extracted += 1
                        stats.bytes += entry.size
                        if verbose and not quiet:
                            print(f"[OK  ] {entry.path}")
                    else:
                        stats.skipped += 1
                        if verbose and not quiet:
                            print(t("extract.skip_existing", path=entry.path))
                except (PxcError, OSError, ValueError) as exc:
                    stats.failed += 1
                    stats.errors.append(f"{entry.path}: {exc}")
                    print(f"[FAIL] {entry.path}: {exc}")
                if not quiet and not verbose:
                    _print_progress(index, len(selected))
            if not quiet and not verbose and sys.stdout.isatty():
                sys.stdout.write("\n")
                sys.stdout.flush()

    if not quiet:
        print(
            t(
                "extract.done",
                extracted=stats.extracted,
                skipped=stats.skipped,
                failed=stats.failed,
                size=human_size(stats.bytes),
            )
        )
    return stats


def _verify_entry(
    reader: ArchiveReader,
    resolver: "_ImageResolver",
    entry: ArchiveEntry,
    quick: bool,
) -> None:
    """校验单个条目；失败时抛异常，由调用方记账。"""
    if entry.kind == "image":
        if quick and entry.codec not in {
            CODEC_DUP, CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL
        }:
            reader.read_entry_data(entry)
        elif not quick and not entry.pixel_sha256:
            # 没有像素哈希（例如 raw 直通且创建时跳过了解码）：
            # 只验 payload 完整性。payload 就是源文件字节，字节级
            # SHA-256 已完整覆盖等价性，再解码一遍没有额外信息。
            reader.read_entry_data(entry)
        else:
            image = resolver.get(entry.id)
            if entry.pixel_sha256 and image.pixel_hash.hex() != entry.pixel_sha256:
                raise PxcsFormatError(t("verify.pixel_hash_mismatch"))
        if entry.codec in {CODEC_RAW, CODEC_JPEGXL}:
            payload = _raw_payload_for_image(reader, entry)
            if entry.source_sha256 and (
                sha256_bytes(payload).hex() != entry.source_sha256
            ):
                raise PxcsFormatError(t("verify.raw_sha_mismatch"))
    else:
        if quick:
            reader.read_entry_data(entry)
        else:
            _original_bytes(reader, entry)


def _verify_range(
    archive: Path,
    password: Optional[str],
    quick: bool,
    start: int,
    stop: int,
) -> tuple[int, list[str]]:
    """校验 [start, stop) 区间的条目，返回 (通过数, 错误列表)。

    每个 worker 开自己的 reader：ArchiveReader 持有文件句柄，共享会互相抢
    读写位置。索引会被重复解析一遍，相对于解码成本可以忽略。
    """
    checked = 0
    errors: list[str] = []
    with ArchiveReader(archive, password=password) as reader:
        resolver = _ImageResolver(reader)
        for entry in reader.entries[start:stop]:
            try:
                _verify_entry(reader, resolver, entry, quick)
                checked += 1
            except (PxcError, OSError, ValueError) as exc:
                errors.append(f"{entry.path}: {exc}")
    return checked, errors


# 少于这么多条时，开线程池的开销比收益还大。
_VERIFY_PARALLEL_MIN = 64


def verify_archive(
    archive: Path,
    *,
    quick: bool = False,
    password: Optional[str] = None,
    threads: int = 0,
) -> VerifyStats:
    from .limits import resolve_threads

    workers = resolve_threads(threads)
    stats = VerifyStats()
    with ArchiveReader(archive, password=password) as reader:
        total = len(reader.entries)
        encrypted = reader.is_encrypted
        lossy = sum(1 for e in reader.entries if e.lossy)

    # 加密归档每次都重新派生密钥（Argon2id 很贵），不值得多开 reader。
    if workers > 1 and total >= _VERIFY_PARALLEL_MIN and not encrypted:
        chunk = (total + workers - 1) // workers
        ranges = [
            (start, min(total, start + chunk))
            for start in range(0, total, chunk)
        ]
        with cf.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            results = list(
                pool.map(
                    lambda rng: _verify_range(
                        archive, password, quick, rng[0], rng[1]
                    ),
                    ranges,
                )
            )
        for checked, errors in results:
            stats.checked += checked
            stats.failed += len(errors)
            stats.errors.extend(errors)
    else:
        with ArchiveReader(archive, password=password) as reader:
            resolver = _ImageResolver(reader)
            for entry in reader.entries:
                try:
                    _verify_entry(reader, resolver, entry, quick)
                    stats.checked += 1
                except (PxcError, OSError, ValueError) as exc:
                    stats.failed += 1
                    stats.errors.append(f"{entry.path}: {exc}")

    for message in stats.errors:
        print(f"[FAIL] {message}")
    extra = t("verify.lossy_note", count=lossy) if lossy else ""
    print(
        t(
            "verify.summary",
            checked=stats.checked,
            total=total,
            failed=stats.failed,
            extra=extra,
        )
    )
    return stats


def archive_info(archive: Path, *, password: Optional[str] = None) -> str:
    from .crypto import crypto_summary

    with ArchiveReader(archive, password=password) as reader:
        size = reader.logical_size
        original = sum(e.size for e in reader.entries)
        images = sum(1 for e in reader.entries if e.kind == "image")
        files = len(reader.entries) - images
        keyframes = sum(
            1 for e in reader.entries
            if e.codec in {CODEC_PXC, CODEC_WEBP_LOSSLESS}
        )
        deltas = sum(
            1 for e in reader.entries
            if e.codec in {CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL}
        )
        raws = sum(
            1 for e in reader.entries
            if e.codec in {CODEC_RAW, CODEC_JPEGXL} and e.kind == "image"
        )
        dups = sum(1 for e in reader.entries if e.codec == CODEC_DUP)
        max_depth = max((e.depth for e in reader.entries), default=0)
        lossy = sum(1 for e in reader.entries if e.lossy)
        lines = [
            t(
                "info.version",
                major=reader.header.version_major,
                minor=reader.header.version_minor,
            ),
            t("info.archive_id", value=reader.header.archive_id.hex()),
            t(
                "info.entries",
                total=len(reader.entries),
                images=images,
                files=files,
            ),
            t(
                "info.codecs",
                keyframes=keyframes,
                deltas=deltas,
                lossy=lossy,
                raw=raws,
                dups=dups,
            ),
            t("info.max_depth", depth=max_depth),
            t("info.original", size=human_size(original)),
            t(
                "info.archive",
                size=human_size(size),
                ratio=ratio_text(size, original),
            ),
            t("info.index", size=human_size(reader.header.index_length)),
        ]
        if reader.volume_count > 1:
            lines.append(
                t(
                    "info.volumes",
                    count=reader.volume_count,
                    first=reader.volume_paths[0].name,
                    last=reader.volume_paths[-1].name,
                    physical=human_size(reader.physical_size),
                )
            )
        if reader.crypto_header is not None:
            lines.append(crypto_summary(reader.crypto_header))
        return "\n".join(lines)
