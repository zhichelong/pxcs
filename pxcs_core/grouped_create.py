"""分辨率/正则分组的并行创建器。

设计目标：
- 每个图片组先写一个 key（JPEG 组是原始 JPEG，PNG 组是 WebP/PXC 关键帧）；
- 组内其余图片在 worker 线程里并行编码，全部引用这个 key；
- 最多同时推进 ``parallel_groups`` 个组，所有组共享 ``threads`` 个 worker；
  这样只有一组时全部 worker 都服务该组，多组时空闲 worker 会自动去别的组；
- 归档条目仍按组、组内自然序写入，引用关系可预测。

这个模块只处理 resolution/regex 模式；natural 模式仍走 archive.py 的
逐张相似度搜索实现。
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pxc_core.config import MODE_SPECS
from pxc_core.errors import PxcError
from pxc_core.i18n import t
from pxc_core.image_io import ImageData, load_image
from pxc_core.utils import crc32, human_size, ratio_text, sha256_bytes

from .archive import (
    CreateOptions,
    CreateStats,
    PxcsFormatError,
    WorkItem,
    _compress_file,
    _crypto_params,
    _is_archive_artifact,
    _is_pxcs_artifact,
    _make_entry,
    _order_work_items,
    _source_ext,
    _RAW_IMAGE_FORMATS,
    _LOSSLESS_FORMATS,
    summary_text,
)
from .codec import (
    encode_keyframe,
    encode_residual_with_preview,
    encode_webp_lossless,
    encode_zlib_residual,
)
from . import jxl
from .container import (
    ArchiveReader,
    ArchiveEntry,
    ArchiveWriter,
    PxcsSelfCheckError,
    compute_flags,
)
from .format import (
    DEFAULT_KEYFRAME_INTERVAL,
    CODEC_DUP,
    CODEC_PXC,
    CODEC_PXC_RESIDUAL,
    CODEC_RAW,
    CODEC_JPEGXL,
    CODEC_STORE,
    CODEC_WEBP_LOSSLESS,
    CODEC_ZLIB,
    CODEC_ZLIB_RESIDUAL,
    RESIDUAL_CODECS,
)
from .limits import (
    MemoryBudget,
    cap_parallel_groups_by_memory,
    estimate_key_retention,
    estimate_raw_lease,
    estimate_working_set,
    resolve_memory_budget,
    resolve_parallel_groups,
    resolve_threads,
)
from .paths import SourceFile
from .progress import ProgressDisplay

# 组 key 读不了时，最多向后尝试多少个成员来当 key。
# 单个坏文件不应该让整组退化成 raw，但也不能无限试下去。
MAX_KEY_CANDIDATES = 8


# 同分辨率的图不一定相似：一个组里可能混着几十个不同作品，全部对着组首图
# 做差分必然失败（reason=residual_larger_than_source）。这里按缩略图把大组
# 再切一刀，每个子簇一个 key。只处理 JPEG：缩略图能走 draft() 快速降采样，
# PNG 没有这条捷径，为分组把每张图完整解码一遍太贵。
_SIMILAR_THUMB = 32
_SIMILAR_THRESHOLD = 10.0
_SIMILAR_MIN_GROUP = 8  # 小组切开收益有限，不值得为它多跑一遍缩略图解码


def _file_thumbnail(path: Path) -> Optional[np.ndarray]:
    """32x32 灰度缩略图；解码失败返回 None（该图自成一组）。"""
    from PIL import Image

    try:
        with Image.open(path) as img:
            try:
                img.draft("L", (_SIMILAR_THUMB * 4, _SIMILAR_THUMB * 4))
            except Exception:  # 非 JPEG 格式没有 draft
                pass
            thumb = img.convert("L").resize(
                (_SIMILAR_THUMB, _SIMILAR_THUMB), Image.BILINEAR
            )
            return np.asarray(thumb, dtype=np.int16)
    except Exception:  # noqa: BLE001 - 单张读不了不应该影响分组
        return None


def _cluster_by_thumbnail(
    group: list[WorkItem], thumbs: list[Optional[np.ndarray]]
) -> list[list[WorkItem]]:
    """按缩略图的平均绝对差做贪心聚类，簇内成员保持自然序。"""
    clusters: list[tuple[Optional[np.ndarray], list[WorkItem]]] = []
    for item, thumb in zip(group, thumbs):
        target = None
        if thumb is not None:
            best = _SIMILAR_THRESHOLD
            for rep, bucket in clusters:
                if rep is None:
                    continue
                dist = float(np.abs(rep - thumb).mean())
                if dist <= best:
                    best = dist
                    target = bucket
        if target is None:
            clusters.append((thumb, [item]))
        else:
            target.append(item)
    return [bucket for _, bucket in clusters]


def _thumbnail_batch_size(group: list[WorkItem], threads: int) -> int:
    """一波同时解码多少张缩略图：按解码后大小折算，上限约 1 GiB。"""
    probe = group[0].probe
    if probe is None or probe.width <= 0 or probe.height <= 0:
        return max(1, min(int(threads), 4))
    channels = int(MODE_SPECS.get(probe.mode, {}).get("channels", 3) or 3)
    # 解码缓冲 + 一份临时副本 ≈ 每样本 2 字节，不要按 4 估——会把
    # 一波的并发压得太低（12 MP JPEG 只剩 7 张并发）。
    per_image = max(1, int(probe.width) * int(probe.height) * channels * 2)
    allowed = max(1, (1 << 30) // per_image)
    return max(1, min(int(threads), allowed))


def _split_similar_groups(
    groups: list[list[WorkItem]], threads: int, enabled: bool = True
) -> tuple[list[list[WorkItem]], int]:
    """把同分辨率的大组按内容切开，返回 (新分组, 被切开的组数)。"""
    if not enabled:
        return groups, 0
    out: list[list[WorkItem]] = []
    split_count = 0
    for group in groups:
        if len(group) < _SIMILAR_MIN_GROUP:
            out.append(group)
            continue
        # 缩略图要解码整张图（JPEG 有 draft 捷径，PNG 没有），一次解 24 张
        # 4K RGBA 就是好几个 GB。按"一波约 1 GiB"分批，避免峰值失控。
        batch = _thumbnail_batch_size(group, threads)
        thumbs: list[Optional[np.ndarray]] = []
        with cf.ThreadPoolExecutor(max_workers=batch) as pool:
            for start in range(0, len(group), batch):
                chunk = group[start:start + batch]
                thumbs.extend(
                    pool.map(_file_thumbnail, [item.source.path for item in chunk])
                )
        clusters = _cluster_by_thumbnail(group, thumbs)
        if len(clusters) > 1:
            split_count += 1
        out.extend(clusters)
    return out, split_count


def _null_lease():
    """没有内存预算时的空上下文管理器。"""
    import contextlib

    return contextlib.nullcontext()
# 与 CreateOptions.search_window 的默认值保持一致，用于判断用户是否显式改过。
DEFAULT_SEARCH_WINDOW = 24


@dataclass
class _Result:
    item: WorkItem
    payload: bytes = b""
    codec: str = CODEC_RAW
    kind: str = "image"
    reason: str = ""
    quant: int = 0
    centered: bool = False
    lossy: bool = False
    approx_hash: str = ""
    source_sha: str = ""
    raw_crc: int = 0
    image_format: str = ""
    constant_alpha: Optional[int] = None
    width: int = 0
    height: int = 0
    mode: str = ""
    image: Optional[ImageData] = None
    error: str = ""


@dataclass
class _GroupState:
    key: tuple
    members: list[WorkItem]
    key_id: int = -1
    key_image: Optional[ImageData] = None
    key_sha: str = ""
    key_depth: int = 0
    futures: dict[cf.Future, int] = field(default_factory=dict)
    results: list[Optional[_Result]] = field(default_factory=list)
    group_bytes: int = 0
    # 实际用作 key 的成员在 members 中的下标；前面那些是读不了被跳过的。
    key_index: int = 0


def _split_groups(items: list[WorkItem]) -> list[list[WorkItem]]:
    groups: list[list[WorkItem]] = []
    current_key = None
    current: list[WorkItem] = []
    for item in items:
        if current_key is None or item.group_key != current_key:
            if current:
                groups.append(current)
            current_key = item.group_key
            current = [item]
        else:
            current.append(item)
    if current:
        groups.append(current)
    return groups


def _image_sha(image: ImageData) -> str:
    if image.source_file_hash:
        return image.source_file_hash.hex()
    return ""


def _raw_result(
    item: WorkItem,
    *,
    codec: str,
    payload: bytes,
    reason: str,
    image: Optional[ImageData] = None,
    keep_image: bool = False,
    source_sha: Optional[str] = None,
    image_format: Optional[str] = None,
) -> _Result:
    """构造 raw/store 结果。

    image 为 None 时不解码，尺寸/模式取自扫描阶段的 header 探测结果
    （probe），因此调用方可以完全没有 ImageData。此时 pixel_sha256 为空，
    verify 走字节级 SHA-256 校验（payload 就是源文件字节，等价性已完整覆盖）。
    """
    src_path = item.source.path
    src_size = item.source.path.stat().st_size
    if image is not None:
        source_sha = _image_sha(image)
        fmt = (image.source_format or "").upper()
        constant_alpha = image.constant_alpha
        width, height, mode = image.width, image.height, image.mode
    else:
        if not source_sha:
            source_sha = sha256_bytes(payload).hex()
        fmt = (image_format or "").upper()
        constant_alpha = None
        width = item.probe.width if item.probe else 0
        height = item.probe.height if item.probe else 0
        mode = item.probe.mode if item.probe else ""
    raw_crc = crc32(payload) if codec in {CODEC_RAW, CODEC_STORE} else 0
    if codec == CODEC_RAW:
        packed = jxl.pack(payload)
        if packed is not None:
            # 体积变小才换编码器；raw_crc 仍然按原始字节记录。
            payload, codec = packed, CODEC_JPEGXL
    return _Result(
        item=item,
        payload=payload,
        codec=codec,
        kind="image",
        reason=reason,
        source_sha=source_sha,
        raw_crc=raw_crc,
        image_format=fmt,
        constant_alpha=constant_alpha,
        width=width,
        height=height,
        mode=mode,
        image=image if keep_image else None,
    )


def _lease_bytes(item: WorkItem, kind: str) -> int:
    """按扫描阶段拿到的图片头估算这个任务要占多少内存。"""
    probe = item.probe
    if probe is not None and probe.width > 0 and probe.height > 0:
        spec = MODE_SPECS.get(probe.mode, {})
        channels = int(spec.get("channels", 3) or 3)
        sample_bytes = 2 if probe.mode.startswith("I;16") else (
            4 if probe.mode in {"I", "F"} else 1
        )
        return estimate_working_set(
            probe.width, probe.height, channels, sample_bytes, kind=kind
        )
    try:
        return estimate_raw_lease(item.source.path.stat().st_size)
    except OSError:
        return estimate_raw_lease(0)


def _read_raw_result(item: WorkItem, reason: str = "raw") -> _Result:
    """读原始字节兜底。

    文件读不到时返回带 ``error`` 的结果而不是抛异常：单个文件不可读不应
    让整次归档作废（这是 1290 张、几分钟任务的最后防线）。
    """
    try:
        payload = item.source.path.read_bytes()
    except OSError as exc:
        return _Result(item=item, error=f"{type(exc).__name__}: {exc}")
    try:
        image = load_image(
            item.source.path,
            compute_file_hash=True,
            compute_source_pixel_hash=False,
        )
    except PxcError:
        image = None
    return _raw_result(item, codec=CODEC_RAW, payload=payload, reason=reason, image=image)


def _encode_key(
    item: WorkItem,
    opts: CreateOptions,
    need_image: bool = True,
    budget: Optional[MemoryBudget] = None,
) -> _Result:
    """编码组 key；任何 I/O 失败都转成 ``_Result.error``，不抛异常。"""
    try:
        return _encode_key_impl(item, opts, need_image, budget)
    except (PxcError, OSError, MemoryError) as exc:
        return _Result(item=item, error=f"{type(exc).__name__}: {exc}")


def _encode_key_impl(
    item: WorkItem,
    opts: CreateOptions,
    need_image: bool = True,
    budget: Optional[MemoryBudget] = None,
) -> _Result:
    """按内存预算记账后调用实际编码。"""
    lease = (
        budget.lease(_lease_bytes(item, "key")) if budget is not None
        else _null_lease()
    )
    with lease:
        return _encode_key_body(item, opts, need_image)


def _encode_key_body(
    item: WorkItem,
    opts: CreateOptions,
    need_image: bool = True,
) -> _Result:
    """加载并编码组 key。JPEG 直接保留原始字节。

    ``need_image=False`` 且确定走 raw 直通时，跳过完整解码。raw 直通存的就是
    源文件字节，字节级 SHA-256 已完整覆盖完整性；解码这一步只为确认"能解出
    像素"并取得 constant_alpha，对 raw 条目的重建没有作用，却要占掉这类归档
    创建耗时的四成以上。需要恢复原先的完整校验时用 ``CreateOptions.decode_raw``。
    """
    src_path = item.source.path
    src_size = src_path.stat().st_size
    probe = item.probe
    probed_format = (probe.format or "").upper() if probe else ""
    probed_ready = bool(
        probe is not None
        and probe.width > 0
        and probe.height > 0
        and bool(probe.mode)
    )
    will_passthrough = opts.algorithm == "passthrough" or (
        opts.algorithm == "auto" and probed_format in _RAW_IMAGE_FORMATS
    )
    if (
        will_passthrough
        and probed_ready
        and not need_image
        and not opts.decode_raw
    ):
        raw = src_path.read_bytes()
        return _raw_result(
            item,
            codec=CODEC_RAW,
            payload=raw,
            reason="key_lossy_passthrough",
            image_format=probed_format,
        )

    try:
        image = load_image(
            src_path,
            compute_file_hash=True,
            compute_source_pixel_hash=False,
        )
    except PxcError as exc:
        raw = src_path.read_bytes()
        return _raw_result(item, codec=CODEC_RAW, payload=raw, reason=f"decode_error: {exc}")

    src_format = (image.source_format or "").upper()
    prefer_raw = (
        opts.algorithm == "passthrough"
        or (opts.algorithm == "auto" and src_format in _RAW_IMAGE_FORMATS)
    )
    if prefer_raw:
        raw = src_path.read_bytes()
        return _raw_result(
            item,
            codec=CODEC_RAW,
            payload=raw,
            reason="key_lossy_passthrough",
            image=image,
            keep_image=True,
        )

    try:
        if (
            opts.keyframe_codec in {"auto", "webp"}
            and image.mode in {"RGB", "RGBA"}
            and src_format in _LOSSLESS_FORMATS
        ):
            payload = encode_webp_lossless(image, method=opts.webp_method)
            codec = CODEC_WEBP_LOSSLESS
        else:
            payload = encode_keyframe(image, opts.tile_size, threads=1)
            codec = CODEC_PXC
    except (PxcError, ValueError, OSError) as exc:
        raw = src_path.read_bytes()
        return _raw_result(
            item,
            codec=CODEC_RAW,
            payload=raw,
            reason=f"key_encode_error: {exc}",
            image=image,
            keep_image=True,
        )

    # 像素编码比源文件大时回退原始字节；它仍然可以充当 key。
    if opts.algorithm == "auto" and src_size > 0 and len(payload) >= src_size * 1.02:
        raw = src_path.read_bytes()
        return _raw_result(
            item,
            codec=CODEC_RAW,
            payload=raw,
            reason="key_pixel_larger",
            image=image,
            keep_image=True,
        )

    return _Result(
        item=item,
        payload=payload,
        codec=codec,
        kind="image",
        reason="key",
        approx_hash=image.pixel_hash.hex(),
        source_sha=_image_sha(image),
        image_format=src_format,
        constant_alpha=image.constant_alpha,
        width=image.width,
        height=image.height,
        mode=image.mode,
        image=image,
    )


def _encode_delta(
    key_image: Optional[ImageData],
    key_id: int,
    item: WorkItem,
    opts: CreateOptions,
    budget: Optional[MemoryBudget] = None,
) -> _Result:
    """编码差分；任何 I/O 失败都转成 ``_Result.error``，不抛异常。"""
    try:
        return _encode_delta_impl(key_image, key_id, item, opts, budget)
    except (PxcError, OSError, MemoryError) as exc:
        return _Result(item=item, error=f"{type(exc).__name__}: {exc}")


def _encode_delta_impl(
    key_image: Optional[ImageData],
    key_id: int,
    item: WorkItem,
    opts: CreateOptions,
    budget: Optional[MemoryBudget] = None,
) -> _Result:
    """按内存预算记账后调用实际编码。"""
    lease = (
        budget.lease(_lease_bytes(item, "delta")) if budget is not None
        else _null_lease()
    )
    with lease:
        return _encode_delta_body(key_image, key_id, item, opts)


def _encode_delta_body(
    key_image: Optional[ImageData],
    key_id: int,
    item: WorkItem,
    opts: CreateOptions,
) -> _Result:
    """在 worker 中把一张图编码成 key 的残差；失败/不划算时回退 raw。"""
    src_path = item.source.path
    src_size = src_path.stat().st_size
    try:
        image = load_image(
            src_path,
            compute_file_hash=True,
            compute_source_pixel_hash=False,
        )
    except PxcError as exc:
        try:
            raw = src_path.read_bytes()
        except OSError as read_exc:
            return _Result(item=item, error=f"{exc}; read: {read_exc}")
        return _raw_result(item, codec=CODEC_RAW, payload=raw, reason=f"decode_error: {exc}")

    src_format = (image.source_format or "").upper()
    prefer_raw = (
        opts.algorithm == "passthrough"
        or (opts.algorithm == "auto" and src_format in _RAW_IMAGE_FORMATS)
    )
    if (
        key_image is None
        or key_id < 0
        or opts.reference == "none"
        or opts.algorithm == "passthrough"
        or not _delta_capable(image)
    ):
        return _raw_result(
            item, codec=CODEC_RAW, payload=src_path.read_bytes(),
            reason="delta_not_allowed", image=image,
        )

    is_lossy_image = src_format in _RAW_IMAGE_FORMATS
    quant_base = (
        int(opts.delta_quant)
        if is_lossy_image and image.data.dtype == np.dtype(np.uint8)
        else 0
    )
    quant_values = [quant_base]
    if quant_base > 0 and opts.delta_quant_max > quant_base:
        quant_values.append(int(opts.delta_quant_max))

    best_payload: Optional[bytes] = None
    best_hash = ""
    best_quant = 0
    best_codec = CODEC_PXC_RESIDUAL
    for quant in quant_values:
        try:
            use_zlib = opts.residual_codec == "zlib" or (
                opts.residual_codec != "pxc" and quant > 0
            )
            if use_zlib:
                # q>0 的残差大部分是 128，zlib 比 PXC tile 快一个数量级，
                # 体积通常只大 20–40%，是 JPEG 差分更实用的默认路径。
                payload, approx_hash = encode_zlib_residual(
                    key_image,
                    image,
                    quant=quant,
                    level=opts.zlib_level,
                    centered=True,
                    min_zero_fraction=opts.residual_min_zero_fraction,
                )
                codec = CODEC_ZLIB_RESIDUAL
            else:
                payload, approx_hash = encode_residual_with_preview(
                    key_image,
                    image,
                    opts.tile_size,
                    threads=1,
                    quant=quant,
                    centered=True,
                )
                codec = CODEC_PXC_RESIDUAL
        except (PxcError, ValueError, MemoryError):
            continue
        if best_payload is None or len(payload) < len(best_payload):
            best_payload = payload
            best_hash = approx_hash.hex()
            best_quant = quant
            best_codec = codec
        if src_size <= 0 or len(payload) < src_size * 0.98:
            break

    if (
        best_payload is None
        or (src_size > 0 and len(best_payload) >= src_size * 0.98)
    ):
        return _raw_result(
            item,
            codec=CODEC_RAW,
            payload=src_path.read_bytes(),
            reason="residual_larger_than_source",
            image=image,
        )

    return _Result(
        item=item,
        payload=best_payload,
        codec=best_codec,
        kind="image",
        reason="residual_lossy" if best_quant > 0 else "residual",
        quant=best_quant,
        centered=True,
        lossy=bool(best_quant > 0),
        approx_hash=best_hash,
        source_sha=_image_sha(image),
        image_format=src_format,
        constant_alpha=image.constant_alpha,
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def _delta_capable(image: ImageData) -> bool:
    return (
        image.mode in {"1", "L", "LA", "RGBA", "RGB", "CMYK", "I;16", "I;16L", "I;16B"}
        and image.width > 0
        and image.height > 0
    )


def _restorable_entry(entries: list[ArchiveEntry], entry_id: int) -> bool:
    """DUP 只能指向能还原原始字节的条目，否则默认提取会丢失源字节。"""
    seen: set[int] = set()
    current = entry_id
    while 0 <= current < len(entries):
        if current in seen:
            return False
        seen.add(current)
        entry = entries[current]
        if entry.codec == CODEC_DUP:
            if entry.ref is None:
                return False
            current = entry.ref
            continue
        return entry.codec in {CODEC_RAW, CODEC_JPEGXL, CODEC_STORE, CODEC_ZLIB}
    return False


def _write_result(
    writer: ArchiveWriter,
    entries: list[ArchiveEntry],
    result: _Result,
    raw_hash_index: dict[str, int],
    *,
    ref_id: Optional[int] = None,
    ref_depth: int = 0,
) -> tuple[int, bool]:
    """把一个编码结果写成条目；返回 (写入字节数, 是否重复)。"""
    item = result.item
    source = item.source
    src_path = source.path
    src_size = src_path.stat().st_size
    st = src_path.stat()
    source_sha = result.source_sha

    # 同字节文件：若前面的条目可还原原始字节，则只写 DUP。
    if (
        source_sha
        and source_sha in raw_hash_index
        and result.kind == "image"
    ):
        ref_entry = entries[raw_hash_index[source_sha]]
        if ref_entry.kind == "image" and _restorable_entry(
            entries, raw_hash_index[source_sha]
        ):
            entry = _make_entry(
                len(entries),
                source.rel,
                "image",
                CODEC_DUP,
                size=src_size,
                mtime_ns=st.st_mtime_ns,
                source_sha256=source_sha,
                ref=raw_hash_index[source_sha],
                depth=1,
                width=result.width,
                height=result.height,
                mode=result.mode,
                format_name=result.image_format,
                ext=_source_ext(src_path),
                constant_alpha=result.constant_alpha,
                reason="duplicate",
            )
            entries.append(entry)
            return 0, True

    if result.error:
        # 读取/解码彻底失败时，仍然把原始字节作为普通文件保存，避免丢数据。
        try:
            raw = src_path.read_bytes()
        except OSError:
            raise
        result = _raw_result(item, codec=CODEC_RAW, payload=raw,
                             reason=f"error: {result.error}")

    payload = result.payload
    if result.codec == CODEC_RAW and result.raw_crc == 0 and payload:
        result.raw_crc = crc32(payload)
    info = writer.add_payload(payload)
    entry = _make_entry(
        len(entries),
        source.rel,
        "image",
        result.codec,
        size=src_size,
        mtime_ns=st.st_mtime_ns,
        source_sha256=source_sha,
        payload_info=info,
        raw_crc32=result.raw_crc,
        ref=ref_id if result.codec in RESIDUAL_CODECS else None,
        depth=(ref_depth + 1) if result.codec in RESIDUAL_CODECS else 0,
        width=result.width,
        height=result.height,
        mode=result.mode,
        format_name=result.image_format,
        ext=_source_ext(src_path),
        constant_alpha=result.constant_alpha,
        pixel_sha256=result.approx_hash,
        quant=result.quant,
        centered=result.centered,
        lossy=result.lossy,
        reason=result.reason or result.codec,
    )
    entries.append(entry)
    if source_sha:
        raw_hash_index.setdefault(source_sha, entry.id)
    return len(payload), False


def _self_check_archive(
    path: Path, expected_entries: int, password: Optional[str]
) -> None:
    """重开刚写完的归档，确认头和索引都能读、条目数对得上。

    成本约为一次 `list`（几毫秒），但能在磁盘满/被杀软改写/网络盘掉线的
    情况下当场报错，而不是等下次 verify 才发现。
    """
    try:
        with ArchiveReader(path, password=password) as reader:
            found = len(reader.entries)
            index_length = int(reader.header.index_length)
    except Exception as exc:  # 任何异常都说明产物不可信
        raise PxcsSelfCheckError(
            t("create.self_check_failed", reason=f"{type(exc).__name__}: {exc}")
        ) from exc
    if index_length <= 0:
        raise PxcsSelfCheckError(
            t("create.self_check_failed", reason="index is empty")
        )
    if found != expected_entries:
        raise PxcsSelfCheckError(
            t(
                "create.self_check_failed",
                reason=f"entries {found} != expected {expected_entries}",
            )
        )


def _print_summary(stats: CreateStats, path: Path, opts: CreateOptions) -> None:
    print("-" * 72)
    extra = ""
    if stats.volumes > 1:
        extra += t("create.volumes_suffix", count=stats.volumes)
    if opts.password:
        extra += t(
            "create.encrypted_suffix"
            if opts.encrypt_filenames
            else "create.encrypted_plain_suffix"
        )
    print(summary_text(stats, path, extra))
    if stats.skipped:
        print(t("create.skipped_warning", count=stats.skipped))
        shown = stats.errors[:20]
        for detail in shown:
            print(t("create.skipped_detail", detail=detail))
        remaining = len(stats.errors) - len(shown)
        if remaining > 0:
            print(t("create.skipped_more", count=remaining))


def create_grouped_archive(
    output: Path,
    inputs: Iterable[str],
    opts: CreateOptions,
) -> CreateStats:
    """并行创建 resolution/regex 分组归档。"""
    if opts.group_mode not in {"resolution", "regex"}:
        raise ValueError(t("create.group_mode_only"))
    if opts.group_regex:
        import re
        try:
            re.compile(opts.group_regex)
        except re.error as exc:
            raise ValueError(t("create.group_regex_invalid", regex=opts.group_regex, reason=exc)) from exc

    if not opts.quiet:
        # 这三个选项只被 natural 模式读取；分组模式下静默忽略最坑人，明确告警。
        ignored: list[str] = []
        if opts.reference in {"chain", "anchor"}:
            ignored.append("--reference")
        if opts.search_window != DEFAULT_SEARCH_WINDOW:
            ignored.append("--search")
        if opts.keyframe_interval != DEFAULT_KEYFRAME_INTERVAL:
            ignored.append("--keyframe")
        if ignored:
            print(t("create.option_ignored", options=", ".join(ignored)))

    # threads=0 → 用满所有核心；预算默认取可用内存的 60%（见 limits.py）。
    threads = resolve_threads(opts.threads)
    budget_bytes = resolve_memory_budget(opts.memory_limit_mb)
    budget = MemoryBudget(budget_bytes)

    set_level = None
    from .codec import set_zlib_level as set_level  # local import keeps startup light
    set_level(opts.zlib_level)

    started = time.perf_counter()
    input_paths = list(inputs)
    from .paths import collect_source_files
    sources = collect_source_files(
        input_paths, recursive=opts.recursive, exclude=output
    )
    skipped_artifacts = [
        item for item in sources
        if _is_archive_artifact(item.path.name, output.name)
        or _is_pxcs_artifact(item.path.name)
    ]
    if skipped_artifacts and not opts.quiet:
        print(
            t("create.skip_artifacts", count=len(skipped_artifacts))
        )
    sources = [
        item for item in sources
        if not _is_archive_artifact(item.path.name, output.name)
        and not _is_pxcs_artifact(item.path.name)
    ]
    if opts.group_mode == "regex":
        import re as _re
        try:
            pat = _re.compile(opts.group_regex)
            hits = sum(1 for s in sources if pat.search(Path(s.rel).stem))
        except _re.error:
            hits = 0
        if hits == 0 and not opts.quiet:
            print(
                t(
                    "create.regex_no_match",
                    regex=repr(opts.group_regex),
                    fallback=opts.group_regex_fallback,
                )
            )
    scan_callback = None
    if not opts.quiet:
        print(t("scan.start", count=len(sources)), flush=True)

        def scan_callback(done: int, total: int) -> None:
            print(f"\r[SCAN] {done}/{total}", end="", flush=True)

    work_items = _order_work_items(
        sources,
        group_mode=opts.group_mode,
        group_regex=opts.group_regex,
        group_regex_fallback=opts.group_regex_fallback,
        scan_callback=scan_callback,
    )
    if not opts.quiet:
        print()
    groups = _split_groups(work_items)
    image_groups = [
        g for g in groups if g and g[0].probe is not None
    ]
    image_groups, split_groups = _split_similar_groups(
        image_groups, threads, enabled=bool(opts.cluster_similar)
    )
    if split_groups and not opts.quiet:
        print(t("create.similar_split", groups=split_groups))
    parallel_groups = resolve_parallel_groups(opts.parallel_groups, threads)
    if image_groups:
        # 每个活跃组会常驻一张参考图。这里按 90 分位（而不是最大值）估算：
        # 用最大值会被个别超大图带偏，把并行度压到 1-3，反而跑不满 CPU。
        key_sizes = sorted(
            estimate_key_retention(
                g[0].probe.width,
                g[0].probe.height,
                int(MODE_SPECS.get(g[0].probe.mode, {}).get("channels", 3) or 3),
                2 if g[0].probe.mode.startswith("I;16") else 1,
            )
            for g in image_groups
            if g[0].probe is not None
        )
        if key_sizes:
            index = min(len(key_sizes) - 1, int(len(key_sizes) * 0.9))
            parallel_groups = cap_parallel_groups_by_memory(
                parallel_groups, budget_bytes, key_sizes[index]
            )
    file_groups = [
        g for g in groups if g and g[0].probe is None
    ]
    total_source = sum(s.path.stat().st_size for s in sources)
    stats = CreateStats(
        files_total=len(sources),
        groups=len(image_groups),
        original_bytes=total_source,
    )
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
    if not opts.quiet and not opts.progress:
        print(
            t(
                "create.info",
                count=len(sources),
                size=human_size(total_source),
                group_mode=opts.group_mode,
                groups=len(image_groups),
                workers=threads,
                parallel_groups=parallel_groups,
                quant=opts.delta_quant,
                memory=human_size(budget_bytes),
            )
        )
        if parallel_groups < threads:
            # 不是 bug，是内存预算把并行度压下来了；说清楚原因。
            print(
                t(
                    "create.memory_limited",
                    groups=parallel_groups,
                    threads=threads,
                    memory=human_size(budget_bytes),
                )
            )
    if image_groups and not opts.quiet and not opts.show_groups:
        largest = max(image_groups, key=len)
        probe = largest[0].probe
        print(
            t(
                "create.groups_summary",
                groups=len(image_groups),
                width=probe.width,
                height=probe.height,
                count=len(largest),
            )
        )
    if opts.show_groups and not opts.quiet:
        for index, group in enumerate(image_groups, 1):
            probe = group[0].probe
            print(
                t(
                    "create.group_line",
                    index=index,
                    width=probe.width,
                    height=probe.height,
                    mode=probe.mode,
                    count=len(group),
                    key=group[0].source.rel,
                )
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
    executor = cf.ThreadPoolExecutor(max_workers=threads)
    states: list[_GroupState] = []
    next_group = 0
    done_entries = 0
    processed_bytes = 0
    tasks_done = 0
    tasks_bytes = 0
    current_name = ""
    detail_done = 0

    def update_progress(name: str = "") -> None:
        nonlocal current_name
        if name:
            current_name = name
        progress.update(
            min(len(work_items), max(done_entries, tasks_done)),
            current_name or "...",
            max(processed_bytes, tasks_bytes),
            writer.position(),
        )

    def record_skipped(item: WorkItem, message: str) -> None:
        """记录一个被跳过的文件：不让整次归档失败，但必须让用户看见。"""
        nonlocal done_entries, tasks_done, tasks_bytes
        stats.errors.append(f"{item.source.rel}: {message}")
        stats.skipped += 1
        done_entries += 1
        tasks_done += 1
        try:
            tasks_bytes += item.source.path.stat().st_size
        except OSError:
            pass
        if opts.verbose and not opts.quiet:
            print(t("create.skipped_line", path=item.source.rel, reason=message))
        update_progress(item.source.rel)

    def encode_group_key(
        state: _GroupState,
    ) -> tuple[
        int, Optional[WorkItem], Optional[_Result], list[tuple[WorkItem, str]]
    ]:
        """在工作线程里挑 key 候选并编码。

        组内第一个成员读不了时向后最多试 ``MAX_KEY_CANDIDATES`` 个候选取一个
        当 key，避免一个坏文件让整组退化成 raw。

        注意：这里**不能**碰 entries / stats / progress，它们只有主线程能写。
        候选失败只收集错误文本，回到主线程再统一 ``record_skipped``。
        """
        need_image = len(state.members) > 1
        limit = min(len(state.members), MAX_KEY_CANDIDATES)
        failures: list[tuple[WorkItem, str]] = []
        for offset in range(limit):
            candidate = state.members[offset]
            result = _encode_key(
                candidate, opts, need_image=need_image, budget=budget
            )
            if not result.error:
                return offset, candidate, result, failures
            failures.append((candidate, result.error))
        return -1, None, None, failures

    def commit_group_key(
        state: _GroupState,
        packed: tuple[
            int, Optional[WorkItem], Optional[_Result], list[tuple[WorkItem, str]]
        ],
    ) -> bool:
        """把已经编码好的 key 落成条目，返回是否成功（只在主线程调用）。"""
        nonlocal done_entries, processed_bytes, tasks_done, tasks_bytes
        for candidate, message in packed[3]:
            record_skipped(candidate, message)
        if packed[0] < 0:
            for candidate in state.members[MAX_KEY_CANDIDATES:]:
                record_skipped(candidate, "group_key_unavailable")
            return False

        offset, key_item, result = packed[0], packed[1], packed[2]
        state.key_index = offset
        state.key_sha = result.source_sha
        if result.codec == CODEC_PXC_RESIDUAL:
            # key 永远不应该是残差；防御性回退。
            result = _read_raw_result(key_item, reason="key_invalid_residual")
            if result.error:
                record_skipped(key_item, result.error)
                return False
        try:
            st = key_item.source.path.stat()
        except OSError as exc:
            record_skipped(key_item, f"{type(exc).__name__}: {exc}")
            return False
        state.key_depth = 0
        state.key_id = len(entries)
        payload = result.payload
        if result.codec == CODEC_RAW and result.raw_crc == 0 and payload:
            result.raw_crc = crc32(payload)
        info = writer.add_payload(payload)
        entry = _make_entry(
            len(entries),
            key_item.source.rel,
            "image",
            result.codec,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            source_sha256=result.source_sha,
            payload_info=info,
            raw_crc32=result.raw_crc,
            width=result.width,
            height=result.height,
            mode=result.mode,
            format_name=result.image_format,
            ext=_source_ext(key_item.source.path),
            constant_alpha=result.constant_alpha,
            pixel_sha256=result.approx_hash,
            quant=0,
            centered=False,
            lossy=False,
            reason=result.reason or "key",
        )
        entries.append(entry)
        if result.source_sha:
            raw_hash_index.setdefault(result.source_sha, entry.id)
        stats.keys += 1
        stats.images += 1
        if result.codec in {CODEC_RAW, CODEC_JPEGXL}:
            stats.raw_images += 1
        elif result.codec in {CODEC_PXC, CODEC_WEBP_LOSSLESS}:
            stats.keyframes += 1
        processed_bytes += entry.size
        done_entries += 1
        tasks_done += 1
        tasks_bytes += entry.size
        state.key_image = result.image
        update_progress(key_item.source.rel)
        progress.detail(
            t(
                "progress.detail_key",
                path=key_item.source.rel,
                original=human_size(entry.size),
                stored=human_size(len(payload)),
                codec=result.codec,
            )
        )
        return True

    def collect_future(fut: cf.Future, state: _GroupState, index: int) -> None:
        nonlocal detail_done, tasks_done, tasks_bytes
        try:
            result = fut.result()
        except Exception as exc:  # worker 内已经尽量兜底；这里防止线程崩溃
            result = _Result(item=state.members[index], error=str(exc))
        state.results[index] = result
        detail_done += 1
        tasks_done += 1
        try:
            tasks_bytes += result.item.source.path.stat().st_size
        except OSError:
            pass
        update_progress(result.item.source.rel)

    def finish_state(state: _GroupState) -> None:
        nonlocal done_entries, processed_bytes
        # key 已写入；按组内自然序写剩余条目。
        for index, item in enumerate(state.members[1:], 1):
            if index <= state.key_index:
                # 这些成员在挑选 key 时已确认读不了，前面已记过跳过。
                continue
            result = state.results[index]
            if result is None:
                result = _read_raw_result(item, reason="missing_worker_result")
            if result.error:
                fallback = _read_raw_result(item, reason=f"error: {result.error}")
                if fallback.error:
                    record_skipped(item, fallback.error)
                    continue
                result = fallback
            before = len(entries)
            try:
                written, is_dup = _write_result(
                    writer,
                    entries,
                    result,
                    raw_hash_index,
                    ref_id=state.key_id if result.codec in RESIDUAL_CODECS else None,
                    ref_depth=state.key_depth,
                )
            except (PxcError, OSError) as exc:
                # 单张失败不应毁掉整个归档；原始字节兜底，再失败就跳过。
                fallback = _read_raw_result(item, reason=f"write_error: {exc}")
                if fallback.error:
                    record_skipped(item, fallback.error)
                    continue
                try:
                    written, is_dup = _write_result(
                        writer, entries, fallback, raw_hash_index,
                        ref_id=None, ref_depth=0,
                    )
                except (PxcError, OSError) as exc2:
                    record_skipped(item, str(exc2))
                    continue
            entry = entries[-1]
            processed_bytes += entry.size
            done_entries += 1
            if is_dup:
                stats.dups += 1
            if entry.codec == CODEC_DUP:
                stats.images += 1
            else:
                stats.images += 1
                if entry.codec in {CODEC_RAW, CODEC_JPEGXL}:
                    stats.raw_images += 1
                elif entry.codec in RESIDUAL_CODECS:
                    stats.deltas += 1
                    if entry.lossy:
                        stats.lossy_deltas += 1
                elif entry.codec in {CODEC_PXC, CODEC_WEBP_LOSSLESS}:
                    stats.keyframes += 1
            if entry.codec in RESIDUAL_CODECS:
                progress.detail(
                    t(
                        "progress.detail_delta",
                        path=entry.path,
                        ref=entry.ref,
                        original=human_size(entry.size),
                        stored=human_size(entry.stored_length),
                    )
                    + (f" q={entry.quant}" if entry.quant else "")
                )
            elif entry.codec == CODEC_DUP:
                progress.detail(
                    t("progress.detail_dup", path=entry.path, ref=entry.ref)
                )
            else:
                progress.detail(
                    t(
                        "progress.detail_raw",
                        path=entry.path,
                        size=human_size(entry.size),
                    )
                )
            update_progress(item.source.rel)

    def write_files(group: list[WorkItem]) -> None:
        nonlocal done_entries, processed_bytes
        for item in group:
            try:
                raw = item.source.path.read_bytes()
            except OSError as exc:
                record_skipped(item, f"{type(exc).__name__}: {exc}")
                continue
            codec, payload, raw_crc = _compress_file(raw, opts.zlib_level)
            source_sha = sha256_bytes(raw).hex()
            if source_sha in raw_hash_index:
                ref_id = raw_hash_index[source_sha]
                ref = entries[ref_id]
                if ref.kind == "file":
                    entry = _make_entry(
                        len(entries), item.source.rel, "file", CODEC_DUP,
                        size=len(raw), mtime_ns=item.source.path.stat().st_mtime_ns,
                        source_sha256=source_sha, ref=ref_id, depth=1,
                        format_name=_source_format(item.source.path),
                        ext=_source_ext(item.source.path), reason="duplicate",
                    )
                    entries.append(entry)
                    stats.dups += 1
                    stats.files += 1
                    done_entries += 1
                    processed_bytes += entry.size
                    update_progress(item.source.rel)
                    continue
            info = writer.add_payload(payload)
            entry = _make_entry(
                len(entries), item.source.rel, "file", codec,
                size=len(raw), mtime_ns=item.source.path.stat().st_mtime_ns,
                source_sha256=source_sha, payload_info=info,
                raw_crc32=raw_crc, format_name=_source_format(item.source.path),
                ext=_source_ext(item.source.path), reason=codec,
            )
            entries.append(entry)
            raw_hash_index.setdefault(source_sha, entry.id)
            stats.files += 1
            done_entries += 1
            processed_bytes += entry.size
            update_progress(item.source.rel)

    try:
        # key 编码原本在主线程串行执行：PNG 组的 webp_lossless 单张要 9 秒，
        # 几百个小组就是几百秒单核（实测只跑出 2.5 核）。现在提交到线程池，
        # 但**严格按提交顺序**落条目，所以条目顺序与串行版本完全一致。
        pending_keys: list[tuple[_GroupState, cf.Future]] = []
        while next_group < len(image_groups) or states or pending_keys:
            # 1) 提交新组的 key 编码，受并行窗口约束
            while (
                len(states) + len(pending_keys) < parallel_groups
                and next_group < len(image_groups)
            ):
                group = image_groups[next_group]
                next_group += 1
                state = _GroupState(key=group[0].group_key, members=group)
                state.results = [None] * len(group)
                pending_keys.append(
                    (state, executor.submit(encode_group_key, state))
                )

            # 2) 按提交顺序落 key：保证条目顺序可复现
            while pending_keys and pending_keys[0][1].done():
                state, key_future = pending_keys.pop(0)
                try:
                    packed = key_future.result()
                except Exception as exc:  # worker 已兜底，这里只防线程崩溃
                    packed = (-1, None, None, [(state.members[0], str(exc))])
                if not commit_group_key(state, packed):
                    # 整组都读不了：成员已逐个记为跳过，不提交差分。
                    continue
                states.append(state)
                first_delta = state.key_index + 1
                for index, item in enumerate(
                    state.members[first_delta:], first_delta
                ):
                    fut = executor.submit(
                        _encode_delta,
                        state.key_image,
                        state.key_id,
                        item,
                        opts,
                        budget,
                    )
                    state.futures[fut] = index

            # 3) 队首组完成即写它的条目（保持组顺序稳定）
            while states and not states[0].futures:
                finish_state(states.pop(0))

            # 4) 等任意一个未完成的工作
            all_futures = [f for st in states for f in st.futures]
            if pending_keys:
                all_futures.append(pending_keys[0][1])
            if not all_futures:
                # 窗口为空但还有组没提交（parallel_groups 小于组数时会发生）：
                # 回到第 1 步继续提交，不能在这里直接收工。
                if next_group < len(image_groups):
                    continue
                break
            done, _ = cf.wait(all_futures, return_when=cf.FIRST_COMPLETED)
            for state in states:
                for fut in list(state.futures):
                    if fut in done:
                        index = state.futures.pop(fut)
                        collect_future(fut, state, index)

        if next_group != len(image_groups):
            # 漏提交/提前退出会让归档静默缺条目。宁可当场失败，也不要写出一个
            # 看起来正常、实际少了文件的归档。
            raise PxcsFormatError(
                t(
                    "create.groups_incomplete",
                    done=next_group,
                    total=len(image_groups),
                )
            )

        for group in file_groups:
            write_files(group)

        # 全部条目写完后收尾。
        flags = compute_flags(entries)
        final_path, archive_size = writer.finish(entries, flags)
        if opts.self_check:
            # 刚写完的归档立刻重开：宁可当场报错，也不要留下一个读不出来的文件。
            _self_check_archive(final_path, len(entries), opts.password)
        stats.archive_bytes = archive_size
        stats.volumes = len(writer.output_files)
        stats.elapsed = time.perf_counter() - started
        progress.update(len(work_items), current_name or str(input_paths[0]),
                        total_source, archive_size)
        progress.finish()
        if opts.verbose and not opts.quiet:
            # 让用户能看到内存护栏是否真的生效，而不是只看到一句"上限"。
            print(
                t(
                    "create.memory_stats",
                    peak=human_size(budget.peak),
                    budget=human_size(budget.total),
                    waits=budget.waits,
                )
            )
    except BaseException:
        writer.abort()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not opts.quiet:
        _print_summary(stats, final_path, opts)
    return stats


def _source_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "JPEG"
    if ext == ".tif":
        return "TIFF"
    return ext.lstrip(".").upper()
