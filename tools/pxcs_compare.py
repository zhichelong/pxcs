"""PXCS 语义对比工具。

用途
----
比较两个 PXCS 归档在“语义层面”是否等价，主要服务于将来 C++ 重实现与
Python 参考实现的差分测试：

- ``ERROR``：语义不等价，必须处理（版本、flags 关键位、条目数量/路径集合、
  codec、明文 payload 长度与字节、``data_crc32`` / ``data_sha256``、
  差分参考 ``ref``、``quant`` / ``centered`` / ``lossy`` 等）；
- ``WARN``：已知可忽略、但需要提示的差异（``archive_id``、payload 物理偏移、
  index 物理偏移/长度、header/footer/index CRC 等）；
- ``INFO``：两侧版本、条目数、原始/归档体积、压缩率、分卷数、加密状态。

为什么必须忽略 ``archive_id``
-----------------------------
``archive_id`` 由 ``ArchiveWriter`` 用 ``uuid.uuid4()`` 随机生成，并被写进
header、footer 和 index JSON；索引压缩后的长度、header/footer/index 的 CRC
也都随之变化。因此同一批输入创建两次的归档**永远不可能字节相同**。
本工具忽略 ``archive_id`` 本身，并把由它派生的 CRC/长度差异降级为
``WARN``（不参与退出码），这正是本工具存在的理由。

payload 如何比较
----------------
一律通过 ``ArchiveReader.read_entry_data()`` 读取**明文**：未加密归档就是
原始 payload 字节，加密归档会先用口令解密并校验 AEAD、CRC 和 SHA-256。
不按文件偏移直接比字节，因此分卷、加密、物理布局差异都不会造成误报。
代价是两侧 payload 会同时进入内存，本工具不追求大归档下的内存占用。

退出码
------
- ``0``：语义等价（允许存在 ``WARN``）；
- ``1``：存在 ``ERROR``；``--strict`` 下“可升级的 WARN”也算；
- ``2``：参数或 IO 错误（归档打不开、缺口令、口令错误、路径不存在等）。

用法
----
::

    python -m tools.pxcs_compare A.pxcs B.pxcs [--password-file PATH]
                                [--password-stdin] [--json] [--strict]

两个位置参数都可以是任意一个分卷文件（如 ``b.pxcs.002``），
``volumes.open_source`` 会自动定位整套分卷。

``--strict`` 的语义
-------------------
默认只把语义字段的差异计为不等价；``--strict`` 额外把“布局/元数据”层面的
``WARN``（payload 偏移、条目顺序、index 偏移、``mtime_ns``、``source_sha256``、
``pixel_sha256``、``format`` / ``ext`` 等索引字段）也计为不等价。
纯 ``archive_id`` 派生的差异（archive_id 本身、三个 CRC、index 长度）在
``--strict`` 下仍然忽略。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pxc_core.errors import PxcError
from pxc_core.i18n import t
from pxc_core.utils import human_size, ratio_text, sha256_bytes

from pxcs_core.container import ArchiveEntry, ArchiveReader
from pxcs_core.crypto import PxcAuthenticationError, PxcPasswordRequired
from pxcs_core.format import (
    CODEC_DUP,
    FLAG_ENCRYPTED,
    FLAG_HAS_DELTA,
    FLAG_HAS_DUP,
    FLAG_HAS_FILES,
    FLAG_HAS_LOSSY_DELTA,
    FLAG_HAS_RAW_IMAGES,
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
)

# TODO i18n: 本工具新增的对比文案暂用中文字面量；等 language/<code>.json
# 增加 "compare" 段落后再统一迁移到 t()。已有的通用键（口令、cli.fail）仍走 t()。

TOOL_NAME = "pxcs_compare"
JSON_SCHEMA_VERSION = 1

EXIT_EQUIVALENT = 0
EXIT_DIFFERENT = 1
EXIT_USAGE_ERROR = 2

_SEVERITY_ERROR = "ERROR"
_SEVERITY_WARN = "WARN"

# 已知 flag 位 --> 可读名；顺序与 format.py 保持一致。
_FLAG_LABELS: tuple[tuple[int, str], ...] = (
    (FLAG_HAS_DELTA, "HAS_DELTA"),
    (FLAG_HAS_FILES, "HAS_FILES"),
    (FLAG_HAS_DUP, "HAS_DUP"),
    (FLAG_HAS_RAW_IMAGES, "HAS_RAW_IMAGES"),
    (FLAG_HAS_LOSSY_DELTA, "HAS_LOSSY_DELTA"),
    (FLAG_ENCRYPTED, "ENCRYPTED"),
)
_KNOWN_FLAGS = 0
for _bit, _name in _FLAG_LABELS:
    _KNOWN_FLAGS |= _bit

# 只影响索引/校验、不影响“解出来的内容”的条目字段；差异记为可升级 WARN。
_METADATA_FIELDS: tuple[str, ...] = (
    "format",
    "ext",
    "reason",
    "mtime_ns",
    "source_sha256",
    "pixel_sha256",
    "raw_crc32",
    "depth",
    "flags",
)


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class Difference:
    """一条对比差异。"""

    severity: str          # ERROR / WARN
    category: str          # 机器可读的类别，如 entry_payload / archive_id
    message: str           # 面向用户的说明（中文）
    entry: Optional[str] = None      # 相关条目路径
    entry_id: Optional[int] = None   # 相关条目 id（公共条目取左侧 id）
    left: Any = None       # 左侧取值，仅用于展示/断言
    right: Any = None      # 右侧取值
    strict_fatal: bool = False       # --strict 下是否计为不等价

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
        }
        if self.entry is not None:
            data["entry"] = self.entry
        if self.entry_id is not None:
            data["entry_id"] = self.entry_id
        if self.left is not None:
            data["left"] = _json_value(self.left)
        if self.right is not None:
            data["right"] = _json_value(self.right)
        if self.strict_fatal:
            data["strict_fatal"] = True
        return data


@dataclass
class ArchiveSummary:
    """单个归档的概览信息（对应输出里的 INFO 行）。"""

    path: Path
    version_major: int
    version_minor: int
    flags: int
    encrypted: bool
    names_encrypted: bool
    volumes: int
    logical_size: int
    physical_size: int
    index_offset: int
    index_length: int
    header_crc32: Optional[int]
    footer_crc32: Optional[int]
    index_crc32: Optional[int]
    archive_id: str
    entry_count: int
    image_count: int
    file_count: int
    original_size: int

    @property
    def version(self) -> str:
        return f"{self.version_major}.{self.version_minor}"

    @property
    def compression_ratio(self) -> Optional[float]:
        if self.original_size <= 0:
            return None
        return self.logical_size / self.original_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "version": self.version,
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "flags": self.flags,
            "flags_set": _flag_names(self.flags),
            "encrypted": self.encrypted,
            "names_encrypted": self.names_encrypted,
            "volumes": self.volumes,
            "archive_id": self.archive_id,
            "entry_count": self.entry_count,
            "image_count": self.image_count,
            "file_count": self.file_count,
            "original_size": self.original_size,
            "logical_size": self.logical_size,
            "physical_size": self.physical_size,
            "compression_ratio": self.compression_ratio,
            "index_offset": self.index_offset,
            "index_length": self.index_length,
            "header_crc32": self.header_crc32,
            "footer_crc32": self.footer_crc32,
            "index_crc32": self.index_crc32,
        }


@dataclass
class CompareReport:
    """两个归档的完整对比结果。"""

    left: ArchiveSummary
    right: ArchiveSummary
    differences: list[Difference] = field(default_factory=list)
    strict: bool = False

    @property
    def errors(self) -> list[Difference]:
        return [d for d in self.differences if d.severity == _SEVERITY_ERROR]

    @property
    def warnings(self) -> list[Difference]:
        return [d for d in self.differences if d.severity == _SEVERITY_WARN]

    @property
    def strict_failures(self) -> list[Difference]:
        """``--strict`` 下会额外导致失败的可升级 WARN。"""
        return [d for d in self.warnings if d.strict_fatal]

    @property
    def equivalent(self) -> bool:
        if self.errors:
            return False
        if self.strict and self.strict_failures:
            return False
        return True

    @property
    def exit_code(self) -> int:
        return EXIT_EQUIVALENT if self.equivalent else EXIT_DIFFERENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": TOOL_NAME,
            "schema_version": JSON_SCHEMA_VERSION,
            "strict": self.strict,
            "equivalent": self.equivalent,
            "exit_code": self.exit_code,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "counts": {
                "error": len(self.errors),
                "warn": len(self.warnings),
                "strict_fatal": len(self.strict_failures),
            },
            "differences": [d.to_dict() for d in self.differences],
        }


# ---------------------------------------------------------------- 对比核心


def compare_archives(
    left_path: Path,
    right_path: Path,
    *,
    password: Optional[str] = None,
    strict: bool = False,
) -> CompareReport:
    """打开两个归档并做语义对比。

    归档本身打不开、缺口令、口令错误等输入问题会直接抛出异常（由 CLI 映射为
    退出码 2）；单条 payload 读取失败则记为一条 ``ERROR``，继续比对其余条目。
    """
    left_path = Path(left_path)
    right_path = Path(right_path)
    with (
        ArchiveReader(left_path, password=password) as left_reader,
        ArchiveReader(right_path, password=password) as right_reader,
    ):
        left = _summarize(left_reader, left_path)
        right = _summarize(right_reader, right_path)
        differences: list[Difference] = []
        _compare_headers(left_reader, right_reader, differences)
        _compare_identity(left, right, differences)
        _compare_entries(left_reader, right_reader, differences)
        _compare_index_layout(left, right, differences)
    return CompareReport(
        left=left,
        right=right,
        differences=differences,
        strict=bool(strict),
    )


def _summarize(reader: ArchiveReader, path: Path) -> ArchiveSummary:
    header = reader.header
    entries = reader.entries
    images = sum(1 for entry in entries if entry.kind == "image")
    header_crc32, footer_crc32, index_crc32 = _raw_crc32(reader)
    return ArchiveSummary(
        path=path,
        version_major=header.version_major,
        version_minor=header.version_minor,
        flags=header.flags,
        encrypted=reader.is_encrypted,
        names_encrypted=reader.names_encrypted,
        volumes=len(reader.volume_paths),
        logical_size=reader.logical_size,
        physical_size=reader.physical_size,
        index_offset=header.index_offset,
        index_length=header.index_length,
        header_crc32=header_crc32,
        footer_crc32=footer_crc32,
        index_crc32=index_crc32,
        archive_id=header.archive_id.hex(),
        entry_count=len(entries),
        image_count=images,
        file_count=len(entries) - images,
        original_size=sum(entry.size for entry in entries),
    )


def _compare_headers(
    left_reader: ArchiveReader,
    right_reader: ArchiveReader,
    out: list[Difference],
) -> None:
    """版本、flags 关键位、条目数量。"""
    left = left_reader.header
    right = right_reader.header
    if (left.version_major, left.version_minor) != (
        right.version_major,
        right.version_minor,
    ):
        out.append(
            Difference(
                _SEVERITY_ERROR,
                "version",
                f"版本不同: {left.version_major}.{left.version_minor} != "
                f"{right.version_major}.{right.version_minor}",
                left=f"{left.version_major}.{left.version_minor}",
                right=f"{right.version_major}.{right.version_minor}",
            )
        )

    differing = (left.flags ^ right.flags) & _KNOWN_FLAGS
    if differing:
        parts: list[str] = []
        for bit, name in _FLAG_LABELS:
            if not differing & bit:
                continue
            left_state = "置位" if left.flags & bit else "未置位"
            right_state = "置位" if right.flags & bit else "未置位"
            parts.append(f"{name}（左 {left_state} / 右 {right_state}）")
        out.append(
            Difference(
                _SEVERITY_ERROR,
                "flags",
                "flags 关键位不同: " + ", ".join(parts),
                left=left.flags,
                right=right.flags,
            )
        )

    unknown = (left.flags ^ right.flags) & ~_KNOWN_FLAGS
    if unknown:
        # 旧程序不认识的新 flag 位:当前不影响已知语义，但重实现必须一致。
        out.append(
            Difference(
                _SEVERITY_WARN,
                "flags_unknown",
                f"flags 未知位不同: 0x{unknown:08x}（当前实现不使用，按设计忽略）",
                left=left.flags,
                right=right.flags,
                strict_fatal=True,
            )
        )

    if left.entry_count != right.entry_count:
        out.append(
            Difference(
                _SEVERITY_ERROR,
                "entry_count",
                f"条目数量不同: {left.entry_count} != {right.entry_count}",
                left=left.entry_count,
                right=right.entry_count,
            )
        )


def _compare_identity(
    left: ArchiveSummary,
    right: ArchiveSummary,
    out: list[Difference],
) -> None:
    """archive_id 及其派生的 CRC。这些差异永远不应导致退出码 1。"""
    if left.archive_id != right.archive_id:
        out.append(
            Difference(
                _SEVERITY_WARN,
                "archive_id",
                "archive_id 不同（uuid4，每次创建随机，按设计忽略）: "
                f"{left.archive_id[:12]}… != {right.archive_id[:12]}…",
                left=left.archive_id,
                right=right.archive_id,
            )
        )
    for category, label, left_value, right_value in (
        ("header_crc", "header CRC32", left.header_crc32, right.header_crc32),
        ("footer_crc", "footer CRC32", left.footer_crc32, right.footer_crc32),
        ("index_crc", "index CRC32", left.index_crc32, right.index_crc32),
    ):
        if left_value is None or right_value is None or left_value == right_value:
            continue
        out.append(
            Difference(
                _SEVERITY_WARN,
                category,
                f"{label} 不同: 0x{left_value:08x} != 0x{right_value:08x}"
                "（覆盖 archive_id/索引内容，按设计忽略）",
                left=left_value,
                right=right_value,
            )
        )


def _compare_entries(
    left_reader: ArchiveReader,
    right_reader: ArchiveReader,
    out: list[Difference],
) -> None:
    """按路径匹配条目，逐条比较语义字段与明文 payload。"""
    left_entries = left_reader.entries
    right_entries = right_reader.entries
    left_by_path = {entry.path: entry for entry in left_entries}
    right_by_path = {entry.path: entry for entry in right_entries}

    left_only = sorted(set(left_by_path) - set(right_by_path))
    right_only = sorted(set(right_by_path) - set(left_by_path))
    for path in left_only:
        entry = left_by_path[path]
        out.append(
            Difference(
                _SEVERITY_ERROR,
                "entry_missing",
                f"条目只存在于左侧（右归档缺失）: {path}",
                entry=path,
                entry_id=entry.id,
            )
        )
    for path in right_only:
        entry = right_by_path[path]
        out.append(
            Difference(
                _SEVERITY_ERROR,
                "entry_extra",
                f"条目只存在于右侧（左归档缺失）: {path}",
                entry=path,
                entry_id=entry.id,
            )
        )

    common = sorted(
        set(left_by_path) & set(right_by_path),
        key=lambda path: left_by_path[path].id,
    )
    for path in common:
        _compare_entry(
            left_reader, right_reader, left_by_path[path], right_by_path[path], out
        )

    if not left_only and not right_only:
        left_order = [entry.path for entry in left_entries]
        right_order = [entry.path for entry in right_entries]
        if left_order != right_order:
            position = next(
                index
                for index, (left_path, right_path) in enumerate(
                    zip(left_order, right_order)
                )
                if left_path != right_path
            )
            out.append(
                Difference(
                    _SEVERITY_WARN,
                    "entry_order",
                    f"条目顺序不同（自第 {position} 条起: "
                    f"{left_order[position]} / {right_order[position]}）；"
                    "顺序只影响物理布局",
                    left=left_order[position],
                    right=right_order[position],
                    strict_fatal=True,
                )
            )


def _compare_entry(
    left_reader: ArchiveReader,
    right_reader: ArchiveReader,
    left_entry: ArchiveEntry,
    right_entry: ArchiveEntry,
    out: list[Difference],
) -> None:
    """比较同一个路径在两边的条目。

    严重度划分:
    - ERROR：改了解出来的东西（codec、payload、ref、quant/lossy、尺寸等）；
    - WARN：只改索引/布局/校验字段（偏移、存储长度、元数据）。
    """
    path = left_entry.path

    def error(
        category: str, message: str, left: Any = None, right: Any = None
    ) -> None:
        out.append(
            Difference(
                _SEVERITY_ERROR,
                category,
                message,
                entry=path,
                entry_id=left_entry.id,
                left=left,
                right=right,
            )
        )

    def warn(
        category: str,
        message: str,
        *,
        left: Any = None,
        right: Any = None,
        strict_fatal: bool = True,
    ) -> None:
        out.append(
            Difference(
                _SEVERITY_WARN,
                category,
                message,
                entry=path,
                entry_id=left_entry.id,
                left=left,
                right=right,
                strict_fatal=strict_fatal,
            )
        )

    if left_entry.kind != right_entry.kind:
        error(
            "entry_kind",
            f"条目类型不同: {left_entry.kind} != {right_entry.kind}",
            left_entry.kind,
            right_entry.kind,
        )
    if left_entry.codec != right_entry.codec:
        error(
            "entry_codec",
            f"codec 不同: {left_entry.codec} != {right_entry.codec}",
            left_entry.codec,
            right_entry.codec,
        )

    left_geometry = (left_entry.width, left_entry.height, left_entry.mode)
    right_geometry = (right_entry.width, right_entry.height, right_entry.mode)
    if left_geometry != right_geometry:
        error(
            "entry_geometry",
            "图像尺寸/模式不同: "
            f"{left_entry.width}x{left_entry.height} {left_entry.mode} != "
            f"{right_entry.width}x{right_entry.height} {right_entry.mode}",
            list(left_geometry),
            list(right_geometry),
        )

    if left_entry.size != right_entry.size:
        error(
            "entry_size",
            f"原始大小不同: {left_entry.size} B != {right_entry.size} B",
            left_entry.size,
            right_entry.size,
        )
    if left_entry.constant_alpha != right_entry.constant_alpha:
        error(
            "entry_constant_alpha",
            f"constant_alpha 不同: {left_entry.constant_alpha} != "
            f"{right_entry.constant_alpha}",
            left_entry.constant_alpha,
            right_entry.constant_alpha,
        )
    if left_entry.quant != right_entry.quant:
        error(
            "entry_quant",
            f"quant 不同: {left_entry.quant} != {right_entry.quant}",
            left_entry.quant,
            right_entry.quant,
        )
    if bool(left_entry.centered) != bool(right_entry.centered):
        error(
            "entry_centered",
            f"centered 不同: {bool(left_entry.centered)} != "
            f"{bool(right_entry.centered)}",
            bool(left_entry.centered),
            bool(right_entry.centered),
        )
    if bool(left_entry.lossy) != bool(right_entry.lossy):
        error(
            "entry_lossy",
            f"lossy 不同: {bool(left_entry.lossy)} != {bool(right_entry.lossy)}",
            bool(left_entry.lossy),
            bool(right_entry.lossy),
        )

    left_ref = _ref_path(left_reader, left_entry)
    right_ref = _ref_path(right_reader, right_entry)
    if left_ref != right_ref:
        error(
            "entry_ref",
            f"差分参考不同: {left_ref or '（无）'} != {right_ref or '（无）'}",
            left_ref,
            right_ref,
        )

    left_plain = left_entry.stored_length
    right_plain = right_entry.stored_length
    if left_plain != right_plain:
        error(
            "entry_plain_length",
            f"明文 payload 长度不同: {left_plain} B != {right_plain} B",
            left_plain,
            right_plain,
        )
    if left_entry.data_crc32 != right_entry.data_crc32:
        error(
            "entry_data_crc32",
            f"data_crc32 不同: 0x{left_entry.data_crc32:08x} != "
            f"0x{right_entry.data_crc32:08x}",
            left_entry.data_crc32,
            right_entry.data_crc32,
        )
    if left_entry.data_sha256 != right_entry.data_sha256:
        error(
            "entry_data_sha256",
            f"data_sha256 不同: {_short_hash(left_entry.data_sha256)} != "
            f"{_short_hash(right_entry.data_sha256)}",
            left_entry.data_sha256,
            right_entry.data_sha256,
        )

    if (
        left_entry.length != right_entry.length
        and left_entry.stored_length == right_entry.stored_length
    ):
        warn(
            "entry_stored_length",
            f"存储长度不同: {left_entry.length} B != {right_entry.length} B"
            "（明文长度相同，仅物理布局差异）",
            left=left_entry.length,
            right=right_entry.length,
        )

    _compare_payload(left_reader, right_reader, left_entry, right_entry, error)

    metadata_left: dict[str, Any] = {}
    metadata_right: dict[str, Any] = {}
    for name in _METADATA_FIELDS:
        left_value = getattr(left_entry, name)
        right_value = getattr(right_entry, name)
        if left_value != right_value:
            metadata_left[name] = left_value
            metadata_right[name] = right_value
    if metadata_left:
        warn(
            "entry_metadata",
            "条目元数据不同（payload 相同，仅索引字段）: "
            + ", ".join(sorted(metadata_left)),
            left=metadata_left,
            right=metadata_right,
        )

    if (
        left_entry.codec != CODEC_DUP
        and left_entry.length == right_entry.length
        and left_entry.offset != right_entry.offset
    ):
        # DUP 没有自己的 payload（offset/length 恒为 0），不比较偏移。
        warn(
            "entry_offset",
            f"payload 逻辑偏移不同（长度相同）: {left_entry.offset} != "
            f"{right_entry.offset}（物理布局差异，按设计忽略）",
            left=left_entry.offset,
            right=right_entry.offset,
        )


def _compare_payload(
    left_reader: ArchiveReader,
    right_reader: ArchiveReader,
    left_entry: ArchiveEntry,
    right_entry: ArchiveEntry,
    error: Any,
) -> None:
    """比较解密后的 payload 明文字节。"""
    left_data, left_error = _read_payload(left_reader, left_entry)
    right_data, right_error = _read_payload(right_reader, right_entry)
    if left_error or right_error:
        error(
            "entry_payload_unreadable",
            "payload 读取失败: "
            f"左={'正常' if not left_error else left_error}； "
            f"右={'正常' if not right_error else right_error}",
            left_error or "ok",
            right_error or "ok",
        )
        return
    if left_data == right_data:
        return
    left_sha = sha256_bytes(left_data or b"").hex()
    right_sha = sha256_bytes(right_data or b"").hex()
    error(
        "entry_payload",
        "payload 明文字节不同: "
        f"{len(left_data or b'')} B/{_short_hash(left_sha)} != "
        f"{len(right_data or b'')} B/{_short_hash(right_sha)}",
        {"length": len(left_data or b""), "sha256": left_sha},
        {"length": len(right_data or b""), "sha256": right_sha},
    )


def _compare_index_layout(
    left: ArchiveSummary,
    right: ArchiveSummary,
    out: list[Difference],
) -> None:
    """index 的物理位置/长度。"""
    if left.index_offset != right.index_offset:
        out.append(
            Difference(
                _SEVERITY_WARN,
                "index_offset",
                f"index 逻辑偏移不同: {left.index_offset} != {right.index_offset}"
                "（payload 布局差异导致）",
                left=left.index_offset,
                right=right.index_offset,
                strict_fatal=True,
            )
        )
    if left.index_length != right.index_length:
        # 索引 JSON 里的 archive_id 不同，压缩后的长度会有几个字节的浮动。
        out.append(
            Difference(
                _SEVERITY_WARN,
                "index_length",
                f"index 长度不同: {left.index_length} B != {right.index_length} B"
                "（archive_id 内容不同导致 zlib 长度浮动，按设计忽略）",
                left=left.index_length,
                right=right.index_length,
                strict_fatal=False,
            )
        )


# ---------------------------------------------------------------- 读取辅助


def _read_payload(
    reader: ArchiveReader, entry: ArchiveEntry
) -> tuple[Optional[bytes], str]:
    """读取条目明文 payload；失败时返回（None，原因），不中断整体对比。"""
    try:
        return reader.read_entry_data(entry), ""
    except (PxcError, OSError, ValueError) as exc:
        return None, str(exc)


def _ref_path(reader: ArchiveReader, entry: ArchiveEntry) -> str:
    """把 ``ref`` 的条目 id 解析成路径，便于跨归档比较差分关系。"""
    if entry.ref is None:
        return ""
    return reader.get_entry(entry.ref).path


def _read_stream(reader: ArchiveReader, offset: int, length: int) -> bytes:
    """读取归档逻辑流的原始字节。

    ``ArchiveReader`` 只解析 header/footer 而不保留 CRC；为了对比这三个 CRC，
    这里读取原始字节。优先使用 reader 内部 source（FileSource/VolumeSource 都
    实现了 ``read_at``），缺失时退回直接读文件（仅单文件归档可用）。
    """
    source = getattr(reader, "_source", None)
    read_at = getattr(source, "read_at", None)
    if callable(read_at):
        return bytes(read_at(offset, length))
    with open(reader.path, "rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def _raw_crc32(
    reader: ArchiveReader,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """返回（header_crc32, footer_crc32, index_crc32）；读不到时为 None。"""
    try:
        raw_header = _read_stream(reader, 0, HEADER_SIZE)
        raw_footer = _read_stream(
            reader, reader.logical_size - FOOTER_SIZE, FOOTER_SIZE
        )
        if len(raw_header) != HEADER_SIZE or len(raw_footer) != FOOTER_SIZE:
            return None, None, None
        header_crc32 = int(HEADER_STRUCT.unpack(raw_header)[-1])
        footer = FOOTER_STRUCT.unpack(raw_footer)
        # (magic, file_size, index_offset, index_length, index_crc, footer_crc, ...)
        return header_crc32, int(footer[5]), int(footer[4])
    except (PxcError, OSError, ValueError):
        return None, None, None


# ---------------------------------------------------------------- 输出


def format_report(report: CompareReport) -> str:
    """把对比结果格式化成人类可读的多行文本。"""
    lines: list[str] = ["PXCS 语义对比" + ("（严格模式）" if report.strict else "")]
    lines.append(_format_summary_line("左侧", report.left))
    lines.append(_format_summary_line("右侧", report.right))
    lines.append("")

    errors = report.errors
    warnings = report.warnings
    if errors:
        lines.append(f"ERROR: {len(errors)} 个（语义不等价，必须处理）")
        lines.extend("  " + _format_difference(diff) for diff in errors)
        lines.append("")
    if warnings:
        escalations = sum(1 for diff in warnings if diff.strict_fatal)
        note = f"；其中 {escalations} 个在 --strict 下计为不等价" if escalations else ""
        lines.append(f"WARN: {len(warnings)} 个（已知差异，默认忽略{note}）")
        lines.extend("  " + _format_difference(diff) for diff in warnings)
        lines.append("")
    if not errors and not warnings:
        lines.append("未发现任何差异。")
        lines.append("")
    lines.append("结果: " + _verdict_text(report))
    return "\n".join(lines)


def _format_summary_line(side: str, summary: ArchiveSummary) -> str:
    volumes = "单文件" if summary.volumes == 0 else f"分卷 {summary.volumes}"
    if not summary.encrypted:
        encrypted = "未加密"
    elif summary.names_encrypted:
        encrypted = "已加密（文件名加密）"
    else:
        encrypted = "已加密（文件名明文）"
    head = f"INFO  {side}: {summary.path}"
    detail = (
        f"      PXCS {summary.version} | 条目 {summary.entry_count}"
        f"（图片 {summary.image_count} / 文件 {summary.file_count}） | "
        f"原始 {human_size(summary.original_size)} -> 归档 "
        f"{human_size(summary.logical_size)}"
        f"({ratio_text(summary.logical_size, summary.original_size)}) | "
        f"物理 {human_size(summary.physical_size)} | {volumes} | {encrypted} | "
        f"archive_id {summary.archive_id[:12]}…"
    )
    return head + "\n" + detail


def _format_difference(diff: Difference) -> str:
    scope = f"[{diff.category}]"
    if diff.entry is not None:
        scope += f"[{diff.entry}]"
    suffix = "  (strict)" if diff.severity == _SEVERITY_WARN and diff.strict_fatal else ""
    return f"{scope} {diff.message}{suffix}"


def _verdict_text(report: CompareReport) -> str:
    if report.equivalent:
        if not report.warnings:
            return "语义等价（未发现任何差异）"
        return f"语义等价（忽略 {len(report.warnings)} 个已知差异）"
    if report.errors:
        return f"语义不等价（{len(report.errors)} 个 ERROR）"
    return (
        f"语义不等价（--strict 下有 {len(report.strict_failures)} 个可升级差异）"
    )


# ---------------------------------------------------------------- 杂项


def _flag_names(flags: int) -> list[str]:
    return [name for bit, name in _FLAG_LABELS if flags & bit]


def _short_hash(value: str) -> str:
    """哈希/长字符串在消息里只保留前 16 位。"""
    if not value:
        return "（空）"
    if len(value) <= 16:
        return value
    return value[:16] + "…"


def _json_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.pxcs_compare",
        description=(
            "比较两个 PXCS 归档在语义层面是否等价；"
            "archive_id 及其派生的 CRC 差异会被忽略。"
        ),
        epilog=(
            "退出码: 0=语义等价；1=存在 ERROR（--strict 下可升级 WARN 也算）；"
            "2=参数或 IO 错误。"
        ),
    )
    parser.add_argument(
        "left",
        metavar="A.pxcs",
        help=t("cli.archive_help"),
    )
    parser.add_argument(
        "right",
        metavar="B.pxcs",
        help=t("cli.archive_help"),
    )
    parser.add_argument(
        "--password-file",
        default=None,
        metavar="PATH",
        help=t("cli.password_file_help"),
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help=t("cli.password_stdin_help"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=t("cli.json_help") + "（结构化结果，便于 CI 断言）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "把布局/元数据层面的 WARN（偏移、条目顺序、mtime、哈希字段等）"
            "也计为不等价；archive_id 派生差异仍忽略"
        ),
    )
    return parser


def _ensure_utf8_stdio() -> None:
    """与 pxcs.py 保持一致：Windows 控制台统一输出 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _read_password_file(path: str) -> str:
    """与 pxcs.py 的 --password-file 行为保持一致：只取第一行。"""
    file_path = Path(path).expanduser()
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PxcError(t("cli.password_file_error", path=file_path, reason=exc)) from exc
    if not raw:
        raise PxcError(t("cli.password_empty_file", path=file_path))
    lines = raw.splitlines()
    return lines[0] if lines else ""


def _password_from_args(args: argparse.Namespace) -> Optional[str]:
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


def _fail(args: argparse.Namespace, reason: str, *, hint: str = "") -> int:
    """参数/IO 错误统一出口：退出码 2，--json 时给结构化错误对象。"""
    if getattr(args, "json", False):
        payload = {
            "tool": TOOL_NAME,
            "schema_version": JSON_SCHEMA_VERSION,
            "error": reason,
            "hint": hint,
            "exit_code": EXIT_USAGE_ERROR,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(t("cli.fail", reason=reason), file=sys.stderr)
        if hint:
            print(hint, file=sys.stderr)
    return EXIT_USAGE_ERROR


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回进程退出码（0/1/2）。"""
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    password_hint = (
        "提示: 加密归档需要口令，用 --password-file PATH 或 --password-stdin 提供。"
    )
    try:
        password = _password_from_args(args)
        report = compare_archives(
            Path(args.left),
            Path(args.right),
            password=password,
            strict=bool(args.strict),
        )
    except (PxcPasswordRequired, PxcAuthenticationError) as exc:
        return _fail(args, str(exc), hint=password_hint)
    except (PxcError, OSError, ValueError) as exc:
        return _fail(args, str(exc))

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
