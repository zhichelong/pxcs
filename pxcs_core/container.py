"""PXCS 二进制容器：Header / payload / Index / Footer。

布局（v1.2 及更早，未加密）：:

    128B Header
    payloads（按 8 字节对齐）
    zlib(JSON Index)
    64B Footer

布局（v1.3，加密）：:

    128B Header
    128B Crypto Header（KDF 参数、盐、校验块、索引 MAC）
    payloads（每条 payload 为 nonce(12) || AES-256-GCM 密文+tag）
    Index（文件名加密时：nonce || AES-GCM(zlib(JSON))；否则 zlib(JSON)）
    64B Footer

分卷不改变以上结构：分卷只是把同一条字节流切成多个文件，
`volumes.VolumeSink/VolumeSource` 负责在逻辑偏移和卷之间映射。

Index 放在末尾，创建时可以边编码边写，最后回填 Header/Footer。
"""
from __future__ import annotations

import hashlib
import json
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pxc_core.errors import PxcCorruptionError, PxcFormatError, PxcVersionError
from pxc_core.i18n import t
from pxc_core.utils import crc32, sha256_bytes

from .crypto import (
    CRYPTO_HEADER_SIZE,
    CryptoContext,
    CryptoHeader,
    CryptoParams,
    new_crypto_context,
    open_from_header,
    pack_header_with_mac,
    unpack_header,
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
    FILE_CODECS,
    FLAG_ENCRYPTED,
    FLAG_HAS_DELTA,
    FLAG_HAS_LOSSY_DELTA,
    FLAG_HAS_DUP,
    FLAG_HAS_FILES,
    FLAG_HAS_RAW_IMAGES,
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    INDEX_FORMAT,
    INDEX_VERSION,
    MAX_DEPTH,
    MAX_ENTRIES,
    MAX_PATH_BYTES,
    PXCS_END_MAGIC,
    PXCS_MAGIC,
    SUPPORTED_MINORS,
    VERSION_MAJOR,
    VERSION_MINOR,
    WRITE_MINOR_CRYPTO,
    WRITE_MINOR_JPEGXL,
    WRITE_MINOR_PLAIN,
)
from .paths import normalize_rel_path
from .volumes import ArchiveSink, ArchiveSource, open_source, volume_path


class PxcsFormatError(PxcFormatError):
    """PXCS 容器格式错误。"""


class PxcsSelfCheckError(PxcsFormatError):
    """归档写完后自检失败：产物不可信（CLI 退出码 4）。"""


class PxcsCorruptionError(PxcCorruptionError):
    """PXCS 数据损坏。"""


_VALID_IMAGE_CODECS = {
    CODEC_PXC, CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL,
    CODEC_WEBP_LOSSLESS, CODEC_RAW, CODEC_JPEGXL, CODEC_DUP,
}
_VALID_FILE_CODECS = FILE_CODECS
_HEX_LEN = 64

DATA_OFFSET_PLAIN = HEADER_SIZE
DATA_OFFSET_CRYPTO = HEADER_SIZE + CRYPTO_HEADER_SIZE


@dataclass
class ArchiveHeader:
    version_major: int
    version_minor: int
    flags: int
    entry_count: int
    index_offset: int
    index_length: int
    data_offset: int
    archive_id: bytes


@dataclass
class ArchiveEntry:
    id: int
    path: str
    kind: str
    codec: str
    offset: int = 0
    length: int = 0        # 归档内实际占用的字节数（加密时含 nonce/tag）
    plain_length: Optional[int] = None  # 解密后的 payload 长度；未加密时与 length 相同
    size: int = 0
    ref: Optional[int] = None
    depth: int = 0
    width: int = 0
    height: int = 0
    mode: str = ""
    format: str = ""
    ext: str = ""
    constant_alpha: Optional[int] = None
    # 新式差分：centered=True 表示残差以 128 为中心；quant>0 表示有损量化步长。
    quant: int = 0
    centered: bool = False
    lossy: bool = False
    pixel_sha256: str = ""
    source_sha256: str = ""
    data_crc32: int = 0
    raw_crc32: int = 0
    data_sha256: str = ""
    mtime_ns: int = 0
    flags: int = 0
    reason: str = ""

    @property
    def stored_length(self) -> int:
        """payload 的明文长度；用于体积统计和长度校验。"""
        return self.length if self.plain_length is None else int(self.plain_length)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "path": self.path,
            "kind": self.kind,
            "codec": self.codec,
            "offset": self.offset,
            "length": self.length,
            "size": self.size,
            "ref": self.ref,
            "depth": self.depth,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "format": self.format,
            "ext": self.ext,
            "constant_alpha": self.constant_alpha,
            "quant": self.quant,
            "centered": self.centered,
            "lossy": self.lossy,
            "pixel_sha256": self.pixel_sha256,
            "source_sha256": self.source_sha256,
            "data_crc32": self.data_crc32,
            "raw_crc32": self.raw_crc32,
            "data_sha256": self.data_sha256,
            "mtime_ns": self.mtime_ns,
            "flags": self.flags,
            "reason": self.reason,
        }
        if self.plain_length is not None:
            # 仅加密归档需要；未加密时不写，保持与 1.2 索引完全一致。
            data["plain_length"] = self.plain_length
        return data

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ArchiveEntry":
        try:
            return cls(
                id=int(obj["id"]),
                path=str(obj["path"]),
                kind=str(obj["kind"]),
                codec=str(obj["codec"]),
                offset=int(obj.get("offset", 0)),
                length=int(obj.get("length", 0)),
                plain_length=(
                    None
                    if obj.get("plain_length") is None
                    else int(obj["plain_length"])
                ),
                size=int(obj.get("size", 0)),
                ref=None if obj.get("ref") is None else int(obj["ref"]),
                depth=int(obj.get("depth", 0)),
                width=int(obj.get("width", 0)),
                height=int(obj.get("height", 0)),
                mode=str(obj.get("mode", "")),
                format=str(obj.get("format", "")),
                ext=str(obj.get("ext", "")),
                constant_alpha=(
                    None
                    if obj.get("constant_alpha") is None
                    else int(obj["constant_alpha"])
                ),
                quant=int(obj.get("quant", 0) or 0),
                centered=bool(obj.get("centered", False)),
                lossy=bool(obj.get("lossy", False)),
                pixel_sha256=str(obj.get("pixel_sha256", "")),
                source_sha256=str(obj.get("source_sha256", "")),
                data_crc32=int(obj.get("data_crc32", 0)),
                raw_crc32=int(obj.get("raw_crc32", 0)),
                data_sha256=str(obj.get("data_sha256", "")),
                mtime_ns=int(obj.get("mtime_ns", 0)),
                flags=int(obj.get("flags", 0)),
                reason=str(obj.get("reason", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PxcsFormatError(t("container.entry_field", reason=exc)) from exc


@dataclass
class ArchiveIndex:
    entries: list[ArchiveEntry]
    archive_id: bytes
    created_by: str = "PXCS Python"
    raw: dict[str, Any] = field(default_factory=dict)


class ArchiveWriter:
    """边编码边写 payload，最后回填 header/index/footer。

    ``password`` 非空时启用 AES-256-GCM；``volume_size`` 非空时写成分卷。
    两者可以同时使用，互不影响。
    """

    def __init__(
        self,
        path: Path,
        overwrite: bool = False,
        zlib_level: int = 6,
        *,
        password: Optional[str] = None,
        encrypt_filenames: bool = True,
        crypto_params: Optional[CryptoParams] = None,
        volume_size: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.zlib_level = max(1, min(9, int(zlib_level)))
        self.archive_id = uuid.uuid4().bytes
        self.volume_size = int(volume_size) if volume_size else None
        self.overwrite = bool(overwrite)
        self.crypto: Optional[CryptoContext] = None
        if password:
            params = crypto_params or CryptoParams()
            params = CryptoParams(
                kdf_id=params.kdf_id,
                param1=params.param1,
                param2=params.param2,
                param3=params.param3,
                encrypt_filenames=bool(encrypt_filenames),
            )
            self.crypto = new_crypto_context(password, self.archive_id, params)
        self.data_offset = DATA_OFFSET_CRYPTO if self.crypto else DATA_OFFSET_PLAIN

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.overwrite:
            if self.path.exists():
                raise PxcsFormatError(t("container.output_exists", path=self.path))
            if self.volume_size and volume_path(self.path, 1).exists():
                raise PxcsFormatError(
                    t("container.output_exists", path=volume_path(self.path, 1))
                )

        from .volumes import make_sink

        self.sink: ArchiveSink = make_sink(
            self.path,
            archive_id=self.archive_id,
            volume_size=self.volume_size,
            overwrite=self.overwrite,
        )
        # 预留 Header（+ Crypto Header）位置，收尾时回填。
        self.sink.write(b"\x00" * self.data_offset)
        self._closed = False
        self.output_files: list[Path] = []

    # ------------------------------------------------------------- payload
    def add_payload(self, payload: bytes) -> tuple[int, int, int, int, str]:
        """写入 payload，返回 (offset, 存储长度, 明文长度, crc32, sha256)。"""
        if self._closed:
            raise RuntimeError(t("container.writer_closed"))
        blob = self.crypto.seal_payload(payload) if self.crypto else payload
        pos = self.sink.tell()
        padding = (-pos) % 8
        if padding:
            self.sink.write(b"\x00" * padding)
            pos += padding
        self.sink.write(blob)
        plain_length = len(payload) if self.crypto is not None else None
        return pos, len(blob), plain_length, crc32(payload), sha256_bytes(payload).hex()

    def position(self) -> int:
        """当前已写入的逻辑字节数（含 header/payload，不含 index/footer）。"""
        return self.sink.tell()

    # ------------------------------------------------------------- finish
    def finish(self, entries: list[ArchiveEntry], flags: int) -> tuple[Path, int]:
        if self._closed:
            raise RuntimeError(t("container.writer_closed"))
        try:
            index_obj = {
                "format": INDEX_FORMAT,
                "version": INDEX_VERSION,
                "archive_id": self.archive_id.hex(),
                "created_by": "PXCS Python 1.3",
                "entries": [e.to_dict() for e in entries],
            }
            index_raw = json.dumps(
                index_obj, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(index_raw) > 256 * 1024 * 1024:
                raise PxcsFormatError(t("container.index_too_large"))
            compressed = zlib.compress(index_raw, 9)
            index_mac = b"\x00" * 32
            if self.crypto is not None:
                if self.crypto.encrypt_filenames:
                    index_blob = self.crypto.seal_index(compressed)
                else:
                    index_blob = compressed
                    index_mac = self.crypto.index_mac(index_raw)
            else:
                index_blob = compressed
            index_offset = self.sink.tell()
            self.sink.write(index_blob)
            index_length = len(index_blob)
            file_size = self.sink.tell() + FOOTER_SIZE

            footer_zero = FOOTER_STRUCT.pack(
                PXCS_END_MAGIC,
                file_size,
                index_offset,
                index_length,
                crc32(index_blob),
                0,
                self.archive_id,
                b"\x00" * 8,
            )
            footer = FOOTER_STRUCT.pack(
                PXCS_END_MAGIC,
                file_size,
                index_offset,
                index_length,
                crc32(index_blob),
                crc32(footer_zero),
                self.archive_id,
                b"\x00" * 8,
            )
            self.sink.write(footer)

            minor = WRITE_MINOR_CRYPTO if self.crypto else WRITE_MINOR_PLAIN
            if any(e.codec == CODEC_JPEGXL for e in entries):
                minor = max(minor, WRITE_MINOR_JPEGXL)
            if self.crypto is not None:
                flags = flags | FLAG_ENCRYPTED
            header_zero = HEADER_STRUCT.pack(
                PXCS_MAGIC,
                VERSION_MAJOR,
                minor,
                flags,
                HEADER_SIZE,
                len(entries),
                index_offset,
                index_length,
                self.data_offset,
                self.archive_id,
                b"\x00" * 4,
                b"\x00" * 62,
                0,
            )
            header = HEADER_STRUCT.pack(
                PXCS_MAGIC,
                VERSION_MAJOR,
                minor,
                flags,
                HEADER_SIZE,
                len(entries),
                index_offset,
                index_length,
                self.data_offset,
                self.archive_id,
                b"\x00" * 4,
                b"\x00" * 62,
                crc32(header_zero),
            )
            self.sink.patch(0, header)
            if self.crypto is not None:
                self.sink.patch(
                    HEADER_SIZE,
                    pack_header_with_mac(self.crypto, index_mac),
                )
            _main_path, size, files = self.sink.finish()
            self.output_files = list(files)
            self._closed = True
            return self.path, size
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if not self._closed:
            try:
                self.sink.abort()
            except Exception:
                pass
        self._closed = True


class ArchiveReader:
    """读取单文件或分卷归档；加密归档需要口令。"""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        password: Optional[str] = None,
        source: Optional[ArchiveSource] = None,
    ) -> None:
        if source is None:
            if path is None:
                raise ValueError("path or source required")
            source = open_source(Path(path))
        self._source = source
        self.path = Path(path) if path is not None else source.path
        self.volume_paths: list[Path] = list(getattr(source, "volume_paths", []))
        self.logical_size = int(source.size)
        self.physical_size = int(getattr(source, "physical_size", source.size))
        self.header: ArchiveHeader
        self.entries: list[ArchiveEntry]
        self.crypto: Optional[CryptoContext] = None
        self.crypto_header: Optional[CryptoHeader] = None
        self._entry_by_id: dict[int, ArchiveEntry]
        self._fh = None  # 兼容旧代码里对 reader._fh 的探测
        try:
            self._read_all(password)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------ context
    @property
    def volume_count(self) -> int:
        return len(self.volume_paths)

    @property
    def is_encrypted(self) -> bool:
        return self.crypto is not None

    @property
    def names_encrypted(self) -> bool:
        return bool(self.crypto and self.crypto.encrypt_filenames)

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._source.close()
        except OSError:
            pass

    # ------------------------------------------------------------ parsing
    def _read_all(self, password: Optional[str]) -> None:
        file_size = self.logical_size
        if file_size < HEADER_SIZE + FOOTER_SIZE:
            raise PxcsFormatError(t("container.too_short"))

        raw_header = self._source.read_at(0, HEADER_SIZE)
        if len(raw_header) != HEADER_SIZE:
            raise PxcsFormatError(t("container.header_truncated"))
        try:
            (
                magic,
                major,
                minor,
                flags,
                header_size,
                entry_count,
                index_offset,
                index_length,
                data_offset,
                archive_id,
                reserved4,
                reserved62,
                header_crc,
            ) = HEADER_STRUCT.unpack(raw_header)
        except struct.error as exc:
            raise PxcsFormatError(t("container.header_parse", reason=exc)) from exc

        if magic != PXCS_MAGIC:
            raise PxcsFormatError(t("container.header_magic"))
        if major != VERSION_MAJOR:
            raise PxcVersionError(
                t(
                    "container.version_unsupported",
                    major=major,
                    minor=minor,
                    supported_major=VERSION_MAJOR,
                    supported_minor=VERSION_MINOR,
                )
            )
        if minor not in SUPPORTED_MINORS:
            raise PxcVersionError(
                t(
                    "container.version_too_new",
                    major=major,
                    minor=minor,
                    supported_major=VERSION_MAJOR,
                    supported_minor=VERSION_MINOR,
                )
            )
        if header_size != HEADER_SIZE:
            raise PxcsFormatError(t("container.header_size", size=header_size))
        if reserved4 != b"\x00" * 4 or reserved62 != b"\x00" * 62:
            raise PxcsFormatError(t("container.header_reserved"))
        if entry_count > MAX_ENTRIES:
            raise PxcsFormatError(t("container.entry_count", count=entry_count))
        if not (HEADER_SIZE <= index_offset <= file_size - FOOTER_SIZE):
            raise PxcsFormatError(t("container.index_offset"))
        if index_length > file_size - FOOTER_SIZE - index_offset:
            raise PxcsFormatError(t("container.index_length"))
        if index_offset + index_length + FOOTER_SIZE != file_size:
            raise PxcsFormatError(t("container.index_layout"))

        header_zero = bytearray(raw_header)
        header_zero[-4:] = b"\x00\x00\x00\x00"
        if crc32(bytes(header_zero)) != header_crc:
            raise PxcsCorruptionError(t("container.header_crc"))

        # ---- 加密头：先派生密钥并验证校验块，再读索引
        encrypted = bool(flags & FLAG_ENCRYPTED)
        if encrypted:
            if minor < WRITE_MINOR_CRYPTO:
                raise PxcsFormatError(t("container.header_reserved"))
            if data_offset != DATA_OFFSET_CRYPTO:
                raise PxcsFormatError(t("container.data_offset"))
            if password is None:
                raise self._password_error()
            raw_crypto = self._source.read_at(HEADER_SIZE, CRYPTO_HEADER_SIZE)
            header = unpack_header(raw_crypto)
            self.crypto_header = header
            self.crypto = open_from_header(header, password, archive_id)
        else:
            if data_offset != DATA_OFFSET_PLAIN:
                raise PxcsFormatError(t("container.data_offset"))

        raw_footer = self._source.read_at(file_size - FOOTER_SIZE, FOOTER_SIZE)
        if len(raw_footer) != FOOTER_SIZE:
            raise PxcsFormatError(t("container.footer_truncated"))
        try:
            (
                end_magic,
                footer_file_size,
                footer_index_offset,
                footer_index_length,
                index_crc,
                footer_crc,
                footer_archive_id,
                footer_reserved,
            ) = FOOTER_STRUCT.unpack(raw_footer)
        except struct.error as exc:
            raise PxcsFormatError(t("container.footer_parse", reason=exc)) from exc
        if end_magic != PXCS_END_MAGIC:
            raise PxcsFormatError(t("container.footer_magic"))
        if footer_file_size != file_size:
            raise PxcsFormatError(t("container.footer_size"))
        if (footer_index_offset, footer_index_length) != (index_offset, index_length):
            raise PxcsFormatError(t("container.footer_index"))
        if footer_archive_id != archive_id:
            raise PxcsFormatError(t("container.footer_archive_id"))
        if footer_reserved != b"\x00" * 8:
            raise PxcsFormatError(t("container.footer_reserved"))
        footer_zero = FOOTER_STRUCT.pack(
            end_magic,
            footer_file_size,
            footer_index_offset,
            footer_index_length,
            index_crc,
            0,
            footer_archive_id,
            b"\x00" * 8,
        )
        if crc32(footer_zero) != footer_crc:
            raise PxcsCorruptionError(t("container.footer_crc"))

        index_blob = self._source.read_at(index_offset, index_length)
        if len(index_blob) != index_length:
            raise PxcsFormatError(t("container.index_truncated"))
        if crc32(index_blob) != index_crc:
            raise PxcsCorruptionError(t("container.index_crc"))

        if self.crypto is not None and self.crypto.encrypt_filenames:
            index_raw = _zlib_decompress(self.crypto.open_index(index_blob))
        else:
            index_raw = _zlib_decompress(index_blob)
            if self.crypto is not None and self.crypto_header is not None:
                self.crypto.verify_index_mac(index_raw, self.crypto_header.index_mac)

        try:
            index_obj = json.loads(index_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PxcsFormatError(t("container.index_json", reason=exc)) from exc
        if not isinstance(index_obj, dict):
            raise PxcsFormatError(t("container.index_root"))
        if index_obj.get("format") != INDEX_FORMAT:
            raise PxcsFormatError(t("container.index_format"))
        if int(index_obj.get("version", 0)) != INDEX_VERSION:
            raise PxcVersionError(t("container.index_version"))
        if index_obj.get("archive_id") != archive_id.hex():
            raise PxcsFormatError(t("container.index_archive_id"))
        raw_entries = index_obj.get("entries")
        if not isinstance(raw_entries, list):
            raise PxcsFormatError(t("container.index_entries"))
        if len(raw_entries) != entry_count:
            raise PxcsFormatError(t("container.index_entry_count"))

        entries: list[ArchiveEntry] = []
        seen_ids: set[int] = set()
        seen_paths: set[str] = set()
        for i, obj in enumerate(raw_entries):
            if not isinstance(obj, dict):
                raise PxcsFormatError(t("container.entry_object"))
            entry = ArchiveEntry.from_dict(obj)
            if entry.id != i:
                raise PxcsFormatError(
                    t("container.entry_id_order", expected=i, actual=entry.id)
                )
            if entry.id in seen_ids:
                raise PxcsFormatError(t("container.entry_id_dup", id=entry.id))
            seen_ids.add(entry.id)
            try:
                norm_path = normalize_rel_path(entry.path)
            except Exception as exc:
                raise PxcsFormatError(
                    t("container.entry_path", path=entry.path, reason=exc)
                ) from exc
            if norm_path != entry.path:
                raise PxcsFormatError(t("container.entry_path_norm", path=entry.path))
            if len(entry.path.encode("utf-8", "surrogateescape")) > MAX_PATH_BYTES:
                raise PxcsFormatError(t("container.entry_path_long"))
            path_key = entry.path.casefold()
            if path_key in seen_paths:
                raise PxcsFormatError(t("container.entry_path_dup", path=entry.path))
            seen_paths.add(path_key)
            if entry.kind not in ("image", "file"):
                raise PxcsFormatError(t("container.entry_kind", kind=entry.kind))
            if entry.kind == "image":
                if entry.codec not in _VALID_IMAGE_CODECS:
                    raise PxcsFormatError(
                        t("container.image_codec", codec=entry.codec)
                    )
            else:
                if entry.codec not in _VALID_FILE_CODECS:
                    raise PxcsFormatError(
                        t("container.file_codec", codec=entry.codec)
                    )
            if entry.size < 0:
                raise PxcsFormatError(t("container.entry_size"))
            if entry.plain_length is not None and entry.plain_length < 0:
                raise PxcsFormatError(t("container.plain_length"))
            if entry.codec == CODEC_DUP:
                if entry.offset != 0 or entry.length != 0:
                    raise PxcsFormatError(t("container.dup_payload"))
                if entry.ref is None:
                    raise PxcsFormatError(t("container.dup_ref"))
            else:
                if entry.length < 0:
                    raise PxcsFormatError(t("container.entry_length"))
                if entry.offset < data_offset:
                    raise PxcsFormatError(t("container.entry_offset"))
                if entry.offset + entry.length > index_offset:
                    raise PxcsFormatError(t("container.entry_range"))
            if entry.ref is not None:
                if not (0 <= entry.ref < entry.id):
                    raise PxcsFormatError(t("container.entry_ref"))
                if entry.depth < 1 or entry.depth > MAX_DEPTH:
                    raise PxcsFormatError(t("container.entry_depth"))
            else:
                if entry.depth != 0:
                    raise PxcsFormatError(t("container.entry_depth_zero"))
            if entry.kind == "image":
                if entry.width <= 0 or entry.height <= 0:
                    raise PxcsFormatError(t("container.image_size"))
                if entry.pixel_sha256 and len(entry.pixel_sha256) != _HEX_LEN:
                    raise PxcsFormatError(t("container.pixel_sha"))
                if not (0 <= entry.quant <= 255):
                    raise PxcsFormatError(t("container.quant"))
                if entry.lossy and entry.codec not in {
                    CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL
                }:
                    raise PxcsFormatError(t("container.lossy_codec"))
            if entry.source_sha256 and len(entry.source_sha256) != _HEX_LEN:
                raise PxcsFormatError(t("container.source_sha"))
            if entry.data_sha256 and len(entry.data_sha256) != _HEX_LEN:
                raise PxcsFormatError(t("container.data_sha"))
            entries.append(entry)

        for entry in entries:
            if entry.ref is None:
                continue
            ref = entries[entry.ref]
            if ref.kind != entry.kind:
                raise PxcsFormatError(t("container.ref_kind"))
            if entry.codec in {CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL}:
                if ref.codec not in (
                    CODEC_PXC, CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL,
                    CODEC_WEBP_LOSSLESS, CODEC_RAW, CODEC_JPEGXL
                ):
                    raise PxcsFormatError(t("container.ref_pixel"))

        self.header = ArchiveHeader(
            version_major=major,
            version_minor=minor,
            flags=flags,
            entry_count=entry_count,
            index_offset=index_offset,
            index_length=index_length,
            data_offset=data_offset,
            archive_id=archive_id,
        )
        self.entries = entries
        self._entry_by_id = {e.id: e for e in entries}

    def _password_error(self) -> Exception:
        from .crypto import PxcPasswordRequired

        return PxcPasswordRequired(t("crypto.password_required", path=self.path))

    # ------------------------------------------------------------ access
    def get_entry(self, entry_id: int) -> ArchiveEntry:
        try:
            return self._entry_by_id[entry_id]
        except KeyError as exc:
            raise PxcsFormatError(t("container.entry_missing", id=entry_id)) from exc

    def read_entry_data(self, entry: ArchiveEntry) -> bytes:
        """返回 payload 明文（加密归档会自动解密并校验认证）。"""
        if entry.codec == CODEC_DUP:
            return b""
        if entry.length == 0:
            return b""
        data = self._source.read_at(entry.offset, entry.length)
        if len(data) != entry.length:
            raise PxcsCorruptionError(t("container.entry_data_short", path=entry.path))
        if self.crypto is not None:
            data = self.crypto.open_payload(data)
        if crc32(data) != entry.data_crc32:
            raise PxcsCorruptionError(t("container.entry_crc", path=entry.path))
        if entry.data_sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual != entry.data_sha256:
                raise PxcsCorruptionError(t("container.entry_sha", path=entry.path))
        return data


def _zlib_decompress(blob: bytes) -> bytes:
    try:
        dec = zlib.decompressobj()
        out = dec.decompress(blob, 256 * 1024 * 1024 + 1)
        out += dec.flush()
        if len(out) > 256 * 1024 * 1024 or dec.unconsumed_tail:
            raise PxcsFormatError(t("container.index_unzip_size"))
        return out
    except zlib.error as exc:
        raise PxcsCorruptionError(t("container.index_zlib_failed", reason=exc)) from exc


def compute_flags(entries: list[ArchiveEntry]) -> int:
    flags = 0
    if any(e.codec in {CODEC_PXC_RESIDUAL, CODEC_ZLIB_RESIDUAL} for e in entries):
        flags |= FLAG_HAS_DELTA
    if any(e.kind == "file" for e in entries):
        flags |= FLAG_HAS_FILES
    if any(e.codec == CODEC_DUP for e in entries):
        flags |= FLAG_HAS_DUP
    if any(
        e.codec in {CODEC_RAW, CODEC_JPEGXL} and e.kind == "image"
        for e in entries
    ):
        flags |= FLAG_HAS_RAW_IMAGES
    if any(e.lossy for e in entries):
        flags |= FLAG_HAS_LOSSY_DELTA
    return flags
