"""PXCS 分卷：卷头、分卷写入器、按逻辑偏移随机读取。

分卷不改变 PXCS 归档格式本身：它只是把同一条字节流切成若干文件。
每一卷的开头有 128 字节卷头（magic、卷号、总卷数、本卷在逻辑流中的起始偏移、
本卷数据长度、archive_id、CRC）。因此：

- 读取时按逻辑偏移映射到 (卷号, 卷内偏移)，单张图片仍可随机提取；
- 缺卷、卷头损坏、卷顺序错误、卷被截断都能给出明确错误；
- 加密与分卷互相独立：AEAD 作用在逻辑流里的 payload 上，分卷只负责切分。

文件命名：``<base>.<卷号>``，卷号从 001 开始，至少 3 位，连续递增。
"""
from __future__ import annotations

import bisect
import os
import re
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from pxc_core.errors import PxcError
from pxc_core.i18n import t
from pxc_core.utils import crc32, human_size

VOLUME_MAGIC = b"PXCSVOL\x00"
VOLUME_VERSION = 1
VOLUME_HEADER_SIZE = 128
VOLUME_FLAG_FINAL = 1 << 0
VOLUME_HEADER_STRUCT = struct.Struct("<8sHHIIQQ16s72sI")
assert VOLUME_HEADER_STRUCT.size == VOLUME_HEADER_SIZE

MIN_VOLUME_SIZE = 1024
DEFAULT_VOLUME_SIZE = 4 * 1024 ** 3
MAX_VOLUMES = 100_000

_VOLUME_NAME_RE = re.compile(r"^(?P<base>.+)\.(?P<index>\d{3,})$")


class PxcsVolumeError(PxcError):
    """分卷缺失、损坏或顺序错误。"""


# ---------------------------------------------------------------- 命名

def volume_path(base: Path, index: int) -> Path:
    """第 index 卷（从 1 开始）的路径。"""
    if index < 1:
        raise ValueError("volume index must be >= 1")
    return base.parent / f"{base.name}.{index:03d}"


def parse_volume_name(name: str) -> Optional[tuple[str, int]]:
    match = _VOLUME_NAME_RE.match(name)
    if not match:
        return None
    return match.group("base"), int(match.group("index"))


def is_volume_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == VOLUME_MAGIC
    except OSError:
        return False


def format_volume_list(paths: list[Path]) -> str:
    if not paths:
        return ""
    if len(paths) == 1:
        return str(paths[0])
    return f"{paths[0]} … {paths[-1]}"


# ---------------------------------------------------------------- 卷头

@dataclass(frozen=True)
class VolumeInfo:
    path: Path
    index: int          # 1-based
    count: int
    logical_offset: int
    length: int         # 本卷承载的逻辑字节数
    archive_id: bytes
    final: bool

    @property
    def physical_size(self) -> int:
        return VOLUME_HEADER_SIZE + self.length


def pack_volume_header(
    index: int,
    count: int,
    logical_offset: int,
    length: int,
    archive_id: bytes,
    final: bool,
) -> bytes:
    flags = VOLUME_FLAG_FINAL if final else 0
    zero = VOLUME_HEADER_STRUCT.pack(
        VOLUME_MAGIC,
        VOLUME_VERSION,
        flags,
        index,
        count,
        logical_offset,
        length,
        archive_id,
        b"\x00" * 72,
        0,
    )
    return VOLUME_HEADER_STRUCT.pack(
        VOLUME_MAGIC,
        VOLUME_VERSION,
        flags,
        index,
        count,
        logical_offset,
        length,
        archive_id,
        b"\x00" * 72,
        crc32(zero),
    )


def read_volume_header(path: Path, expect_index: Optional[int] = None) -> VolumeInfo:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw = f.read(VOLUME_HEADER_SIZE)
    except OSError as exc:
        raise PxcsVolumeError(t("volume.unreadable", path=path, reason=exc)) from exc
    if len(raw) != VOLUME_HEADER_SIZE:
        raise PxcsVolumeError(t("volume.truncated", path=path))
    if raw[:8] != VOLUME_MAGIC:
        raise PxcsVolumeError(t("volume.bad_magic", path=path))
    try:
        (
            _magic,
            version,
            flags,
            index,
            count,
            logical_offset,
            length,
            archive_id,
            reserved,
            header_crc,
        ) = VOLUME_HEADER_STRUCT.unpack(raw)
    except struct.error as exc:
        raise PxcsVolumeError(t("volume.bad_header", path=path, reason=exc)) from exc
    if version != VOLUME_VERSION:
        raise PxcsVolumeError(
            t("volume.bad_version", path=path, version=version, supported=VOLUME_VERSION)
        )
    if reserved != b"\x00" * 72:
        raise PxcsVolumeError(t("volume.bad_header", path=path, reason="reserved"))
    zero = bytearray(raw)
    zero[-4:] = b"\x00" * 4
    if crc32(bytes(zero)) != header_crc:
        raise PxcsVolumeError(t("volume.header_crc", path=path))
    if expect_index is not None and index != expect_index:
        raise PxcsVolumeError(
            t("volume.order", path=path, expected=expect_index, actual=index)
        )
    if index < 1 or index > count or count > MAX_VOLUMES:
        raise PxcsVolumeError(
            t("volume.bad_index", path=path, index=index, count=count)
        )
    if size != VOLUME_HEADER_SIZE + length:
        raise PxcsVolumeError(
            t(
                "volume.size_mismatch",
                path=path,
                expected=VOLUME_HEADER_SIZE + length,
                actual=size,
            )
        )
    return VolumeInfo(
        path=path,
        index=index,
        count=count,
        logical_offset=logical_offset,
        length=length,
        archive_id=bytes(archive_id),
        final=bool(flags & VOLUME_FLAG_FINAL),
    )


def find_volume_set(path: Path) -> Optional[list[Path]]:
    """给定任意一卷或 base 路径，返回按顺序排列的卷文件列表。"""
    path = Path(path)
    base: Optional[Path] = None

    if path.exists() and path.is_file() and is_volume_file(path):
        parsed = parse_volume_name(path.name)
        if parsed is None:
            raise PxcsVolumeError(t("volume.unnamed", path=path))
        base = path.parent / parsed[0]
    elif (path.parent / (path.name + ".001")).is_file():
        base = path
    else:
        parsed = parse_volume_name(path.name)
        if parsed is not None:
            candidate = path.parent / parsed[0]
            if (path.parent / f"{parsed[0]}.001").is_file():
                base = candidate
    if base is None:
        return None

    directory = base.parent
    found: dict[int, Path] = {}
    for candidate in directory.glob(base.name + ".*"):
        parsed = parse_volume_name(candidate.name)
        if parsed is None or parsed[0] != base.name:
            continue
        found[parsed[1]] = candidate
    if not found:
        return None

    first = found.get(1)
    if first is None:
        raise PxcsVolumeError(
            t("volume.missing", missing="001", expected="001")
        )
    info = read_volume_header(first, expect_index=1)
    count = info.count
    missing = [f"{i:03d}" for i in range(1, count + 1) if i not in found]
    if missing:
        raise PxcsVolumeError(
            t("volume.missing", missing=", ".join(missing), expected=f"001..{count:03d}")
        )
    return [found[i] for i in range(1, count + 1)]


# ---------------------------------------------------------------- 读

class ArchiveSource:
    """归档的逻辑字节流；单文件或分卷都实现同一接口。"""

    path: Path
    size: int
    volume_paths: list[Path]

    def read_at(self, offset: int, length: int) -> bytes:  # pragma: no cover - 抽象
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    @property
    def volume_count(self) -> int:
        return len(self.volume_paths)

    @property
    def physical_size(self) -> int:
        raise NotImplementedError

    def __enter__(self) -> "ArchiveSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FileSource(ArchiveSource):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.volume_paths: list[Path] = []
        try:
            self._fh: Optional[BinaryIO] = open(self.path, "rb")
            self.size = os.fstat(self._fh.fileno()).st_size
        except OSError as exc:
            raise PxcsVolumeError(t("volume.unreadable", path=path, reason=exc)) from exc

    def read_at(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if offset < 0 or offset + length > self.size:
            raise PxcsVolumeError(
                t("volume.read_out_of_range", offset=offset, length=length, size=self.size)
            )
        if self._fh is None:
            raise RuntimeError("source 已关闭")
        self._fh.seek(offset)
        data = self._fh.read(length)
        if len(data) != length:
            raise PxcsVolumeError(t("volume.read_short", expected=length, actual=len(data)))
        return data

    @property
    def physical_size(self) -> int:
        return self.size

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


class VolumeSource(ArchiveSource):
    def __init__(self, paths: list[Path]) -> None:
        self.volume_paths = [Path(p) for p in paths]
        self.path = self.volume_paths[0]
        self._infos: list[VolumeInfo] = []
        self._starts: list[int] = []
        archive_id: Optional[bytes] = None
        expected_offset = 0
        for index, path in enumerate(self.volume_paths, 1):
            info = read_volume_header(path, expect_index=index)
            if archive_id is None:
                archive_id = info.archive_id
            elif info.archive_id != archive_id:
                raise PxcsVolumeError(t("volume.archive_mismatch", path=path))
            if info.logical_offset != expected_offset:
                raise PxcsVolumeError(
                    t(
                        "volume.order",
                        path=path,
                        expected=expected_offset,
                        actual=info.logical_offset,
                    )
                )
            if info.count != len(self.volume_paths):
                raise PxcsVolumeError(
                    t("volume.count_mismatch", path=path, count=info.count)
                )
            self._infos.append(info)
            self._starts.append(info.logical_offset)
            expected_offset += info.length
        if not self._infos:
            raise PxcsVolumeError(t("volume.none"))
        if not self._infos[-1].final:
            raise PxcsVolumeError(t("volume.not_final", path=self.volume_paths[-1]))
        if any(info.final for info in self._infos[:-1]):
            raise PxcsVolumeError(t("volume.early_final"))
        self.size = expected_offset
        self.archive_id = archive_id or b""
        self._handles: dict[int, BinaryIO] = {}

    def _handle(self, index: int) -> BinaryIO:
        handle = self._handles.get(index)
        if handle is None:
            try:
                handle = open(self.volume_paths[index], "rb")
            except OSError as exc:
                raise PxcsVolumeError(
                    t("volume.unreadable", path=self.volume_paths[index], reason=exc)
                ) from exc
            self._handles[index] = handle
        return handle

    def read_at(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if offset < 0 or offset + length > self.size:
            raise PxcsVolumeError(
                t("volume.read_out_of_range", offset=offset, length=length, size=self.size)
            )
        parts: list[bytes] = []
        remaining = length
        cursor = offset
        while remaining > 0:
            index = bisect.bisect_right(self._starts, cursor) - 1
            if index < 0 or index >= len(self._infos):
                raise PxcsVolumeError(t("volume.gap", offset=cursor))
            info = self._infos[index]
            local = cursor - info.logical_offset
            take = min(remaining, info.length - local)
            if take <= 0:
                raise PxcsVolumeError(t("volume.gap", offset=cursor))
            handle = self._handle(index)
            handle.seek(VOLUME_HEADER_SIZE + local)
            block = handle.read(take)
            if len(block) != take:
                raise PxcsVolumeError(
                    t("volume.read_short", expected=take, actual=len(block))
                )
            parts.append(block)
            cursor += take
            remaining -= take
        return b"".join(parts)

    @property
    def physical_size(self) -> int:
        return sum(info.physical_size for info in self._infos)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except OSError:
                pass
        self._handles.clear()


def open_source(path: Path) -> ArchiveSource:
    """按路径自动识别单文件归档或分卷归档。"""
    path = Path(path)
    if not path.exists():
        volumes = find_volume_set(path)
        if volumes:
            return VolumeSource(volumes)
        raise PxcsVolumeError(t("volume.not_found", path=path))
    if path.is_dir():
        raise PxcsVolumeError(t("volume.is_dir", path=path))
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as exc:
        raise PxcsVolumeError(t("volume.unreadable", path=path, reason=exc)) from exc
    if head == VOLUME_MAGIC:
        volumes = find_volume_set(path)
        if not volumes:
            raise PxcsVolumeError(t("volume.missing", missing="001", expected="001"))
        return VolumeSource(volumes)
    return FileSource(path)


# ---------------------------------------------------------------- 写

class ArchiveSink:
    """写入抽象：单文件或分卷。"""

    path: Path
    max_volume_size: Optional[int]

    def tell(self) -> int:  # pragma: no cover - 抽象
        raise NotImplementedError

    def write(self, data: bytes) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def patch(self, logical_offset: int, data: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def finish(self) -> tuple[Path, int, list[Path]]:  # pragma: no cover
        raise NotImplementedError

    def abort(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError


class FileSink(ArchiveSink):
    def __init__(self, path: Path, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise PxcError(t("container.output_exists", path=self.path))
        self.max_volume_size = None
        self.volume_paths: list[Path] = []
        self._tmp = self.path.with_name(
            self.path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self._fh: Optional[BinaryIO] = open(self._tmp, "w+b")
        self._closed = False

    def tell(self) -> int:
        return 0 if self._fh is None else int(self._fh.tell())

    def write(self, data: bytes) -> None:
        if self._closed or self._fh is None:
            raise RuntimeError("sink 已关闭")
        self._fh.write(data)

    def patch(self, logical_offset: int, data: bytes) -> None:
        if self._closed or self._fh is None:
            raise RuntimeError("sink 已关闭")
        current = self._fh.tell()
        self._fh.seek(logical_offset)
        self._fh.write(data)
        self._fh.seek(current)

    def finish(self) -> tuple[Path, int, list[Path]]:
        if self._closed or self._fh is None:
            raise RuntimeError("sink 已关闭")
        size = int(self._fh.tell())
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        self._closed = True
        os.replace(self._tmp, self.path)
        return self.path, size, [self.path]

    def abort(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        self._closed = True
        try:
            if self._tmp.exists():
                self._tmp.unlink()
        except OSError:
            pass


class VolumeSink(ArchiveSink):
    def __init__(
        self,
        path: Path,
        max_volume_size: int,
        archive_id: bytes,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_volume_size = int(max_volume_size)
        if self.max_volume_size < MIN_VOLUME_SIZE:
            raise PxcsVolumeError(
                t("volume.too_small", value=self.max_volume_size, minimum=MIN_VOLUME_SIZE)
            )
        if self.max_volume_size <= VOLUME_HEADER_SIZE + 256:
            raise PxcsVolumeError(
                t("volume.too_small", value=self.max_volume_size, minimum=MIN_VOLUME_SIZE)
            )
        self.archive_id = bytes(archive_id)
        self.overwrite = bool(overwrite)
        self._data_limit = self.max_volume_size - VOLUME_HEADER_SIZE
        self._tag = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        # 每一卷：[path, tmp_path, fh, logical_start, length]
        self._volumes: list[dict] = []
        self.volume_paths: list[Path] = []
        self._closed = False
        self._open_next()

    # ---------------------------------------------------------- 内部
    def _open_next(self) -> None:
        index = len(self._volumes) + 1
        if index > MAX_VOLUMES:
            raise PxcsVolumeError(t("volume.too_many", maximum=MAX_VOLUMES))
        final = volume_path(self.path, index)
        tmp = final.with_name(final.name + f".tmp-{self._tag}")
        fh = open(tmp, "w+b")
        fh.write(b"\x00" * VOLUME_HEADER_SIZE)
        self._volumes.append(
            {
                "path": final,
                "tmp": tmp,
                "fh": fh,
                "start": self.tell(),
                "length": 0,
            }
        )

    def _current(self) -> dict:
        return self._volumes[-1]

    # ---------------------------------------------------------- 接口
    def tell(self) -> int:
        return sum(int(v["length"]) for v in self._volumes)

    def write(self, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("sink 已关闭")
        view = memoryview(data)
        while view:
            current = self._current()
            room = self._data_limit - int(current["length"])
            if room <= 0:
                self._open_next()
                continue
            take = min(room, len(view))
            current["fh"].write(view[:take])
            current["length"] = int(current["length"]) + take
            view = view[take:]

    def patch(self, logical_offset: int, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("sink 已关闭")
        starts = [int(v["start"]) for v in self._volumes]
        index = bisect.bisect_right(starts, logical_offset) - 1
        if index < 0:
            raise RuntimeError("patch offset 越界")
        volume = self._volumes[index]
        local = logical_offset - int(volume["start"])
        if local + len(data) > int(volume["length"]):
            # 头部只可能落在第 1 卷，且该区域一定已经写过。
            raise RuntimeError("patch 超出已写区域")
        fh: BinaryIO = volume["fh"]
        end = fh.tell()
        fh.seek(VOLUME_HEADER_SIZE + local)
        fh.write(data)
        fh.seek(end)

    def finish(self) -> tuple[Path, int, list[Path]]:
        if self._closed:
            raise RuntimeError("sink 已关闭")
        total = self.tell()
        count = len(self._volumes)
        # 1. 回填每一卷的卷头。
        for index, info in enumerate(self._volumes, 1):
            fh: BinaryIO = info["fh"]
            header = pack_volume_header(
                index,
                count,
                int(info["start"]),
                int(info["length"]),
                self.archive_id,
                final=(index == count),
            )
            fh.flush()
            fh.seek(0)
            fh.write(header)
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
        self._closed = True
        # 2. 全部落盘后再改名，避免中途失败留下“半套”正式文件名。
        finalized: list[Path] = []
        for info in self._volumes:
            final = info["path"]
            os.replace(info["tmp"], final)
            finalized.append(final)
        self.volume_paths = finalized
        # 3. 覆盖模式下删除多出来的旧分卷，避免下次读取时数量对不上。
        self._remove_stale(len(finalized))
        return finalized[0], total, finalized

    def _remove_stale(self, count: int) -> None:
        if not self.overwrite:
            return
        directory = self.path.parent
        for candidate in directory.glob(self.path.name + ".*"):
            parsed = parse_volume_name(candidate.name)
            if parsed is None or parsed[0] != self.path.name:
                continue
            if parsed[1] > count:
                try:
                    candidate.unlink()
                except OSError:
                    pass

    def abort(self) -> None:
        self._closed = True
        for info in self._volumes:
            fh: BinaryIO = info["fh"]
            try:
                fh.close()
            except OSError:
                pass
        for info in self._volumes:
            try:
                tmp = info["tmp"]
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        self._volumes = []


def make_sink(
    path: Path,
    *,
    archive_id: bytes,
    volume_size: Optional[int],
    overwrite: bool,
) -> ArchiveSink:
    if volume_size:
        return VolumeSink(path, int(volume_size), archive_id, overwrite=overwrite)
    return FileSink(path, overwrite=overwrite)


def volume_summary(paths: list[Path]) -> str:
    if not paths:
        return ""
    total = sum(p.stat().st_size for p in paths if p.exists())
    return t(
        "volume.summary",
        count=len(paths),
        first=paths[0].name,
        last=paths[-1].name,
        size=human_size(total),
    )


def describe_layout(path: Path) -> str:
    """调试/日志用：返回分卷布局摘要（不含文件内容）。"""
    volumes = find_volume_set(path)
    if not volumes:
        return ""
    return volume_summary(volumes)

