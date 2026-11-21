"""zlib 解压边界与残差 dtype 的回归测试。

这里钉住三件事：

1. 解压必须在**分配之前**就有限制，炸弹只能报错，不能把进程撑爆；
2. 尾部垃圾、提前结束、声明长度不符都算归档损坏，不算"能读的归档"；
3. 写不出"能创建、解不开"的残差条目——16-bit 走 zlib 残差必须在编码阶段拒绝。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from pxc_core.errors import PxcCorruptionError
from pxc_core.image_io import ImageData
from pxcs_core.archive import CreateOptions, create_archive, verify_archive
from pxcs_core.codec import PxcsImageError, encode_zlib_residual
from pxcs_core.container import ArchiveReader
from pxcs_core.format import (
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    PXCS_END_MAGIC,
    PXCS_MAGIC,
)
from pxcs_core.zlibsafe import decompress_bounded


def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _build_archive(path: Path, entries: list[dict], payloads: list[bytes | None]) -> None:
    """写一个只满足容器结构的最小归档，用于构造恶意样本。"""
    archive_id = os.urandom(16)
    body = bytearray()
    cur = HEADER_SIZE
    for entry, payload in zip(entries, payloads):
        if payload is None:
            entry["offset"] = 0
            entry["length"] = 0
            continue
        pad = (-cur) % 8
        if pad:
            body += b"\x00" * pad
            cur += pad
        entry["offset"] = cur
        entry["length"] = len(payload)
        entry["data_crc32"] = _crc(payload)
        body += payload
        cur += len(payload)
    index_raw = json.dumps(
        {
            "format": "PXCS",
            "version": 1,
            "archive_id": archive_id.hex(),
            "created_by": "test",
            "entries": entries,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    index_blob = zlib.compress(index_raw, 9)
    index_offset = HEADER_SIZE + len(body)
    file_size = index_offset + len(index_blob) + FOOTER_SIZE
    header_zero = HEADER_STRUCT.pack(
        PXCS_MAGIC, 1, 2, 0, HEADER_SIZE, len(entries), index_offset,
        len(index_blob), HEADER_SIZE, archive_id, b"\x00" * 4, b"\x00" * 62, 0,
    )
    header = HEADER_STRUCT.pack(
        PXCS_MAGIC, 1, 2, 0, HEADER_SIZE, len(entries), index_offset,
        len(index_blob), HEADER_SIZE, archive_id, b"\x00" * 4, b"\x00" * 62,
        _crc(header_zero),
    )
    footer_zero = FOOTER_STRUCT.pack(
        PXCS_END_MAGIC, file_size, index_offset, len(index_blob),
        _crc(index_blob), 0, archive_id, b"\x00" * 8,
    )
    footer = FOOTER_STRUCT.pack(
        PXCS_END_MAGIC, file_size, index_offset, len(index_blob),
        _crc(index_blob), _crc(footer_zero), archive_id, b"\x00" * 8,
    )
    path.write_bytes(header + bytes(body) + index_blob + footer)


def _file_entry(size: int, name: str = "big.bin") -> dict:
    return {
        "id": 0, "path": name, "kind": "file", "codec": "zlib",
        "offset": 0, "length": 0, "size": size, "ref": None, "depth": 0,
        "width": 0, "height": 0, "mode": "", "format": "", "ext": "",
        "quant": 0, "centered": False, "lossy": False,
        "pixel_sha256": "", "source_sha256": "", "data_crc32": 0,
        "raw_crc32": 0, "data_sha256": "", "mtime_ns": 0, "flags": 0, "reason": "",
    }


class BoundedDecompressTest(unittest.TestCase):
    def test_bomb_rejected_at_limit(self) -> None:
        bomb = zlib.compress(b"\x00" * (1 << 20), 9)
        with self.assertRaises(PxcCorruptionError) as ctx:
            decompress_bounded(bomb, limit=4096, label="bomb")
        self.assertIn("4096", str(ctx.exception))

    def test_exact_length_accepted(self) -> None:
        raw = bytes(range(256)) * 8
        out = decompress_bounded(
            zlib.compress(raw), limit=len(raw), expected=len(raw), label="ok"
        )
        self.assertEqual(out, raw)

    def test_trailing_garbage_rejected(self) -> None:
        blob = zlib.compress(b"hello") + b"junk"
        with self.assertRaises(PxcCorruptionError):
            decompress_bounded(blob, limit=64, label="trailing")

    def test_truncated_stream_rejected(self) -> None:
        blob = zlib.compress(b"x" * 4096)[:-6]
        with self.assertRaises(PxcCorruptionError):
            decompress_bounded(blob, limit=8192, label="short")

    def test_declared_length_mismatch_rejected(self) -> None:
        with self.assertRaises(PxcCorruptionError):
            decompress_bounded(
                zlib.compress(b"abc"), limit=16, expected=4, label="len"
            )

    def test_empty_payload_rejected(self) -> None:
        with self.assertRaises(PxcCorruptionError):
            decompress_bounded(b"", limit=16, label="empty")


class CraftedArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_file_entry_bomb_is_rejected_by_declared_size(self) -> None:
        """声明 16 字节、实际可展开 1 MiB：必须在 16 字节处停下。"""
        payload = zlib.compress(b"\x00" * (1 << 20), 9)
        path = self.tmp / "bomb.pxcs"
        _build_archive(path, [_file_entry(16)], [payload])
        from pxcs_core.archive import _original_bytes

        with ArchiveReader(path) as reader:
            with self.assertRaises(PxcCorruptionError):
                _original_bytes(reader, reader.get_entry(0))

    def test_file_entry_with_trailing_bytes_is_rejected(self) -> None:
        raw = b"payload" * 16
        payload = zlib.compress(raw) + b"\x00" * 8
        path = self.tmp / "trailing.pxcs"
        _build_archive(path, [_file_entry(len(raw), "t.bin")], [payload])
        from pxcs_core.archive import _original_bytes

        with ArchiveReader(path) as reader:
            with self.assertRaises(PxcCorruptionError):
                _original_bytes(reader, reader.get_entry(0))


class ResidualDtypeTest(unittest.TestCase):
    def _image16(self, offset: int = 0) -> ImageData:
        data = (
            np.arange(16, dtype=np.uint16).reshape(4, 4, 1) * 9 + offset
        ).astype(np.uint16)
        return ImageData(mode="I;16", width=4, height=4, data=data)

    def test_encode_zlib_rejects_16bit(self) -> None:
        with self.assertRaises(PxcsImageError):
            encode_zlib_residual(self._image16(0), self._image16(3), quant=0)

    def test_fast_mode_16bit_create_stays_verifiable(self) -> None:
        """--fast（residual_codec=zlib）不能把 16-bit 组写成坏归档。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            base = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) * 7)
            Image.fromarray(base).save(src / "a.png")
            Image.fromarray((base + 3).astype(np.uint16)).save(src / "b.png")
            out = root / "out.pxcs"
            create_archive(
                out,
                [str(src)],
                CreateOptions(
                    quiet=True, progress=False, threads=2,
                    residual_codec="zlib", self_check=True,
                ),
            )
            stats = verify_archive(out)
            self.assertEqual(stats.failed, 0, stats.errors)
            with ArchiveReader(out) as reader:
                self.assertFalse(
                    any(e.codec == "zlib_residual" for e in reader.entries),
                    "16-bit 组不应写出 zlib 残差",
                )
                # 无损源必须仍能逐像素恢复：解出来和源图一致。
                from pxcs_core.archive import _ImageResolver

                resolver = _ImageResolver(reader)
                got = resolver.get(reader.entries[-1].id)
                expected = np.asarray(Image.open(src / "b.png"))
                self.assertTrue(np.array_equal(got.data[:, :, 0], expected))
                self.assertEqual(
                    hashlib.sha256(got.data.tobytes()).hexdigest(),
                    hashlib.sha256(
                        np.ascontiguousarray(expected[:, :, None]).tobytes()
                    ).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
