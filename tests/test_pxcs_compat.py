"""格式兼容性测试：旧版归档可读，新版未加密归档仍是旧布局。"""
from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pxc_core.utils import crc32, sha256_bytes
from pxcs_core.archive import (
    CreateOptions,
    archive_info,
    create_archive,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxcs_core.container import ArchiveReader
from pxcs_core.format import (
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    PXCS_END_MAGIC,
    PXCS_MAGIC,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_v1_archive(path: Path, root: Path) -> None:
    """手工构造一个 1.0 布局的归档（无 quant/centered/lossy 字段）。"""
    archive_id = bytes(range(16))
    payloads: list[bytes] = []
    entries: list[dict] = []
    data = bytearray(b"\x00" * HEADER_SIZE)

    note = "legacy v1.0 archive\n" * 20
    blob = zlib.compress(note.encode("utf-8"), 6)
    offset = len(data)
    data += blob
    payloads.append(blob)
    entries.append(
        {
            "id": 0, "path": "note.txt", "kind": "file", "codec": "zlib",
            "offset": offset, "length": len(blob), "size": len(note),
            "data_crc32": crc32(blob),
            "raw_crc32": crc32(note.encode("utf-8")),
            "source_sha256": sha256_bytes(note.encode("utf-8")).hex(),
        }
    )

    jpg = root / "photo.jpg"
    img = Image.fromarray(
        np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (48, 1, 3)), "RGB"
    )
    img.save(jpg, quality=90)
    raw = jpg.read_bytes()
    pad = (-len(data)) % 8
    data += b"\x00" * pad
    offset = len(data)
    data += raw
    entries.append(
        {
            "id": 1, "path": "photo.jpg", "kind": "image", "codec": "raw",
            "offset": offset, "length": len(raw), "size": len(raw),
            "width": 64, "height": 48, "mode": "RGB", "format": "JPEG",
            "ext": ".jpg", "data_crc32": crc32(raw),
            "data_sha256": sha256_bytes(raw).hex(),
            "source_sha256": sha256_bytes(raw).hex(),
        }
    )

    index_obj = {
        "format": "PXCS",
        "version": 1,
        "archive_id": archive_id.hex(),
        "created_by": "PXCS Python 1.0",
        "entries": entries,
    }
    index_raw = json.dumps(index_obj, separators=(",", ":")).encode("utf-8")
    index_blob = zlib.compress(index_raw, 9)
    index_offset = len(data)
    data += index_blob
    file_size = len(data) + FOOTER_SIZE

    footer_zero = FOOTER_STRUCT.pack(
        PXCS_END_MAGIC, file_size, index_offset, len(index_blob),
        crc32(index_blob), 0, archive_id, b"\x00" * 8,
    )
    footer = FOOTER_STRUCT.pack(
        PXCS_END_MAGIC, file_size, index_offset, len(index_blob),
        crc32(index_blob), crc32(footer_zero), archive_id, b"\x00" * 8,
    )
    header_zero = HEADER_STRUCT.pack(
        PXCS_MAGIC, 1, 0, 0, HEADER_SIZE, len(entries), index_offset,
        len(index_blob), HEADER_SIZE, archive_id, b"\x00" * 4, b"\x00" * 62, 0,
    )
    header = HEADER_STRUCT.pack(
        PXCS_MAGIC, 1, 0, 0, HEADER_SIZE, len(entries), index_offset,
        len(index_blob), HEADER_SIZE, archive_id, b"\x00" * 4, b"\x00" * 62,
        crc32(header_zero),
    )
    data[:HEADER_SIZE] = header
    data += footer
    path.write_bytes(bytes(data))


class CompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-compat-"))
        self.src = self.tmp / "src"
        self.src.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fixture_created_by_1_2_still_works(self) -> None:
        fixture = FIXTURES / "legacy_v1_2.pxcs"
        self.assertTrue(fixture.is_file(), fixture)
        with ArchiveReader(fixture) as reader:
            self.assertEqual(reader.header.version_minor, 2)
            self.assertEqual(reader.volume_count, 0)
            self.assertFalse(reader.is_encrypted)
        self.assertIn("001.png", list_entries(fixture))
        self.assertEqual(verify_archive(fixture).failed, 0)
        out = self.tmp / "fixture-out"
        stats = extract_archive(fixture, out, quiet=True)
        self.assertEqual(stats.failed, 0)
        self.assertTrue((out / "001.png").is_file())
        self.assertTrue((out / "readme.txt").is_file())

    def test_v1_0_archive_is_readable(self) -> None:
        archive = self.tmp / "legacy_v1_0.pxcs"
        build_v1_archive(archive, self.tmp)
        with ArchiveReader(archive) as reader:
            self.assertEqual(reader.header.version_minor, 0)
            self.assertEqual(len(reader.entries), 2)
            self.assertEqual(reader.entries[0].plain_length, None)
        text = list_entries(archive)
        self.assertIn("note.txt", text)
        self.assertIn("photo.jpg", text)
        self.assertEqual(verify_archive(archive).failed, 0)
        self.assertEqual(verify_archive(archive, quick=True).failed, 0)
        out = self.tmp / "v1-out"
        stats = extract_archive(archive, out, quiet=True)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(
            (self.src / "unused").exists(), False
        )
        self.assertEqual((out / "note.txt").read_text(encoding="utf-8"),
                         "legacy v1.0 archive\n" * 20)

    def test_new_plain_archive_keeps_v1_layout(self) -> None:
        Image.fromarray(np.zeros((32, 32, 3), np.uint8), "RGB").save(
            self.src / "a.png"
        )
        archive = self.tmp / "new.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        raw = archive.read_bytes()
        magic, major, minor, flags = struct.unpack_from("<4sHHI", raw, 0)
        data_offset = HEADER_STRUCT.unpack_from(raw, 0)[8]
        self.assertEqual(magic, PXCS_MAGIC)
        self.assertEqual(major, 1)
        self.assertEqual(minor, 2)
        self.assertEqual(flags & 0x20, 0)
        self.assertEqual(data_offset, HEADER_SIZE)
        self.assertEqual(len(list(self.tmp.glob("new.pxcs.*"))), 0)

    def test_encrypted_archive_uses_minor_3(self) -> None:
        Image.fromarray(np.zeros((32, 32, 3), np.uint8), "RGB").save(
            self.src / "a.png"
        )
        archive = self.tmp / "enc.pxcs"
        create_archive(
            archive, [str(self.src)],
            CreateOptions(
                quiet=True, password="passphrase-123",
                kdf_memory_kib=8192, kdf_iterations=1, kdf_lanes=1,
            ),
        )
        raw = archive.read_bytes()
        _magic, major, minor, flags = struct.unpack_from("<4sHHI", raw, 0)
        self.assertEqual(minor, 3)
        self.assertTrue(flags & 0x20)
        self.assertIn("PXCS 1.3", archive_info(archive, password="passphrase-123"))

    def test_future_minor_is_rejected(self) -> None:
        Image.fromarray(np.zeros((32, 32, 3), np.uint8), "RGB").save(
            self.src / "a.png"
        )
        archive = self.tmp / "future.pxcs"
        create_archive(archive, [str(self.src)], CreateOptions(quiet=True))
        raw = bytearray(archive.read_bytes())
        struct.pack_into("<H", raw, 6, 9)  # minor = 9
        # 重算 header CRC，让错误来自版本检查而不是 CRC
        head = bytearray(raw[:HEADER_SIZE])
        head[-4:] = b"\x00\x00\x00\x00"
        struct.pack_into("<I", raw, HEADER_SIZE - 4, crc32(bytes(head)))
        archive.write_bytes(bytes(raw))
        from pxc_core.errors import PxcVersionError

        with self.assertRaises(PxcVersionError):
            list_entries(archive)


if __name__ == "__main__":
    unittest.main()
