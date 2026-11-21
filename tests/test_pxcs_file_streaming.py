"""普通文件流式处理的回归。

大文件（≥64 MiB）不能整块读入再 zlib：视频/压缩包会白烧 CPU 和内存，
日志类又值得压缩。策略是抽样判断可压缩性，再流式 store 或流式 zlib。
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive, extract_archive, verify_archive
from pxcs_core.container import ArchiveReader, ArchiveWriter, PxcsFormatError, compute_flags
from pxcs_core.archive import _make_entry
from pxcs_core.format import CODEC_STORE, CODEC_ZLIB
from pxcs_core.grouped_create import _STREAM_FILE_THRESHOLD


def _write_random(path: Path, size: int) -> None:
    with path.open("wb") as fh:
        remaining = size
        while remaining > 0:
            chunk = os.urandom(min(1 << 20, remaining))
            fh.write(chunk)
            remaining -= len(chunk)


def _write_text(path: Path, size: int) -> None:
    line = b"the quick brown fox jumps over the lazy dog 0123456789\n"
    block = line * ((1 << 20) // len(line) + 1)
    with path.open("wb") as fh:
        remaining = size
        while remaining > 0:
            chunk = block[: min(len(block), remaining)]
            fh.write(chunk)
            remaining -= len(chunk)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class AddStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_store_does_not_buffer_whole_file(self) -> None:
        size = 64 * 1024 * 1024
        src = self.root / "big.bin"
        _write_random(src, size)
        out = self.root / "a.pxcs"
        writer = ArchiveWriter(out, overwrite=True)
        tracemalloc.start()
        info = writer.add_stream(src, size)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        entry = _make_entry(
            0, "big.bin", "file", CODEC_STORE, size=size, mtime_ns=0,
            source_sha256=info[5],
            payload_info=(info[0], info[1], None, info[2], info[3]),
            raw_crc32=info[4],
        )
        writer.finish([entry], compute_flags([entry]))
        self.assertLess(peak, 16 * 1024 * 1024, "add_stream 不应整块读入")
        with ArchiveReader(out) as reader:
            from pxcs_core.archive import _original_bytes

            self.assertEqual(
                hashlib.sha256(_original_bytes(reader, reader.get_entry(0))).hexdigest(),
                _sha(src),
            )

    def test_streaming_zlib_roundtrip(self) -> None:
        size = 8 * 1024 * 1024
        src = self.root / "text.log"
        _write_text(src, size)
        out = self.root / "b.pxcs"
        writer = ArchiveWriter(out, overwrite=True)
        info = writer.add_stream(src, size, compress_level=6)
        self.assertLess(info[1], size // 4, "可压缩文件应该明显变小")
        entry = _make_entry(
            0, "text.log", "file", CODEC_ZLIB, size=size, mtime_ns=0,
            source_sha256=info[5],
            payload_info=(info[0], info[1], None, info[2], info[3]),
            raw_crc32=info[4],
        )
        writer.finish([entry], compute_flags([entry]))
        with ArchiveReader(out) as reader:
            from pxcs_core.archive import _original_bytes

            self.assertEqual(
                hashlib.sha256(_original_bytes(reader, reader.get_entry(0))).hexdigest(),
                _sha(src),
            )

    def test_encrypted_writer_rejects_streaming(self) -> None:
        src = self.root / "small.bin"
        src.write_bytes(b"x" * 1024)
        writer = ArchiveWriter(self.root / "c.pxcs", overwrite=True, password="pw",
                               crypto_params=None)
        try:
            with self.assertRaises(PxcsFormatError):
                writer.add_stream(src, 1024)
        finally:
            writer.abort()


class LargeFileCreateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "src"
        self.src.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create(self) -> tuple[Path, object]:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(self.src / "a.png")
        out = self.root / "out.pxcs"
        stats = create_archive(
            out, [str(self.src)], CreateOptions(quiet=True, progress=False, threads=2)
        )
        return out, stats

    def test_incompressible_large_file_is_stored(self) -> None:
        big = self.src / "video.bin"
        _write_random(big, _STREAM_FILE_THRESHOLD + (1 << 20))
        out, _ = self._create()
        with ArchiveReader(out) as reader:
            entry = next(e for e in reader.entries if e.path.endswith("video.bin"))
            self.assertEqual(entry.codec, CODEC_STORE)
            self.assertEqual(entry.reason, "store_stream")
        self.assertEqual(verify_archive(out).failed, 0)

    def test_compressible_large_file_is_stream_compressed(self) -> None:
        big = self.src / "app.log"
        _write_text(big, _STREAM_FILE_THRESHOLD + (1 << 20))
        out, _ = self._create()
        with ArchiveReader(out) as reader:
            entry = next(e for e in reader.entries if e.path.endswith("app.log"))
            self.assertEqual(entry.codec, CODEC_ZLIB)
            self.assertLess(entry.stored_length, entry.size // 4)
        self.assertEqual(verify_archive(out).failed, 0)
        dest = self.root / "out"
        extract_archive(out, dest, quiet=True, threads=1)
        self.assertEqual(_sha(dest / "app.log"), _sha(big))


if __name__ == "__main__":
    unittest.main()
