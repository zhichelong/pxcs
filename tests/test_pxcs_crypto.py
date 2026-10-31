"""PXCS 加密测试：AEAD、口令校验、文件名加密、篡改检测、与分卷组合。"""
from __future__ import annotations

import contextlib
import io
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pxc_core.utils import crc32
from pxcs_core.archive import (
    CreateOptions,
    CreateStats,
    archive_info,
    create_archive,
    extract_archive,
    list_entries,
    verify_archive,
)
from pxc_core.errors import PxcFormatError
from pxcs_core.container import ArchiveReader, PxcsCorruptionError, PxcsFormatError
from pxcs_core.crypto import (
    CRYPTO_HEADER_SIZE,
    PxcAuthenticationError,
    PxcPasswordRequired,
)
from pxcs_core.format import FLAG_ENCRYPTED, FOOTER_STRUCT, HEADER_SIZE
from pxcs_core.volumes import volume_path

PASSWORD = "correct horse battery staple"
FAST_KDF = dict(kdf_memory_kib=8192, kdf_iterations=1, kdf_lanes=1)


def _write_sources(root: Path, count: int = 3) -> None:
    yy, xx = np.mgrid[0:120, 0:120]
    arr = np.zeros((120, 120, 3), np.uint8)
    arr[..., 0] = xx
    arr[..., 1] = yy
    arr[..., 2] = 90
    base = Image.fromarray(arr, "RGB")
    for i in range(count):
        img = base.copy()
        ImageDraw.Draw(img).rectangle([5 + i, 5, 45, 45], fill=(200, i * 50, 30))
        img.save(root / f"{i:03d}.png")
    (root / "note.txt").write_text("encrypted archive\n" * 30, encoding="utf-8")


def _index_blob(path: Path) -> bytes:
    raw = path.read_bytes()
    footer = raw[-64:]
    _magic, _size, index_offset, index_length, *_rest = FOOTER_STRUCT.unpack(footer)
    return raw[index_offset:index_offset + index_length]


class CryptoOptionMixin:
    def make_options(self, **kwargs) -> CreateOptions:
        params = dict(FAST_KDF)
        params.update(kwargs)
        return CreateOptions(quiet=True, **params)


class CryptoBasicsTests(CryptoOptionMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-crypto-"))
        self.src = self.tmp / "src"
        self.src.mkdir()
        _write_sources(self.src)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create(self, name: str = "enc.pxcs", **kwargs) -> Path:
        archive = self.tmp / name
        password = kwargs.pop("password", PASSWORD)
        create_archive(
            archive,
            [str(self.src)],
            self.make_options(password=password, **kwargs),
        )
        return archive

    # ------------------------------------------------------------ 正常流程
    def test_roundtrip_with_filename_encryption(self) -> None:
        archive = self._create()
        text = list_entries(archive, password=PASSWORD)
        self.assertIn("000.png", text)
        self.assertIn("加密（文件名已加密）", text)
        info = archive_info(archive, password=PASSWORD)
        self.assertIn("PXCS 1.3", info)
        self.assertIn("aes-256-gcm", info.lower())
        self.assertEqual(verify_archive(archive, password=PASSWORD).failed, 0)
        out = self.tmp / "out"
        stats = extract_archive(archive, out, quiet=True, password=PASSWORD)
        self.assertEqual(stats.failed, 0)
        for name in ("000.png", "002.png", "note.txt"):
            self.assertEqual(
                (self.src / name).read_bytes(), (out / name).read_bytes(), name
            )

    def test_roundtrip_without_filename_encryption(self) -> None:
        archive = self._create("plain_names.pxcs", encrypt_filenames=False)
        # 文件名模式下索引只是 zlib(JSON)，不带口令也能看出结构；
        # 但 payload 仍是 AEAD 密文。
        blob = _index_blob(archive)
        decoded = zlib.decompress(blob).decode("utf-8")
        self.assertIn("000.png", decoded)
        self.assertEqual(verify_archive(archive, password=PASSWORD).failed, 0)
        out = self.tmp / "out"
        extract_archive(archive, out, quiet=True, password=PASSWORD)
        self.assertEqual(
            (self.src / "000.png").read_bytes(), (out / "000.png").read_bytes()
        )

    def test_encrypted_filenames_hide_names(self) -> None:
        archive = self._create()
        blob = _index_blob(archive)
        self.assertNotIn(b"000.png", blob)
        with self.assertRaises(zlib.error):
            zlib.decompress(blob)

    def test_single_entry_extract(self) -> None:
        archive = self._create()
        out = self.tmp / "one"
        stats = extract_archive(archive, out, names=["001.png"], quiet=True, password=PASSWORD)
        self.assertEqual(stats.extracted, 1)
        self.assertTrue((out / "001.png").is_file())
        self.assertFalse((out / "000.png").exists())

    def test_scrypt_kdf(self) -> None:
        archive = self.tmp / "scrypt.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(
                quiet=True, password=PASSWORD, kdf="scrypt",
                kdf_memory_kib=1024, kdf_iterations=8, kdf_lanes=1,
            ),
        )
        self.assertIn("scrypt", archive_info(archive, password=PASSWORD))
        self.assertEqual(verify_archive(archive, password=PASSWORD).failed, 0)

    # ------------------------------------------------------------ 错误处理
    def test_password_required(self) -> None:
        archive = self._create()
        for call in (
            lambda: list_entries(archive),
            lambda: archive_info(archive),
            lambda: verify_archive(archive),
            lambda: extract_archive(archive, self.tmp / "x", quiet=True),
        ):
            with self.assertRaises(PxcPasswordRequired):
                call()

    def test_wrong_password(self) -> None:
        archive = self._create()
        with self.assertRaises(PxcAuthenticationError) as ctx:
            list_entries(archive, password="wrong password here")
        message = str(ctx.exception)
        self.assertIn("认证失败", message)
        self.assertNotIn(PASSWORD, message)

    def test_wrong_password_without_names_encryption(self) -> None:
        archive = self._create("plain_names.pxcs", encrypt_filenames=False)
        with self.assertRaises(PxcAuthenticationError):
            verify_archive(archive, password="nope nope nope")

    def test_tampered_payload_detected(self) -> None:
        archive = self._create("tamper.pxcs")
        with ArchiveReader(archive, password=PASSWORD) as reader:
            offset = reader.entries[0].offset
        raw = bytearray(archive.read_bytes())
        raw[offset + 3] ^= 0xFF
        archive.write_bytes(bytes(raw))
        stats = verify_archive(archive, password=PASSWORD)
        self.assertGreater(stats.failed, 0)
        joined = "\n".join(stats.errors)
        self.assertTrue("认证失败" in joined or "CRC" in joined, joined)

    def test_tampered_index_detected(self) -> None:
        archive = self._create("tamper_index.pxcs")
        raw = bytearray(archive.read_bytes())
        # 索引块在末尾 footer 之前；改最后一字节（属于 AEAD tag 区）
        raw[-FOOTER_STRUCT.size - 1] ^= 0xFF
        archive.write_bytes(bytes(raw))
        with self.assertRaises((PxcsCorruptionError, PxcAuthenticationError)):
            list_entries(archive, password=PASSWORD)

    def test_tampered_crypto_header(self) -> None:
        archive = self._create("tamper_hdr.pxcs")
        raw = bytearray(archive.read_bytes())
        # 改校验块（44..88），保持 header CRC 有效时会进入 AEAD 校验失败
        raw[HEADER_SIZE + 60] ^= 0xFF
        # 同时修正 header CRC，避免先被 CRC 拦下
        block = bytearray(raw[HEADER_SIZE:HEADER_SIZE + CRYPTO_HEADER_SIZE])
        block[124:128] = b"\x00\x00\x00\x00"
        crc = crc32(bytes(block))
        struct.pack_into("<I", raw, HEADER_SIZE + CRYPTO_HEADER_SIZE - 4, crc)
        archive.write_bytes(bytes(raw))
        with self.assertRaises(PxcAuthenticationError):
            list_entries(archive, password=PASSWORD)

    def test_truncated_crypto_header(self) -> None:
        archive = self._create("trunc.pxcs")
        raw = bytearray(archive.read_bytes())
        raw[HEADER_SIZE + 10] ^= 0xFF  # 破坏加密头 magic 附近的字节
        archive.write_bytes(bytes(raw))
        with self.assertRaises(PxcFormatError):
            list_entries(archive, password=PASSWORD)

    # ------------------------------------------------------------ 组合
    def test_encryption_with_volumes(self) -> None:
        archive = self._create("vol.pxcs", volume_size=2048)
        parts = sorted(self.tmp.glob("vol.pxcs.*"))
        self.assertGreaterEqual(len(parts), 2)
        self.assertNotIn(b"000.png", b"".join(p.read_bytes() for p in parts))
        self.assertEqual(verify_archive(archive, password=PASSWORD).failed, 0)
        out = self.tmp / "vol-out"
        stats = extract_archive(
            archive, out, quiet=True, password=PASSWORD, names=["002.png"]
        )
        self.assertEqual(stats.extracted, 1)
        self.assertEqual(
            (self.src / "002.png").read_bytes(), (out / "002.png").read_bytes()
        )
        with self.assertRaises(PxcPasswordRequired):
            list_entries(archive)
        with self.assertRaises(PxcAuthenticationError):
            list_entries(archive, password="bad password 123")

    def test_flags_mark_encryption(self) -> None:
        archive = self._create()
        raw = archive.read_bytes()
        magic, major, minor, flags = struct.unpack_from("<4sHHI", raw, 0)
        self.assertEqual(magic, b"PXCS")
        self.assertEqual(minor, 3)
        self.assertTrue(flags & FLAG_ENCRYPTED)

    def test_mixed_archive_files_still_work(self) -> None:
        plain = self.tmp / "plain.pxcs"
        create_archive(plain, [str(self.src)], CreateOptions(quiet=True))
        enc = self._create("enc2.pxcs")
        # 未加密归档不受新加密逻辑影响
        self.assertEqual(verify_archive(plain).failed, 0)
        self.assertEqual(verify_archive(enc, password=PASSWORD).failed, 0)


class CryptoSecrecyTests(CryptoOptionMixin, unittest.TestCase):
    """口令绝不能出现在日志、错误信息或普通输出里。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-secret-"))
        self.src = self.tmp / "src"
        self.src.mkdir()
        _write_sources(self.src, 2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_password_not_in_output(self) -> None:
        password = "very-secret-passphrase-42"
        archive = self.tmp / "secret.pxcs"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            opts = CreateOptions(quiet=False, password=password, **FAST_KDF)
            stats = create_archive(archive, [str(self.src)], opts)
            print(stats)
            print(opts)
            list_entries(archive, password=password)
            archive_info(archive, password=password)
            verify_archive(archive, password=password)
            extract_archive(archive, self.tmp / "out", quiet=False, password=password)
            try:
                list_entries(archive, password="wrong-" + password)
            except PxcAuthenticationError as exc:
                print(f"[FAIL] {exc}")
        text = buffer.getvalue()
        self.assertNotIn(password, text)
        self.assertIn("认证失败", text)

    def test_options_repr_hides_password(self) -> None:
        opts = CreateOptions(password="hunter2-hunter2")
        self.assertNotIn("hunter2", repr(opts))
        stats = CreateStats()
        self.assertNotIn("hunter2", repr(stats))

    def test_password_file_reader(self) -> None:
        from pxcs import _read_password_file

        path = self.tmp / "pw.txt"
        path.write_text("file-passphrase\nignored\n", encoding="utf-8")
        self.assertEqual(_read_password_file(str(path)), "file-passphrase")
        empty = self.tmp / "empty.txt"
        empty.write_bytes(b"")
        with self.assertRaises(Exception):
            _read_password_file(str(empty))


if __name__ == "__main__":
    unittest.main()
