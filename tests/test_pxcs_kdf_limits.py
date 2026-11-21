"""KDF 资源消耗上限的回归。

默认参数（256 MiB / 3 轮 / 4 lane）必须能直接打开；自定义高开销参数在
派生密钥之前就要拦住，只有显式 ``--allow-expensive-kdf`` 才放行。
"""
from __future__ import annotations

import os
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxc_core.errors import PxcFormatError
from pxcs_core.crypto import (
    CRYPTO_HEADER_STRUCT,
    KDF_ARGON2ID,
    CryptoParams,
    PxcKdfTooExpensive,
    check_kdf_cost,
    new_crypto_context,
    pack_header,
    set_allow_expensive_kdf,
    unpack_header,
)
from pxcs_core.container import ArchiveReader


def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


class KdfLimitsTest(unittest.TestCase):
    def setUp(self) -> None:
        set_allow_expensive_kdf(False)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        set_allow_expensive_kdf(False)
        self._tmp.cleanup()

    def _make_archive(self, **kwargs) -> Path:
        src = self.root / "src"
        src.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(src / "a.png")
        out = self.root / "enc.pxcs"
        create_archive(
            out,
            [str(src)],
            CreateOptions(
                quiet=True, progress=False, overwrite=True, threads=1,
                password="pw", **kwargs
            ),
        )
        return out

    def test_default_params_open_without_flag(self) -> None:
        archive = self._make_archive()
        with ArchiveReader(archive, password="pw") as reader:
            self.assertEqual(len(reader.entries), 1)

    def test_expensive_params_need_opt_in(self) -> None:
        # 只把迭代次数抬到 9（读侧上限 8），内存保持最小值，派生很快。
        archive = self._make_archive(kdf="argon2id", kdf_memory_kib=8192, kdf_iterations=9)
        with self.assertRaises(PxcKdfTooExpensive) as ctx:
            ArchiveReader(archive, password="pw")
        self.assertIn("argon2id", str(ctx.exception))
        set_allow_expensive_kdf(True)
        with ArchiveReader(archive, password="pw") as reader:
            self.assertEqual(len(reader.entries), 1)

    def test_scrypt_above_read_cap_rejected(self) -> None:
        params = CryptoParams(
            kdf_id=2, param1=19, param2=8, param3=1  # log2N=19 > 18
        )
        with self.assertRaises(PxcKdfTooExpensive):
            check_kdf_cost(params)
        set_allow_expensive_kdf(True)
        check_kdf_cost(params)  # 不应再抛

    def test_unknown_flag_bits_rejected(self) -> None:
        params = CryptoParams(kdf_id=KDF_ARGON2ID, param1=8192, param2=1, param3=1)
        ctx = new_crypto_context("pw", os.urandom(16), params)
        raw = bytearray(pack_header(ctx))
        # 结构：magic(8) version(2) kdf(2) aead(2) flags(2) ...
        raw[14:16] = (0x02).to_bytes(2, "little")
        raw[124:128] = b"\x00" * 4
        raw[124:128] = _crc(bytes(raw)).to_bytes(4, "little")
        with self.assertRaises(PxcFormatError):
            unpack_header(bytes(raw))


if __name__ == "__main__":
    unittest.main()
