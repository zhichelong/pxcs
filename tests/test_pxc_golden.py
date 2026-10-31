"""PXC 字节级黄金向量回归测试。

见 `tests/fixtures/golden/README.md`。两级断言：

1. 解码器：`expected/*.pxc` 解出的像素必须与 `source/*.bin` 完全一致；
2. 编码器：在 zlib 版本一致时，重新编码必须与 `expected/*.pxc` 逐字节相同。

tile payload 的 zlib 输出跨版本不保证字节一致，因此第 2 级在版本不符时
自动降级为提示，而不是失败（CODEC.md §8.4）。
"""
from __future__ import annotations

import hashlib
import json
import unittest
import zlib
from pathlib import Path

import numpy as np

from pxc_core.codec import decode_image_data, encode_tiles
from pxc_core.config import TILE_MODE_NAMES
from pxc_core.container import build_pxc, parse_pxc_bytes
from pxc_core.image_io import ImageData, canonical_pixel_hash

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
NAMES = ("SOLID", "PALETTE_PLAIN", "PALETTE_RLE", "PRED", "RUN", "FILTER", "RAW")


def _load_manifest() -> dict:
    path = GOLDEN_DIR / "manifest.json"
    if not path.is_file():
        raise unittest.SkipTest(f"缺少黄金向量 {path}，请运行 python -m tools.gen_golden")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case(entry: dict, manifest: dict) -> tuple[ImageData, int]:
    """从 manifest 重建 ImageData 与 tile_size。"""
    dtype = np.dtype(entry["dtype"])
    raw = (GOLDEN_DIR / entry["source"]).read_bytes()
    height, width, channels = entry["height"], entry["width"], entry["channels"]
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["source_sha256"]:
        raise AssertionError(f"{entry['name']}: source 文件被改动（sha256 不符）")

    arr = np.frombuffer(raw, dtype=dtype.newbyteorder("<"))
    arr = arr.reshape(height, width, channels).astype(dtype)
    arr = np.ascontiguousarray(arr)

    palette = None
    if "palette_hex" in entry:
        palette = bytes.fromhex(entry["palette_hex"])
    transparency = None
    if "transparency_hex" in entry:
        transparency = bytes.fromhex(entry["transparency_hex"])

    image = ImageData(
        mode=entry["mode"],
        width=width,
        height=height,
        data=arr,
        palette=palette,
        transparency=transparency,
        original_filename=manifest["original_filename"],
        source_format=manifest["source_format"],
        source_ext=manifest["source_ext"],
        source_file_size=0,
        pixel_hash=canonical_pixel_hash(entry["mode"], width, height, arr),
        constant_alpha=entry.get("constant_alpha"),
    )
    return image, int(entry["tile_size"])


def _reencode(image: ImageData, tile_size: int) -> bytes:
    data_payload = encode_tiles(image, tile_size, threads=1)
    return build_pxc(image, data_payload, None, tile_size=tile_size)


class GoldenVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        cls.zlib_matches = cls.manifest.get("zlib_version") == zlib.ZLIB_RUNTIME_VERSION

    def test_manifest_coverage_is_complete(self) -> None:
        """7 种 tile 模式都必须有向量覆盖，否则解码器分支可能失守。"""
        coverage = self.manifest.get("coverage", {})
        missing = [name for name in NAMES if name not in coverage]
        self.assertEqual(missing, [], f"黄金向量缺少模式覆盖: {missing}")

    def test_decoder_matches_source(self) -> None:
        """每个 expected/*.pxc 解出的像素必须与 source 完全一致。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                image, _tile_size = _load_case(entry, self.manifest)
                expected = (GOLDEN_DIR / entry["expected"]).read_bytes()
                decoded = decode_image_data(parse_pxc_bytes(expected))
                self.assertEqual(decoded.mode, image.mode)
                self.assertEqual(decoded.width, image.width)
                self.assertEqual(decoded.height, image.height)
                np.testing.assert_array_equal(
                    decoded.data, image.data, err_msg=f"{entry['name']} 像素不一致"
                )
                self.assertEqual(
                    hashlib.sha256(decoded.data.tobytes()).hexdigest(),
                    hashlib.sha256(image.data.tobytes()).hexdigest(),
                )

    def test_encoder_is_byte_identical(self) -> None:
        """重新编码必须与 expected/*.pxc 逐字节相同（zlib 版本一致时）。"""
        if not self.zlib_matches:
            self.skipTest(
                f"zlib 版本不同（向量 {self.manifest.get('zlib_version')} / "
                f"本机 {zlib.ZLIB_RUNTIME_VERSION}），跳过字节级断言"
            )
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                image, tile_size = _load_case(entry, self.manifest)
                produced = _reencode(image, tile_size)
                expected = (GOLDEN_DIR / entry["expected"]).read_bytes()
                if produced != expected:
                    self.fail(
                        f"{entry['name']}: 编码结果与黄金向量不同 "
                        f"（期望 {len(expected)} 字节，实际 {len(produced)} 字节）；"
                        f"tile 模式分布={entry.get('tile_modes')}"
                    )

    def test_expected_size_matches_manifest(self) -> None:
        """向量文件本身没有被截断或替换。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                blob = (GOLDEN_DIR / entry["expected"]).read_bytes()
                self.assertEqual(len(blob), entry["expected_size"])
                self.assertEqual(
                    hashlib.sha256(blob).hexdigest(), entry["expected_sha256"]
                )

    def test_tile_mode_histogram_matches_manifest(self) -> None:
        """向量里实际出现的 tile 模式必须与 manifest 记录一致。"""
        import struct

        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                blob = (GOLDEN_DIR / entry["expected"]).read_bytes()
                chunk_dir_offset = struct.unpack_from("<Q", blob, 40)[0]
                chunk_count = struct.unpack_from("<I", blob, 48)[0]
                data_payload = b""
                for i in range(chunk_count):
                    base = chunk_dir_offset + 32 * i
                    if blob[base : base + 4] == b"DATA":
                        offset, length = struct.unpack_from("<QQ", blob, base + 8)
                        data_payload = blob[offset : offset + length]
                        break
                self.assertTrue(data_payload, f"{entry['name']} 缺少 DATA chunk")

                _, _, _, _, tile_count = struct.unpack_from("<IIIII", data_payload, 0)
                offset = 20
                hist: dict[str, int] = {}
                for _ in range(tile_count):
                    mode, submode = struct.unpack_from("<BB", data_payload, offset)
                    _, comp_len = struct.unpack_from("<II", data_payload, offset + 4)
                    if mode == 1:
                        key = "PALETTE_RLE" if submode == 1 else "PALETTE_PLAIN"
                    else:
                        key = TILE_MODE_NAMES.get(mode, f"UNKNOWN_{mode}")
                    hist[key] = hist.get(key, 0) + 1
                    offset += 20 + comp_len
                self.assertEqual(hist, entry["tile_modes"])


if __name__ == "__main__":
    unittest.main()
