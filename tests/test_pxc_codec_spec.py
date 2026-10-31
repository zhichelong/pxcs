"""CODEC.md 的“可执行规格”测试。

本文件的解析器与 tile 解码器**完全依据 `CODEC.md` 的文字实现**，不导入
`pxc_core` 的任何解码逻辑（只借用黄金向量的样本数据作为对照）。

它的作用是防止文档与代码漂移：
- 若某天代码改了字节布局而文档没改，或反过来，这里会失败；
- 将来若有人只读文档写第二实现，本文件就是"读文档能得到什么"的证明。

因此：**不要为了通过测试而改这里去迁就实现**。要么改实现，要么改文档。
"""
from __future__ import annotations

import json
import struct
import unittest
import zlib
from pathlib import Path

import numpy as np

from pxc_core.config import (
    TILE_MODE_ORDER,
    TILE_MODE_SOLID,
    TILE_MODE_PALETTE,
    TILE_MODE_PRED,
    TILE_MODE_RUN,
    TILE_MODE_RAW,
    TILE_MODE_FILTER,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"

HEADER_SIZE = 128
CHUNK_ENTRY_SIZE = 32
FOOTER_SIZE = 16
DATA_HEADER_SIZE = 20
TILE_HEADER_SIZE = 20

# CODEC.md §4.6：滤波器类型
FILTER_NONE, FILTER_SUB, FILTER_UP, FILTER_AVG, FILTER_PAETH = 0, 1, 2, 3, 4

# CODEC.md §4：tile 模式数值
TILE_MODE_NAMES_DOC = {
    0: "SOLID",
    1: "PALETTE",
    2: "PRED",
    3: "RUN",
    4: "RAW",
    5: "FILTER",
}


# ---------------------------------------------------------------------------
# 完全独立的 PXC 解析器（依据 CODEC.md §2）
# ---------------------------------------------------------------------------
def parse_pxc_doc(blob: bytes) -> dict:
    """按 CODEC.md §2 的偏移表解析 PXC 文件。"""
    if len(blob) < HEADER_SIZE + FOOTER_SIZE:
        raise AssertionError("PXC 过短")
    assert blob[0:4] == b"PXC1", "magic 应为 PXC1"
    (
        major,
        minor,
        flags,
        header_size,
        header_crc,
        file_size,
        meta_offset,
        meta_len,
        dir_offset,
        chunk_count,
        width,
        height,
        mode_id,
        bit_depth,
        channels,
        color_model,
        alpha_kind,
        compression_id,
        tile_size,
        original_size,
        data_length,
        pixel_hash,
        reserved,
    ) = struct.unpack_from("<BBHIIQQQQIIIHHHHHHIQQ32sI", blob, 4)

    assert header_size == HEADER_SIZE, f"header_size 应为 128，实际 {header_size}"
    assert file_size == len(blob), f"file_size {file_size} != 实际 {len(blob)}"
    assert meta_offset == HEADER_SIZE, f"metadata_offset 应为 128，实际 {meta_offset}"
    assert dir_offset == meta_offset + meta_len, "chunk_dir_offset 关系不成立"
    assert reserved == 0, "header.reserved 应为 0"

    # header_crc：把 [12:16] 置零后对前 128 字节求 CRC32
    probe = bytearray(blob[:HEADER_SIZE])
    probe[12:16] = b"\x00\x00\x00\x00"
    assert zlib.crc32(bytes(probe)) & 0xFFFFFFFF == header_crc, "header_crc 不符"

    # Metadata 块（§2.2）
    meta_blob = blob[meta_offset : meta_offset + meta_len]
    (meta_count,) = struct.unpack_from("<I", meta_blob, 0)
    metadata: dict[int, tuple[int, bytes]] = {}
    cursor = 4
    previous_key = -1
    for _ in range(meta_count):
        key, mtype, mreserved, length = struct.unpack_from("<HBBI", meta_blob, cursor)
        cursor += 8
        assert mreserved == 0, "metadata.reserved 应为 0"
        assert key > previous_key, "metadata 必须按 key 严格升序"
        previous_key = key
        metadata[key] = (mtype, meta_blob[cursor : cursor + length])
        cursor += length
    assert cursor == len(meta_blob), "metadata 块长度与 count 不匹配"

    # Chunk 目录（§2.3）
    chunks: dict[bytes, tuple[int, int]] = {}
    for i in range(chunk_count):
        base = dir_offset + CHUNK_ENTRY_SIZE * i
        chunk_id = blob[base : base + 4]
        cflags, offset, length, ccrc, creserved = struct.unpack_from(
            "<IQQII", blob, base + 4
        )
        assert cflags == 0, "chunk.flags 应为 0"
        assert creserved == 0, "chunk.reserved 应为 0"
        payload = blob[offset : offset + length]
        assert zlib.crc32(payload) & 0xFFFFFFFF == ccrc, f"{chunk_id} chunk CRC 不符"
        chunks[chunk_id] = (offset, length)
    assert b"META" in chunks and b"DATA" in chunks, "必须同时存在 META 与 DATA"
    assert chunks[b"META"][0] == HEADER_SIZE, "META chunk 应紧接 header"
    first_payload = dir_offset + CHUNK_ENTRY_SIZE * chunk_count
    assert chunks[b"DATA"][0] >= first_payload, "DATA chunk 应与块目录不重叠"

    # Footer（§2.4）
    footer = blob[-FOOTER_SIZE:]
    assert footer[0:8] == b"PXCEND\0\0", "footer magic 不符"
    file_crc, footer_reserved = struct.unpack_from("<II", footer, 8)
    assert footer_reserved == 0, "footer.reserved 应为 0"
    assert zlib.crc32(blob[:-FOOTER_SIZE]) & 0xFFFFFFFF == file_crc, "file_crc 不符"

    data_offset, data_len = chunks[b"DATA"]
    assert data_len == data_length, "header.data_length 与 DATA chunk 长度不符"

    return {
        "major": major,
        "minor": minor,
        "flags": flags,
        "width": width,
        "height": height,
        "mode_id": mode_id,
        "bit_depth": bit_depth,
        "channels": channels,
        "color_model": color_model,
        "alpha_kind": alpha_kind,
        "compression_id": compression_id,
        "tile_size": tile_size,
        "original_size": original_size,
        "pixel_hash": pixel_hash,
        "metadata": metadata,
        "data_payload": blob[data_offset : data_offset + data_len],
        "total_size": len(blob),
    }


# ---------------------------------------------------------------------------
# 完全独立的 tile 解码器（依据 CODEC.md §3、§4）
# ---------------------------------------------------------------------------
def read_sample(raw: bytes, offset: int, width_bytes: int, signed: bool) -> tuple[int, int]:
    chunk = raw[offset : offset + width_bytes]
    assert len(chunk) == width_bytes, "样本读取越界"
    return int.from_bytes(chunk, "little", signed=signed), offset + width_bytes


def pack_sample(value: int, width_bytes: int, signed: bool) -> int:
    bits = width_bytes * 8
    if signed:
        value %= 1 << bits
    return value & ((1 << bits) - 1)


def storage_width(dtype: np.dtype) -> int:
    """CODEC.md §1.2 存储类型。"""
    if dtype == np.dtype(np.uint8):
        return 1
    if dtype == np.dtype(np.uint16):
        return 2
    if dtype == np.dtype(np.int32):
        return 4
    raise AssertionError(f"CODEC.md 未定义该样本类型的存储宽度: {dtype}")


def _decode_tile_doc(
    mode: int,
    submode: int,
    raw: bytes,
    shape: tuple[int, int, int],
    dtype: np.dtype,
) -> np.ndarray:
    """按 CODEC.md §4 解码单个 tile，返回 (H,W,C) 样本数组。"""
    height, width, channels = shape
    n = height * width
    sample_bytes = storage_width(dtype)
    signed = dtype == np.dtype(np.int32)
    out = [[0] * (width * channels) for _ in range(height)]

    if mode == TILE_MODE_SOLID:  # §4.2
        assert len(raw) == channels * sample_bytes, "SOLID payload 长度错误"
        color = []
        cursor = 0
        for _ in range(channels):
            value, cursor = read_sample(raw, cursor, sample_bytes, signed)
            color.append(value)
        for y in range(height):
            row = []
            for _ in range(width):
                row.extend(color)
            out[y] = row
        return _to_array(out, shape, dtype)

    if mode == TILE_MODE_PALETTE:  # §4.3
        (k,) = struct.unpack_from("<H", raw, 0)
        assert 1 <= k <= 256, f"调色板条目数越界: {k}"
        cursor = 2
        palette = []
        for _ in range(k):
            entry = []
            for _ in range(channels):
                value, cursor = read_sample(raw, cursor, sample_bytes, signed)
                entry.append(value)
            palette.append(entry)
        body = raw[cursor:]
        if submode == 0:
            assert len(body) == n, f"PLAIN 索引长度应为 {n}，实际 {len(body)}"
            indices = list(body)
        elif submode == 1:
            indices = []
            pos = 0
            while pos < len(body):
                assert pos + 5 <= len(body), "RLE 数据不完整"
                (run,) = struct.unpack_from("<I", body, pos)
                idx = body[pos + 4]
                pos += 5
                assert run > 0, "RLE 行程长度不得为 0"
                assert len(indices) + run <= n, "RLE 行程越界"
                indices.extend([idx] * run)
            assert len(indices) == n, "RLE 未铺满像素"
        else:
            raise AssertionError(f"未知 PALETTE 子模式 {submode}")
        assert max(indices) < k, "调色板索引越界"
        for y in range(height):
            row = []
            for x in range(width):
                row.extend(palette[indices[y * width + x]])
            out[y] = row
        return _to_array(out, shape, dtype)

    if mode == TILE_MODE_RUN:  # §4.5
        assert len(raw) > 0, "RUN payload 为空"
        flat = []
        pos = 0
        while pos < len(raw):
            assert pos + 4 <= len(raw), "RUN 头不完整"
            (run,) = struct.unpack_from("<I", raw, pos)
            pos += 4
            assert run > 0, "RUN 行程长度不得为 0"
            color = []
            for _ in range(channels):
                value, pos = read_sample(raw, pos, sample_bytes, signed)
                color.append(value)
            for _ in range(run):
                flat.extend(color)
        assert len(flat) == n * channels, "RUN 像素数不符"
        for y in range(height):
            out[y] = flat[y * width * channels : (y + 1) * width * channels]
        return _to_array(out, shape, dtype)

    if mode == TILE_MODE_RAW:  # §4.1
        expected = n * channels * sample_bytes
        assert len(raw) == expected, f"RAW 长度应为 {expected}，实际 {len(raw)}"
        cursor = 0
        for y in range(height):
            row = []
            for _ in range(n // height * channels):
                value, cursor = read_sample(raw, cursor, sample_bytes, signed)
                row.append(value)
            out[y] = row
        return _to_array(out, shape, dtype)

    if mode == TILE_MODE_FILTER:  # §4.6
        data_bytes = n * channels * sample_bytes
        assert len(raw) == height + data_bytes, "FILTER 长度不符"
        types = list(raw[:height])
        body = raw[height:]
        mod = 1 << (sample_bytes * 8)
        assert all(0 <= t <= 4 for t in types), "未知滤波类型"

        filtered = []
        cursor = 0
        for y in range(height):
            row = []
            for _ in range(width * channels):
                value, cursor = read_sample(body, cursor, sample_bytes, signed)
                row.append(value)
            filtered.append(row)

        decoded_rows: list[list[int]] = []
        for y in range(height):
            frow = filtered[y]
            prev = decoded_rows[y - 1] if y > 0 else None
            row: list[int] = []
            ft = types[y]
            for i, f in enumerate(frow):
                left = row[i - channels] if i >= channels else 0
                up = prev[i] if prev is not None else 0
                ul = prev[i - channels] if (prev is not None and i >= channels) else 0
                if ft == FILTER_NONE:
                    base = 0
                elif ft == FILTER_SUB:
                    base = left
                elif ft == FILTER_UP:
                    base = up
                elif ft == FILTER_AVG:
                    base = (left + up) // 2
                elif ft == FILTER_PAETH:
                    p = left + up - ul
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                    if pa <= pb and pa <= pc:
                        base = left
                    elif pb <= pc:
                        base = up
                    else:
                        base = ul
                else:
                    raise AssertionError(f"未知滤波类型 {ft}")
                row.append((f + base) % mod)
            decoded_rows.append(row)
        for y in range(height):
            out[y] = decoded_rows[y]
        return _to_array(out, shape, dtype)

    if mode == TILE_MODE_PRED:  # §4.4
        residual_bytes = 2 if sample_bytes == 1 else (4 if sample_bytes == 2 else 8)
        expected = n * channels * residual_bytes
        assert len(raw) == expected, f"PRED 长度应为 {expected}，实际 {len(raw)}"
        # zigzag 反解
        residuals: list[list[int]] = []
        cursor = 0
        for y in range(height):
            row = []
            for _ in range(width * channels):
                u, cursor = read_sample(raw, cursor, residual_bytes, False)
                value = (u >> 1) if u % 2 == 0 else -((u >> 1) + 1)
                row.append(value)
            residuals.append(row)

        # MED 重建（tile 内闭合）
        recon: list[list[int]] = []
        for y in range(height):
            row: list[int] = []
            prev = recon[y - 1] if y > 0 else None
            for i in range(width * channels):
                if y == 0 and i < channels:
                    pred = 0
                elif y == 0:
                    pred = row[i - channels]
                elif i < channels:
                    pred = prev[i]
                else:
                    a = row[i - channels]
                    b = prev[i]
                    c = prev[i - channels]
                    if c >= max(a, b):
                        pred = min(a, b)
                    elif c <= min(a, b):
                        pred = max(a, b)
                    else:
                        pred = a + b - c
                row.append(pred + residuals[y][i])
            recon.append(row)

        info = np.iinfo(dtype)
        for row in recon:
            for value in row:
                assert info.min <= value <= info.max, "PRED 重建结果越界"
        for y in range(height):
            out[y] = recon[y]
        return _to_array(out, shape, dtype)

    raise AssertionError(f"未知 tile 模式 {mode}")


def _to_array(rows: list[list[int]], shape: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    height, width, channels = shape
    arr = np.array(rows, dtype=np.int64).reshape(height, width, channels)
    return arr.astype(dtype)


def decode_pxc_doc(blob: bytes) -> tuple[np.ndarray, dict]:
    """按 CODEC.md 独立解码整个 PXC，返回 (样本数组, 解析信息)。"""
    info = parse_pxc_doc(blob)
    assert info["compression_id"] == 1, "本测试只覆盖 tile 流"
    payload = info["data_payload"]
    version, tile_size, tiles_x, tiles_y, tile_count = struct.unpack_from(
        "<IIIII", payload, 0
    )
    assert version == 1, f"DATA version 应为 1，实际 {version}"
    assert tile_count == tiles_x * tiles_y, "tile_count 与网格不符"
    width, height, channels = info["width"], info["height"], info["channels"]
    assert tiles_x == (width + tile_size - 1) // tile_size, "tiles_x 与尺寸不符"
    assert tiles_y == (height + tile_size - 1) // tile_size, "tiles_y 与尺寸不符"
    assert tile_size == info["tile_size"], "DATA 与 header 的 tile_size 不一致"

    dtype = _mode_dtype(info["mode_id"])
    canvas = np.zeros((height, width, channels), dtype=dtype)
    cursor = DATA_HEADER_SIZE
    modes_seen: dict[str, int] = {}

    for index in range(tile_count):
        ty, tx = divmod(index, tiles_x)
        x0, y0 = tx * tile_size, ty * tile_size
        x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)

        mode, submode, tflags, raw_len, comp_len, raw_crc, payload_crc = (
            struct.unpack_from("<BBHIIII", payload, cursor)
        )
        cursor += TILE_HEADER_SIZE
        stored = payload[cursor : cursor + comp_len]
        assert len(stored) == comp_len, "tile payload 越界"
        cursor += comp_len
        assert zlib.crc32(stored) & 0xFFFFFFFF == payload_crc, "tile payload CRC 不符"
        raw = zlib.decompress(stored) if (tflags & 1) else stored
        assert len(raw) == raw_len, "tile 解压长度不符"
        assert zlib.crc32(raw) & 0xFFFFFFFF == raw_crc, "tile 原始 CRC 不符"

        key = (
            f"PALETTE_{'RLE' if submode else 'PLAIN'}"
            if mode == TILE_MODE_PALETTE
            else TILE_MODE_NAMES_DOC.get(mode, f"UNKNOWN_{mode}")
        )
        modes_seen[key] = modes_seen.get(key, 0) + 1

        tile = _decode_tile_doc(
            mode, submode, raw, (y1 - y0, x1 - x0, channels), dtype
        )
        canvas[y0:y1, x0:x1, :] = tile

    assert cursor == len(payload), "DATA payload 存在多余数据"

    # §2.1：alpha_kind == 3 表示 alpha 平面被省略，解码时必须用常量补回
    if info["alpha_kind"] == 3:
        mtype, blob = info["metadata"][18]
        assert mtype == 3, "constant_alpha 必须是 u32"
        (alpha_value,) = struct.unpack("<I", blob)
        assert 0 <= alpha_value <= 255, "constant_alpha 越界"
        plane = np.full((height, width, 1), alpha_value, dtype=dtype)
        canvas = np.ascontiguousarray(np.concatenate([canvas, plane], axis=2))
        info = {**info, "channels": channels + 1}

    return canvas, {**info, "tiles_x": tiles_x, "tiles_y": tiles_y,
                    "tile_modes": modes_seen}


def _mode_dtype(mode_id: int) -> np.dtype:
    table = {
        1: np.uint8, 2: np.uint8, 3: np.uint8, 4: np.uint8, 5: np.uint8,
        6: np.uint8, 7: np.uint8, 8: np.int32, 9: np.uint16, 10: np.uint16,
        11: np.uint16, 12: np.float32,
    }
    return np.dtype(table[mode_id])


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
class DocSpecConformanceTests(unittest.TestCase):
    """逐条验证 CODEC.md 的描述与真实产物一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        manifest_path = GOLDEN_DIR / "manifest.json"
        if not manifest_path.is_file():
            raise unittest.SkipTest("缺少黄金向量")
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def _case_blob(self, entry: dict) -> bytes:
        return (GOLDEN_DIR / entry["expected"]).read_bytes()

    def _case_source(self, entry: dict) -> np.ndarray:
        dtype = np.dtype(entry["dtype"])
        raw = (GOLDEN_DIR / entry["source"]).read_bytes()
        arr = np.frombuffer(raw, dtype=dtype.newbyteorder("<"))
        return arr.reshape(
            entry["height"], entry["width"], entry["channels"]
        ).astype(dtype)

    def test_struct_sizes_match_doc(self) -> None:
        """CODEC.md 里写死的结构体尺寸必须与实现一致。"""
        self.assertEqual(struct.calcsize("<4sBBHIIQQQQIIIHHHHHHIQQ32sI"), HEADER_SIZE)
        self.assertEqual(struct.calcsize("<4sIQQII"), CHUNK_ENTRY_SIZE)
        self.assertEqual(struct.calcsize("<8sII"), FOOTER_SIZE)
        self.assertEqual(struct.calcsize("<IIIII"), DATA_HEADER_SIZE)
        self.assertEqual(struct.calcsize("<BBHIIII"), TILE_HEADER_SIZE)
        self.assertEqual(struct.calcsize("<HBBI"), 8)

    def test_mode_rank_table_matches_doc(self) -> None:
        """§3.4 的平局排名：FILTER 排在 RAW 之前（与模式数值不同）。"""
        self.assertEqual(
            list(TILE_MODE_ORDER),
            [TILE_MODE_SOLID, TILE_MODE_PALETTE, TILE_MODE_PRED,
             TILE_MODE_RUN, TILE_MODE_FILTER, TILE_MODE_RAW],
        )
        self.assertEqual(TILE_MODE_RAW, 4, "§3.4：RAW 的模式数值是 4")
        self.assertEqual(TILE_MODE_FILTER, 5, "§3.4：FILTER 的模式数值是 5")

    def test_independent_parser_decodes_all_vectors(self) -> None:
        """纯按文档实现的解析器必须能解出全部黄金向量的像素。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                blob = self._case_blob(entry)
                canvas, info = decode_pxc_doc(blob)
                expected = self._case_source(entry)
                self.assertEqual(info["width"], entry["width"])
                self.assertEqual(info["height"], entry["height"])
                self.assertEqual(info["mode_id"], _mode_id(entry["mode"]))
                np.testing.assert_array_equal(
                    canvas, expected, err_msg=f"{entry['name']} 文档解码器像素不符"
                )

    def test_tile_mode_histogram_matches_manifest(self) -> None:
        """独立解析器统计出的模式分布必须与生成时记录的一致。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                _, info = decode_pxc_doc(self._case_blob(entry))
                self.assertEqual(info["tile_modes"], entry["tile_modes"])

    def test_header_flags_and_fields(self) -> None:
        """§2.1 的字段语义：常量 alpha 会减少 header.channels。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                info = parse_pxc_doc(self._case_blob(entry))
                if "constant_alpha" in entry:
                    self.assertEqual(info["alpha_kind"], 3)
                    self.assertEqual(
                        info["channels"], entry["header_channels"],
                        "常量 alpha 时 header.channels 应比 mode 少 1",
                    )
                    self.assertIn(18, info["metadata"], "缺少 constant_alpha 元数据")
                    mtype, payload = info["metadata"][18]
                    self.assertEqual(mtype, 3, "constant_alpha 必须是 u32")
                    self.assertEqual(
                        struct.unpack("<I", payload)[0], entry["constant_alpha"]
                    )
                else:
                    self.assertEqual(
                        info["channels"], entry["channels"],
                        "header.channels 应等于 mode 通道数",
                    )

    def test_metadata_required_keys(self) -> None:
        """§2.2：这 6 个键必须总是存在。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                info = parse_pxc_doc(self._case_blob(entry))
                for key in (1, 2, 3, 4, 11, 12, 13):
                    self.assertIn(key, info["metadata"], f"缺少元数据键 {key}")
                mtype, payload = info["metadata"][4]
                self.assertEqual(mtype, 2, "mode_name 必须是 utf8")
                self.assertEqual(payload.decode("utf-8"), entry["mode"])

    def test_pixel_hash_definition(self) -> None:
        """§7.1：SHA256(b"PXC-PIXELS\\0" + mode + LE32 w + LE32 h + samples)。"""
        import hashlib

        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                blob = self._case_blob(entry)
                info = parse_pxc_doc(blob)
                samples = self._case_source(entry)
                digest = hashlib.sha256()
                digest.update(b"PXC-PIXELS\x00")
                digest.update(entry["mode"].encode("utf-8"))
                digest.update(struct.pack("<II", entry["width"], entry["height"]))
                digest.update(np.ascontiguousarray(samples).tobytes())
                self.assertEqual(
                    digest.digest(), info["pixel_hash"],
                    "header.pixel_hash 与 §7.1 的定义不符",
                )

    def test_palette_ordering_rule(self) -> None:
        """§4.3：单通道数值升序；多通道按通道顺序的字节序。"""
        for entry in self.manifest["cases"]:
            if "PALETTE_PLAIN" not in entry["tile_modes"] and \
               "PALETTE_RLE" not in entry["tile_modes"]:
                continue
            with self.subTest(case=entry["name"]):
                info = parse_pxc_doc(self._case_blob(entry))
                payload = info["data_payload"]
                _, _, _, _, tile_count = struct.unpack_from("<IIIII", payload, 0)
                cursor = DATA_HEADER_SIZE
                dtype = _mode_dtype(info["mode_id"])
                sample_bytes = storage_width(dtype)
                signed = dtype == np.dtype(np.int32)
                channels = info["channels"]
                checked = 0
                for _ in range(tile_count):
                    mode, submode, tflags, _, comp_len, _, _ = struct.unpack_from(
                        "<BBHIIII", payload, cursor
                    )
                    cursor += TILE_HEADER_SIZE
                    stored = payload[cursor : cursor + comp_len]
                    cursor += comp_len
                    if mode != TILE_MODE_PALETTE:
                        continue
                    raw = zlib.decompress(stored) if (tflags & 1) else stored
                    (k,) = struct.unpack_from("<H", raw, 0)
                    entries = []
                    pos = 2
                    for _ in range(k):
                        pixel = []
                        for _ in range(channels):
                            value, pos = read_sample(raw, pos, sample_bytes, signed)
                            pixel.append(value)
                        entries.append(tuple(pixel))
                    self.assertEqual(
                        entries, sorted(entries),
                        f"{entry['name']}: 调色板必须升序排列",
                    )
                    checked += 1
                self.assertGreater(checked, 0, "该向量应当包含 PALETTE tile")

    def test_zlib_flag_semantics(self) -> None:
        """§3.3：flags bit0 表示已压缩，且压缩后必须严格更小。"""
        for entry in self.manifest["cases"]:
            with self.subTest(case=entry["name"]):
                info = parse_pxc_doc(self._case_blob(entry))
                payload = info["data_payload"]
                _, _, _, _, tile_count = struct.unpack_from("<IIIII", payload, 0)
                cursor = DATA_HEADER_SIZE
                for _ in range(tile_count):
                    mode, submode, tflags, raw_len, comp_len, _, _ = (
                        struct.unpack_from("<BBHIIII", payload, cursor)
                    )
                    cursor += TILE_HEADER_SIZE
                    stored = payload[cursor : cursor + comp_len]
                    cursor += comp_len
                    if tflags & 1:
                        self.assertLess(
                            comp_len, raw_len,
                            "仅当压缩后严格更小时才允许置 zlib 标志",
                        )
                        self.assertEqual(len(zlib.decompress(stored)), raw_len)
                    else:
                        self.assertEqual(comp_len, raw_len, "未压缩时长度必须相等")


def _mode_id(mode: str) -> int:
    table = {
        "1": 1, "L": 2, "LA": 3, "P": 4, "RGB": 5, "RGBA": 6,
        "CMYK": 7, "I": 8, "I;16": 9, "I;16L": 10, "I;16B": 11, "F": 12,
    }
    return table[mode]


if __name__ == "__main__":
    unittest.main()
