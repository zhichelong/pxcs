"""算法组件单元测试：MED、PALETTE、RUN、逐行滤波。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pxc_core.filters import (
    FILTER_AVG,
    FILTER_NONE,
    FILTER_PAETH,
    FILTER_SUB,
    FILTER_TYPES,
    FILTER_UP,
    choose_row_filters,
    filter_tile,
    unfilter_rows,
    unfilter_tile,
)
from pxc_core.palette import (
    PALETTE_PLAIN,
    PALETTE_RLE,
    decode_palette_payload,
    decode_run_payload,
    palette_payload,
    run_payload,
)
from pxc_core.predictor import decode_residuals, encode_residuals


class PredictorTest(unittest.TestCase):
    def test_med_roundtrip_shapes_and_dtypes(self) -> None:
        rng = np.random.default_rng(123)
        cases = [
            (np.uint8, (1, 1, 1)),
            (np.uint8, (1, 7, 3)),
            (np.uint8, (7, 1, 3)),
            (np.uint8, (19, 23, 3)),
            (np.uint8, (8, 8, 4)),
            (np.uint16, (13, 17, 1)),
            (np.uint16, (13, 17, 4)),
            (np.int32, (9, 11, 1)),
        ]
        for dtype, shape in cases:
            with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
                if np.dtype(dtype) == np.dtype(np.uint8):
                    tile = rng.integers(0, 256, shape, dtype=np.uint8)
                elif np.dtype(dtype) == np.dtype(np.uint16):
                    tile = rng.integers(0, 65536, shape, dtype=np.uint16)
                else:
                    tile = rng.integers(-100000, 100000, shape, dtype=np.int32)
                raw, _ = encode_residuals(tile)
                out = decode_residuals(raw, shape, np.dtype(dtype))
                np.testing.assert_array_equal(out, tile)

    def test_med_gradient(self) -> None:
        y, x = np.mgrid[0:64, 0:80]
        tile = np.stack([x * 3, y * 2, (x + y)], axis=-1).astype(np.uint8)
        raw, _ = encode_residuals(tile)
        out = decode_residuals(raw, tile.shape, np.dtype(np.uint8))
        np.testing.assert_array_equal(out, tile)


class PaletteRunTest(unittest.TestCase):
    def test_palette_plain_and_rle(self) -> None:
        rng = np.random.default_rng(5)
        for shape, c in [((17, 19, 1), 1), ((17, 19, 3), 3), ((10, 12, 4), 4)]:
            with self.subTest(shape=shape):
                palette = rng.integers(0, 256, (5, c), dtype=np.uint8)
                idx = rng.integers(0, 5, (shape[0] * shape[1],), dtype=np.uint8)
                tile = palette[idx].reshape(shape)
                for rle in (False, True):
                    raw, submode = palette_payload(tile, np.dtype(np.uint8), rle)
                    self.assertEqual(submode, PALETTE_RLE if rle else PALETTE_PLAIN)
                    out = decode_palette_payload(raw, shape, np.dtype(np.uint8), submode)
                    np.testing.assert_array_equal(out, tile)

    def test_palette_16bit(self) -> None:
        rng = np.random.default_rng(6)
        palette = rng.integers(0, 65536, (7, 1), dtype=np.uint16)
        idx = rng.integers(0, 7, (15, 9), dtype=np.uint8)
        tile = palette[idx].reshape(15, 9, 1)
        raw, submode = palette_payload(tile, np.dtype(np.uint16), False)
        out = decode_palette_payload(raw, tile.shape, np.dtype(np.uint16), submode)
        np.testing.assert_array_equal(out, tile)

    def test_run_roundtrip(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        tile[:5] = (1, 2, 3)
        tile[5:10] = (4, 5, 6)
        tile[10:] = (7, 8, 9)
        raw = run_payload(tile, np.dtype(np.uint8))
        out = decode_run_payload(raw, tile.shape, np.dtype(np.uint8))
        np.testing.assert_array_equal(out, tile)

    def test_run_16bit(self) -> None:
        tile = np.zeros((7, 11, 1), dtype=np.uint16)
        tile[2:5] = 60000
        tile[5:] = 3
        raw = run_payload(tile, np.dtype(np.uint16))
        out = decode_run_payload(raw, tile.shape, np.dtype(np.uint16))
        np.testing.assert_array_equal(out, tile)


class FilterTest(unittest.TestCase):
    def test_filter_all_types_uint8(self) -> None:
        rng = np.random.default_rng(9)
        tile = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)
        for ft in FILTER_TYPES:
            with self.subTest(ft=ft):
                f = filter_tile(tile, ft)
                out = unfilter_tile(f, ft)
                np.testing.assert_array_equal(out, tile)

    def test_filter_all_types_uint16(self) -> None:
        rng = np.random.default_rng(10)
        tile = rng.integers(0, 65536, (9, 11, 1), dtype=np.uint16)
        for ft in FILTER_TYPES:
            with self.subTest(ft=ft):
                f = filter_tile(tile, ft)
                out = unfilter_tile(f, ft)
                np.testing.assert_array_equal(out, tile)

    def test_row_filters(self) -> None:
        rng = np.random.default_rng(11)
        for dtype, maxv in [(np.uint8, 256), (np.uint16, 65536)]:
            with self.subTest(dtype=np.dtype(dtype).name):
                tile = rng.integers(0, maxv, (16, 21, 3), dtype=dtype)
                types, filtered = choose_row_filters(tile)
                out = unfilter_rows(filtered, types)
                np.testing.assert_array_equal(out, tile)

    def test_row_filters_single_row_and_column(self) -> None:
        for shape in [(1, 9, 3), (9, 1, 3), (1, 1, 1)]:
            with self.subTest(shape=shape):
                tile = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) % 251
                types, filtered = choose_row_filters(tile)
                out = unfilter_rows(filtered, types)
                np.testing.assert_array_equal(out, tile)

    def test_specific_filter_modes(self) -> None:
        tile = np.arange(5 * 7 * 2, dtype=np.uint8).reshape(5, 7, 2) % 199
        for ft in (FILTER_NONE, FILTER_SUB, FILTER_UP, FILTER_AVG, FILTER_PAETH):
            with self.subTest(ft=ft):
                np.testing.assert_array_equal(unfilter_tile(filter_tile(tile, ft), ft), tile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
