"""--preset 的映射与端到端行为。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pxcs import _apply_create_preset
from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.container import ArchiveReader


def _args(**kw):
    base = dict(preset="custom", quant_list=None, tree_depth=0)
    base.update(kw)
    return SimpleNamespace(**base)


class PresetMappingTest(unittest.TestCase):
    def test_lossless_disables_quantization(self) -> None:
        opts = _apply_create_preset(CreateOptions(), _args(preset="lossless"))
        self.assertEqual(opts.algorithm, "pixel")
        self.assertEqual(opts.delta_quant, 0)
        self.assertEqual(opts.delta_quant_max, 0)
        self.assertEqual(opts.tree_depth, 8)

    def test_max_uses_wider_quant_list(self) -> None:
        opts = _apply_create_preset(CreateOptions(), _args(preset="max"))
        self.assertEqual(opts.quant_list, (8, 12, 16, 24))
        self.assertEqual(opts.zlib_level, 9)
        self.assertEqual(opts.webp_method, 6)

    def test_explicit_flags_win(self) -> None:
        # CLI 会先把显式参数放进 opts，再应用预设。
        opts = _apply_create_preset(
            CreateOptions(quant_list=(0, 24), tree_depth=3),
            _args(preset="max", quant_list=(0, 24), tree_depth=3),
        )
        self.assertEqual(opts.quant_list, (0, 24))
        self.assertEqual(opts.tree_depth, 3)

    def test_custom_changes_nothing(self) -> None:
        base = CreateOptions(delta_quant=20)
        opts = _apply_create_preset(base, _args(preset="custom"))
        self.assertEqual(opts.delta_quant, 20)
        self.assertEqual(opts.algorithm, "auto")


class PresetEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        src = self.root / "src"
        src.mkdir()
        base = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        other = base.copy()
        other[:16, :16] = (other[:16, :16] // 2).astype(np.uint8)
        Image.fromarray(base).save(src / "a.jpg", quality=92)
        Image.fromarray(other).save(src / "b.jpg", quality=92)
        self.src = src

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lossless_preset_avoids_lossy_path(self) -> None:
        opts = _apply_create_preset(CreateOptions(quiet=True, progress=False, threads=2),
                                    _args(preset="lossless"))
        out = self.root / "l.pxcs"
        stats = create_archive(out, [str(self.src)], opts)
        self.assertEqual(stats.lossy_deltas, 0)
        self.assertEqual(stats.raw_images, 0)
        with ArchiveReader(out) as reader:
            self.assertTrue(all(e.codec != "raw" for e in reader.entries))


if __name__ == "__main__":
    unittest.main()
