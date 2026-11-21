"""P1 计划/预算的回归。

- 分块参考树规划必须与旧的四维矩阵版本给出**完全相同**的 parents；
- ``--memory-limit`` 显式小值不能再被静默抬到 512 MiB；
- 峰值 RSS 要能真读到值（不是永远 0）。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.grouped_create import _plan_reference_tree
from pxcs_core.limits import (
    peak_rss_bytes,
    resolve_memory_budget,
)


def _old_plan(thumbs, key_index, max_depth=6, max_children=8):
    """1.4.1 的实现（(n,n,32,32) 四维差矩阵），仅用于对照。"""
    n = len(thumbs)
    parents = [-1] * n
    grid = np.stack(thumbs).astype(np.int16)
    dist = np.abs(grid[:, None, :, :] - grid[None, :, :, :]).mean(axis=(2, 3))
    depths = [0] * n
    children = [0] * n
    for i in range(key_index + 1, n):
        best_parent = key_index
        best_dist = float(dist[i, key_index])
        for j in range(key_index, i):
            if depths[j] >= max_depth or children[j] >= max_children:
                continue
            d = float(dist[i, j])
            if d < best_dist:
                best_parent, best_dist = j, d
        parents[i] = best_parent
        depths[i] = depths[best_parent] + 1
        children[best_parent] += 1
    return parents


class ChunkedTreePlannerTest(unittest.TestCase):
    def test_matches_old_formula(self) -> None:
        rng = np.random.default_rng(5)
        for n in (4, 40, 180):
            for key_index in (0, n // 2):
                thumbs = [
                    rng.integers(0, 256, (32, 32), dtype=np.uint8)
                    for _ in range(n)
                ]
                self.assertEqual(
                    _plan_reference_tree(thumbs, key_index),
                    _old_plan(thumbs, key_index),
                    f"n={n} key={key_index} 规划结果与旧实现不同",
                )

    def test_depth_and_children_limits_respected(self) -> None:
        rng = np.random.default_rng(9)
        thumbs = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(120)]
        parents = _plan_reference_tree(thumbs, 0, max_depth=3, max_children=2)
        depths = [0] * len(parents)
        children = [0] * len(parents)
        for i in range(1, len(parents)):
            self.assertLess(parents[i], i)
            depths[i] = depths[parents[i]] + 1
            children[parents[i]] += 1
        self.assertLessEqual(max(depths), 3)
        # key 是兜底锚点，允许超过 max_children；其余父节点必须受限。
        self.assertLessEqual(max(children[1:]), 2)


class MemoryBudgetTest(unittest.TestCase):
    def test_explicit_small_budget_is_honored(self) -> None:
        self.assertEqual(resolve_memory_budget(64), 64 * 1024 * 1024)
        self.assertEqual(resolve_memory_budget(1), 16 * 1024 * 1024)  # 下限 16 MiB

    def test_auto_budget_has_floor(self) -> None:
        self.assertGreaterEqual(resolve_memory_budget(0), 512 * 1024 * 1024)

    def test_peak_rss_is_reported(self) -> None:
        self.assertGreater(peak_rss_bytes(), 0)

    def test_create_with_tiny_budget_succeeds_and_reports_rss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            Image.fromarray(
                np.random.default_rng(1).integers(0, 256, (96, 96, 3), dtype=np.uint8)
            ).save(src / "a.png")
            Image.fromarray(
                np.random.default_rng(2).integers(0, 256, (96, 96, 3), dtype=np.uint8)
            ).save(src / "b.png")
            out = root / "tiny.pxcs"
            stats = create_archive(
                out,
                [str(src)],
                CreateOptions(
                    quiet=True, progress=False, threads=2,
                    memory_limit_mb=32, self_check=True,
                ),
            )
            self.assertEqual(stats.files_total, 2)
            self.assertGreater(stats.peak_rss_bytes, 0)


if __name__ == "__main__":
    unittest.main()
