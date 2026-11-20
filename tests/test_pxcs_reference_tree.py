"""组内参考树与有损差分链的回归测试。

这里钉住两条容易静默退化的性质：

1. **小组也要走参考树**：低于相似度切分门槛（8 张）的组曾经完全不算缩略图，
   ``_plan_reference_tree`` 拿不到距离矩阵就直接退回星形，收益悄悄消失。
2. **有损差分的第 d 层必须用第 d-1 层重建后的像素当参考**：用原始像素会让每
   一层量化误差各自独立累加，``verify`` 的像素哈希检查会拒绝整个归档。

同时验证拓扑约束 ``ref < id``，它保证条目可以按成员序写出、读取端不需要
第二遍扫描。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import (
    CreateOptions,
    create_archive,
    list_entries,
    verify_archive,
)

SIZE = (96, 72)


def _write_tinted(path: Path, tint: int) -> None:
    """渐变噪声图。tint 越接近，两张图的缩略图距离越小。"""
    width, height = SIZE
    xs = np.arange(width, dtype=np.int32)[None, :]
    ys = np.arange(height, dtype=np.int32)[:, None]
    r = (tint + xs) % 256
    g = (tint + ys * 3) % 256
    b = (tint + xs * ys) % 256
    planes = np.broadcast_arrays(r, g, b)
    arr = np.stack(planes, axis=-1).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path, format="JPEG", quality=88)


class ReferenceTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _entries(self, name: str = "tree.pxcs") -> list[dict]:
        archive = self.tmp / name
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(quiet=True, progress=False, overwrite=True),
        )
        data = json.loads(list_entries(archive, as_json=True))
        return data["entries"]

    def _make_cluster_chain(self) -> None:
        """两簇色调，簇内相邻：自然的最近邻链应该是 0->1->2 和 3->4->5。"""
        for i, tint in enumerate((100, 103, 106, 140, 143, 146)):
            _write_tinted(self.src / f"a{i}_p0.jpg", tint)

    def test_cluster_chain_reaches_depth_two_or_more(self) -> None:
        self._make_cluster_chain()
        depths = [e["depth"] for e in self._entries()]
        self.assertGreaterEqual(
            max(depths), 2, f"参考树没有生效，depths={depths}"
        )

    def test_every_ref_points_backwards(self) -> None:
        self._make_cluster_chain()
        for entry in self._entries():
            ref = entry["ref"]
            if ref is not None:
                self.assertLess(ref, entry["id"], f"ref 没有指向更早的条目: {entry}")

    def test_lossy_reference_chain_verifies(self) -> None:
        """有损差分跨层时必须用重建像素当参考，否则 verify 会失败。"""
        self._make_cluster_chain()
        archive = self.tmp / "lossy.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(quiet=True, progress=False, overwrite=True),
        )
        stats = verify_archive(archive, threads=1)
        self.assertEqual(stats.failed, 0, f"校验失败: {stats.errors}")

    def test_group_below_split_threshold_still_gets_a_tree(self) -> None:
        """3 张图低于 8 张切分门槛：不切簇，但参考树仍要生效。"""
        for i, tint in enumerate((100, 103, 106)):
            _write_tinted(self.src / f"s{i}_p0.jpg", tint)
        depths = [e["depth"] for e in self._entries("small.pxcs")]
        self.assertGreaterEqual(
            max(depths), 2, f"小组退回了星形，depths={depths}"
        )

    def test_no_cluster_keeps_star_topology(self) -> None:
        """--no-cluster 的语义是“整组对着首图”，不应悄悄变成参考树。"""
        self._make_cluster_chain()
        archive = self.tmp / "star.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(
                quiet=True, progress=False, overwrite=True, cluster_similar=False
            ),
        )
        data = json.loads(list_entries(archive, as_json=True))
        depths = [e["depth"] for e in data["entries"]]
        self.assertEqual(max(depths), 1, f"--no-cluster 仍建了深树: depths={depths}")


if __name__ == "__main__":
    unittest.main()
