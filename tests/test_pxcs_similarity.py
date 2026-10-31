"""同分辨率大组按内容切分的单元测试。

背景：按分辨率分组时，一个组里可能混着几十个不同作品；全部对着组首图做
差分必然失败（实测 A10 的 347 张里有 ~35 个作品）。切分后
96.73% -> 25.32%。这里锁住聚类本身的行为。
"""
from __future__ import annotations

import unittest

import numpy as np

from pxcs_core.grouped_create import (
    _SIMILAR_THRESHOLD,
    _cluster_by_thumbnail,
)


def _thumb(value: int) -> np.ndarray:
    return np.full((32, 32), value, dtype=np.int16)


class SimilarityClusterTests(unittest.TestCase):
    def test_close_images_share_a_cluster(self) -> None:
        a = _thumb(0)
        b = _thumb(3)  # 距离 3，低于阈值
        c = _thumb(200)  # 距离 200，远
        clusters = _cluster_by_thumbnail([0, 1, 2], [a, b, c])
        self.assertEqual(clusters, [[0, 1], [2]])

    def test_unreadable_image_gets_its_own_cluster(self) -> None:
        a = _thumb(0)
        clusters = _cluster_by_thumbnail([0, 1, 2], [a, None, _thumb(100)])
        self.assertEqual(clusters, [[0], [1], [2]])

    def test_threshold_is_the_boundary(self) -> None:
        a = _thumb(0)
        near = _thumb(int(_SIMILAR_THRESHOLD))
        far = _thumb(int(_SIMILAR_THRESHOLD) + 1)
        self.assertEqual(_cluster_by_thumbnail([0, 1], [a, near]), [[0, 1]])
        self.assertEqual(_cluster_by_thumbnail([0, 1], [a, far]), [[0], [1]])

    def test_members_keep_natural_order(self) -> None:
        a = _thumb(0)
        same = ["x", "y", "z"]
        clusters = _cluster_by_thumbnail(same + ["w"], [a, _thumb(1), _thumb(2), _thumb(99)])
        self.assertEqual(clusters, [["x", "y", "z"], ["w"]])


if __name__ == "__main__":
    unittest.main()
