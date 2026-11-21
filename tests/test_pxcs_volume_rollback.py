"""多卷覆盖提交的回滚测试。

多卷没有"单文件 rename 原子性"：中途失败会留下新 001 + 旧 002 的混合体。
提交前把旧卷改成 .bak，失败时撤掉已提交的新卷并还原 .bak。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.container import ArchiveReader, ArchiveWriter, compute_flags


class VolumeRollbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "src"
        self.src.mkdir()
        rng = np.random.default_rng(7)
        for i in range(4):
            arr = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(self.src / f"{i}.png")
        self.archive = self.root / "multi.pxcs"
        create_archive(
            self.archive, [str(self.src)],
            CreateOptions(quiet=True, progress=False, threads=2,
                          volume_size=8 * 1024, self_check=False),
        )
        with ArchiveReader(self.archive) as reader:
            self.original_count = len(reader.entries)
            self.volume_count = reader.volume_count
        self.assertGreaterEqual(self.volume_count, 3)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_failed_commit_rolls_back_old_volumes(self) -> None:
        writer = ArchiveWriter(
            self.archive, overwrite=True, volume_size=8 * 1024
        )
        for _ in range(12):
            writer.add_payload(os.urandom(20000))

        real_replace = os.replace
        state = {"commits": 0}

        def fake_replace(src, dst):
            src_s, dst_s = str(src), str(dst)
            if ".bak-" in dst_s:
                return real_replace(src, dst)
            state["commits"] += 1
            if state["commits"] == 2:
                raise OSError("injected commit failure")
            return real_replace(src, dst)

        with mock.patch("pxcs_core.volumes.os.replace", side_effect=fake_replace):
            with self.assertRaises(OSError):
                writer.finish([], compute_flags([]))

        # 旧归档必须仍然可读、条目数不变；临时/备份文件不能留在正式目录。
        with ArchiveReader(self.archive) as reader:
            self.assertEqual(len(reader.entries), self.original_count)
            self.assertEqual(reader.volume_count, self.volume_count)
        leftovers = [
            p.name for p in self.root.iterdir()
            if ".bak-" in p.name or ".tmp-" in p.name
        ]
        self.assertEqual(leftovers, [])

    def test_successful_overwrite_removes_backups(self) -> None:
        writer = ArchiveWriter(
            self.archive, overwrite=True, volume_size=8 * 1024
        )
        for _ in range(12):
            writer.add_payload(os.urandom(20000))
        final_path, _size = writer.finish([], compute_flags([]))
        self.assertEqual(final_path, self.archive)  # 分卷归档的“路径”是基名
        volumes = sorted(self.root.glob("multi.pxcs.*"))
        self.assertGreaterEqual(len(volumes), 2)
        self.assertTrue(volumes[0].exists())
        leftovers = [
            p.name for p in self.root.iterdir()
            if ".bak-" in p.name or ".tmp-" in p.name
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
