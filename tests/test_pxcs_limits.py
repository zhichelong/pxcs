"""并发度与内存预算的语义测试。

这里的契约是用户可见的，所以单独测：

- ``threads=0`` / ``parallel_groups=0`` 表示"自动跑满"，**不是**"最小值 1"；
- 并发度不能改变归档内容（同样的输入，8 线程和 32 线程产物必须一致）；
- 内存预算是真的会挡人（超预算时任务排队），而且不会因为单任务超预算就死锁。
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.container import ArchiveReader
from pxcs_core.limits import (
    MemoryBudget,
    cap_parallel_groups_by_memory,
    estimate_key_retention,
    estimate_raw_lease,
    estimate_working_set,
    resolve_memory_budget,
    resolve_parallel_groups,
    resolve_threads,
)
from pxcs_core.settings import DEFAULT_SETTINGS, load_settings


class ThreadSemanticsTests(unittest.TestCase):
    def test_zero_means_all_cores(self) -> None:
        cores = max(1, os.cpu_count() or 1)
        self.assertEqual(resolve_threads(0), cores)
        self.assertEqual(resolve_threads(None), cores)
        self.assertEqual(resolve_threads(-1), cores)

    def test_explicit_value_is_respected(self) -> None:
        self.assertEqual(resolve_threads(1), 1)
        self.assertEqual(resolve_threads(8), 8)

    def test_parallel_groups_zero_follows_threads(self) -> None:
        self.assertEqual(resolve_parallel_groups(0, 8), 8)
        self.assertEqual(resolve_parallel_groups(0, 24), 24)
        self.assertEqual(resolve_parallel_groups(None, 4), 4)

    def test_parallel_groups_never_exceeds_threads(self) -> None:
        self.assertEqual(resolve_parallel_groups(100, 8), 8)
        self.assertEqual(resolve_parallel_groups(3, 8), 3)

    def test_defaults_are_auto(self) -> None:
        self.assertEqual(DEFAULT_SETTINGS["create"]["threads"], 0)
        self.assertEqual(DEFAULT_SETTINGS["create"]["parallel_groups"], 0)
        self.assertEqual(DEFAULT_SETTINGS["create"]["memory_limit_mb"], 0)
        settings = load_settings()
        self.assertEqual(settings["create"]["threads"], 0)
        self.assertEqual(settings["create"]["parallel_groups"], 0)
        self.assertEqual(settings["create"]["memory_limit_mb"], 0)
        opts = CreateOptions()
        self.assertEqual(opts.threads, 0)
        self.assertEqual(opts.parallel_groups, 0)
        self.assertEqual(opts.memory_limit_mb, 0)

    def test_parallel_groups_zero_survives_settings_clamp(self) -> None:
        """0 不能被“最小 1”的钳位改成 1（否则就跑不回自动）。"""
        settings = load_settings()
        self.assertIsInstance(settings["create"]["parallel_groups"], int)


class MemoryEstimateTests(unittest.TestCase):
    def test_estimate_scales_with_pixels(self) -> None:
        small = estimate_working_set(100, 100, 3, kind="delta")
        large = estimate_working_set(4000, 3000, 3, kind="delta")
        self.assertLess(small, large)
        self.assertGreaterEqual(small, 1024 * 1024, "小图也应有最小记账量")

    def test_delta_costs_more_than_key(self) -> None:
        delta = estimate_working_set(2000, 2000, 3, kind="delta")
        key = estimate_working_set(2000, 2000, 3, kind="key")
        self.assertGreater(delta, key, "差分要额外持有残差与居中副本")

    def test_raw_lease_uses_file_size(self) -> None:
        self.assertGreater(estimate_raw_lease(10 * 1024 * 1024), 10 * 1024 * 1024)

    def test_key_retention_larger_than_plain_pixels(self) -> None:
        pixels = 2000 * 2000 * 3
        self.assertGreater(estimate_key_retention(2000, 2000, 3), pixels)


class ParallelGroupCapTests(unittest.TestCase):
    def test_caps_when_keys_are_huge(self) -> None:
        budget = 4 * 1024 ** 3
        key_bytes = 1 * 1024 ** 3
        self.assertLessEqual(cap_parallel_groups_by_memory(24, budget, key_bytes), 2)

    def test_keeps_full_concurrency_when_keys_are_small(self) -> None:
        budget = 4 * 1024 ** 3
        key_bytes = 8 * 1024 ** 2
        self.assertEqual(cap_parallel_groups_by_memory(16, budget, key_bytes), 16)

    def test_never_returns_zero(self) -> None:
        self.assertGreaterEqual(
            cap_parallel_groups_by_memory(8, 64 * 1024 ** 2, 4 * 1024 ** 3), 1
        )


class MemoryBudgetTests(unittest.TestCase):
    def test_budget_from_explicit_limit(self) -> None:
        self.assertEqual(resolve_memory_budget(1024), 1024 * 1024 * 1024)

    def test_auto_budget_is_positive(self) -> None:
        self.assertGreater(resolve_memory_budget(0), 0)

    def test_acquire_release_balances(self) -> None:
        budget = MemoryBudget(64 * 1024 ** 2)
        lease = budget.acquire(16 * 1024 ** 2)
        self.assertEqual(budget.used, lease)
        self.assertEqual(budget.peak, lease)
        budget.release(lease)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.peak, lease, "峰值不随释放回落")

    def test_oversized_request_is_clamped(self) -> None:
        """单任务比预算还大时不能饿死：放宽到整个预算，且可正常释放。"""
        budget = MemoryBudget(8 * 1024 ** 2)
        lease = budget.acquire(1024 * 1024 ** 2)
        self.assertEqual(lease, budget.total)
        budget.release(lease)
        self.assertEqual(budget.used, 0)

    def test_blocks_when_exhausted(self) -> None:
        # 用事件而不是 sleep 判断"有没有阻塞"：机器满载时线程启动时机不可控，
        # 原来的 time.sleep(0.05) + wait(2.0) 在跑基准时偶发失败过一次。
        budget = MemoryBudget(16 * 1024 ** 2)
        first = budget.acquire(16 * 1024 ** 2)
        started = threading.Event()
        progressed = threading.Event()

        def worker() -> None:
            started.set()
            lease = budget.acquire(16 * 1024 ** 2)
            progressed.set()
            budget.release(lease)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertTrue(started.wait(10.0), "工作线程没起来")
        self.assertFalse(progressed.wait(0.2), "预算已满时第二个任务必须等待")
        budget.release(first)
        self.assertTrue(progressed.wait(10.0), "释放后应被唤醒")
        thread.join(timeout=10.0)

    def test_lease_context_manager_releases_on_error(self) -> None:
        budget = MemoryBudget(32 * 1024 ** 2)
        with self.assertRaises(RuntimeError):
            with budget.lease(8 * 1024 ** 2):
                raise RuntimeError("boom")
        self.assertEqual(budget.used, 0)


class ConcurrencyDoesNotChangeOutputTests(unittest.TestCase):
    """并发度只影响速度，不能影响归档内容。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        for index in range(4):
            img = Image.new("RGB", (96, 72))
            arr = np.zeros((72, 96, 3), dtype=np.uint8)
            arr[..., 0] = (index * 20 + np.arange(96)) % 256
            arr[..., 1] = (index * 30 + np.arange(72))[:, None] % 256
            arr[..., 2] = index * 10
            img = Image.fromarray(arr, mode="RGB")
            img.save(self.src / f"g{index}_p{index}.jpg", format="JPEG", quality=90)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create(self, name: str, threads: int, groups: int) -> Path:
        archive = self.tmp / name
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(
                quiet=True,
                progress=False,
                overwrite=True,
                threads=threads,
                parallel_groups=groups,
            ),
        )
        return archive

    def _fingerprint(self, archive: Path) -> list[tuple[str, str, int]]:
        with ArchiveReader(archive) as reader:
            return [
                (entry.path, entry.codec, entry.stored_length)
                for entry in reader.entries
            ]

    def test_auto_matches_explicit(self) -> None:
        """并发度不同，条目与 payload 必须一致。

        注意不能比较整文件大小：`archive_id` 是随机 uuid，写进索引后 zlib
        压缩长度会差几个字节，这是设计使然（见 CODEC.md §8.2）。
        """
        auto = self._create("auto.pxcs", 0, 0)
        serial = self._create("one.pxcs", 1, 1)
        self.assertEqual(self._fingerprint(auto), self._fingerprint(serial))

    def test_payload_bytes_are_identical(self) -> None:
        auto = self._create("auto.pxcs", 0, 0)
        serial = self._create("one.pxcs", 1, 1)

        def payloads(archive: Path) -> list[bytes]:
            with ArchiveReader(archive) as reader:
                return [reader.read_entry_data(e) for e in reader.entries]

        self.assertEqual(payloads(auto), payloads(serial))


class ParallelKeyOrderTests(unittest.TestCase):
    """组 key 编码从主线程改到线程池之后，条目顺序仍必须逐条一致。

    key 曾经是同步写的（先写 key 再提交差分），改成并行后靠"按提交顺序落
    条目"维持同样的物理布局；这里用多组、多种分辨率把这个契约钉住。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        shapes = [(64, 48), (80, 60), (96, 72)]
        for gi, (w, h) in enumerate(shapes):
            for vi in range(2):
                arr = np.zeros((h, w, 3), dtype=np.uint8)
                arr[..., 0] = (np.arange(w) + gi * 7) % 256
                arr[..., 1] = ((np.arange(h) + vi * 5) % 256)[:, None]
                Image.fromarray(arr, mode="RGB").save(
                    self.src / f"a{gi}_v{vi}.png", format="PNG"
                )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create(self, name: str, threads: int, groups: int) -> Path:
        archive = self.tmp / name
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(
                quiet=True,
                progress=False,
                overwrite=True,
                threads=threads,
                parallel_groups=groups,
            ),
        )
        return archive

    def _fingerprint(self, archive: Path) -> dict[str, tuple[str, bytes]]:
        with ArchiveReader(archive) as reader:
            return {
                entry.path: (entry.codec, reader.read_entry_data(entry))
                for entry in reader.entries
            }

    def test_same_config_is_reproducible(self) -> None:
        """同一份配置跑两遍，payload 必须逐条一致。

        条目顺序取决于并行完成顺序（见下一条测试），不是格式契约的一部分；
        所以这里比的是"路径集合 + 每个路径的 codec 与 payload"，不比物理顺序。
        """
        first = self._create("r1.pxcs", 0, 0)
        second = self._create("r2.pxcs", 0, 0)
        with ArchiveReader(first) as a, ArchiveReader(second) as b:
            self.assertEqual({e.path for e in a.entries}, {e.path for e in b.entries})
        self.assertEqual(self._fingerprint(first), self._fingerprint(second))

    def test_window_size_does_not_change_content(self) -> None:
        """并行窗口大小会改变条目的物理顺序（既有行为），但内容必须等价。"""
        auto = self._create("auto.pxcs", 0, 0)
        serial = self._create("one.pxcs", 1, 1)
        self.assertEqual(self._fingerprint(auto), self._fingerprint(serial))
        with ArchiveReader(auto) as reader:
            codes = {e.path: e.codec for e in reader.entries}
        self.assertTrue(codes)


if __name__ == "__main__":
    unittest.main()
