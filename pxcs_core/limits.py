"""CPU 与内存的并发上限计算。

两件事在这里集中处理，避免散落在各处的 `min(8, cpu)` 之类魔法数字：

1. **线程数语义**：``threads=0`` 表示"用满所有核心"，``parallel_groups=0``
   表示"跟线程数一样多"。0 不再是"最小值 1"。
2. **内存预算**：并发跑满 CPU 时，单张图的解码缓冲、残差缓冲会同时放大
   N 倍。这里按"预估工作集"给出一个预算，超了就排队等，而不是让进程
   被系统 OOM 杀掉或让 Pillow 报 MemoryError。

工作集估算刻意保守（像素数 × 系数），因为它只需要保证"不会明显超"，
不需要精确——多留一点余量比精确但危险更划算。
"""
from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

# 单张图在编码过程中的峰值倍数（相对"解码后的原始像素字节数"）。
#   delta：解码图 + 残差 + 居中/量化副本 + tobytes 副本 + zlib 输出
#   key  ：解码图 + tile 输出 + 源字节
# 数值来自 pxcs_core.codec 的实际分配路径，取偏大的估计。
_FACTORS = {"delta": 5, "key": 3, "raw": 2, "file": 2}

# 单任务至少按这么多字节记账，避免大量小图把计数打到 0 而失去保护意义。
_MIN_LEASE = 1 * 1024 * 1024

# 自动预算占"可用物理内存"的比例。
_AUTO_BUDGET_RATIO = 0.6
_MIN_BUDGET = 512 * 1024 * 1024
# 键图（每个活跃组的参考图）最多占用预算的这个比例，剩下留给 worker。
# 这样总内存上界 ≈ 预算 + 0.5×预算 = 1.5×预算，而预算本身是可用内存的 60%，
# 所以最坏情况约占可用内存的 90%，仍在安全区。
_KEY_BUDGET_RATIO = 0.5


def resolve_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 4))


def resolve_threads(value: Optional[int]) -> int:
    """``0`` / ``None`` → 全部核心；其余按用户给的值（至少 1）。"""
    try:
        threads = int(value or 0)
    except (TypeError, ValueError):
        threads = 0
    if threads <= 0:
        return resolve_cpu_count()
    return max(1, threads)


def resolve_parallel_groups(value: Optional[int], threads: int) -> int:
    """``0`` / ``None`` → 跟线程数一致；不超过线程数。"""
    try:
        groups = int(value or 0)
    except (TypeError, ValueError):
        groups = 0
    threads = max(1, int(threads))
    if groups <= 0:
        return threads
    return max(1, min(groups, threads))


def available_memory_bytes() -> int:
    """可用物理内存（字节）；拿不到时返回保守估计。

    Windows 走 ``GlobalMemoryStatusEx``，不引入 psutil 依赖。
    其它平台用 ``os.sysconf``，再不行给 2 GiB。
    """
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (OSError, AttributeError, ValueError):
            pass
    else:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
        except (OSError, ValueError, AttributeError):
            pass
    return 2 * 1024 ** 3


def resolve_memory_budget(limit_mb: Optional[int]) -> int:
    """``0`` → 可用内存的 60%（不低于 512 MiB）；>0 用指定值。"""
    try:
        value = int(limit_mb or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return max(_MIN_BUDGET, value * 1024 * 1024)
    return max(_MIN_BUDGET, int(available_memory_bytes() * _AUTO_BUDGET_RATIO))


def estimate_working_set(
    width: int,
    height: int,
    channels: int,
    sample_bytes: int = 1,
    *,
    kind: str = "delta",
) -> int:
    """估算单个任务的工作集字节数。"""
    pixels = max(0, int(width)) * max(0, int(height)) * max(1, int(channels))
    factor = _FACTORS.get(kind, 3)
    return max(_MIN_LEASE, pixels * max(1, int(sample_bytes)) * factor)


def estimate_raw_lease(file_size: int) -> int:
    """raw 直通：读一次 + 写一次。"""
    return max(_MIN_LEASE, int(file_size) * 2)


def estimate_key_retention(width: int, height: int, channels: int, sample_bytes: int = 1) -> int:
    """参考图常驻内存的量（每个活跃组一份）。"""
    pixels = max(0, int(width)) * max(0, int(height)) * max(1, int(channels))
    return max(_MIN_LEASE, pixels * max(1, int(sample_bytes)) * 4)


def cap_parallel_groups_by_memory(
    parallel_groups: int, budget_bytes: int, key_bytes: int
) -> int:
    """按"键图最多占预算 40%"折算并行组数。"""
    groups = max(1, int(parallel_groups))
    key_bytes = max(_MIN_LEASE, int(key_bytes))
    allowed = max(
        1, int(max(1024 * 1024, budget_bytes * _KEY_BUDGET_RATIO) // key_bytes)
    )
    return max(1, min(groups, allowed))


class MemoryBudget:
    """按字节记账的并发闸门。

    worker 在开始解码前 ``acquire``，结束（``finally``）``release``。
    预算不足时阻塞等待，而不是让系统 OOM。注意：只给**线程池里的任务**用，
    主线程的"参考图常驻"改用 ``cap_parallel_groups_by_memory`` 预留额度，
    否则主线程等待会导致完成状态无法回收，形成死锁。
    """

    def __init__(self, total_bytes: int) -> None:
        self.total = max(_MIN_LEASE, int(total_bytes))
        self._used = 0
        self._peak = 0
        self._waits = 0
        self._lock = threading.Condition()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    @property
    def waits(self) -> int:
        with self._lock:
            return self._waits

    def acquire(self, size: int) -> int:
        """占用 ``size`` 字节，返回实际记账量，用于 release。"""
        want = max(1, min(int(size), self.total))
        with self._lock:
            if self._used > 0 and self._used + want > self.total:
                self._waits += 1
                while self._used > 0 and self._used + want > self.total:
                    self._lock.wait()
            self._used += want
            self._peak = max(self._peak, self._used)
            return want

    def release(self, size: int) -> None:
        with self._lock:
            self._used = max(0, self._used - max(0, int(size)))
            self._lock.notify_all()

    @contextmanager
    def lease(self, size: int) -> Iterator[None]:
        taken = self.acquire(size)
        try:
            yield
        finally:
            self.release(taken)
