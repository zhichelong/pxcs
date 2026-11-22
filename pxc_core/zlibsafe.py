"""有界 zlib 解压：PXC tile 与 PXCS 归档的所有 zlib 数据都走这里。

旧实现的问题不在于"忘了检查长度"，而在于检查得太晚：

    out = dec.decompress(blob, LIMIT)
    out += dec.flush()          # flush 没有长度参数，剩余输入会被整块展开
    if len(out) > LIMIT: ...

``decompressobj.flush()`` 会把 ``unconsumed_tail`` 里的内容一次性解完，所以上限
只挡住了第一次调用。这里的做法是：先确定"最多允许多少字节"，循环时把
``max_length`` 一直压在剩余空间上，一旦超出立刻抛错；结束条件同时要求
``eof``、没有 ``unconsumed_tail``、没有 ``unused_data``，最后再核对精确长度。

所有异常统一成 ``PxcCorruptionError``（``PxcsCorruptionError`` 是它的子类），
调用方不需要再区分 zlib 层错误和归档层错误。
"""
from __future__ import annotations

import zlib
from typing import Optional, Union

from .errors import PxcCorruptionError
from .i18n import t

# 每次喂给 decompressobj 的输入块大小。太小会增加 Python 调用开销，太大
# 会让单次解压的临时缓冲变大。
_CHUNK = 1 << 20

# 索引本身的上限；与写入端 ``container.finish`` 的 256 MiB 限制一致。
MAX_INDEX_BYTES = 256 * 1024 * 1024


def decompress_bounded(
    blob: Union[bytes, bytearray, memoryview],
    *,
    limit: int,
    expected: Optional[int] = None,
    label: str = "",
) -> bytes:
    """解压 ``blob``，输出绝不超过 ``limit`` 字节。

    ``expected`` 非空时它同时是"必须精确等于"的目标：既用来收紧 ``limit``，
    也用来做最后的长短校验。``label`` 只用于错误信息（通常是条目路径）。
    """
    if expected is not None:
        expected = int(expected)
        if expected < 0:
            raise PxcCorruptionError(
                t("zlib.length", expected=expected, actual=0, label=label)
            )
        if expected < int(limit):
            limit = expected
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit 不能为负")

    dec = zlib.decompressobj()
    # 用小块列表而不是 bytearray 累加：bytearray 扩容时会整块复制，
    # 256 MiB 的索引在超限前会先吃掉两倍内存，分块列表只占输出本身。
    chunks: list[bytes] = []
    total = 0
    view = memoryview(blob)
    try:
        for start in range(0, len(view), _CHUNK):
            data: Union[bytes, memoryview] = view[start:start + _CHUNK]
            while True:
                # 至少留 1 字节余量：这样"刚好等于 limit"不会误报，
                # 而多出来的第 1 字节就能触发超限。
                room = limit + 1 - total
                if room <= 0:  # pragma: no cover - 上面的检查已经拦住
                    raise PxcCorruptionError(
                        t("zlib.limit", limit=limit, label=label)
                    )
                piece = dec.decompress(data, room)
                if piece:
                    chunks.append(piece)
                    total += len(piece)
                    if total > limit:
                        raise PxcCorruptionError(
                            t("zlib.limit", limit=limit, label=label)
                        )
                data = dec.unconsumed_tail
                if not data:
                    break
            if dec.eof:
                break
    except zlib.error as exc:
        raise PxcCorruptionError(
            t("zlib.bad_stream", label=label, reason=exc)
        ) from exc

    if not dec.eof:
        raise PxcCorruptionError(t("zlib.incomplete", label=label))
    if dec.unused_data:
        raise PxcCorruptionError(
            t("zlib.trailing", label=label, size=len(dec.unused_data))
        )
    if expected is not None and total != expected:
        raise PxcCorruptionError(
            t("zlib.length", expected=expected, actual=total, label=label)
        )
    return b"".join(chunks)
