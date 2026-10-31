"""实验：残差在 zlib 之前做行滤波，能省多少字节。

现状（pxcs_core.codec.encode_zlib_residual）：把居中量化后的残差数组
原样 tobytes 交给 zlib。图像数据普遍适合先做 PNG 那套行滤波再压，
这里量化/居中后大量样本等于 128，理论上更值得滤。

用法：
    python -m tools.exp_residual_filter --dir "D:\\pictures\\jpeg-variants" --pairs 8
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pxc_core.image_io import ImageDecodeError, load_image  # noqa: E402
from pxcs_core.codec import (  # noqa: E402
    center_residual,
    make_residual,
    quantize_residual,
)


def _sub(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[:, 1:] = a[:, 1:] - a[:, :-1]
    return out


def _up(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[1:] = a[1:] - a[:-1]
    return out


def _avg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    left = np.zeros_like(a)
    left[:, 1:] = a[:, :-1]
    abv = np.zeros_like(a)
    abv[1:] = a[:-1]
    out = a - ((left.astype(np.uint16) + abv.astype(np.uint16)) // 2).astype(np.uint8)
    return out


def _paeth(a: np.ndarray) -> np.ndarray:
    left = np.zeros_like(a)
    left[:, 1:] = a[:, :-1]
    abv = np.zeros_like(a)
    abv[1:] = a[:-1]
    ul = np.zeros_like(a)
    ul[1:, 1:] = a[:-1, :-1]
    l = left.astype(np.int16)
    b = abv.astype(np.int16)
    c = ul.astype(np.int16)
    p = l + b - c
    pa, pb, pc = np.abs(p - l), np.abs(p - b), np.abs(p - c)
    pred = np.where((pa <= pb) & (pa <= pc), l, np.where(pb <= pc, b, c))
    return (a.astype(np.int16) - pred).astype(np.uint8)


FILTERS = {
    "none": lambda a: a,
    "sub": _sub,
    "up": _up,
    "avg": _avg,
    "paeth": _paeth,
}


def _adaptive(a: np.ndarray) -> np.ndarray:
    """逐行选最优滤波（PNG 启发式：非零字节的绝对值之和最小）。"""
    h = a.shape[0]
    cands = np.stack([fn(a) for fn in FILTERS.values()], axis=0)  # (F,H,W,C)
    signed = cands.astype(np.int16)
    signed = np.where(signed > 127, signed - 256, signed)
    cost = np.abs(signed).sum(axis=(2, 3))  # (F, H)
    choice = cost.argmin(axis=0)  # (H,)
    rows = np.arange(h)
    return cands[choice, rows]


def _group_by_size(paths: list[Path], max_groups: int) -> list[list[Path]]:
    from PIL import Image

    buckets: dict[tuple, list[Path]] = {}
    order: list[tuple] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                key = (img.size, img.mode)
        except Exception:  # noqa: BLE001
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(path)
    groups = [buckets[k] for k in order if len(buckets[k]) >= 2]
    groups.sort(key=len, reverse=True)
    return groups[:max_groups]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--pairs", type=int, default=6, help="每组取几张做差分")
    parser.add_argument("--groups", type=int, default=3, help="取几个最大的同尺寸组")
    parser.add_argument("--quant", type=int, default=12)
    args = parser.parse_args(argv)

    root = Path(args.dir)
    paths = sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
        key=lambda p: str(p),
    )
    groups = _group_by_size(paths, args.groups)
    if not groups:
        print("没有找到同尺寸组")
        return 1

    print(f"{'组':>3} {'序号':>4} {'滤波':<8}{'lvl6':>10}{'lvl9':>10}{'占比':>8}")
    totals: dict[str, list[int]] = {k: [0, 0] for k in list(FILTERS) + ["adaptive"]}
    for gi, group in enumerate(groups):
        members = group[: args.pairs + 1]
        try:
            images = [load_image(p, compute_file_hash=False, compute_source_pixel_hash=False)
                      for p in members]
        except (ImageDecodeError, OSError) as exc:
            print(f"{gi:>3} 跳过: {exc}")
            continue
        key = images[0]
        for idx, img in enumerate(images[1:], start=1):
            if img.data.shape != key.data.shape or img.mode != key.mode:
                continue
            residual = quantize_residual(make_residual(key, img), args.quant)
            residual = center_residual(residual)
            data = residual.data
            base6 = len(zlib.compress(data.tobytes(), 6))
            for name, fn in list(FILTERS.items()) + [("adaptive", _adaptive)]:
                filtered = fn(data)
                s6 = len(zlib.compress(filtered.tobytes(), 6))
                s9 = len(zlib.compress(filtered.tobytes(), 9))
                totals[name][0] += s6
                totals[name][1] += s9
                print(
                    f"{gi:>3} {idx:>4} {name:<8}{s6:>10}{s9:>10}"
                    f"{s6 / base6 * 100:>7.1f}%"
                )
    print(f"\n{'滤波':<10}{'lvl6 合计':>14}{'lvl9 合计':>14}{'相对 none':>12}")
    base = totals["none"][0] or 1
    for name, (s6, s9) in totals.items():
        print(f"{name:<10}{s6:>14}{s9:>14}{s6 / base * 100:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
