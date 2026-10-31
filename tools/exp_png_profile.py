"""剖析：大尺寸 RGBA PNG 的差分编码各阶段耗时，以及多线程能否并行。

背景：A09（397 张 5588x3388 RGBA PNG）跑了 25 分钟只有 2.5 核实测 CPU，
说明某一段被串行化了。本脚本把一张图的处理拆成 5 段分别计时，
再对比 1 线程 / N 线程的墙钟时间。

用法：
    python -m tools.exp_png_profile --dir "D:\\pictures\\png-variants" --count 6
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pxc_core.image_io import canonical_pixel_hash, load_image  # noqa: E402
from pxcs_core.codec import (  # noqa: E402
    center_residual,
    encode_webp_lossless,
    make_residual,
    quantize_residual,
)

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _stages(key: object, path: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    img = load_image(path, compute_file_hash=True, compute_source_pixel_hash=True)
    t1 = time.perf_counter()
    res = make_residual(key, img)  # type: ignore[arg-type]
    t2 = time.perf_counter()
    res = quantize_residual(res, 12)
    res = center_residual(res)
    t3 = time.perf_counter()
    payload = zlib.compress(res.data.tobytes(), 6)
    t4 = time.perf_counter()
    canonical_pixel_hash(img.mode, img.width, img.height, img.data)
    t5 = time.perf_counter()
    return {
        "load": t1 - t0,
        "residual": t2 - t1,
        "quant": t3 - t2,
        "zlib": t4 - t3,
        "hash": t5 - t4,
        "total": t5 - t0,
        "payload": len(payload),
    }


def _same_size_samples(root: Path, count: int) -> list[Path]:
    from PIL import Image

    pool = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not pool:
        return []
    with Image.open(pool[0]) as first:
        want = (first.size, first.mode)
    picked: list[Path] = []
    for path in pool:
        if len(picked) >= count + 1:
            break
        try:
            with Image.open(path) as img:
                if (img.size, img.mode) == want:
                    picked.append(path)
        except Exception:  # noqa: BLE001
            continue
    print(f"同尺寸样本 {len(picked)} 张（{want[0][0]}x{want[0][1]} {want[1]}）")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    files = _same_size_samples(Path(args.dir), args.count)
    if len(files) < 2:
        print("同尺寸图片不足")
        return 1

    key = load_image(files[0], compute_file_hash=False, compute_source_pixel_hash=False)
    print(
        f"基准图: {key.width}x{key.height} {key.mode}, "
        f"{key.data.nbytes / 1048576:.0f} MB 像素"
    )

    t0 = time.perf_counter()
    webp = encode_webp_lossless(key, method=4)
    t1 = time.perf_counter()
    print(f"webp_lossless 单张: {t1 - t0:.2f}s -> {len(webp) / 1048576:.2f} MB")

    print(f"\n{'i':>2} {'load':>7}{'resid':>7}{'quant':>7}{'zlib':>7}{'hash':>7}{'total':>7}")
    seq: list[dict[str, float]] = []
    t_all = time.perf_counter()
    for i, path in enumerate(files[1:], 1):
        st = _stages(key, path)
        seq.append(st)
        print(
            f"{i:>2} {st['load']:>7.2f}{st['residual']:>7.2f}{st['quant']:>7.2f}"
            f"{st['zlib']:>7.2f}{st['hash']:>7.2f}{st['total']:>7.2f}"
        )
    seq_wall = time.perf_counter() - t_all
    mean = {
        k: statistics.mean(s[k] for s in seq)
        for k in ("load", "residual", "quant", "zlib", "hash", "total")
    }
    print(
        f"\n单线程合计 {seq_wall:.2f}s  平均每张 {mean['total']:.2f}s"
        f"（load {mean['load']:.2f} / resid {mean['residual']:.2f} / "
        f"quant {mean['quant']:.2f} / zlib {mean['zlib']:.2f} / hash {mean['hash']:.2f}）"
    )

    n = len(files) - 1
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(lambda p: _stages(key, p), files[1:]))
    par_wall = time.perf_counter() - t0
    print(
        f"{args.threads} 线程合计 {par_wall:.2f}s（{n} 张） "
        f"加速比 {seq_wall / par_wall:.2f}x  理论上限 {min(args.threads, n)}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
