"""生成 PXC 字节级黄金向量（tests/fixtures/golden/）。

黄金向量是"第二实现"的验收基准：给定同一份样本数组与同一组参数，
任何实现都必须编码出完全相同的 PXC 字节。

用法：

    python -m tools.gen_golden            # 重新生成全部向量
    python -m tools.gen_golden --check    # 只校验现有向量是否仍然一致

注意：tile payload 里的 zlib 输出与 zlib 版本有关。manifest 记录了生成时的
zlib 运行时版本，测试在看到版本不一致时会降级为"语义等价"检查而不是失败
（见 CODEC.md §8.4）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pxc_core.codec import encode_tiles
from pxc_core.container import build_pxc
from pxc_core.image_io import ImageData, canonical_pixel_hash
from pxc_core.config import TILE_MODE_NAMES, PALETTE_PLAIN, PALETTE_RLE

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden"

# 用固定元数据，避免文件名/格式名随环境变化而改变字节。
ORIGINAL_FILENAME = "golden-source.png"
SOURCE_FORMAT = "PNG"
SOURCE_EXT = ".png"


# ---------------------------------------------------------------------------
# 样本数组生成器：全部用 numpy 构造，不依赖 Pillow 解码结果。
# ---------------------------------------------------------------------------
def _solid(height: int, width: int, channels: int, value: int | tuple) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[:, :] = value
    return arr


def _checker(height: int, width: int, values: list[int]) -> np.ndarray:
    """棋盘格：相邻像素不同，PALETTE-PLAIN 优于 PALETTE-RLE。"""
    arr = np.zeros((height, width, 1), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = values[(y + x) % len(values)]
    return arr


def _bands(height: int, width: int, channels: int, colors: list[tuple]) -> np.ndarray:
    """横向色带：同一行同色，行与行之间换色，利于 RUN / SUB 滤波。"""
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for y in range(height):
        arr[y, :] = colors[y % len(colors)]
    return arr


def _row_stripes(height: int, width: int, channels: int) -> np.ndarray:
    """每行一个颜色，行内完全一致：SUB 滤波后整行为 0。"""
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for y in range(height):
        base = (y * 37) % 256
        arr[y, :] = (base, (base + 11) % 256, (base + 23) % 256)
    return arr


def _gradient_gray(height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width, 1), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x * 3 + y * 5) % 256
    return arr


def _noise_rgb(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _gradient_u16(height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width, 1), dtype=np.uint16)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x * 700 + y * 1300) % 65536
    return arr


def _palette_order_rgb8(height: int, width: int) -> np.ndarray:
    """三通道 8 位：验证调色板按“通道顺序的字节字典序”排序。

    颜色 (1,2,3)、(0,255,0)、(255,0,0)、(0,0,255) 的期望顺序是
    (0,0,255) < (0,255,0) < (1,2,3) < (255,0,0)，
    与“按 R、G、B 依次比较”一致，但与“按亮度/数值和”完全不同。
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(1, 2, 3), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
    for y in range(height):
        for x in range(width):
            arr[y, x] = colors[(y * width + x) % len(colors)]
    return arr


def _palette_order_u16(height: int, width: int) -> np.ndarray:
    """单通道 16 位：验证调色板按数值升序（1 在 256 之前，而不是按字节序）。"""
    arr = np.zeros((height, width, 1), dtype=np.uint16)
    values = [256, 1, 2, 65535, 0]
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = values[(y * width + x) % len(values)]
    return arr


def _int32_gradient(height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width, 1), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x - width // 2) * 1000 + (y - height // 2) * 37
    return arr


def _palette_256(height: int, width: int) -> np.ndarray:
    """恰好 256 色，覆盖调色板条目数上界。"""
    arr = np.zeros((height, width, 1), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (y * width + x) % 256
    return arr


def _rle_runs_rgb(
    height: int, width: int, seed: int, colors: int, run_len: int
) -> np.ndarray:
    """构造"短行程 + 多种随机颜色"的图，使 PALETTE-RLE 成为成本最小者。

    这类图在真实数据上很罕见（4000 次随机试验只有 3 次），但属于合法产物，
    解码器必须支持，因此需要专门的向量覆盖。行程短 + 颜色随机会让 PRED 残差
    稀疏但体积翻倍（u16），从而输给 RLE。
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    n = height * width
    idx = np.repeat(rng.integers(0, colors, size=n // run_len + 1), run_len)[:n]
    return np.ascontiguousarray(palette[idx].reshape(height, width, 3))


def _build_cases() -> list[dict[str, Any]]:
    """返回全部向量定义。每个 case 自带样本数组与编码参数。"""
    cases: list[dict[str, Any]] = []

    def add(
        name: str,
        mode: str,
        arr: np.ndarray,
        tile_size: int,
        *,
        palette: bytes | None = None,
        transparency: bytes | None = None,
        note: str = "",
    ) -> None:
        height, width = arr.shape[0], arr.shape[1]
        cases.append(
            {
                "name": name,
                "mode": mode,
                "width": width,
                "height": height,
                "tile_size": tile_size,
                "data": arr,
                "palette": palette,
                "transparency": transparency,
                "note": note,
            }
        )

    # --- SOLID ---
    add("solid_rgb8", "RGB", _solid(32, 32, 3, (17, 34, 51)), 8, note="全同色 → SOLID")
    add("solid_gray_u16", "I;16", np.full((16, 16, 1), 1234, dtype=np.uint16), 8)

    # --- PALETTE ---
    p8 = bytes(bytearray(range(256)) * 3)
    add("palette_plain_l8", "L", _checker(16, 16, [0, 60, 120, 180]), 8,
        note="棋盘格，索引无长行程 → PLAIN")
    add("palette_rle_l8", "L", _bands(32, 32, 1, [(0,), (90,), (200,)]), 8,
        note="整行同色 → RLE")
    add("palette_p_mode", "P", _checker(16, 16, [3, 7, 11, 15]), 8, palette=p8,
        note="P 模式带调色板元数据")
    add("palette_256_l8", "L", _palette_256(16, 16), 8, note="恰好 256 色")
    add(
        "palette_rle_rgb8",
        "RGB",
        _rle_runs_rgb(34, 26, seed=12345, colors=12, run_len=7),
        34,
        note="短行程 + 12 色随机调色板 → PALETTE-RLE（否则会被 PRED 压过）",
    )

    # --- PRED ---
    add("pred_gray8", "L", _gradient_gray(32, 32), 16, note="平滑梯度 → PRED")
    add("pred_u16", "I;16", _gradient_u16(24, 24), 8, note="16 位残差存储为 u32")
    add("pred_int32", "I", _int32_gradient(24, 24), 8, note="int32 残差存储为 u64")

    # --- RUN ---
    add("run_rgb8", "RGB", _bands(48, 48, 3, [(10, 20, 30), (200, 100, 50)]), 8)

    # --- FILTER ---
    add("filter_rgb8", "RGB", _row_stripes(32, 32, 3), 8,
        note="行内恒定，SUB 后整行为 0")

    # --- RAW ---
    add("raw_noise_rgb8", "RGB", _noise_rgb(24, 24, seed=7), 8, note="噪声 → RAW")
    add("raw_noise_gray8", "L", _noise_rgb(24, 24, seed=11)[:, :, :1], 8)

    # --- 调色板排序 ---
    add("palette_order_rgb8", "RGB", _palette_order_rgb8(16, 16), 8,
        note="多通道按通道顺序字节序")
    add("palette_order_u16", "I;16", _palette_order_u16(16, 16), 8,
        note="单通道按数值序")

    # --- 边界 ---
    add("nondiv_tile_rgb8", "RGB", _noise_rgb(30, 18, seed=3), 8,
        note="宽高非 tile_size 整数倍")
    add("tile_size_min", "L", _gradient_gray(16, 16), 8, note="tile_size 下界")
    add("tile_size_max_tile", "L", _gradient_gray(40, 40), 40, note="单 tile 覆盖整图")

    # --- 常量 alpha ---
    rgba = np.zeros((24, 24, 4), dtype=np.uint8)
    rgba[..., :3] = _gradient_gray(24, 24)
    rgba[..., 3] = 255
    cases.append(
        {
            "name": "constant_alpha_rgba",
            "mode": "RGBA",
            "width": 24,
            "height": 24,
            "tile_size": 8,
            "data": rgba,
            "palette": None,
            "transparency": None,
            "note": "alpha 全 255 → 只存 3 通道 + constant_alpha",
            "header_channels": 3,
            "constant_alpha": 255,
        }
    )

    # --- 透明度元数据 ---
    add("transparency_p_mode", "P", _checker(16, 16, [0, 1, 2, 3]), 8, palette=p8,
        transparency=b"\x00" + struct.pack("<I", 2),
        note="P 模式单索引透明")

    return cases


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------
def _make_image(case: dict[str, Any]) -> ImageData:
    arr = np.ascontiguousarray(case["data"])
    height, width = arr.shape[0], arr.shape[1]
    constant_alpha = case.get("constant_alpha")
    image = ImageData(
        mode=case["mode"],
        width=width,
        height=height,
        data=arr,
        palette=case.get("palette"),
        transparency=case.get("transparency"),
        original_filename=ORIGINAL_FILENAME,
        source_format=SOURCE_FORMAT,
        source_ext=SOURCE_EXT,
        source_file_size=0,
        pixel_hash=b"",
        constant_alpha=constant_alpha,
    )
    # ImageData 会按 mode 强制 dtype；若向量定义用了 mode 不支持的 dtype，
    # 这里会静默截断，导致像素哈希与实际数据不符。必须显式拦住。
    if image.data.dtype != arr.dtype:
        raise ValueError(
            f"{case['name']}: mode {case['mode']} 的样本类型是 "
            f"{image.data.dtype}，与向量定义的 {arr.dtype} 不符（会被静默截断）"
        )
    # 像素哈希必须覆盖 ImageData 实际存储的数据，而不是传入的数组。
    image.pixel_hash = canonical_pixel_hash(
        image.mode, image.width, image.height, image.data
    )
    return image


def _encode(case: dict[str, Any]) -> bytes:
    image = _make_image(case)
    tile_size = case["tile_size"]
    data_payload = encode_tiles(image, tile_size, threads=1)
    return build_pxc(image, data_payload, None, tile_size=tile_size)


def _tile_payload_of(pxc_bytes: bytes) -> bytes:
    """从 PXC 字节里取出 DATA chunk（不依赖 codec 的解码器）。"""
    header_size = struct.unpack_from("<I", pxc_bytes, 8)[0]
    chunk_dir_offset = struct.unpack_from("<Q", pxc_bytes, 40)[0]
    chunk_count = struct.unpack_from("<I", pxc_bytes, 48)[0]
    for i in range(chunk_count):
        base = chunk_dir_offset + 32 * i
        chunk_id = pxc_bytes[base : base + 4]
        offset, length = struct.unpack_from("<QQ", pxc_bytes, base + 8)
        if chunk_id == b"DATA":
            return pxc_bytes[offset : offset + length]
    raise RuntimeError("PXC 中没有 DATA chunk")


def _mode_histogram(data_payload: bytes) -> dict[str, int]:
    """统计 tile 模式分布，用来确认向量覆盖了所有模式。"""
    _, _, _, _, tile_count = struct.unpack_from("<IIIII", data_payload, 0)
    offset = 20
    hist: dict[str, int] = {}
    for _ in range(tile_count):
        mode, submode = struct.unpack_from("<BB", data_payload, offset)
        _, comp_len = struct.unpack_from("<II", data_payload, offset + 4)
        if mode == 1:
            key = f"PALETTE_{'RLE' if submode == PALETTE_RLE else 'PLAIN'}"
        else:
            key = TILE_MODE_NAMES.get(mode, f"UNKNOWN_{mode}")
        hist[key] = hist.get(key, 0) + 1
        offset += 20 + comp_len
    return hist


# ---------------------------------------------------------------------------
# 生成 / 校验
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_vectors() -> dict[str, Any]:
    expected_dir = GOLDEN_DIR / "expected"
    source_dir = GOLDEN_DIR / "source"
    expected_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}

    for case in _build_cases():
        arr = np.ascontiguousarray(case["data"])
        payload = _encode(case)
        hist = _mode_histogram(_tile_payload_of(payload))
        for key, count in hist.items():
            coverage[key] = coverage.get(key, 0) + count

        # 样本数组按小端落盘，测试侧据此重建同一份数组。
        source_bytes = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        source_name = f"{case['name']}.bin"
        expected_name = f"{case['name']}.pxc"
        (source_dir / source_name).write_bytes(source_bytes)
        (expected_dir / expected_name).write_bytes(payload)

        entry: dict[str, Any] = {
            "name": case["name"],
            "note": case.get("note", ""),
            "mode": case["mode"],
            "width": case["width"],
            "height": case["height"],
            "dtype": str(arr.dtype),
            "channels": int(arr.shape[2]),
            "tile_size": case["tile_size"],
            "source": f"source/{source_name}",
            "source_sha256": _sha256(source_bytes),
            "expected": f"expected/{expected_name}",
            "expected_size": len(payload),
            "expected_sha256": _sha256(payload),
            "tile_modes": hist,
        }
        if case.get("palette") is not None:
            entry["palette_hex"] = case["palette"].hex()
            entry["palette_sha256"] = _sha256(case["palette"])
        if case.get("transparency") is not None:
            entry["transparency_hex"] = case["transparency"].hex()
        if case.get("constant_alpha") is not None:
            entry["constant_alpha"] = case["constant_alpha"]
            entry["header_channels"] = case["header_channels"]
        entries.append(entry)

    manifest = {
        "format": "PXC-GOLDEN",
        "version": 1,
        "generator": "tools/gen_golden.py",
        "zlib_version": zlib.ZLIB_RUNTIME_VERSION,
        "numpy_version": np.__version__,
        "original_filename": ORIGINAL_FILENAME,
        "source_format": SOURCE_FORMAT,
        "source_ext": SOURCE_EXT,
        "coverage": dict(sorted(coverage.items())),
        "cases": entries,
    }
    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _check_vectors() -> int:
    """校验磁盘上的向量是否与当前实现一致；返回失败数。"""
    manifest_path = GOLDEN_DIR / "manifest.json"
    if not manifest_path.is_file():
        print(f"缺少 {manifest_path}，请先运行 python -m tools.gen_golden")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = 0
    for case in _build_cases():
        payload = _encode(case)
        entry = next(
            (item for item in manifest["cases"] if item["name"] == case["name"]), None
        )
        if entry is None:
            print(f"[MISS] {case['name']} 不在 manifest 中")
            failures += 1
            continue
        digest = _sha256(payload)
        if digest != entry["expected_sha256"]:
            print(
                f"[DIFF] {case['name']}: 期望 {entry['expected_size']} 字节 "
                f"{entry['expected_sha256'][:16]}，实际 {len(payload)} 字节 {digest[:16]}"
            )
            failures += 1
    if failures == 0:
        print(f"OK: {len(manifest['cases'])} 个向量与当前实现一致")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成/校验 PXC 字节级黄金向量")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    if args.check:
        return 1 if _check_vectors() else 0

    manifest = _write_vectors()
    print(f"已生成 {len(manifest['cases'])} 个向量 → {GOLDEN_DIR}")
    print("tile 模式覆盖：")
    for key, count in manifest["coverage"].items():
        print(f"  {key:<16} {count}")
    missing = [
        name
        for name in ("SOLID", "PALETTE_PLAIN", "PALETTE_RLE", "PRED", "RUN", "FILTER", "RAW")
        if name not in manifest["coverage"]
    ]
    if missing:
        print(f"警告：以下模式未被任何向量覆盖：{', '.join(missing)}")
        return 2
    print("全部 7 种 tile 模式均已覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
