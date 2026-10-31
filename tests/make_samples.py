"""生成测试图片。所有图片都是程序生成，不依赖外部素材。"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, PngImagePlugin


def _gradient_rgb(w: int, h: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(1, w - 1)).astype(np.uint8)
    g = (y * 255 // max(1, h - 1)).astype(np.uint8)
    b = (((x + y) * 255) // max(1, w + h - 2)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _add_noise(arr: np.ndarray, seed: int = 1, step: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = arr.copy()
    mask = np.zeros(out.shape[:2], dtype=bool)
    mask[::step, ::step] = True
    noise = rng.integers(0, 256, (mask.sum(), out.shape[2]), dtype=np.uint8)
    out[mask] = noise
    return out


def _save_icc_exif_png(path: Path) -> None:
    img = Image.new("RGB", (64, 48), (20, 40, 60))
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[0x010E] = "PXC test image"
    exif[0x0131] = "PXC test software"
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(
        "XML:com.adobe.xmp",
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>PXC test</x:xmpmeta>",
        zip=False,
    )
    img.save(
        path,
        pnginfo=pnginfo,
        icc_profile=icc,
        exif=exif.tobytes(),
        dpi=(300, 300),
    )


def create_all(root: Path, include_large: bool = True) -> dict[str, Path]:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # 纯色
    p = root / "solid_rgb.png"
    Image.new("RGB", (320, 240), (30, 144, 255)).save(p)
    out["solid_rgb"] = p

    # 渐变
    p = root / "gradient_h.png"
    Image.fromarray(_gradient_rgb(320, 240), "RGB").save(p)
    out["gradient_h"] = p

    p = root / "gradient_v.png"
    y, x = np.mgrid[0:240, 0:320]
    arr = np.stack(
        [
            (y * 255 // 239).astype(np.uint8),
            (x * 255 // 319).astype(np.uint8),
            ((x * 2 + y) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(arr, "RGB").save(p)
    out["gradient_v"] = p

    # 调色板 + transparency
    p = root / "palette.png"
    img = Image.new("P", (256, 192), 0)
    pal = []
    for i in range(8):
        pal += [i * 30, 255 - i * 20, i * 10]
    img.putpalette(pal)
    data = [((x // 16) + (y // 16)) % 8 for y in range(192) for x in range(256)]
    img.putdata(data)
    img.info["transparency"] = bytes([0] + [255] * 7)
    img.save(p, transparency=bytes([0] + [255] * 7))
    out["palette"] = p

    # RGBA 渐变 alpha
    p = root / "rgba.png"
    h, w = 150, 200
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = _gradient_rgb(w, h)
    rgba[..., 3] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    Image.fromarray(rgba, "RGBA").save(p)
    out["rgba"] = p

    # 全透明但 RGB 不同
    p = root / "rgba_invisible.png"
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (128, 128, 4), dtype=np.uint8)
    arr[..., 3] = 0
    Image.fromarray(arr, "RGBA").save(p)
    out["rgba_invisible"] = p

    # 灰度 / LA
    p = root / "gray.png"
    y, x = np.mgrid[0:200, 0:256]
    Image.fromarray(((x * 2 + y) % 256).astype(np.uint8), "L").save(p)
    out["gray"] = p

    p = root / "gray_alpha.png"
    la = np.dstack(
        [((x + y) % 256).astype(np.uint8), (255 - x).astype(np.uint8)]
    )
    Image.fromarray(la, "LA").save(p)
    out["gray_alpha"] = p

    # 1-bit
    p = root / "mono.png"
    w, h = 256, 192
    y, x = np.mgrid[0:h, 0:w]
    bits = ((x // 8 + y // 8) % 2).astype(bool)
    packed = np.packbits(bits, axis=1, bitorder="big")
    Image.frombytes("1", (w, h), packed.tobytes()).save(p)
    out["mono"] = p

    # 噪声
    p = root / "noise.png"
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8), "RGB").save(p)
    out["noise"] = p

    # 多瓦片边界的大图
    if include_large:
        p = root / "large.png"
        arr = _add_noise(_gradient_rgb(1200, 900), seed=3, step=11)
        Image.fromarray(arr, "RGB").save(p)
        out["large"] = p

    # 16-bit
    p = root / "gray16.png"
    y, x = np.mgrid[0:240, 0:320]
    a16 = ((x * 300 + y * 17) % 65536).astype(np.uint16)
    Image.fromarray(a16).save(p)
    out["gray16"] = p

    p = root / "gray16be.tif"
    a16 = ((x * 123 + y * 7) % 65536).astype(np.uint16)
    raw = a16.astype(">u2").tobytes()
    Image.frombytes("I;16B", (320, 240), raw).save(p)
    out["gray16be"] = p

    # CMYK / I
    p = root / "cmyk.tif"
    ac = np.stack(
        [
            (x * 2) % 256,
            (y * 3) % 256,
            (x + y) % 256,
            (x * y) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(ac, "CMYK").save(p)
    out["cmyk"] = p

    p = root / "int32.tif"
    ai = (x.astype(np.int32) * 1000 - y.astype(np.int32) * 7)
    Image.fromarray(ai, "I").save(p)
    out["int32"] = p

    # JPEG 有损源
    p = root / "photo.jpg"
    arr = _add_noise(_gradient_rgb(640, 480), seed=5, step=9)
    Image.fromarray(arr, "RGB").save(p, quality=90)
    out["photo_jpeg"] = p

    # WebP lossless / BMP / TIFF
    p = root / "lossless.webp"
    Image.fromarray(_gradient_rgb(320, 240), "RGB").save(p, lossless=True)
    out["webp"] = p

    p = root / "rgb.bmp"
    Image.fromarray(_gradient_rgb(320, 240), "RGB").save(p)
    out["bmp"] = p

    p = root / "rgb.tif"
    Image.fromarray(_gradient_rgb(320, 240), "RGB").save(p, compression="tiff_deflate")
    out["tiff"] = p

    # 元数据
    p = root / "meta.png"
    _save_icc_exif_png(p)
    out["meta"] = p

    # 动画 GIF / 多页 TIFF：用于“明确拒绝”
    p = root / "animated.gif"
    frames = [Image.new("P", (32, 24), i) for i in range(3)]
    for frame in frames:
        frame.putpalette([i for i in range(256)] * 3)
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=100, loop=0)
    out["animated_gif"] = p

    p = root / "multipage.tif"
    a = Image.new("RGB", (32, 24), (1, 2, 3))
    b = Image.new("RGB", (32, 24), (4, 5, 6))
    a.save(p, save_all=True, append_images=[b])
    out["multipage_tif"] = p

    # 目录结构 / 同名冲突
    nested = root / "nested"
    (nested / "sub").mkdir(parents=True)
    Image.fromarray(_gradient_rgb(96, 64), "RGB").save(nested / "a.png")
    Image.fromarray(_gradient_rgb(96, 64)[::-1], "RGB").save(nested / "sub" / "b.png")
    out["nested_dir"] = nested

    d1 = root / "dir1"
    d2 = root / "dir2"
    d1.mkdir()
    d2.mkdir()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(d1 / "same.png")
    Image.new("RGB", (64, 64), (30, 20, 10)).save(d2 / "same.png")
    out["dir1"] = d1
    out["dir2"] = d2

    (root / "notes.txt").write_text("not an image\n", encoding="utf-8")
    out["notes"] = root / "notes.txt"

    return out


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests") / "_samples"
    samples = create_all(target)
    for key, path in sorted(samples.items()):
        print(f"{key}: {path}")
