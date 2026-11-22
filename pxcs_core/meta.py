"""把图片元数据编码进 PXCS 索引。

zlib 残差只存像素数组，没有地方放 ICC/EXIF/XMP；PXC 残差靠容器自带
metadata 区解决，zlib 残差则把这些字段放进 entry.meta（JSON，base64）。
索引本身是 zlib 压缩的，重复的 ICC 在索引里会重复但能被压掉。
"""
from __future__ import annotations

import base64
from typing import Any, Optional

from pxc_core.image_io import ImageData


def _b64(value: Optional[bytes]) -> Optional[str]:
    if not value:
        return None
    return base64.b64encode(bytes(value)).decode("ascii")


def _unb64(value: Any) -> Optional[bytes]:
    if not value:
        return None
    try:
        return base64.b64decode(str(value).encode("ascii"))
    except (ValueError, TypeError):
        return None


def encode_image_meta(image: ImageData) -> Optional[dict]:
    """返回可写进索引的元数据字典；没有元数据时返回 None。"""
    meta: dict[str, Any] = {}
    icc = _b64(image.icc_profile)
    if icc:
        meta["icc"] = icc
    exif = _b64(image.exif)
    if exif:
        meta["exif"] = exif
    xmp = _b64(image.xmp)
    if xmp:
        meta["xmp"] = xmp
    comment = _b64(image.comment)
    if comment:
        meta["comment"] = comment
    transparency = _b64(image.transparency)
    if transparency:
        meta["transparency"] = transparency
    if image.dpi:
        dpi = [float(image.dpi[0]), float(image.dpi[1])]
        if dpi != [0.0, 0.0]:
            meta["dpi"] = dpi
    if image.gamma is not None:
        meta["gamma"] = float(image.gamma)
    return meta or None


def decode_image_meta(meta: Optional[dict]) -> dict[str, Any]:
    """把 entry.meta 还原成 ImageData 的关键字参数。"""
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for key, field in (
        ("icc", "icc_profile"),
        ("exif", "exif"),
        ("xmp", "xmp"),
        ("comment", "comment"),
        ("transparency", "transparency"),
    ):
        value = _unb64(meta.get(key))
        if value is not None:
            out[field] = value
    dpi = meta.get("dpi")
    if isinstance(dpi, (list, tuple)) and len(dpi) == 2:
        try:
            out["dpi"] = (float(dpi[0]), float(dpi[1]))
        except (TypeError, ValueError):
            pass
    gamma = meta.get("gamma")
    if gamma is not None:
        try:
            out["gamma"] = float(gamma)
        except (TypeError, ValueError):
            pass
    return out
