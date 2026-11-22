"""兼容层：有界解压实现已移到 pxc_core.zlibsafe（PXC tile 也要用）。"""
from pxc_core.zlibsafe import MAX_INDEX_BYTES, decompress_bounded

__all__ = ["MAX_INDEX_BYTES", "decompress_bounded"]
