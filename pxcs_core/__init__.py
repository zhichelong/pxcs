"""PXCS 核心包。"""
from pxc_core.i18n import ensure_stdio_utf8
from .format import VERSION_STRING

ensure_stdio_utf8()

__all__ = ["VERSION_STRING"]
