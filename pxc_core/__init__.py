"""PXC 核心包。"""
from .config import VERSION_STRING
from .i18n import ensure_stdio_utf8

ensure_stdio_utf8()

__all__ = ["VERSION_STRING"]
