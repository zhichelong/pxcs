# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把 pxcs.py 打成一个单文件 exe。

用法（在仓库根目录）：

    python -m pip install pyinstaller
    python -m PyInstaller --noconfirm --clean packaging/pxcs.spec

产物是 `dist/pxcs.exe`。`imagecodecs` 会连带 libjxl / libwebp / zlib 一起打进去，
所以体积约 50 MB；没有这个依赖也能跑，只是少了 `jpegxl` 编码器。

`language/` 打进 exe 内部；如果把 `language/` 或 `config/` 放在 exe 旁边，
它们会覆盖内置版本。
"""
import os

ROOT = os.path.dirname(SPECPATH)  # noqa: F821 - SPECPATH 由 PyInstaller 注入

from PyInstaller.utils.hooks import collect_all

datas = [(os.path.join(ROOT, "language"), "language")]
binaries = []
hiddenimports = []
_imagecodecs = collect_all("imagecodecs")
datas += _imagecodecs[0]
binaries += _imagecodecs[1]
hiddenimports += _imagecodecs[2]

a = Analysis(
    [os.path.join(ROOT, "pxcs.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="pxcs",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
