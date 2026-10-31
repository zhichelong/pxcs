# Third-party notices

PXCS itself is MIT-licensed (see [LICENSE](LICENSE)). It depends on the following
components. The prebuilt Windows executable bundles them, so their license texts must
travel with it.

## Runtime

| Component | License | Used for |
|---|---|---|
| [Pillow](https://python-pillow.org/) | MIT-CMU | image decoding and encoding |
| [numpy](https://numpy.org/) | BSD-3-Clause | pixel and residual arrays |
| [cryptography](https://cryptography.io/) | Apache-2.0 OR BSD-3-Clause | AES-256-GCM, Argon2id, HKDF |
| [imagecodecs](https://www.lfd.uci.edu/~gohlke/codecs/) | BSD-3-Clause | JPEG XL lossless JPEG transcoding |

`imagecodecs` bundles further codecs (libjxl, libwebp, zlib, brotli, lcms2 and others).
Their notices are shipped in its own distribution; consult the `imagecodecs` package for
the complete list.

## Development

| Component | License |
|---|---|
| [coverage.py](https://coverage.readthedocs.io/) | Apache-2.0 |

## Codecs used through Pillow

libjpeg-turbo (IJG/BSD), libpng (libpng-2.0), libtiff (libtiff), libwebp (BSD-3-Clause)
and zlib (zlib) are linked by the Pillow wheels. See Pillow's `LICENSE` for the exact
notices.
