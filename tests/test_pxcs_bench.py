"""``tools.pxcs_bench`` 性能基线工具的测试。"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pxcs_core.archive import CreateOptions, create_archive
from tools.pxcs_bench import main, run_benchmark


def _tiny_image(path: Path, shift: int = 0) -> None:
    """写一张 32x32 的确定性小图，用于快速跑通完整流程。"""
    yy, xx = np.mgrid[0:32, 0:32]
    arr = np.zeros((32, 32, 3), np.uint8)
    arr[..., 0] = (xx * 8 + shift) % 256
    arr[..., 1] = (yy * 8 + shift) % 256
    arr[..., 2] = ((xx + yy) * 4 + shift) % 256
    image = Image.fromarray(arr, "RGB")
    ImageDraw.Draw(image).rectangle(
        [2 + shift % 6, 3, 10 + shift % 6, 12],
        fill=(220, 40, 60),
    )
    image.save(path)


class PxcsBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-bench-test-"))
        self.src = self.tmp / "src"
        self.src.mkdir()
        _tiny_image(self.src / "001.png", 0)
        _tiny_image(self.src / "002.png", 17)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_directory_benchmark_returns_structured_data(self) -> None:
        result = run_benchmark(self.src, repeat=1, include_hostname=False)

        self.assertEqual(result.mode, "directory")
        self.assertEqual(result.entries, 2)
        self.assertEqual(result.images, 2)
        self.assertEqual(result.files, 0)
        self.assertGreater(result.input_bytes, 0)
        self.assertGreater(result.archive_bytes, 0)
        self.assertIsNone(result.archive_path)
        for step in ("create", "list", "verify", "extract"):
            self.assertIn(step, result.timings)
            self.assertGreaterEqual(result.timings[step], 0.0)
        self.assertEqual(result.verify_failed, 0)
        self.assertEqual(result.extract_failed, 0)
        self.assertEqual(len(result.runs), 1)
        self.assertIn("create", result.runs[0])
        self.assertNotIn("hostname", result.environment)

        data = result.to_dict()
        self.assertEqual(data["mode"], "directory")
        self.assertEqual(data["archive"]["entries"], 2)
        self.assertIn("environment", data)
        self.assertIn("throughput_mb_s", data)
        # 临时目录必须已被清理。
        self.assertFalse(Path(result.temp_root).exists())

    def test_json_output_is_parseable(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(
                [str(self.src), "--json", "--repeat", "1", "--no-hostname"]
            )
        self.assertEqual(code, 0)
        data = json.loads(buffer.getvalue())
        self.assertEqual(data["tool"]["name"], "pxcs_bench")
        self.assertEqual(data["tool"]["schema"], 1)
        self.assertEqual(data["mode"], "directory")
        self.assertEqual(data["repeat"], 1)
        self.assertEqual(data["archive"]["entries"], 2)
        self.assertGreater(data["archive"]["bytes"], 0)
        for key in (
            "python",
            "numpy",
            "pillow",
            "zlib_runtime",
            "cpu_count",
            "os",
            "pxcs_version",
        ):
            self.assertIn(key, data["environment"])
        for step in ("create", "list", "verify", "extract"):
            self.assertIn(step, data["timings"])
            self.assertIn(step, data["throughput_mb_s"])

    def test_archive_mode_only_reads_and_cleans_up(self) -> None:
        archive = self.tmp / "bench.pxcs"
        create_archive(
            archive,
            [str(self.src)],
            CreateOptions(quiet=True, progress=False),
        )
        result = run_benchmark(archive, repeat=1, include_hostname=False)

        self.assertEqual(result.mode, "archive")
        self.assertEqual(result.entries, 2)
        self.assertEqual(result.archive_path, str(archive))
        self.assertNotIn("create", result.timings)
        for step in ("list", "verify", "extract"):
            self.assertIn(step, result.timings)
        self.assertGreater(result.input_bytes, 0)
        self.assertGreater(result.archive_bytes, 0)
        self.assertEqual(result.verify_failed, 0)
        self.assertEqual(result.extract_failed, 0)
        self.assertTrue(archive.is_file())
        self.assertFalse(Path(result.temp_root).exists())


if __name__ == "__main__":
    unittest.main()
