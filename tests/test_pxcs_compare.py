"""PXCS 语义对比工具测试：等价判定、错误定位、加密、分卷、严格模式。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
from PIL import Image

from pxcs_core.archive import CreateOptions, create_archive
from pxcs_core.container import ArchiveReader
from pxcs_core.volumes import volume_path
from tools.pxcs_compare import (
    EXIT_DIFFERENT,
    EXIT_EQUIVALENT,
    EXIT_USAGE_ERROR,
    compare_archives,
    main,
)


def _write_images(root: Path, count: int = 3, seed: int = 0) -> None:
    """生成随机像素样本，保证编码结果与文件名一一对应。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        array = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        Image.fromarray(array, "RGB").save(root / f"{index:02d}.png")


def _run(argv: list[str]) -> tuple[int, str, str]:
    """在进程内执行 CLI，返回 (退出码, stdout, stderr)。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CompareToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pxcs-compare-"))
        self.src = self.tmp / "src"
        _write_images(self.src)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------ 辅助
    def _make(
        self,
        name: str,
        source: Optional[Path] = None,
        **options: object,
    ) -> Path:
        archive = self.tmp / name
        opts = CreateOptions(quiet=True, progress=False, **options)
        create_archive(archive, [str(source or self.src)], opts)
        return archive

    def _make_encrypted(self, name: str, password: str) -> Path:
        return self._make(
            name,
            password=password,
            kdf_memory_kib=8192,
            kdf_iterations=1,
            kdf_lanes=1,
        )

    # ------------------------------------------------------------ 等价
    def test_self_compare_is_equivalent(self) -> None:
        archive = self._make("a.pxcs")
        code, out, err = _run([str(archive), str(archive)])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)
        self.assertIn("语义等价", out)
        self.assertNotIn("ERROR:", out)
        self.assertNotIn("WARN:", out)

    def test_two_creates_are_equivalent_despite_random_archive_id(self) -> None:
        """同一批输入创建两次，archive_id 必然不同，但必须判定为等价。"""
        first = self._make("first.pxcs")
        second = self._make("second.pxcs")
        with (
            ArchiveReader(first) as left_reader,
            ArchiveReader(second) as right_reader,
        ):
            self.assertNotEqual(
                left_reader.header.archive_id,
                right_reader.header.archive_id,
            )

        code, out, err = _run([str(first), str(second)])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)
        self.assertIn("archive_id 不同", out)
        self.assertIn("语义等价", out)
        self.assertNotIn("ERROR:", out)

        report = compare_archives(first, second)
        self.assertTrue(report.equivalent)
        self.assertEqual(report.errors, [])
        self.assertTrue(
            any(diff.category == "archive_id" for diff in report.warnings)
        )

        # 结构化输出便于 CI:两边条目、版本、体积一致
        code, out, _ = _run([str(first), str(second), "--json"])
        self.assertEqual(code, EXIT_EQUIVALENT)
        data = json.loads(out)
        self.assertTrue(data["equivalent"])
        self.assertEqual(data["counts"]["error"], 0)
        self.assertEqual(data["left"]["entry_count"], data["right"]["entry_count"])
        self.assertEqual(data["left"]["version"], data["right"]["version"])
        self.assertNotEqual(data["left"]["archive_id"], data["right"]["archive_id"])

        # --strict 也不能因 archive_id 及其派生的 CRC/长度判为不等价
        code, out, err = _run([str(first), str(second), "--strict"])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)

    def test_strict_treats_metadata_difference_as_failure(self) -> None:
        first = self._make("first.pxcs")
        copied = self.tmp / "copied"
        shutil.copytree(self.src, copied)
        stamp = 1_600_000_000_000_000_000  # 2020-09-13，与源文件不同
        for path in copied.iterdir():
            os.utime(path, ns=(stamp, stamp))
        second = self._make("second.pxcs", copied)

        # mtime 只影响索引元数据:默认判定等价(带 WARN)
        code, out, err = _run([str(first), str(second)])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)
        self.assertIn("entry_metadata", out)

        # --strict 下同一差异应计为不等价
        code, out, err = _run([str(first), str(second), "--strict"])
        self.assertEqual(code, EXIT_DIFFERENT, out + err)
        self.assertIn("mtime_ns", out)

    # ------------------------------------------------------------ 不等价
    def test_different_content_is_error_and_names_entry(self) -> None:
        first = self._make("first.pxcs")
        other_src = self.tmp / "other"
        other_src.mkdir()
        for index in range(3):
            array = np.full((48, 64, 3), 32 + index * 40, dtype=np.uint8)
            Image.fromarray(array, "RGB").save(other_src / f"{index:02d}.png")
        second = self._make("second.pxcs", other_src)

        code, out, err = _run([str(first), str(second), "--json"])
        self.assertEqual(code, EXIT_DIFFERENT, out + err)
        data = json.loads(out)
        self.assertFalse(data["equivalent"])
        errors = [item for item in data["differences"] if item["severity"] == "ERROR"]
        self.assertTrue(errors)
        # 必须定位到具体条目，并归类为 payload 内容差异
        self.assertTrue(any(item.get("entry") == "00.png" for item in errors))
        self.assertTrue(
            any(item["category"] == "entry_payload" for item in errors)
        )

        code, out, err = _run([str(first), str(second)])
        self.assertEqual(code, EXIT_DIFFERENT)
        self.assertIn("ERROR:", out)
        self.assertIn("00.png", out)
        self.assertIn("语义不等价", out)

    def test_lossy_delta_metadata_difference_is_error(self) -> None:
        """有损差分的 codec/quant/lossy 不同必须报 ERROR（残差条目也要参与比较）。"""
        jpeg_src = self.tmp / "jpeg-src"
        jpeg_src.mkdir()
        base = np.tile(np.arange(64, dtype=np.int16)[None, :, None], (48, 1, 3))
        for index in range(3):
            array = np.clip(base + index, 0, 255).astype(np.uint8)
            Image.fromarray(array, "RGB").save(
                jpeg_src / f"{index:02d}.jpg", quality=90
            )

        lossy = self._make("lossy.pxcs", jpeg_src)  # delta_quant=12 -> zlib_residual
        lossy_again = self._make("lossy-again.pxcs", jpeg_src)
        code, out, err = _run([str(lossy), str(lossy_again)])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)

        plain = self._make("plain-delta.pxcs", jpeg_src, delta_quant=0)  # pxc_residual
        with ArchiveReader(lossy) as reader:
            self.assertIn("zlib_residual", {entry.codec for entry in reader.entries})
        with ArchiveReader(plain) as reader:
            self.assertIn("pxc_residual", {entry.codec for entry in reader.entries})

        report = compare_archives(lossy, plain)
        self.assertFalse(report.equivalent)
        categories = {diff.category for diff in report.errors}
        self.assertIn("entry_codec", categories)
        self.assertIn("entry_quant", categories)
        self.assertIn("entry_lossy", categories)
        # 残差条目本身也要被定位（key 是 00.jpg，残差从 01.jpg 开始）
        self.assertTrue(any(diff.entry == "01.jpg" for diff in report.errors))

    def test_entry_set_difference_is_error(self) -> None:
        first = self._make("first.pxcs")
        extra_src = self.tmp / "extra"
        shutil.copytree(self.src, extra_src)
        (extra_src / "notes.txt").write_text("extra file\n" * 10, encoding="utf-8")
        second = self._make("second.pxcs", extra_src)

        code, out, err = _run([str(first), str(second), "--json"])
        self.assertEqual(code, EXIT_DIFFERENT, out + err)
        data = json.loads(out)
        categories = {
            item["category"]
            for item in data["differences"]
            if item["severity"] == "ERROR"
        }
        self.assertIn("entry_count", categories)
        self.assertIn("entry_extra", categories)
        self.assertTrue(
            any(
                item.get("entry") == "notes.txt"
                for item in data["differences"]
                if item["severity"] == "ERROR"
            )
        )

    # ------------------------------------------------------------ 加密
    def test_missing_password_is_usage_error(self) -> None:
        password = "compare-tool-passphrase"
        first = self._make_encrypted("enc-first.pxcs", password)
        second = self._make_encrypted("enc-second.pxcs", password)

        code, out, err = _run([str(first), str(second)])
        self.assertEqual(code, EXIT_USAGE_ERROR, out + err)
        self.assertIn("口令", err)

        # --json 下也要给出结构化错误
        code, out, _ = _run([str(first), str(second), "--json"])
        self.assertEqual(code, EXIT_USAGE_ERROR)
        data = json.loads(out)
        self.assertEqual(data["exit_code"], EXIT_USAGE_ERROR)
        self.assertIn("error", data)

    def test_encrypted_archives_with_password_are_equivalent(self) -> None:
        password = "compare-tool-passphrase"
        first = self._make_encrypted("enc-first.pxcs", password)
        second = self._make_encrypted("enc-second.pxcs", password)

        password_file = self.tmp / "password.txt"
        password_file.write_text(password + "\n", encoding="utf-8")
        code, out, err = _run(
            [str(first), str(second), "--password-file", str(password_file)]
        )
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)
        self.assertIn("已加密", out)

        with mock.patch("sys.stdin", io.StringIO(password + "\n")):
            code, out, err = _run([str(first), str(second), "--password-stdin"])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)

        # 错误口令属于输入问题:退出码 2
        wrong_file = self.tmp / "wrong.txt"
        wrong_file.write_text("totally-wrong-password\n", encoding="utf-8")
        code, out, err = _run(
            [str(first), str(second), "--password-file", str(wrong_file)]
        )
        self.assertEqual(code, EXIT_USAGE_ERROR, out + err)

    # ------------------------------------------------------------ 分卷
    def test_volume_archives_are_equivalent(self) -> None:
        first = self._make("vol-first.pxcs", volume_size=4096)
        second = self._make("vol-second.pxcs", volume_size=4096)
        self.assertTrue(volume_path(first, 2).is_file())
        self.assertGreater(len(list(self.tmp.glob("vol-first.pxcs.*"))), 1)

        # 任意一卷作为输入都要能自动找到整套
        code, out, err = _run([str(volume_path(first, 2)), str(second)])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)
        self.assertIn("分卷", out)

        # 分卷归档同样能读到原始 header/footer(CRC 差异应被识别为 WARN)
        report = compare_archives(volume_path(first, 2), second)
        categories = {diff.category for diff in report.warnings}
        self.assertIn("header_crc", categories)
        self.assertIn("footer_crc", categories)
        self.assertIn("index_crc", categories)

        # base 路径(文件本身不存在)同样可用
        code, out, err = _run([str(first), str(volume_path(second, 2))])
        self.assertEqual(code, EXIT_EQUIVALENT, out + err)

    # ------------------------------------------------------------ 参数/IO
    def test_missing_archive_is_usage_error(self) -> None:
        code, out, err = _run(
            [str(self.tmp / "missing.pxcs"), str(self.tmp / "missing2.pxcs")]
        )
        self.assertEqual(code, EXIT_USAGE_ERROR, out + err)
        self.assertIn("[FAIL]", err)

    def test_help_exits_zero(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(["--help"])
        self.assertEqual(context.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
