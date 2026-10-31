"""统一的本地检查入口。

一条命令完成"这个仓库是否健康"的全部检查：

    python -m tools.check              # 单元测试 + 黄金向量字节级自检
    python -m tools.check --coverage   # 额外统计行覆盖率
    python -m tools.check --fast       # 只跑单元测试

退出码：0 = 全绿；1 = 有失败。

之所以需要它：这个项目是本地单机工具，不接 CI，靠一条可复现的命令保证
"任何改动都能被一套固定检查验证"。将来若接入 CI，直接调用本脚本即可。
"""
from __future__ import annotations

import argparse
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 覆盖率统计范围：整个仓库，但排除测试、工具与打包产物。
# 必须用绝对路径：coverage 按 cwd 解析相对 name，容易匹配不到而报 0%。
COVERAGE_SOURCE = str(ROOT)
COVERAGE_OMIT = ("tests/*", "tools/*", "build/*", "dist/*", "*.spec")


def _run_unittests(verbosity: int = 1) -> bool:
    """跑 tests/ 下的全部测试，返回是否全绿。"""
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    if getattr(loader, "errors", None):
        for error in loader.errors:
            print(error, file=sys.stderr)
        return False
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    result = runner.run(suite)
    text = stream.getvalue()
    # 只回显失败详情与最终统计，避免刷屏
    lines = text.splitlines()
    for line in lines:
        if line.startswith(("FAIL:", "ERROR:", "Ran ", "OK", "FAILED")):
            print(line)
    if not result.wasSuccessful():
        for case, trace in list(result.failures) + list(result.errors):
            print(f"\n--- {case} ---")
            print(trace)
    return result.wasSuccessful()


def _run_compare_selfcheck() -> bool:
    """用同一批输入创建两次归档，确认对比工具判定"语义等价"。

    这条检查同时守住两个契约：
    - 同一批输入两次创建，除 `archive_id` 外完全确定性；
    - 跨实现比对的正确标准是 payload 级，而不是整文件哈希。
    """
    import contextlib
    import tempfile

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow 是运行时依赖
        print("未安装 Pillow，跳过对比自检", file=sys.stderr)
        return True

    from pxcs_core.archive import CreateOptions, create_archive
    from tools import pxcs_compare

    with tempfile.TemporaryDirectory(prefix="pxcs-check-") as tmp:
        root = Path(tmp)
        src = root / "in"
        src.mkdir()
        for index in range(2):
            img = Image.new("RGB", (32, 24), (index * 40, 60, 90))
            img.save(src / f"g{index}.jpg", format="JPEG", quality=90)
        first = root / "a.pxcs"
        second = root / "b.pxcs"
        for target in (first, second):
            create_archive(
                target, [str(src)], CreateOptions(quiet=True, progress=False)
            )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = pxcs_compare.main([str(first), str(second), "--json"])
    if code == pxcs_compare.EXIT_EQUIVALENT:
        return True
    print(f"对比自检失败（退出码 {code}）：")
    print(buffer.getvalue())
    return False


def _run_golden_check() -> bool:
    """校验黄金向量与当前实现是否仍然字节一致。"""
    from tools import gen_golden

    failures = gen_golden._check_vectors()
    return failures == 0


def _run_coverage() -> bool:
    """跑一遍测试并报告产品代码的行覆盖率。"""
    try:
        import coverage
    except ImportError:
        print(
            "未安装 coverage，请先执行：python -m pip install coverage",
            file=sys.stderr,
        )
        return False

    cov = coverage.Coverage(
        source=[COVERAGE_SOURCE], omit=list(COVERAGE_OMIT), branch=True
    )
    cov.start()
    try:
        ok = _run_unittests(verbosity=0)
    finally:
        cov.stop()
        cov.save()

    print("\n覆盖率（行 + 分支）：")
    buffer = io.StringIO()
    cov.report(file=buffer, show_missing=False)
    print(buffer.getvalue())
    cov.erase()
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PXCS 本地统一检查")
    parser.add_argument("--coverage", action="store_true", help="统计行覆盖率")
    parser.add_argument("--fast", action="store_true", help="只跑单元测试")
    parser.add_argument(
        "--verbose", action="store_true", help="逐条打印测试名（较慢）"
    )
    args = parser.parse_args(argv)

    if args.coverage:
        return 0 if _run_coverage() else 1

    ok = _run_unittests(verbosity=2 if args.verbose else 1)
    if args.fast:
        return 0 if ok else 1

    print("\n黄金向量自检：")
    golden_ok = _run_golden_check()
    print("\n归档对比自检（两次创建应判定语义等价）：")
    compare_ok = _run_compare_selfcheck()
    if compare_ok:
        print("OK: 两次创建语义等价（仅 archive_id 不同）")
    if not ok or not golden_ok or not compare_ok:
        print("\n结果：失败")
        return 1
    print("\n结果：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
