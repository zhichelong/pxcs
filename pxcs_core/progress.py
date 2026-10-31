"""PXCS 创建过程中的终端进度显示。

默认只显示一个原地刷新的进度块；--details 时在下方滚动显示最近 N 条
[KEY]/[DELT]/[RAW] 等详情。重定向到文件时降级为每 5 秒一行，避免刷屏。
"""
from __future__ import annotations

import sys
import time
from collections import deque
from typing import Optional, TextIO

from pxc_core.i18n import t
from pxc_core.utils import human_size


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressDisplay:
    def __init__(
        self,
        title: str,
        total_files: int,
        total_bytes: int,
        *,
        enabled: bool = True,
        details: bool = False,
        detail_lines: int = 3,
        refresh_seconds: float = 0.2,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.title = title
        self.total_files = max(0, total_files)
        self.total_bytes = max(0, total_bytes)
        self.enabled = bool(enabled)
        self.details = bool(details)
        self.detail_lines = max(0, int(detail_lines))
        self.refresh_seconds = max(0.05, float(refresh_seconds))
        self.stream = stream or sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._start = time.monotonic()
        self._last_render = 0.0
        self._drawn_lines = 0
        self._detail_queue: deque[str] = deque(maxlen=self.detail_lines or 1)
        self._done = 0
        self._current = ""
        self._processed = 0
        self._archive = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._render(force=True)

    def detail(self, message: str) -> None:
        if not self.enabled or not self.details or self.detail_lines <= 0:
            return
        self._detail_queue.append(message)

    def update(
        self,
        done: int,
        current: str,
        processed_bytes: int,
        archive_bytes: int,
    ) -> None:
        self._done = done
        self._current = current
        self._processed = max(0, processed_bytes)
        self._archive = max(0, archive_bytes)
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_render < self.refresh_seconds and done < self.total_files:
            return
        self._render()

    def finish(self) -> None:
        if not self.enabled:
            return
        self._render(force=True)
        if self.is_tty and self._drawn_lines:
            self.stream.write("\n")
            self.stream.flush()
            self._drawn_lines = 0

    def _lines(self) -> list[str]:
        elapsed = time.monotonic() - self._start
        pct = 0.0
        if self.total_bytes:
            pct = min(100.0, self._processed * 100.0 / self.total_bytes)
        elif self.total_files:
            pct = min(100.0, self._done * 100.0 / self.total_files)
        filled = int(pct // 5)
        bar = "#" * filled + "-" * (20 - filled)
        speed = self._processed / elapsed if elapsed > 0 else 0.0
        ratio = (
            self._archive * 100.0 / self._processed
            if self._processed > 0
            else 0.0
        )
        lines = [
            t("progress.title", title=self.title),
            t("progress.adding", current=self._current or "..."),
            t("progress.percent", pct=pct, bar=bar),
            t("progress.files", done=self._done, total=self.total_files),
            t("progress.speed", speed=human_size(int(speed))),
            t("progress.total", total=human_size(self.total_bytes)),
            t("progress.processed", processed=human_size(self._processed)),
            t("progress.archive", archive=human_size(self._archive)),
            t("progress.ratio", ratio=ratio),
            t("progress.elapsed", elapsed=_elapsed_text(elapsed)),
            t("progress.cancel_hint"),
        ]
        if self.details and self._detail_queue:
            lines.extend(self._detail_queue)
        return lines

    def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < self.refresh_seconds:
            return
        lines = self._lines()
        if self.is_tty:
            if self._drawn_lines:
                self.stream.write("\x1b[1A\x1b[2K" * self._drawn_lines)
            self.stream.write("\n".join(lines) + "\n")
            self.stream.flush()
            self._drawn_lines = len(lines)
        else:
            # 非 TTY 不做光标回退；每 5 秒或结束时打一块。
            if force or now - self._last_render >= 5.0:
                self.stream.write("\n".join(lines) + "\n")
                self.stream.flush()
        self._last_render = now
