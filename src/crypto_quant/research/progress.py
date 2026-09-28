"""Append-only, terminal-visible progress for research runs.

Progress is operational telemetry, kept beside sealed run directories so later
updates cannot change frozen research evidence.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path


_active_progress: ContextVar[ProgressLog | None] = ContextVar("active_research_progress", default=None)


class ProgressLog:
    def __init__(self, path: Path, *, heartbeat_seconds: float = 30):
        self.path = Path(path)
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._announced = False

    @classmethod
    def for_run(cls, root: Path) -> ProgressLog:
        root = Path(root)
        if root.parent.name == "runs" and (root.parent.parent / "goal.json").is_file():
            return cls(root.parent.parent / "run_progress" / f"{root.name}.jsonl")
        return cls(root.parent / f"{root.name}.progress.jsonl")

    @staticmethod
    def current() -> ProgressLog | None:
        return _active_progress.get()

    def emit(self, event: str, step: str, **details) -> None:
        # Call sites supply only fixed labels/counts. Never write prompts, replies,
        # exception messages, credentials, or arbitrary model diagnostics here.
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
                  "step": step, **details}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        labels = {"started": "开始", "heartbeat": "进行中", "completed": "完成",
                  "failed": "失败", "interrupted": "中断", "retry": "重试"}
        extra = " ".join(f"{key}={value}" for key, value in details.items())
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            if not self._announced:
                print(f"[进度] 日志：{self.path.resolve()}", file=sys.stderr, flush=True)
                self._announced = True
            clock = datetime.now().astimezone().strftime("%H:%M:%S")
            print(f"{clock} [进度] {labels[event]} {step}" + (f" {extra}" if extra else ""),
                  file=sys.stderr, flush=True)

    @contextmanager
    def span(self, step: str, *, heartbeat: bool = False, **details):
        started = time.monotonic()
        self.emit("started", step, **details)
        token = _active_progress.set(self)
        stop = threading.Event()
        thread = None
        if heartbeat:
            def pulse():
                while not stop.wait(self.heartbeat_seconds):
                    self.emit("heartbeat", step, elapsed_seconds=round(time.monotonic() - started, 3), **details)

            thread = threading.Thread(target=pulse, name="research-progress", daemon=True)
            thread.start()
        event, error_type = "completed", None
        try:
            yield
        except BaseException as exc:
            event = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error_type = type(exc).__name__
            raise
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            fields = {"elapsed_seconds": round(time.monotonic() - started, 3), **details}
            if error_type is not None:
                fields["error_type"] = error_type
            try:
                self.emit(event, step, **fields)
            finally:
                _active_progress.reset(token)

    def track(self, step: str, action, *, heartbeat: bool = True):
        def tracked(state):
            with self.span(step, heartbeat=heartbeat):
                return action(state)
        return tracked
