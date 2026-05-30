"""按日期+启动批次滚动的日志 Handler。

每次进程启动创建独立子目录: logs/YYYY-MM-DD/HH-MM-SS/
按级别分文件: all.log / warning.log / error.log
"""
import os
import logging
from datetime import datetime

# 级别 → 文件名
_LEVEL_FILES = {
    logging.NOTSET: "all.log",
    logging.WARNING: "warning.log",
    logging.ERROR: "error.log",
}


class DailyDirectoryHandler(logging.Handler):
    """按日期目录 + 启动批次组织日志文件，按级别分流。"""

    def __init__(self, base_dir: str = "logs", level: int = logging.NOTSET):
        super().__init__()
        self._file = None
        self.setLevel(level)
        # 进程启动时锁定一次 run_dir，之后不再变化
        now = datetime.now()
        self._run_dir = os.path.join(
            base_dir,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H-%M-%S"),
        )
        os.makedirs(self._run_dir, exist_ok=True)
        filename = _LEVEL_FILES.get(level, "all.log")
        self._path = os.path.join(self._run_dir, filename)

    def _ensure_file(self):
        if self._file is None:
            self._file = open(self._path, "a", encoding="utf-8")

    def emit(self, record):
        try:
            self._ensure_file()
            msg = self.format(record)
            if msg:
                self._file.write(msg + "\n")
                self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._file:
            self._file.close()
        super().close()
