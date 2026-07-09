"""MCI World Model 统一日志配置。

生产环境使用 JSON 结构化日志，开发环境使用人类可读格式。
通过环境变量 MCI_LOG_LEVEL 和 MCI_LOG_FORMAT 控制。

用法:
    from mci_world_model._logging import setup_logging
    setup_logging()  # 在应用入口调用一次

    # 各模块中:
    import logging
    logger = logging.getLogger(__name__)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON 结构化日志格式，便于 ELK/Loki 等日志系统采集。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """配置全局日志。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)。默认从 MCI_LOG_LEVEL 环境变量读取。
        json_format: 是否使用 JSON 格式。默认从 MCI_LOG_FORMAT 环境变量判断。
    """
    if level is None:
        level = os.environ.get("MCI_LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.environ.get("MCI_LOG_FORMAT", "json") == "json"

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
