import logging
import logging.config
import os
import structlog
import sys

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

def setup_logging():
    '''
    初始化日志系统，配置日志格式和输出方式
    '''
    # 1. 定义共用的处理器
    shared_processors = [
        structlog.stdlib.add_log_level,                 # 添加日志级别 (info, error)
        structlog.stdlib.add_logger_name,               # 添加 Logger 名称
        structlog.processors.TimeStamper(fmt="iso"),    # ISO 8601 格式时间戳
        structlog.processors.StackInfoRenderer(),       # 错误发生时的调用栈
        structlog.processors.format_exc_info,           # 格式化 Exception
        structlog.processors.UnicodeDecoder(),          # 统一字符编码
    ]

    # 2. 配置标准 logging 模块
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console_formatter": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=True),
                ],
                "foreign_pre_chain": shared_processors,
            },
            "json_formatter": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(ensure_ascii=False),
                ],
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            # 控制台静默 — 避免干扰流式输出。日志全部走 DailyDirectoryHandler 落盘。
            "console": {
                "class": "logging.NullHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": "INFO",
            },
            # 第三方库静默 — 避免 httpx/openai/chromadb 的请求日志污染终端
            "httpx": {"level": "WARNING"},
            "openai": {"level": "WARNING"},
            "chromadb": {"level": "WARNING"},
            "bm25s": {"level": "WARNING"},
        }
    })

    # 按日期目录写入（dictConfig 不支持自定义类，手动挂到 root logger）
    from utils.daily_handler import DailyDirectoryHandler

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        foreign_pre_chain=shared_processors,
    )

    for h in (
        DailyDirectoryHandler(base_dir="logs", level=logging.NOTSET),   # all.log
        DailyDirectoryHandler(base_dir="logs", level=logging.WARNING),  # warning.log
        DailyDirectoryHandler(base_dir="logs", level=logging.ERROR),    # error.log
    ):
        h.setFormatter(json_formatter)
        logging.root.addHandler(h)

    # 3. 配置 Structlog 顶层包装
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# 在模块导入时自动执行配置
setup_logging()

# 导出一个便捷获取 logger 的函数
def get_logger(name: str = __name__):
    return structlog.get_logger(name)