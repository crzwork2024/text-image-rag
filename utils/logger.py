"""
日志管理模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：配置和管理系统日志记录
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        """格式化日志记录"""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_format: str = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s',
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    设置日志记录器

    参数:
        name: 日志记录器名称
        log_level: 日志级别
        log_format: 日志格式
        log_dir: 日志文件目录
        enable_file_logging: 是否启用文件日志

    返回:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 清除现有的处理器
    logger.handlers.clear()

    # 控制台处理器（彩色输出）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if enable_file_logging and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        # 按日期创建日志文件
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file_path = log_dir / log_filename

        file_handler = logging.FileHandler(
            log_file_path,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    参数:
        name: 日志记录器名称

    返回:
        日志记录器实例
    """
    return logging.getLogger(name)
