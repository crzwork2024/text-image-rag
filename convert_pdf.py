"""
PDF 转换快捷脚本 - RAG 智能问答系统
作者: RAG 项目团队
描述: 便捷的命令行工具，用于将 PDF 转换为 Markdown 并准备导入
"""

import sys
from pathlib import Path
from pdf_converter import PDFConverter, main as converter_main

if __name__ == "__main__":
    # 直接调用 pdf_converter 的 main 函数
    converter_main()
