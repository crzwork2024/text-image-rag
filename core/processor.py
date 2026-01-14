"""
文档处理模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：处理 Markdown 文档，分割为可检索的文本块
"""

import re
import uuid
import json
import hashlib
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from config import config
from utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


def get_hash(text: str) -> str:
    """
    计算文本的 MD5 哈希值

    参数:
        text: 输入文本

    返回:
        MD5 哈希字符串
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class DocumentProcessor:
    """文档处理器 - 负责文档分块和预处理"""

    @staticmethod
    def process_markdown_to_chunks(
        md_text: str,
        export_debug: bool = True
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        将 Markdown 文档分割为父节点（章节）和子节点（段落）

        处理流程:
        1. 按标题分割文档为多个章节（父节点）
        2. 每个章节按段落分割为多个文本块（子节点）
        3. 为每个父节点生成哈希值作为标识
        4. 子节点保存父节点哈希的引用

        参数:
            md_text: Markdown 文档文本
            export_debug: 是否导出调试文件

        返回:
            (向量项列表, 父节点映射字典)
            - 向量项: 包含 id, text, metadata 的字典列表
            - 父节点映射: 哈希值到完整章节文本的映射

        异常:
            DocumentProcessingError: 文档处理失败时抛出
        """
        try:
            logger.info("开始处理 Markdown 文档")

            # 按标题分割文档（父节点）
            # 正则表达式匹配 Markdown 标题: # 或 ## 或 ### 等
            sections = re.split(r'\n(?=#+ )', md_text)
            vector_items = []
            parent_map = {}

            logger.info(f"文档分割为 {len(sections)} 个章节")

            for idx, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue

                # 为章节生成唯一哈希标识
                section_hash = get_hash(section)
                parent_map[section_hash] = section

                # 按双换行符分割段落（子节点）
                paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]

                # 跟踪有效段落数
                valid_paragraphs = 0

                for para in paragraphs:
                    # 跳过标题行（已包含在父节点中）
                    if para.startswith('#'):
                        continue

                    # 跳过过短的段落（可能是噪声）
                    if len(para) < 10:
                        continue

                    # 创建向量项
                    vector_items.append({
                        "id": str(uuid.uuid4()),
                        "text": para,
                        "metadata": {
                            "parent_hash": section_hash,
                            "section_index": idx
                        }
                    })
                    valid_paragraphs += 1

                logger.debug(
                    f"章节 {idx + 1}: "
                    f"提取 {valid_paragraphs} 个有效段落 "
                    f"(Hash: {section_hash[:8]}...)"
                )

            logger.info(f"文档处理完成: {len(vector_items)} 个文本块, {len(parent_map)} 个章节")

            # 导出调试文件
            if export_debug:
                DocumentProcessor._export_debug_chunks(vector_items)

            return vector_items, parent_map

        except Exception as e:
            error_msg = f"文档处理失败: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg, details=str(e))

    @staticmethod
    def _export_debug_chunks(vector_items: List[Dict]):
        """
        导出调试用的文本块文件

        参数:
            vector_items: 向量项列表
        """
        try:
            debug_path = config.DEBUG_EXPORT_PATH
            logger.info(f"导出调试文件到: {debug_path}")

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(vector_items, f, ensure_ascii=False, indent=2)

            logger.debug(f"调试文件导出成功，大小: {Path(debug_path).stat().st_size} 字节")

        except Exception as e:
            logger.warning(f"导出调试文件失败: {e}")

    @staticmethod
    def validate_markdown(md_text: str) -> bool:
        """
        验证 Markdown 文档的有效性

        参数:
            md_text: Markdown 文本

        返回:
            True 如果文档有效，否则 False
        """
        if not md_text or len(md_text.strip()) == 0:
            logger.error("Markdown 文档为空")
            return False

        # 检查是否包含标题
        if not re.search(r'^#+ ', md_text, re.MULTILINE):
            logger.warning("Markdown 文档不包含任何标题")
            return False

        return True


# 向后兼容的函数接口
def process_markdown_to_chunks(md_text: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    处理 Markdown 文档为文本块（向后兼容接口）

    参数:
        md_text: Markdown 文档文本

    返回:
        (向量项列表, 父节点映射字典)
    """
    return DocumentProcessor.process_markdown_to_chunks(md_text)
