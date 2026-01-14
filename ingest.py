"""
数据摄取模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：处理文档导入，包括文档分块、向量化和存储
"""

import json
import logging
from pathlib import Path
from config import config
from core.processor import process_markdown_to_chunks
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from utils.logger import setup_logger
from utils.exceptions import DocumentProcessingError

# 初始化日志
logger = setup_logger(
    "Ingestion",
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_dir=config.LOG_DIR
)


def run_ingestion(
    md_file_path: Path = None,
    force_reingest: bool = False
) -> bool:
    """
    执行文档摄取流程

    流程:
    1. 读取 Markdown 文档
    2. 分割文档为文本块
    3. 生成文本向量
    4. 存储到向量数据库
    5. 保存父节点映射

    参数:
        md_file_path: Markdown 文件路径，默认使用配置值
        force_reingest: 是否强制重新摄取（清空现有数据）

    返回:
        True 如果摄取成功，否则 False

    异常:
        DocumentProcessingError: 文档处理失败时抛出
    """
    logger.info("=" * 60)
    logger.info("开始数据摄取流程")
    logger.info("=" * 60)

    # 使用默认路径
    if md_file_path is None:
        md_file_path = config.MD_FILE_PATH

    try:
        # 步骤 1: 验证文件存在
        if not md_file_path.exists():
            error_msg = f"Markdown 文件不存在: {md_file_path}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

        logger.info(f"读取 Markdown 文件: {md_file_path}")
        file_size = md_file_path.stat().st_size / 1024  # KB
        logger.info(f"文件大小: {file_size:.2f} KB")

        # 步骤 2: 读取文件内容
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"文件内容长度: {len(content)} 字符")

        # 步骤 3: 处理文档分块
        logger.info("-" * 60)
        logger.info("步骤 1/4: 文档分块处理")
        logger.info("-" * 60)

        vector_items, parent_store = process_markdown_to_chunks(content)

        logger.info(f"✓ 生成 {len(vector_items)} 个文本块")
        logger.info(f"✓ 生成 {len(parent_store)} 个父节点映射")

        # 步骤 4: 保存父节点映射
        logger.info("-" * 60)
        logger.info("步骤 2/4: 保存父节点映射")
        logger.info("-" * 60)

        with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 父节点映射已保存: {config.PARENT_STORE_PATH}")

        # 步骤 5: 生成向量
        logger.info("-" * 60)
        logger.info("步骤 3/4: 生成文本向量")
        logger.info("-" * 60)

        documents = [item["text"] for item in vector_items]
        logger.info(f"正在为 {len(documents)} 个文本块生成向量...")

        embeddings = embedding_engine.encode(
            documents,
            batch_size=32,
            show_progress_bar=True
        )

        logger.info(f"✓ 向量生成完成，维度: {len(embeddings[0]) if embeddings else 0}")

        # 步骤 6: 存储到向量数据库
        logger.info("-" * 60)
        logger.info("步骤 4/4: 存储到向量数据库")
        logger.info("-" * 60)

        # 如果强制重新摄取，先清空数据库
        if force_reingest:
            logger.warning("强制重新摄取模式：清空现有数据")
            vector_db.reset()

        vector_db.add_documents(
            ids=[item["id"] for item in vector_items],
            embeddings=embeddings,
            documents=documents,
            metadatas=[item["metadata"] for item in vector_items]
        )

        logger.info(f"✓ 数据已存储到 ChromaDB")
        logger.info(f"✓ 当前数据库文档总数: {vector_db.count()}")

        # 完成
        logger.info("=" * 60)
        logger.info("数据摄取流程完成！")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"数据摄取失败: {str(e)}")
        logger.error("=" * 60)
        raise


def main():
    """主函数 - 命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 文档摄取工具")
    parser.add_argument(
        "--file",
        type=str,
        help="Markdown 文件路径（默认使用配置值）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新摄取（清空现有数据）"
    )

    args = parser.parse_args()

    # 解析文件路径
    md_file_path = Path(args.file) if args.file else None

    try:
        success = run_ingestion(
            md_file_path=md_file_path,
            force_reingest=args.force
        )

        if success:
            logger.info("摄取成功，系统已就绪")
            exit(0)
        else:
            logger.error("摄取失败")
            exit(1)

    except Exception as e:
        logger.error(f"程序异常: {e}")
        exit(1)


if __name__ == "__main__":
    main()
