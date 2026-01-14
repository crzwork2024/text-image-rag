"""
主应用模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：FastAPI 应用主入口，提供问答 API 和 Web 界面
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from core.llm_client import call_llm
from core.reranker import rerank_engine
from ingest import run_ingestion
from utils.logger import setup_logger
from utils.responses import QueryResponse, error_response
from utils.exceptions import RetrievalError, LLMAPIError

# 初始化日志
logger = setup_logger(
    "API",
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_dir=config.LOG_DIR
)

# 全局变量：父节点存储映射
parent_store = {}


class QueryRequest(BaseModel):
    """查询请求模型"""
    prompt: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    use_rerank: bool = Field(True, description="是否使用重排优化")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时:
    - 加载父节点映射
    - 检查向量数据库，如果为空则自动执行摄取

    关闭时:
    - 清理资源
    """
    global parent_store

    logger.info("=" * 60)
    logger.info("系统启动中...")
    logger.info("=" * 60)

    try:
        # 创建必要的目录
        config.create_directories()

        # 加载父节点映射
        if config.PARENT_STORE_PATH.exists():
            logger.info(f"加载父节点映射: {config.PARENT_STORE_PATH}")
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)
            logger.info(f"✓ 加载 {len(parent_store)} 个父节点")
        else:
            logger.warning("父节点映射文件不存在，将在首次摄取时创建")

        # 检查向量数据库
        doc_count = vector_db.count()
        logger.info(f"向量数据库文档数量: {doc_count}")

        if doc_count == 0:
            logger.warning("向量数据库为空，开始自动摄取...")
            run_ingestion()

            # 重新加载父节点映射
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)

            logger.info("✓ 自动摄取完成")

        logger.info("=" * 60)
        logger.info("系统启动完成，服务就绪")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise

    finally:
        logger.info("=" * 60)
        logger.info("系统关闭")
        logger.info("=" * 60)


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 智能问答系统 API",
    description="基于检索增强生成的智能问答服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """
    RAG 问答接口

    流程:
    1. 向量检索：从向量数据库召回候选文档
    2. 初步过滤：根据相似度阈值过滤
    3. 重排（可选）：使用 Rerank 模型精确排序
    4. 上下文组装：根据父节点哈希获取完整章节
    5. LLM 生成：调用大语言模型生成回答

    参数:
        req: 查询请求，包含问题和是否使用重排

    返回:
        QueryResponse: 包含答案、最高分数、来源数量
    """
    mode = "精排模式" if req.use_rerank else "直取模式"
    logger.info("=" * 60)
    logger.info(f"[新查询] {mode}")
    logger.info(f"问题: {req.prompt[:100]}{'...' if len(req.prompt) > 100 else ''}")
    logger.info("=" * 60)

    try:
        # ==================== 步骤 1: 向量检索 ====================
        logger.info(f"步骤 1/5: 向量检索 (召回数: {config.RETRIEVAL_COUNT})")

        query_vec = embedding_engine.encode([req.prompt])
        results = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)

        raw_docs = results["documents"][0]
        raw_metas = results["metadatas"][0]
        raw_dists = results["distances"][0]

        logger.info(f"✓ 召回 {len(raw_docs)} 个候选文档")

        # ==================== 步骤 2: 向量初筛 ====================
        logger.info(f"步骤 2/5: 向量初筛 (阈值: {config.VECTOR_SEARCH_THRESHOLD})")

        candidates = []
        candidates_meta = []

        for i in range(len(raw_docs)):
            sim = 1 - raw_dists[i]
            sim_pct = f"{round(sim * 100, 2)}%"
            h = raw_metas[i].get("parent_hash", "N/A")

            if sim >= config.VECTOR_SEARCH_THRESHOLD:
                logger.debug(f"  [✓] 片段 #{i+1:02d}: {sim_pct} (通过) [Hash: {h[:8]}...]")
                candidates.append(raw_docs[i])
                candidates_meta.append(raw_metas[i])
            else:
                logger.debug(f"  [✗] 片段 #{i+1:02d}: {sim_pct} (过滤) [Hash: {h[:8]}...]")

        logger.info(f"✓ 筛选后剩余 {len(candidates)} 个候选文档")

        # 如果没有候选文档，直接返回
        if not candidates:
            logger.warning("未找到相关内容")
            return QueryResponse(
                answer="抱歉，未找到与您问题相关的内容。",
                best_score="0%",
                sources_count=0
            )

        final_hashes = []
        score_summaries = []

        # ==================== 步骤 3: 重排（可选）====================
        if req.use_rerank and rerank_engine.is_available():
            logger.info(f"步骤 3/5: 执行精排 (候选数: {len(candidates)})")

            rerank_data = rerank_engine.rerank(req.prompt, candidates)

            if rerank_data:
                for i, res in enumerate(rerank_data):
                    orig_idx = res["index"]
                    score = res["relevance_score"]
                    p_hash = candidates_meta[orig_idx].get("parent_hash", "N/A")

                    score_pct = f"{round(score * 100, 2)}%"
                    logger.debug(
                        f"  [排名 {i+1}] 原索引 #{orig_idx+1:02d}, "
                        f"分数: {score_pct} [Hash: {p_hash[:8]}...]"
                    )

                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})

                    # 根据阈值过滤
                    if score >= config.RERANK_THRESHOLD:
                        final_hashes.append(p_hash)

                logger.info(f"✓ 精排完成，保留 {len(final_hashes)} 个高相关文档")
            else:
                logger.warning("精排失败，降级为直取模式")
                req.use_rerank = False

        # ==================== 步骤 4: 直取模式（备选）====================
        if not req.use_rerank or not rerank_engine.is_available():
            logger.info(f"步骤 3/5: 直取模式，选择前 {config.RERANK_TOP_K} 名")

            for i in range(min(config.RERANK_TOP_K, len(candidates_meta))):
                p_hash = candidates_meta[i].get("parent_hash", "N/A")
                sim = 1 - raw_dists[i]
                score_pct = f"{round(sim * 100, 2)}%"

                logger.debug(f"  [采用 {i+1}] 相似度: {score_pct} [Hash: {p_hash[:8]}...]")

                final_hashes.append(p_hash)
                score_summaries.append({"rank": i+1, "rerank_score": score_pct})

            logger.info(f"✓ 直取 {len(final_hashes)} 个文档")

        # ==================== 步骤 5: 获取完整上下文 ====================
        logger.info("步骤 4/5: 组装上下文")

        unique_hashes = list(dict.fromkeys(final_hashes))
        retrieved_sections = [
            parent_store.get(h) for h in unique_hashes
            if parent_store.get(h)
        ]

        logger.info(f"✓ 提取 {len(retrieved_sections)} 个完整章节")

        if not retrieved_sections:
            logger.warning("未能获取有效上下文")
            return QueryResponse(
                answer="抱歉，未能找到足够相关的内容。",
                best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
                sources_count=0
            )

        # ==================== 步骤 6: LLM 生成回答 ====================
        logger.info("步骤 5/5: LLM 生成回答")

        context = "\n\n---\n\n".join(retrieved_sections)
        logger.debug(f"上下文总长度: {len(context)} 字符")

        try:
            answer = call_llm(context, req.prompt)
            logger.info(f"✓ 回答生成成功，长度: {len(answer)} 字符")
        except LLMAPIError as e:
            logger.error(f"LLM 生成失败: {e.message}")
            answer = "抱歉，AI 生成回答时出现错误，请稍后重试。"

        # ==================== 返回结果 ====================
        logger.info("=" * 60)
        logger.info("[查询完成]")
        logger.info("=" * 60)

        return QueryResponse(
            answer=answer,
            best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
            sources_count=len(retrieved_sections)
        )

    except Exception as e:
        logger.error(f"查询处理异常: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                error="查询处理失败",
                details=str(e),
                code="QUERY_ERROR"
            )
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "vector_db_docs": vector_db.count(),
        "parent_store_size": len(parent_store)
    }


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "vector_db": {
            "document_count": vector_db.count(),
            "collection_name": config.CHROMA_COLLECTION_NAME
        },
        "parent_store": {
            "section_count": len(parent_store)
        },
        "config": {
            "retrieval_count": config.RETRIEVAL_COUNT,
            "rerank_top_k": config.RERANK_TOP_K,
            "vector_threshold": config.VECTOR_SEARCH_THRESHOLD,
            "rerank_threshold": config.RERANK_THRESHOLD
        }
    }


# ==================== 静态文件挂载 ====================
# 图片目录
if config.IMAGE_DIR.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(config.IMAGE_DIR)),
        name="images"
    )
    logger.info(f"✓ 挂载图片目录: {config.IMAGE_DIR}")

# 前端静态文件
if config.STATIC_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(config.STATIC_DIR), html=True),
        name="static"
    )
    logger.info(f"✓ 挂载静态文件目录: {config.STATIC_DIR}")


def main():
    """主函数 - 启动服务器"""
    import uvicorn

    logger.info("=" * 60)
    logger.info(f"启动服务器: http://{config.APP_HOST}:{config.APP_PORT}")
    logger.info(f"API 文档: http://{config.APP_HOST}:{config.APP_PORT}/docs")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=config.APP_RELOAD,
        log_level="info"
    )


if __name__ == "__main__":
    main()
