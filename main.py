import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import config
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from core.llm_client import call_llm
from core.reranker import rerank_engine
from ingest import run_ingestion

# Setup Logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("API")

parent_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ 生命周期管理：初始化与资源清理 """
    global parent_store
    logger.info("=== [系统启动] 执行初始化任务 ===")
    
    if config.PARENT_STORE_PATH.exists():
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
        logger.info(f"System: 已加载父文档库，记录数: {len(parent_store)}")

    try:
        if vector_db.count() == 0:
            logger.info("System: 向量库为空，触发自动摄取流程...")
            run_ingestion()
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)
    except Exception as e:
        logger.error(f"System: 启动检查失败: {e}")

    yield
    logger.info("=== [系统关闭] 清理资源 ===")

app = FastAPI(title="Professional RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query_rag(req: QueryRequest):
    logger.info(f"===> [新查询] 用户问题: '{req.prompt[:50]}...' <===")
    
    # --- 阶段 1: 向量召回 (K=10) ---
    logger.info(f"步骤 1: 向量检索 (K={config.RETRIEVAL_COUNT})")
    query_vec = embedding_engine.encode([req.prompt])
    results = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)
    
    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    # --- 阶段 2: 向量初筛 (附带 Hash 记录) ---
    candidates = []
    candidates_meta = []
    logger.info(f"步骤 2: 向量初筛结果 (阈值: {config.VECTOR_SEARCH_THRESHOLD})")
    
    for i in range(len(raw_docs)):
        sim = 1 - raw_dists[i]
        sim_pct = f"{round(sim * 100, 2)}%"
        curr_hash = raw_metas[i].get("parent_hash", "N/A")
        
        if sim >= config.VECTOR_SEARCH_THRESHOLD:
            # 关键日志：添加 [Hash: xxx]
            logger.info(f"  [√] 片段 #{i+1:02d}: 相似度 {sim_pct} (通过) [Hash: {curr_hash}]")
            candidates.append(raw_docs[i])
            candidates_meta.append(raw_metas[i])
        else:
            logger.warning(f"  [×] 片段 #{i+1:02d}: 相似度 {sim_pct} (屏蔽) [Hash: {curr_hash}]")

    if not candidates:
        logger.error("!!! [拦截] 向量阶段无合格候选片段 !!!")
        return {"answer": "书库中未找到相关内容。", "sources_count": 0}

    # --- 阶段 3: Rerank 精排 (附带 Hash 记录) ---
    logger.info(f"步骤 3: 启动 Rerank 精排 (输入数: {len(candidates)})")
    rerank_results = rerank_engine.rerank(req.prompt, candidates)
    
    final_hashes = []
    score_summaries = []

    

    if rerank_results is not None:
        for i, res in enumerate(rerank_results):
            idx = res["index"]
            score = res["relevance_score"]
            p_hash = candidates_meta[idx].get("parent_hash", "N/A")
            
            score_summary = {"rank": i+1, "rerank_score": f"{round(score*100, 2)}%"}
            score_summaries.append(score_summary)
            
            # 关键日志：添加 [Hash: xxx]
            logger.info(f"  [精排 {i+1}] 原始索引: {idx+1}, 得分: {score_summary['rerank_score']} [Hash: {p_hash}]")
            
            if score >= config.RERANK_THRESHOLD:
                final_hashes.append(p_hash)
            else:
                logger.warning(f"    --> 片段 {idx} 分数过低被剔除 [Hash: {p_hash}]")
    else:
        logger.warning("!!! [降级] Rerank 异常，直接使用向量 Top-3 !!!")
        final_hashes = [m["parent_hash"] for m in candidates_meta[:3]]
        score_summaries = [{"info": "System Fallback"}]

    # --- 阶段 4: 获取最终上下文 ---
    unique_hashes = list(dict.fromkeys(final_hashes))
    retrieved_sections = [parent_store.get(h) for h in unique_hashes if parent_store.get(h)]
    
    if not retrieved_sections:
        return {"answer": "检索结果相关度不足。", "scores": score_summaries}

    # --- 阶段 5: LLM 生成答案 ---
    logger.info(f"步骤 4: 调用 LLM 生成答案 (上下文块数: {len(retrieved_sections)})")
    context_text = "\n\n---\n\n".join(retrieved_sections)
    
    try:
        answer = call_llm(context_text, req.prompt)
        logger.info("===> [查询成功] 流程结束 <===")
    except Exception as e:
        logger.error(f"LLM 错误: {e}")
        answer = "AI 助手暂时无法响应。"

    return {
        "answer": answer, 
        "scores": score_summaries,
        "best_score": score_summaries[0].get("rerank_score", "0%") if score_summaries else "0%",
        "sources_count": len(retrieved_sections)
    }

# 静态资源挂载
if config.IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")
if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)