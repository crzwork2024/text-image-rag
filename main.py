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

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("API")

parent_store = {}

class QueryRequest(BaseModel):
    prompt: str
    use_rerank: bool = True  # 新增字段，默认为 True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global parent_store
    logger.info("=== [System Startup] ===")
    if config.PARENT_STORE_PATH.exists():
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
    if vector_db.count() == 0:
        run_ingestion()
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
    yield
    logger.info("=== [System Shutdown] ===")

app = FastAPI(title="Professional RAG API", lifespan=lifespan)

@app.post("/query")
async def query_rag(req: QueryRequest):
    logger.info(f"===> [新查询] 模式: {'精排' if req.use_rerank else '直取'} | 问题: '{req.prompt[:50]}'")
    
    # 1. 向量检索 (Recall 10)
    query_vec = embedding_engine.encode([req.prompt])
    results = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)
    
    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    # 2. 向量初筛
    candidates = []
    candidates_meta = []
    logger.info(f"步骤 1: 向量初筛 (Threshold: {config.VECTOR_SEARCH_THRESHOLD})")
    
    for i in range(len(raw_docs)):
        sim = 1 - raw_dists[i]
        sim_pct = f"{round(sim * 100, 2)}%"
        h = raw_metas[i].get("parent_hash", "N/A")
        
        if sim >= config.VECTOR_SEARCH_THRESHOLD:
            logger.info(f"  [√] 片段 #{i+1:02d}: {sim_pct} (通过) [Hash: {h}]")
            candidates.append(raw_docs[i])
            candidates_meta.append(raw_metas[i])
            # 如果不使用精排，且已经拿够了 Top-K，可以提前记录，但为了日志完整通常跑完循环
        else:
            logger.warning(f"  [×] 片段 #{i+1:02d}: {sim_pct} (屏蔽) [Hash: {h}]")

    if not candidates:
        return {"answer": "未找到相关内容。", "best_score": "0%", "sources_count": 0}

    final_hashes = []
    score_summaries = []

    # 3. 核心分歧点：是否使用 Rerank
    if req.use_rerank:
        logger.info(f"步骤 2: 执行 Rerank 精排 (输入数: {len(candidates)})")
        rerank_data = rerank_engine.rerank(req.prompt, candidates)
        
        if rerank_data:
            for i, res in enumerate(rerank_data):
                orig_idx = res["index"]
                score = res["relevance_score"]
                p_hash = candidates_meta[orig_idx].get("parent_hash", "N/A")
                
                # 人性化索引：orig_idx + 1 对应初筛的 #序号
                human_idx = orig_idx + 1
                score_pct = f"{round(score * 100, 2)}%"
                
                logger.info(f"  [精排名次 {i+1}] 对应初筛 #{human_idx:02d}, 得分: {score_pct} [Hash: {p_hash}]")
                score_summaries.append({"rank": i+1, "rerank_score": score_pct})
                
                if score >= config.RERANK_THRESHOLD:
                    final_hashes.append(p_hash)
        else:
            req.use_rerank = False # 自动降级

    # 4. 非精排模式或降级模式
    if not req.use_rerank:
        logger.info("步骤 2: 跳过精排，直接取向量初筛前 3 名")
        for i in range(min(3, len(candidates_meta))):
            p_hash = candidates_meta[i].get("parent_hash", "N/A")
            sim = 1 - raw_dists[i]
            logger.info(f"  [直接采用] 对应初筛 #{i+1:02d}, 相似度: {round(sim*100,2)}% [Hash: {p_hash}]")
            final_hashes.append(p_hash)
            score_summaries.append({"rank": i+1, "rerank_score": f"{round(sim*100,2)}%"})

    # 5. 获取文本并回答
    unique_hashes = list(dict.fromkeys(final_hashes))
    retrieved_sections = [parent_store.get(h) for h in unique_hashes if parent_store.get(h)]
    
    logger.info(f"步骤 3: 最终提取上下文块: {len(retrieved_sections)}")
    
    if not retrieved_sections:
        return {"answer": "匹配度不足。", "best_score": "N/A", "sources_count": 0}

    try:
        answer = call_llm("\n\n---\n\n".join(retrieved_sections), req.prompt)
        logger.info("===> [查询成功] <===")
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        answer = "AI 生成失败。"

    return {
        "answer": answer, 
        "best_score": score_summaries[0]["rerank_score"] if score_summaries else "0%",
        "sources_count": len(retrieved_sections)
    }

# 挂载...
if config.IMAGE_DIR.exists(): app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")
if config.STATIC_DIR.exists(): app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)