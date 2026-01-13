import json
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import config
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from core.llm_client import call_llm
from ingest import run_ingestion

# Setup Logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("API")

app = FastAPI(title="Professional RAG API")
parent_store = {}

class QueryRequest(BaseModel):
    prompt: str

@app.on_event("startup")
def startup_event():
    global parent_store
    
    if config.PARENT_STORE_PATH.exists():
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
        logger.info("Parent store loaded into memory.")

    if vector_db.count() == 0:
        logger.info("Vector database is empty. Triggering initial ingestion...")
        run_ingestion()
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)

@app.post("/query")
async def query_rag(req: QueryRequest):
    logger.info(f"--- [新查询] 用户问题: '{req.prompt[:50]}...' ---")
    
    # 1. 向量化
    query_vec = embedding_engine.encode([req.prompt])
    
    # 2. 检索 ChromaDB
    results = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)
    
    distances = results.get("distances", [[]])[0]
    metadatas = results["metadatas"][0]
    
    logger.info(f"检索完成，ChromaDB 返回了 {len(distances)} 条候选片段")

    # 3. 过滤逻辑与日志记录
    filtered_hashes = []
    score_summaries = []
    
    for i, dist in enumerate(distances):
        similarity = 1 - dist
        similarity_pct = round(similarity * 100, 2)
        
        # 记录每条结果的分数到日志
        logger.info(f"候选片段 {i+1}: 相似度 {similarity_pct}% (阈值: {round(config.SIMILARITY_THRESHOLD * 100)}%)")
        
        score_summaries.append({
            "rank": i + 1,
            "similarity_score": f"{similarity_pct}%"
        })
        
        # 只有超过阈值才加入处理队列
        if similarity >= config.SIMILARITY_THRESHOLD:
            filtered_hashes.append(metadatas[i]["parent_hash"])
        else:
            logger.warning(f"  --> 片段 {i+1} 分数过低，已排除。")

    # 4. 检查拦截条件
    unique_hashes = list(dict.fromkeys(filtered_hashes))
    logger.info(f"Unique Parent Hashes retrieved: {unique_hashes}")
    retrieved_sections = [parent_store.get(h) for h in unique_hashes if parent_store.get(h)]

    if not retrieved_sections:
        # --- 核心拦截点：节省 Token，保护隐私，防止幻觉 ---
        logger.error("!!! [拦截] 所有检索片段均低于阈值，不调用 LLM !!!")
        return {
            "answer": "抱歉，在书库中未找到与您问题相关的核心内容。请尝试使用更具体的术语，或调整您的搜索范围。", 
            "scores": score_summaries,
            "best_similarity": score_summaries[0]["similarity_score"] if score_summaries else "0%",
            "sources_count": 0
        }

    # 5. 调用 LLM
    logger.info(f"找到 {len(retrieved_sections)} 条有效上下文，正在调用 LLM 生成答案...")
    context_text = "\n\n---\n\n".join(retrieved_sections)
    
    try:
        answer = call_llm(context_text, req.prompt)
        logger.info("LLM 响应成功。")
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        answer = "AI 暂时无法回答，请稍后再试。"

    return {
        "answer": answer, 
        "scores": score_summaries,
        "best_similarity": score_summaries[0]["similarity_score"],
        "sources_count": len(retrieved_sections)
    }

if config.IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")
    logger.info(f"Mounted image directory: {config.IMAGE_DIR}")


if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")
    logger.info(f"Mounted static frontend: {config.STATIC_DIR}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)