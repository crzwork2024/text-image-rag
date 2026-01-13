import requests
import logging
from config import config

logger = logging.getLogger("Reranker")

class SiliconFlowReranker:
    def __init__(self):
        self.api_key = config.SILICONFLOW_API_KEY
        self.url = config.SILICONFLOW_RERANK_URL
        self.model = config.SILICONFLOW_RERANK_MODEL

    def rerank(self, query: str, documents: list):
        if not documents:
            logger.warning("Reranker: 候选列表为空，跳过调用。")
            return []

        logger.info(f"Reranker: 正在对 {len(documents)} 条文档进行深度打分 (Model: {self.model})...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 注意：top_n 决定了 API 最终返回多少条排序结果
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": config.RERANK_TOP_K
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            results = response.json().get("results", [])
            logger.info(f"Reranker: API 成功返回 {len(results)} 条最高相关性的重排结果。")
            return results
        except Exception as e:
            logger.error(f"Reranker: API 调用故障: {e}")
            return None

rerank_engine = SiliconFlowReranker()