import requests
import logging
from config import config

logger = logging.getLogger(__name__)

def call_llm(context: str, user_query: str) -> str:
    headers = {"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}"}
    
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    
    payload = {
        "model": config.SILICONFLOW_MODEL_ID,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE
    }
    
    try:
        logger.info(f"Sending request to SiliconFlow API ({config.SILICONFLOW_MODEL_ID})")
        resp = requests.post(config.SILICONFLOW_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM API Call failed: {e}")
        return "Error: Unable to generate a response from the LLM."