"""
主应用模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：FastAPI 应用主入口，提供问答 API 和 Web 界面
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime, timedelta
import secrets

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import hashlib

from config import config
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from core.llm_client import call_llm
from core.reranker import rerank_engine
from core.query_enhancer import query_enhancer
from core.semantic_cache import SemanticCache
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

# 全局变量：语义缓存
semantic_cache = None

# 全局变量：管理员会话存储
admin_sessions = {}  # {token: expire_time}

# 安全配置
security = HTTPBearer(auto_error=False)


# ==================== 管理员认证辅助函数 ====================
def hash_password(password: str) -> str:
    """哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_admin(username: str, password: str) -> bool:
    """验证管理员凭证"""
    admin_username = config.ADMIN_USERNAME if hasattr(config, 'ADMIN_USERNAME') else "admin"
    admin_password_hash = config.ADMIN_PASSWORD_HASH if hasattr(config, 'ADMIN_PASSWORD_HASH') else hash_password("admin123")
    
    return username == admin_username and hash_password(password) == admin_password_hash


def generate_admin_token() -> str:
    """生成管理员 token"""
    return secrets.token_urlsafe(32)


def verify_admin_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """验证管理员 token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    
    token = credentials.credentials
    if token not in admin_sessions:
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")
    
    # 检查是否过期
    if datetime.now() > admin_sessions[token]:
        del admin_sessions[token]
        raise HTTPException(status_code=401, detail="认证令牌已过期")
    
    return True


class QueryRequest(BaseModel):
    """查询请求模型"""
    prompt: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    use_rerank: bool = Field(True, description="是否使用重排优化")
    use_query_enhancement: bool = Field(False, description="是否使用查询增强（HyDE）")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话ID")


class CacheConfirmRequest(BaseModel):
    """缓存确认请求模型"""
    confirmation_id: str = Field(..., description="确认ID")
    user_confirmed: bool = Field(..., description="用户是否确认使用缓存")


class FeedbackRequest(BaseModel):
    """用户反馈请求模型"""
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="系统答案")
    satisfied: bool = Field(..., description="用户是否满意")
    source_hashes: Optional[List[str]] = Field(None, description="源文档Hash列表")


class AdminLoginRequest(BaseModel):
    """管理员登录请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class ManualCacheRequest(BaseModel):
    """手动添加缓存请求模型"""
    question: str = Field(..., description="问题文本", min_length=1)
    answer: str = Field(..., description="答案文本", min_length=1)
    quality_score: int = Field(10, description="质量分数", ge=0, le=10)
    source_info: Optional[str] = Field(None, description="源文件信息（管理员手动填写）")


class ClearCacheRequest(BaseModel):
    """清除缓存请求模型"""
    cache_types: Optional[List[str]] = Field(None, description="要清除的缓存类型")
    confirm: bool = Field(False, description="确认清除")

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
    global parent_store, semantic_cache

    logger.info("=" * 60)
    logger.info("系统启动中...")
    logger.info("=" * 60)

    try:
        # 创建必要的目录
        config.create_directories()

        # 初始化语义缓存
        logger.info("初始化语义缓存...")
        semantic_cache = SemanticCache(embedding_engine)
        if semantic_cache.is_available():
            logger.info("✓ 语义缓存已启用")
        else:
            logger.warning("⚠️ 语义缓存不可用（Redis连接失败），将跳过缓存功能")

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
    # 构建模式标识
    modes = []
    if req.use_query_enhancement:
        modes.append("查询增强")
    modes.append("精排模式" if req.use_rerank else "直取模式")
    mode_str = " + ".join(modes)

    logger.info("=" * 60)
    logger.info(f"[新查询] {mode_str}")
    logger.info(f"问题: {req.prompt[:100]}{'...' if len(req.prompt) > 100 else ''}")
    logger.info("=" * 60)

    try:
        # ==================== 步骤 0: 语义缓存查询 ====================
        if semantic_cache and semantic_cache.is_available():
            logger.info("步骤 0: 查询语义缓存")
            cache_result = await semantic_cache.query(req.prompt, req.session_id)

            if cache_result["status"] == "hit":
                # 缓存直接命中
                logger.info("⚡ 缓存直接命中，返回缓存答案")
                return QueryResponse(
                    answer=cache_result["answer"],
                    best_score=f"{cache_result['similarity']:.2%}",
                    sources_count=0,
                    metadata={
                        "from_cache": True,
                        "cache_type": "direct_hit",
                        "cached_question": cache_result["cached_question"],
                        "similarity": f"{cache_result['similarity']:.2%}"
                    }
                )

            elif cache_result["status"] == "pending_confirm":
                # 需要用户确认
                logger.info("⏸️ 发现相似问题，等待用户确认")
                return {
                    "need_confirmation": True,
                    "cached_question": cache_result["cached_question"],
                    "similarity": f"{cache_result['similarity']:.2%}",
                    "confirmation_id": cache_result["confirmation_id"],
                    "message": "发现相似问题，是否使用缓存答案？"
                }

            # cache_result["status"] == "miss" → 继续正常流程
            logger.info("🔄 缓存未命中，执行完整检索流程")

        # ==================== 步骤 1: 查询增强（可选）====================
        enhanced_query = None
        if req.use_query_enhancement and query_enhancer.is_available():
            logger.info("步骤 1/6: 查询增强（生成假设关键词）")
            enhanced_query = query_enhancer.generate_hypothetical_keywords(req.prompt)

            if enhanced_query:
                logger.info(f"✓ 生成关键词: {enhanced_query[:100]}...")
            else:
                logger.warning("✗ 查询增强失败，降级为普通检索")
                req.use_query_enhancement = False

        # ==================== 步骤 2: 向量检索 ====================
        step_num = "2/6" if req.use_query_enhancement else "1/5"
        logger.info(f"步骤 {step_num}: 向量检索 (召回数: {config.RETRIEVAL_COUNT})")

        # 原始问题检索
        query_vec = embedding_engine.encode([req.prompt])
        results_query = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)

        raw_docs = results_query["documents"][0]
        raw_metas = results_query["metadatas"][0]
        raw_dists = results_query["distances"][0]
        raw_ids = results_query["ids"][0]

        logger.info(f"✓ 原问题召回 {len(raw_docs)} 个候选文档")

        # 如果启用查询增强，执行第二次检索
        if req.use_query_enhancement and enhanced_query:
            logger.info(f"步骤 2.5/6: 使用关键词进行二次检索")

            enhanced_vec = embedding_engine.encode([enhanced_query])
            results_enhanced = vector_db.query(enhanced_vec, n_results=config.RETRIEVAL_COUNT)

            enhanced_docs = results_enhanced["documents"][0]
            enhanced_metas = results_enhanced["metadatas"][0]
            enhanced_dists = results_enhanced["distances"][0]
            enhanced_ids = results_enhanced["ids"][0]

            logger.info(f"✓ 关键词召回 {len(enhanced_docs)} 个候选文档")

            # 融合两次检索结果 - 使用加权平均
            logger.info("融合两次检索结果（加权融合）...")

            # 构建ID到分数的映射
            query_scores = {}
            enhanced_scores = {}

            for i, doc_id in enumerate(raw_ids):
                sim = 1 - raw_dists[i]
                query_scores[doc_id] = {
                    'similarity': sim,
                    'doc': raw_docs[i],
                    'meta': raw_metas[i]
                }

            for i, doc_id in enumerate(enhanced_ids):
                sim = 1 - enhanced_dists[i]
                enhanced_scores[doc_id] = {
                    'similarity': sim,
                    'doc': enhanced_docs[i],
                    'meta': enhanced_metas[i]
                }

            # 合并并加权
            all_doc_ids = set(query_scores.keys()) | set(enhanced_scores.keys())
            merged_results = []

            query_weight = config.QUERY_ENHANCEMENT_WEIGHT
            enhanced_weight = 1 - query_weight

            for doc_id in all_doc_ids:
                q_sim = query_scores.get(doc_id, {}).get('similarity', 0)
                e_sim = enhanced_scores.get(doc_id, {}).get('similarity', 0)

                # 加权融合
                final_sim = query_weight * q_sim + enhanced_weight * e_sim

                # 使用原问题的文档内容（优先）
                doc_content = query_scores.get(doc_id, {}).get('doc') or enhanced_scores.get(doc_id, {}).get('doc')
                doc_meta = query_scores.get(doc_id, {}).get('meta') or enhanced_scores.get(doc_id, {}).get('meta')

                merged_results.append({
                    'id': doc_id,
                    'similarity': final_sim,
                    'distance': 1 - final_sim,
                    'doc': doc_content,
                    'meta': doc_meta
                })

            # 按融合后的相似度排序
            merged_results.sort(key=lambda x: x['similarity'], reverse=True)

            # 重新组织为原格式，保留前10个
            raw_docs = [r['doc'] for r in merged_results[:10]]
            raw_metas = [r['meta'] for r in merged_results[:10]]
            raw_dists = [r['distance'] for r in merged_results[:10]]

            logger.info(f"✓ 融合完成，保留前 10 个结果")

            # 显示融合后的分数
            logger.info("融合后的前10个结果:")
            for i, r in enumerate(merged_results[:10], 1):
                h = r['meta'].get('parent_hash', 'N/A')
                logger.info(f"  [{i:2d}] 融合分数: {r['similarity']*100:>6.2f}% | 父Hash: {h[:16]}...")

        # ==================== 步骤 3: 向量初筛 ====================
        step_num = "3/6" if req.use_query_enhancement else "2/5"

        # 根据是否使用精排选择不同的阈值
        if req.use_rerank and rerank_engine.is_available():
            threshold = config.VECTOR_SEARCH_THRESHOLD_WITH_RERANK
            threshold_mode = "宽松(精排模式)"
        else:
            threshold = config.VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK
            threshold_mode = "严格(直取模式)"

        logger.info(f"步骤 {step_num}: 向量初筛 (阈值: {threshold} - {threshold_mode})")

        # 显示前10个候选的相似度分数和父Hash
        logger.info("=" * 60)
        logger.info("【向量检索】前10个候选（按相似度排序）:")
        for i in range(min(10, len(raw_docs))):
            sim = 1 - raw_dists[i]
            sim_pct = f"{round(sim * 100, 2)}%"
            h = raw_metas[i].get("parent_hash", "N/A")
            pass_mark = "✓" if sim >= threshold else "✗"
            logger.info(f"  [{pass_mark}] [{i+1:2d}] 相似度: {sim_pct:>7s} | 父Hash: {h[:16]}...")
        logger.info("=" * 60)

        candidates = []
        candidates_meta = []

        for i in range(len(raw_docs)):
            sim = 1 - raw_dists[i]
            h = raw_metas[i].get("parent_hash", "N/A")

            if sim >= threshold:
                candidates.append(raw_docs[i])
                candidates_meta.append(raw_metas[i])

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

        # ==================== 步骤 4: 重排（可选）====================
        if req.use_rerank and rerank_engine.is_available():
            step_num = "4/6" if req.use_query_enhancement else "3/5"
            logger.info(f"步骤 {step_num}: 执行精排 (候选数: {len(candidates)})")

            rerank_data = rerank_engine.rerank(req.prompt, candidates)

            if rerank_data:
                # 显示所有精排结果
                logger.info("=" * 60)
                logger.info("【精排结果】按相关度排序:")

                for i, res in enumerate(rerank_data):
                    orig_idx = res["index"]
                    score = res["relevance_score"]
                    p_hash = candidates_meta[orig_idx].get("parent_hash", "N/A")

                    score_pct = f"{round(score * 100, 2)}%"

                    # 根据阈值过滤
                    if score >= config.RERANK_THRESHOLD:
                        logger.info(f"  [✓ {i+1:2d}] 精排分数: {score_pct:>7s} | 父Hash: {p_hash[:16]}... (已选入)")
                        final_hashes.append(p_hash)
                    else:
                        logger.info(f"  [✗ {i+1:2d}] 精排分数: {score_pct:>7s} | 父Hash: {p_hash[:16]}... (未达阈值)")

                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})

                logger.info("=" * 60)
                logger.info(f"✓ 精排完成，保留 {len(final_hashes)} 个高相关文档")
            else:
                logger.warning("精排失败，降级为直取模式")
                req.use_rerank = False

        # ==================== 步骤 4: 直取模式（备选）====================
        if not req.use_rerank or not rerank_engine.is_available():
            step_num = "4/6" if req.use_query_enhancement else "3/5"
            logger.info(f"步骤 {step_num}: 直取模式，选择前 {config.RERANK_TOP_K} 名")

            # 显示直取的结果
            logger.info("=" * 60)
            logger.info("【直取模式】按向量相似度排序:")

            for i in range(min(config.RERANK_TOP_K, len(candidates_meta))):
                p_hash = candidates_meta[i].get("parent_hash", "N/A")
                # 直取模式需要从candidates中获取，因为已经过滤了
                # 找到这个candidate在原始列表中的位置
                candidate_idx = i
                for j in range(len(raw_docs)):
                    if raw_metas[j].get("parent_hash") == p_hash:
                        candidate_idx = j
                        break

                sim = 1 - raw_dists[candidate_idx]
                score_pct = f"{round(sim * 100, 2)}%"
                
                # 直取模式使用严格阈值（默认50%）
                direct_threshold = config.VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK
                
                if sim >= direct_threshold:
                    logger.info(f"  [✓ {i+1}] 相似度: {score_pct:>7s} | 父Hash: {p_hash[:16]}... (已选入)")
                    final_hashes.append(p_hash)
                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})
                else:
                    logger.info(f"  [✗ {i+1}] 相似度: {score_pct:>7s} | 父Hash: {p_hash[:16]}... (低于阈值{direct_threshold*100:.0f}%，已过滤)")
                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})

            logger.info("=" * 60)
            logger.info(f"✓ 直取 {len(final_hashes)} 个文档")

        # ==================== 检查是否有有效结果 ====================
        if not final_hashes:
            logger.warning("所有结果均低于相关度阈值，未找到足够相关的内容")
            return QueryResponse(
                answer="抱歉，未找到与您问题足够相关的内容。\n\n建议：\n1. 尝试重新表述问题\n2. 使用不同的关键词\n3. 检查知识库是否包含相关信息",
                best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
                sources_count=0
            )

        # ==================== 步骤 5: 获取完整上下文 ====================
        step_num = "5/6" if req.use_query_enhancement else "4/5"
        logger.info(f"步骤 {step_num}: 组装上下文")

        unique_hashes = list(dict.fromkeys(final_hashes))

        # 记录所有最终选定的父hash
        logger.info(f"最终选定的父Hash列表 (共 {len(unique_hashes)} 个):")
        for idx, h in enumerate(unique_hashes, 1):
            logger.info(f"  [{idx}] Hash: {h}")

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
        step_num = "6/6" if req.use_query_enhancement else "5/5"
        logger.info(f"步骤 {step_num}: LLM 生成回答")

        context = "\n\n---\n\n".join(retrieved_sections)
        logger.info(f"上下文总长度: {len(context)} 字符")
        logger.info(f"传给LLM的父节点数量: {len(retrieved_sections)}")

        try:
            answer = call_llm(context, req.prompt)
            logger.info(f"✓ 回答生成成功，长度: {len(answer)} 字符")
        except LLMAPIError as e:
            logger.error(f"LLM 生成失败: {e.message}")
            answer = "抱歉，AI 生成回答时出现错误，请稍后重试。"

        # ==================== 添加到缓存 ====================
        # 注意：目前自动缓存已禁用，只有用户确认满意或管理员手动添加才会缓存
        # if semantic_cache and semantic_cache.is_available():
        #     logger.info("💾 添加答案到语义缓存")
        #     semantic_cache.set(req.prompt, answer)

        # ==================== 返回结果 ====================
        logger.info("=" * 60)
        logger.info("[查询完成]")
        logger.info("=" * 60)

        return QueryResponse(
            answer=answer,
            best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
            sources_count=len(retrieved_sections),
            source_hashes=unique_hashes  # 返回源文档的 parent_hash 列表
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

@app.post("/cache/confirm")
async def confirm_cache(req: CacheConfirmRequest):
    """
    处理用户的缓存确认

    参数:
        req: 确认请求，包含确认ID和用户决定

    返回:
        如果用户确认，返回缓存答案；否则提示重新查询
    """
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="缓存服务不可用")

    cached_answer = await semantic_cache.confirm_cache(
        req.confirmation_id,
        req.user_confirmed
    )

    if req.user_confirmed and cached_answer:
        # 用户确认使用缓存
        return {
            "answer": cached_answer,
            "from_cache": True,
            "best_score": "95%+",
            "sources_count": 0,
            "message": "已使用缓存答案"
        }
    else:
        # 用户拒绝或确认ID过期 → 需要前端重新发起查询
        return {
            "need_requery": True,
            "message": "请重新提问以获取新答案"
        }


@app.post("/cache/feedback")
async def cache_feedback(req: FeedbackRequest):
    """
    用户满意度反馈

    当用户对答案满意时，将问答对添加到高质量缓存
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"status": "unavailable", "message": "缓存服务不可用"}

    try:
        if req.satisfied:
            # 用户满意，添加到高质量缓存
            # 将 source_hashes 转换为 JSON 字符串
            import json
            source_info = json.dumps(req.source_hashes) if req.source_hashes else None
            
            semantic_cache.set(
                req.question,
                req.answer,
                cache_type="confirmed",
                quality_score=5,
                source_info=source_info
            )
            logger.info(f"✅ 用户反馈满意，已添加到高质量缓存: {req.question[:50]}")
            return {
                "status": "success",
                "message": "感谢反馈！已保存到精选问答"
            }
        else:
            # 用户不满意，记录但不缓存
            logger.info(f"❌ 用户反馈不满意: {req.question[:50]}")
            return {
                "status": "success",
                "message": "感谢反馈！我们会改进"
            }
    except Exception as e:
        logger.error(f"处理用户反馈时出错: {e}")
        raise HTTPException(status_code=500, detail="处理反馈失败")


@app.get("/cache/popular")
async def get_popular_questions():
    """
    获取热门问题（供前端显示）

    返回:
        热门问题列表，包含问题、访问次数、相似问题数
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"popular_questions": []}

    popular_questions = semantic_cache.get_popular_questions(3)
    return {"popular_questions": popular_questions}


@app.get("/cache/stats")
async def get_cache_stats():
    """
    获取缓存统计信息

    返回:
        缓存统计数据，包含缓存条目数、命中次数等
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {
            "available": False,
            "message": "缓存服务不可用"
        }

    stats = semantic_cache.get_cache_stats()
    return stats


@app.get("/health")
async def health_check():
    """健康检查接口"""
    cache_available = semantic_cache and semantic_cache.is_available()
    return {
        "status": "healthy",
        "vector_db_docs": vector_db.count(),
        "parent_store_size": len(parent_store),
        "cache_available": cache_available
    }


# ==================== 管理员 API ====================

@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest):
    """
    管理员登录

    返回:
        认证令牌和过期时间
    """
    if not verify_admin(req.username, req.password):
        logger.warning(f"管理员登录失败: {req.username}")
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 生成 token
    token = generate_admin_token()
    expire_time = datetime.now() + timedelta(hours=1)
    admin_sessions[token] = expire_time

    logger.info(f"✅ 管理员登录成功: {req.username}")
    return {
        "token": token,
        "expires_in": 3600,
        "expires_at": expire_time.isoformat()
    }


@app.post("/admin/logout")
async def admin_logout(authorized: bool = Depends(verify_admin_token),
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
    """管理员登出"""
    token = credentials.credentials
    if token in admin_sessions:
        del admin_sessions[token]
        logger.info("✅ 管理员登出成功")
    
    return {"status": "success", "message": "已登出"}


@app.get("/admin/hot-questions")
async def get_hot_questions(
    limit: int = 50,
    min_count: int = 1,
    authorized: bool = Depends(verify_admin_token)
):
    """
    获取热门问题列表（管理员）

    参数:
        limit: 返回数量
        min_count: 最小提问次数

    返回:
        热门问题列表
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"hot_questions": []}

    try:
        # 获取所有热门问题
        popular = semantic_cache.redis.zrevrange("cache:popular", 0, limit - 1, withscores=True)
        
        result = []
        for question_bytes, count in popular:
            if count < min_count:
                continue
                
            question = question_bytes.decode('utf-8') if isinstance(question_bytes, bytes) else question_bytes
            cache_id = semantic_cache._compute_hash(question)
            
            # 检查是否已缓存
            cached_data = semantic_cache.redis.hgetall(f"cache:question:{cache_id}")
            is_cached = bool(cached_data)
            cache_type = None
            
            if is_cached:
                cache_type = cached_data.get(b'cache_type', b'auto').decode('utf-8')
            
            result.append({
                "question": question,
                "count": int(count),
                "cached": is_cached,
                "cache_type": cache_type,
                "cache_id": cache_id
            })
        
        return {"hot_questions": result}
        
    except Exception as e:
        logger.error(f"获取热门问题时出错: {e}")
        raise HTTPException(status_code=500, detail="获取热门问题失败")


@app.get("/admin/cache/list")
async def get_cache_list(
    limit: int = 100,
    authorized: bool = Depends(verify_admin_token)
):
    """
    获取所有缓存列表（管理员）

    返回:
        缓存问题列表
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"cached_questions": []}

    try:
        cached_questions = semantic_cache.get_all_cached_questions(limit)
        return {"cached_questions": cached_questions}
        
    except Exception as e:
        logger.error(f"获取缓存列表时出错: {e}")
        raise HTTPException(status_code=500, detail="获取缓存列表失败")


@app.post("/admin/cache/add")
async def add_manual_cache(
    req: ManualCacheRequest,
    authorized: bool = Depends(verify_admin_token)
):
    """
    手动添加缓存（管理员）

    用于添加精选问答对
    """
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="缓存服务不可用")

    try:
        # 添加到高优先级缓存
        semantic_cache.set(
            req.question,
            req.answer,
            cache_type="manual",
            quality_score=req.quality_score,
            source_info=req.source_info  # 管理员填写的源文件信息
        )
        
        cache_id = semantic_cache._compute_hash(req.question)
        logger.info(f"✅ 管理员手动添加缓存: {req.question[:50]}")
        
        return {
            "status": "success",
            "cache_id": cache_id,
            "message": "已添加到高优先级缓存"
        }
        
    except Exception as e:
        logger.error(f"手动添加缓存时出错: {e}")
        raise HTTPException(status_code=500, detail="添加缓存失败")


@app.delete("/admin/cache/clear")
async def clear_cache(
    req: ClearCacheRequest,
    authorized: bool = Depends(verify_admin_token)
):
    """
    清除缓存（管理员）

    参数:
        cache_types: 要清除的缓存类型列表（None 表示全部）
        confirm: 必须为 true 才能执行
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="必须确认清除操作")

    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="缓存服务不可用")

    try:
        deleted_count = semantic_cache.clear_cache(req.cache_types)
        
        cache_types_str = ", ".join(req.cache_types) if req.cache_types else "所有"
        logger.warning(f"🗑️ 管理员清除缓存: {cache_types_str} ({deleted_count} 条)")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"已清除 {deleted_count} 条缓存"
        }
        
    except Exception as e:
        logger.error(f"清除缓存时出错: {e}")
        raise HTTPException(status_code=500, detail="清除缓存失败")


@app.delete("/admin/cache/{cache_id}")
async def delete_cache_item(
    cache_id: str,
    authorized: bool = Depends(verify_admin_token)
):
    """删除单个缓存条目（管理员）"""
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="缓存服务不可用")

    try:
        semantic_cache._evict_cache(cache_id)
        logger.info(f"🗑️ 管理员删除缓存: {cache_id}")
        
        return {
            "status": "success",
            "message": "已删除缓存条目"
        }
        
    except Exception as e:
        logger.error(f"删除缓存时出错: {e}")
        raise HTTPException(status_code=500, detail="删除缓存失败")


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
