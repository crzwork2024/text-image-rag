"""
Main Application Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: FastAPI application entry point, providing Q&A API and Web interface.
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

# Initialize Logger
logger = setup_logger(
    "API",
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_dir=config.LOG_DIR
)

# Global: Parent Node Storage Map
parent_store = {}

# Global: Semantic Cache
semantic_cache = None

# Global: Admin Session Storage
admin_sessions = {}  # {token: expire_time}

# Security Config
security = HTTPBearer(auto_error=False)


# ==================== Admin Auth Helpers ====================
def hash_password(password: str) -> str:
    """Hash password"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_admin(username: str, password: str) -> bool:
    """Verify admin credentials"""
    admin_username = config.ADMIN_USERNAME if hasattr(config, 'ADMIN_USERNAME') else "admin"
    admin_password_hash = config.ADMIN_PASSWORD_HASH if hasattr(config, 'ADMIN_PASSWORD_HASH') else hash_password("admin123")
    
    return username == admin_username and hash_password(password) == admin_password_hash


def generate_admin_token() -> str:
    """Generate admin token"""
    return secrets.token_urlsafe(32)


def verify_admin_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify admin token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication token not provided")
    
    token = credentials.credentials
    if token not in admin_sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Check expiration
    if datetime.now() > admin_sessions[token]:
        del admin_sessions[token]
        raise HTTPException(status_code=401, detail="Token expired")
    
    return True


class QueryRequest(BaseModel):
    """Query Request Model"""
    prompt: str = Field(..., description="User question", min_length=1, max_length=1000)
    use_rerank: bool = Field(True, description="Whether to use Rerank optimization")
    use_query_enhancement: bool = Field(False, description="Whether to use Query Enhancement (HyDE)")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session ID")


class CacheConfirmRequest(BaseModel):
    """Cache Confirmation Request Model"""
    confirmation_id: str = Field(..., description="Confirmation ID")
    user_confirmed: bool = Field(..., description="Whether user confirmed to use cache")


class FeedbackRequest(BaseModel):
    """User Feedback Request Model"""
    session_id: str = Field(..., description="Session ID")
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="System answer")
    satisfied: bool = Field(..., description="Is user satisfied")
    source_hashes: Optional[List[str]] = Field(None, description="Source document hashes")


class AdminLoginRequest(BaseModel):
    """Admin Login Request Model"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class ManualCacheRequest(BaseModel):
    """Manual Cache Add Request Model"""
    question: str = Field(..., description="Question text", min_length=1)
    answer: str = Field(..., description="Answer text", min_length=1)
    quality_score: int = Field(10, description="Quality score", ge=0, le=10)
    source_info: Optional[str] = Field(None, description="Source info (manually entered by admin)")


class ClearCacheRequest(BaseModel):
    """Clear Cache Request Model"""
    cache_types: Optional[List[str]] = Field(None, description="Cache types to clear")
    confirm: bool = Field(False, description="Confirm clear")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App Lifecycle Manager

    On Startup:
    - Load parent node mapping
    - Check vector DB, run auto-ingestion if empty

    On Shutdown:
    - Clean up resources
    """
    global parent_store, semantic_cache

    logger.info("=" * 60)
    logger.info("System Starting...")
    logger.info("=" * 60)

    try:
        # Create necessary directories
        config.create_directories()

        # Initialize Semantic Cache
        logger.info("Initializing Semantic Cache...")
        semantic_cache = SemanticCache(embedding_engine)
        if semantic_cache.is_available():
            logger.info("✓ Semantic Cache Enabled")
        else:
            logger.warning("⚠️ Semantic Cache Unavailable (Redis Connection Failed), skipping cache feature")

        # Load Parent Node Map
        if config.PARENT_STORE_PATH.exists():
            logger.info(f"Loading Parent Node Map: {config.PARENT_STORE_PATH}")
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)
            logger.info(f"✓ Loaded {len(parent_store)} parent nodes")
        else:
            logger.warning("Parent node map file not found, will be created on first ingestion")

        # Check Vector Database
        doc_count = vector_db.count()
        logger.info(f"Vector Database Document Count: {doc_count}")

        if doc_count == 0:
            logger.warning("Vector database is empty, starting auto-ingestion...")
            run_ingestion()

            # Reload parent node map
            with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)

            logger.info("✓ Auto-ingestion completed")

        logger.info("=" * 60)
        logger.info("System Startup Complete, Service Ready")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"System Startup Failed: {e}")
        raise

    finally:
        logger.info("=" * 60)
        logger.info("System Shutdown")
        logger.info("=" * 60)


# Create FastAPI App
app = FastAPI(
    title="RAG Intelligent Q&A System API",
    description="Intelligent Q&A Service based on Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS Middleware
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
    RAG Query Interface

    Workflow:
    1. Vector Search: Recall candidate documents from vector DB
    2. Preliminary Filtering: Filter by similarity threshold
    3. Rerank (Optional): Precise sorting using Rerank model
    4. Context Assembly: Retrieve full sections using parent node hashes
    5. LLM Generation: Call LLM to generate answer

    Args:
        req: Query request containing question and options

    Returns:
        QueryResponse: Contains answer, best score, source count
    """
    # Build mode identifier
    modes = []
    if req.use_query_enhancement:
        modes.append("Query Enhancement")
    modes.append("Precision Mode" if req.use_rerank else "Fast Mode")
    mode_str = " + ".join(modes)

    logger.info("=" * 60)
    logger.info(f"[New Query] {mode_str}")
    logger.info(f"Question: {req.prompt[:100]}{'...' if len(req.prompt) > 100 else ''}")
    logger.info("=" * 60)

    try:
        # ==================== Step 0: Semantic Cache Query ====================
        if semantic_cache and semantic_cache.is_available():
            logger.info("Step 0: Query Semantic Cache")
            cache_result = await semantic_cache.query(req.prompt, req.session_id)

            if cache_result["status"] == "hit":
                # Cache Direct Hit
                logger.info("⚡ Cache Direct Hit, returning cached answer")
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
                # Needs User Confirmation
                logger.info("⏸️ Found similar question, waiting for user confirmation")
                return {
                    "need_confirmation": True,
                    "cached_question": cache_result["cached_question"],
                    "similarity": f"{cache_result['similarity']:.2%}",
                    "confirmation_id": cache_result["confirmation_id"],
                    "message": "Found a similar question, use cached answer?"
                }

            # cache_result["status"] == "miss" -> Continue
            logger.info("🔄 Cache Miss, proceeding with full retrieval")

        # ==================== Step 1: Query Enhancement (Optional) ====================
        enhanced_query = None
        if req.use_query_enhancement and query_enhancer.is_available():
            logger.info("Step 1/6: Query Enhancement (Generate Hypothetical Keywords)")
            enhanced_query = query_enhancer.generate_hypothetical_keywords(req.prompt)

            if enhanced_query:
                logger.info(f"✓ Keywords Generated: {enhanced_query[:100]}...")
            else:
                logger.warning("✗ Query Enhancement Failed, falling back to standard retrieval")
                req.use_query_enhancement = False

        # ==================== Step 2: Vector Search ====================
        step_num = "2/6" if req.use_query_enhancement else "1/5"
        logger.info(f"Step {step_num}: Vector Search (Recall: {config.RETRIEVAL_COUNT})")

        # Original Question Search
        query_vec = embedding_engine.encode([req.prompt])
        results_query = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)

        raw_docs = results_query["documents"][0]
        raw_metas = results_query["metadatas"][0]
        raw_dists = results_query["distances"][0]
        raw_ids = results_query["ids"][0]

        logger.info(f"✓ Original Question recalled {len(raw_docs)} candidates")

        # If Query Enhancement Enabled, run second search
        if req.use_query_enhancement and enhanced_query:
            logger.info(f"Step 2.5/6: Secondary Search with Keywords")

            enhanced_vec = embedding_engine.encode([enhanced_query])
            results_enhanced = vector_db.query(enhanced_vec, n_results=config.RETRIEVAL_COUNT)

            enhanced_docs = results_enhanced["documents"][0]
            enhanced_metas = results_enhanced["metadatas"][0]
            enhanced_dists = results_enhanced["distances"][0]
            enhanced_ids = results_enhanced["ids"][0]

            logger.info(f"✓ Keywords recalled {len(enhanced_docs)} candidates")

            # Fuse results - Weighted Average
            logger.info("Fusing results (Weighted)...")

            # Build ID to Score Map
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

            # Merge and Weight
            all_doc_ids = set(query_scores.keys()) | set(enhanced_scores.keys())
            merged_results = []

            query_weight = config.QUERY_ENHANCEMENT_WEIGHT
            enhanced_weight = 1 - query_weight

            for doc_id in all_doc_ids:
                q_sim = query_scores.get(doc_id, {}).get('similarity', 0)
                e_sim = enhanced_scores.get(doc_id, {}).get('similarity', 0)

                # Weighted Fusion
                final_sim = query_weight * q_sim + enhanced_weight * e_sim

                # Use original doc content (priority)
                doc_content = query_scores.get(doc_id, {}).get('doc') or enhanced_scores.get(doc_id, {}).get('doc')
                doc_meta = query_scores.get(doc_id, {}).get('meta') or enhanced_scores.get(doc_id, {}).get('meta')

                merged_results.append({
                    'id': doc_id,
                    'similarity': final_sim,
                    'distance': 1 - final_sim,
                    'doc': doc_content,
                    'meta': doc_meta
                })

            # Sort by fused similarity
            merged_results.sort(key=lambda x: x['similarity'], reverse=True)

            # Reconstruct original format, keep top 10
            raw_docs = [r['doc'] for r in merged_results[:10]]
            raw_metas = [r['meta'] for r in merged_results[:10]]
            raw_dists = [r['distance'] for r in merged_results[:10]]

            logger.info(f"✓ Fusion complete, keeping top 10 results")

            # Show fused scores
            logger.info("Fused Top 10 Results:")
            for i, r in enumerate(merged_results[:10], 1):
                h = r['meta'].get('parent_hash', 'N/A')
                logger.info(f"  [{i:2d}] Fusion Score: {r['similarity']*100:>6.2f}% | ParentHash: {h[:16]}...")

        # ==================== Step 3: Preliminary Filtering ====================
        step_num = "3/6" if req.use_query_enhancement else "2/5"

        # Choose threshold based on rerank usage
        if req.use_rerank and rerank_engine.is_available():
            threshold = config.VECTOR_SEARCH_THRESHOLD_WITH_RERANK
            threshold_mode = "Loose (Precision Mode)"
        else:
            threshold = config.VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK
            threshold_mode = "Strict (Fast Mode)"

        logger.info(f"Step {step_num}: Preliminary Filtering (Threshold: {threshold} - {threshold_mode})")

        # Show top 10 candidates scores
        logger.info("=" * 60)
        logger.info("【Vector Search】Top 10 Candidates (Sorted by Similarity):")
        for i in range(min(10, len(raw_docs))):
            sim = 1 - raw_dists[i]
            sim_pct = f"{round(sim * 100, 2)}%"
            h = raw_metas[i].get("parent_hash", "N/A")
            pass_mark = "✓" if sim >= threshold else "✗"
            logger.info(f"  [{pass_mark}] [{i+1:2d}] Similarity: {sim_pct:>7s} | ParentHash: {h[:16]}...")
        logger.info("=" * 60)

        candidates = []
        candidates_meta = []

        for i in range(len(raw_docs)):
            sim = 1 - raw_dists[i]
            if sim >= threshold:
                candidates.append(raw_docs[i])
                candidates_meta.append(raw_metas[i])

        logger.info(f"✓ {len(candidates)} candidates remaining after filtering")

        # If no candidates, return early
        if not candidates:
            logger.warning("No relevant content found")
            return QueryResponse(
                answer="Sorry, I couldn't find any content related to your question.",
                best_score="0%",
                sources_count=0
            )

        final_hashes = []
        score_summaries = []

        # ==================== Step 4: Rerank (Optional) ====================
        if req.use_rerank and rerank_engine.is_available():
            step_num = "4/6" if req.use_query_enhancement else "3/5"
            logger.info(f"Step {step_num}: Executing Rerank (Candidates: {len(candidates)})")

            rerank_data = rerank_engine.rerank(req.prompt, candidates)

            if rerank_data:
                # Show all rerank results
                logger.info("=" * 60)
                logger.info("【Rerank Results】Sorted by Relevance:")

                for i, res in enumerate(rerank_data):
                    orig_idx = res["index"]
                    score = res["relevance_score"]
                    p_hash = candidates_meta[orig_idx].get("parent_hash", "N/A")

                    score_pct = f"{round(score * 100, 2)}%"

                    # Filter by threshold
                    if score >= config.RERANK_THRESHOLD:
                        logger.info(f"  [✓ {i+1:2d}] Rerank Score: {score_pct:>7s} | ParentHash: {p_hash[:16]}... (Selected)")
                        final_hashes.append(p_hash)
                    else:
                        logger.info(f"  [✗ {i+1:2d}] Rerank Score: {score_pct:>7s} | ParentHash: {p_hash[:16]}... (Below Threshold)")

                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})

                logger.info("=" * 60)
                logger.info(f"✓ Rerank complete, keeping {len(final_hashes)} high-relevance docs")
            else:
                logger.warning("Rerank failed, falling back to Fast Mode")
                req.use_rerank = False

        # ==================== Step 4: Fast Mode (Fallback) ====================
        if not req.use_rerank or not rerank_engine.is_available():
            step_num = "4/6" if req.use_query_enhancement else "3/5"
            logger.info(f"Step {step_num}: Fast Mode, selecting Top {config.RERANK_TOP_K}")

            # Show results
            logger.info("=" * 60)
            logger.info("【Fast Mode】Sorted by Vector Similarity:")

            for i in range(min(config.RERANK_TOP_K, len(candidates_meta))):
                p_hash = candidates_meta[i].get("parent_hash", "N/A")
                
                # Find candidate index in raw list
                candidate_idx = i
                for j in range(len(raw_docs)):
                    if raw_metas[j].get("parent_hash") == p_hash:
                        candidate_idx = j
                        break

                sim = 1 - raw_dists[candidate_idx]
                score_pct = f"{round(sim * 100, 2)}%"
                
                # Strict threshold for Fast Mode
                direct_threshold = config.VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK
                
                if sim >= direct_threshold:
                    logger.info(f"  [✓ {i+1}] Similarity: {score_pct:>7s} | ParentHash: {p_hash[:16]}... (Selected)")
                    final_hashes.append(p_hash)
                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})
                else:
                    logger.info(f"  [✗ {i+1}] Similarity: {score_pct:>7s} | ParentHash: {p_hash[:16]}... (Below Threshold {direct_threshold*100:.0f}%, Filtered)")
                    score_summaries.append({"rank": i+1, "rerank_score": score_pct})

            logger.info("=" * 60)
            logger.info(f"✓ Directly selected {len(final_hashes)} documents")

        # ==================== Check for Valid Results ====================
        if not final_hashes:
            logger.warning("All results below relevance threshold, not enough context")
            return QueryResponse(
                answer="Sorry, I couldn't find enough relevant content.\n\nSuggestions:\n1. Rephrase your question\n2. Use different keywords\n3. Check if the knowledge base contains this info",
                best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
                sources_count=0
            )

        # ==================== Step 5: Get Full Context ====================
        step_num = "5/6" if req.use_query_enhancement else "4/5"
        logger.info(f"Step {step_num}: Assembling Context")

        unique_hashes = list(dict.fromkeys(final_hashes))

        # Log all final parent hashes
        logger.info(f"Final Parent Hashes (Total {len(unique_hashes)}):")
        for idx, h in enumerate(unique_hashes, 1):
            logger.info(f"  [{idx}] Hash: {h}")

        retrieved_sections = [
            parent_store.get(h) for h in unique_hashes
            if parent_store.get(h)
        ]

        logger.info(f"✓ Extracted {len(retrieved_sections)} full sections")

        if not retrieved_sections:
            logger.warning("Failed to retrieve valid context")
            return QueryResponse(
                answer="Sorry, unable to retrieve relevant content.",
                best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
                sources_count=0
            )

        # ==================== Step 6: LLM Generation ====================
        step_num = "6/6" if req.use_query_enhancement else "5/5"
        logger.info(f"Step {step_num}: LLM Generating Answer")

        context = "\n\n---\n\n".join(retrieved_sections)
        logger.info(f"Total Context Length: {len(context)} chars")
        logger.info(f"Parent Nodes sent to LLM: {len(retrieved_sections)}")

        try:
            answer = call_llm(context, req.prompt)
            logger.info(f"✓ Answer generated successfully, length: {len(answer)} chars")
        except LLMAPIError as e:
            logger.error(f"LLM Generation Failed: {e.message}")
            answer = "Sorry, an error occurred while generating the answer. Please try again later."

        # ==================== Add to Cache ====================
        # Note: Auto-cache disabled, only explicit user confirmation or admin add works
        # if semantic_cache and semantic_cache.is_available():
        #     logger.info("💾 Adding answer to semantic cache")
        #     semantic_cache.set(req.prompt, answer)

        # ==================== Return Results ====================
        logger.info("=" * 60)
        logger.info("[Query Complete]")
        logger.info("=" * 60)

        return QueryResponse(
            answer=answer,
            best_score=score_summaries[0]["rerank_score"] if score_summaries else "0%",
            sources_count=len(retrieved_sections),
            source_hashes=unique_hashes
        )

    except Exception as e:
        logger.error(f"Query Processing Exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                error="Query Processing Failed",
                details=str(e),
                code="QUERY_ERROR"
            )
        )

@app.post("/cache/confirm")
async def confirm_cache(req: CacheConfirmRequest):
    """
    Handle user cache confirmation

    Args:
        req: Confirmation request

    Returns:
        Cached answer if confirmed, else message to re-query
    """
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")

    cached_answer = await semantic_cache.confirm_cache(
        req.confirmation_id,
        req.user_confirmed
    )

    if req.user_confirmed and cached_answer:
        # User confirmed usage
        return {
            "answer": cached_answer,
            "from_cache": True,
            "best_score": "95%+",
            "sources_count": 0,
            "message": "Used cached answer"
        }
    else:
        # User denied or ID expired -> Re-query
        return {
            "need_requery": True,
            "message": "Please re-ask to get a new answer"
        }


@app.post("/cache/feedback")
async def cache_feedback(req: FeedbackRequest):
    """
    User Satisfaction Feedback

    If user is satisfied, add Q&A pair to high-quality cache
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"status": "unavailable", "message": "Cache service unavailable"}

    try:
        if req.satisfied:
            # User satisfied, add to high quality cache
            import json
            source_info = json.dumps(req.source_hashes) if req.source_hashes else None
            
            semantic_cache.set(
                req.question,
                req.answer,
                cache_type="confirmed",
                quality_score=5,
                source_info=source_info
            )
            logger.info(f"✅ User feedback satisfied, added to HQ cache: {req.question[:50]}")
            return {
                "status": "success",
                "message": "Thank you! Saved to featured Q&A"
            }
        else:
            # User unsatisfied, log but don't cache
            logger.info(f"❌ User feedback unsatisfied: {req.question[:50]}")
            return {
                "status": "success",
                "message": "Thank you for your feedback!"
            }
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail="Feedback processing failed")


@app.get("/cache/popular")
async def get_popular_questions():
    """
    Get popular questions (for frontend)

    Returns:
        List of popular questions
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"popular_questions": []}

    popular_questions = semantic_cache.get_popular_questions(3)
    return {"popular_questions": popular_questions}


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics

    Returns:
        Cache stats
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {
            "available": False,
            "message": "Cache service unavailable"
        }

    stats = semantic_cache.get_cache_stats()
    return stats


@app.get("/health")
async def health_check():
    """Health Check Endpoint"""
    cache_available = semantic_cache and semantic_cache.is_available()
    return {
        "status": "healthy",
        "vector_db_docs": vector_db.count(),
        "parent_store_size": len(parent_store),
        "cache_available": cache_available
    }


# ==================== Admin API ====================

@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest):
    """
    Admin Login

    Returns:
        Auth token and expiration
    """
    if not verify_admin(req.username, req.password):
        logger.warning(f"Admin login failed: {req.username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate token
    token = generate_admin_token()
    expire_time = datetime.now() + timedelta(hours=1)
    admin_sessions[token] = expire_time

    logger.info(f"✅ Admin login successful: {req.username}")
    return {
        "token": token,
        "expires_in": 3600,
        "expires_at": expire_time.isoformat()
    }


@app.post("/admin/logout")
async def admin_logout(authorized: bool = Depends(verify_admin_token),
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Admin Logout"""
    token = credentials.credentials
    if token in admin_sessions:
        del admin_sessions[token]
        logger.info("✅ Admin logout successful")
    
    return {"status": "success", "message": "Logged out"}


@app.get("/admin/hot-questions")
async def get_hot_questions(
    limit: int = 50,
    min_count: int = 1,
    authorized: bool = Depends(verify_admin_token)
):
    """
    Get Hot Questions (Admin)

    Args:
        limit: Result limit
        min_count: Min query count

    Returns:
        Hot questions list
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"hot_questions": []}

    try:
        # Get all popular questions
        popular = semantic_cache.redis.zrevrange("cache:popular", 0, limit - 1, withscores=True)
        
        result = []
        for question_bytes, count in popular:
            if count < min_count:
                continue
                
            question = question_bytes.decode('utf-8') if isinstance(question_bytes, bytes) else question_bytes
            cache_id = semantic_cache._compute_hash(question)
            
            # Check if cached
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
        logger.error(f"Error getting hot questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get hot questions")


@app.get("/admin/cache/list")
async def get_cache_list(
    limit: int = 100,
    authorized: bool = Depends(verify_admin_token)
):
    """
    Get all cached questions (Admin)

    Returns:
        Cached questions list
    """
    if not semantic_cache or not semantic_cache.is_available():
        return {"cached_questions": []}

    try:
        cached_questions = semantic_cache.get_all_cached_questions(limit)
        return {"cached_questions": cached_questions}
        
    except Exception as e:
        logger.error(f"Error getting cache list: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache list")


@app.post("/admin/cache/add")
async def add_manual_cache(
    req: ManualCacheRequest,
    authorized: bool = Depends(verify_admin_token)
):
    """
    Manually add cache (Admin)

    Used for adding curated Q&A pairs
    """
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")

    try:
        # Add to HQ cache
        semantic_cache.set(
            req.question,
            req.answer,
            cache_type="manual",
            quality_score=req.quality_score,
            source_info=req.source_info
        )
        
        cache_id = semantic_cache._compute_hash(req.question)
        logger.info(f"✅ Admin manually added cache: {req.question[:50]}")
        
        return {
            "status": "success",
            "cache_id": cache_id,
            "message": "Added to high-priority cache"
        }
        
    except Exception as e:
        logger.error(f"Error manually adding cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to add cache")


@app.delete("/admin/cache/clear")
async def clear_cache(
    req: ClearCacheRequest,
    authorized: bool = Depends(verify_admin_token)
):
    """
    Clear Cache (Admin)

    Args:
        cache_types: List of types to clear (None for all)
        confirm: Must be true
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")

    try:
        deleted_count = semantic_cache.clear_cache(req.cache_types)
        
        cache_types_str = ", ".join(req.cache_types) if req.cache_types else "ALL"
        logger.warning(f"🗑️ Admin cleared cache: {cache_types_str} ({deleted_count} entries)")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Cleared {deleted_count} cache entries"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.delete("/admin/cache/{cache_id}")
async def delete_cache_item(
    cache_id: str,
    authorized: bool = Depends(verify_admin_token)
):
    """Delete single cache entry (Admin)"""
    if not semantic_cache or not semantic_cache.is_available():
        raise HTTPException(status_code=503, detail="Cache service unavailable")

    try:
        semantic_cache._evict_cache(cache_id)
        logger.info(f"🗑️ Admin deleted cache: {cache_id}")
        
        return {
            "status": "success",
            "message": "Cache entry deleted"
        }
        
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete cache")


@app.get("/stats")
async def get_stats():
    """Get System Stats"""
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


# ==================== Static Files Mount ====================
# Images
if config.IMAGE_DIR.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(config.IMAGE_DIR)),
        name="images"
    )
    logger.info(f"✓ Mounted Image Directory: {config.IMAGE_DIR}")

# Frontend Static Files
if config.STATIC_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(config.STATIC_DIR), html=True),
        name="static"
    )
    logger.info(f"✓ Mounted Static Directory: {config.STATIC_DIR}")


def main():
    """Main Function - Start Server"""
    import uvicorn

    logger.info("=" * 60)
    logger.info(f"Starting Server: http://{config.APP_HOST}:{config.APP_PORT}")
    logger.info(f"API Docs: http://{config.APP_HOST}:{config.APP_PORT}/docs")
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
