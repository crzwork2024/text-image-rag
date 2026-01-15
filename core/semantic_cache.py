"""
Semantic Cache Module - Smart caching based on Redis and Embeddings
Author: RAG Project Team
Description: Matches questions using vector similarity, supports user confirmation mechanism.
"""

import redis
import numpy as np
import hashlib
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from config import config


logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic Cache Class

    Features:
    - Semantic similarity matching based on Embeddings
    - Three-layer threshold strategy (Direct Hit / Pending Confirm / Miss)
    - LRU Cache Eviction
    - Popular questions statistics
    - Similar questions clustering
    """

    def __init__(self, embedding_engine):
        """
        Initialize Semantic Cache

        Args:
            embedding_engine: Embedding engine instance (for calculating question vectors)
        """
        try:
            # Redis connection config
            redis_config = {
                'host': config.REDIS_HOST,
                'port': config.REDIS_PORT,
                'db': config.REDIS_DB,
                'decode_responses': False,  # Keep binary data (for storing embeddings)
                'socket_timeout': 5,
                'socket_connect_timeout': 5
            }

            # Add password only if configured
            if config.REDIS_PASSWORD:
                redis_config['password'] = config.REDIS_PASSWORD

            self.redis = redis.Redis(**redis_config)

            # Test connection
            self.redis.ping()
            self._available = True
            logger.info(f"✅ Redis connected successfully: {config.REDIS_HOST}:{config.REDIS_PORT}")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"⚠️ Redis connection failed, cache disabled: {e}")
            self._available = False
            self.redis = None

        # Embedding Engine
        self.embedding_engine = embedding_engine

        # Similarity Thresholds (from config)
        self.threshold_direct = config.CACHE_THRESHOLD_DIRECT      # 0.98 - Direct return
        self.threshold_confirm = config.CACHE_THRESHOLD_CONFIRM    # 0.95 - Needs confirmation

        # Cache Config
        self.cache_ttl = config.CACHE_TTL                         # TTL (seconds)
        self.max_cache_size = config.CACHE_MAX_SIZE               # Max cache entries

        if self._available:
            logger.info(f"📦 Cache Config - Direct Threshold: {self.threshold_direct}, "
                       f"Confirm Threshold: {self.threshold_confirm}, "
                       f"TTL: {self.cache_ttl}s, "
                       f"Max Capacity: {self.max_cache_size}")

    def is_available(self) -> bool:
        """Check if cache service is available"""
        return self._available

    def _compute_hash(self, text: str) -> str:
        """
        Compute hash ID for text

        Args:
            text: Input text

        Returns:
            16-character hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            emb1: Vector 1
            emb2: Vector 2

        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def query(
        self,
        question: str,
        session_id: str
    ) -> Dict:
        """
        Query the cache

        Args:
            question: User question
            session_id: Session ID

        Returns:
            Dict containing:
            - status: "hit" | "pending_confirm" | "miss"
            - answer: Answer content (only on hit)
            - cached_question: Similar question (on hit or pending_confirm)
            - similarity: Similarity score (on hit or pending_confirm)
            - confirmation_id: Confirmation ID (only on pending_confirm)
        """
        if not self._available:
            return {"status": "miss"}

        try:
            # 1. Calculate question embedding
            question_embedding_list = self.embedding_engine.encode([question])
            question_embedding = np.array(question_embedding_list[0], dtype=np.float32)

            # 2. Get all cached question IDs
            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)

            if not cached_ids or len(cached_ids) == 0:
                logger.debug("💭 Cache is empty, first query")
                return {"status": "miss"}

            logger.info(f"🔍 Starting semantic cache query - Current entries: {len(cached_ids)}")

            # 3. Iterate through cache entries, calculate similarity
            best_match = None
            best_similarity = 0.0
            best_id = None

            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

                # Get cached data
                cached_data = self.redis.hgetall(f"cache:question:{cache_id_str}")
                if not cached_data:
                    continue

                # Deserialize embedding
                cached_embedding = np.frombuffer(
                    cached_data[b'embedding'],
                    dtype=np.float32
                )

                # Calculate cosine similarity
                similarity = self._cosine_similarity(question_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_data
                    best_id = cache_id_str

            # 4. Layered handling based on similarity
            logger.info(f"🎯 Max Similarity: {best_similarity:.4f} "
                       f"(Direct: {self.threshold_direct}, Confirm: {self.threshold_confirm})")

            if best_similarity >= self.threshold_direct:
                # ✅ High similarity -> Direct hit
                cached_question = best_match[b'question'].decode('utf-8')
                cached_answer = best_match[b'answer'].decode('utf-8')

                logger.info(f"✅ Cache Direct Hit! Similarity: {best_similarity:.2%}")
                logger.info(f"   Cached Question: {cached_question[:50]}...")
                logger.info(f"   Current Question: {question[:50]}...")

                # Update stats
                self._update_hit_stats(best_id, cached_question)

                return {
                    "status": "hit",
                    "answer": cached_answer,
                    "cached_question": cached_question,
                    "similarity": best_similarity
                }

            elif best_similarity >= self.threshold_confirm:
                # ⚠️ Moderate similarity -> Needs confirmation
                cached_question = best_match[b'question'].decode('utf-8')
                cached_answer = best_match[b'answer'].decode('utf-8')

                logger.info(f"⚠️ Found similar question, waiting for confirmation (Similarity: {best_similarity:.2%})")
                logger.info(f"   Cached Question: {cached_question[:50]}...")
                logger.info(f"   Current Question: {question[:50]}...")

                # Generate unique confirmation ID
                confirmation_id = f"{session_id}_{int(datetime.now().timestamp() * 1000)}"

                # Store pending data (expires in 5 minutes)
                pending_data = {
                    "question": question,
                    "cached_question": cached_question,
                    "cached_id": best_id,
                    "similarity": best_similarity,
                    "cached_answer": cached_answer,
                    "timestamp": datetime.now().isoformat()
                }

                self.redis.setex(
                    f"cache:pending:{confirmation_id}",
                    300,  # 5 minutes
                    json.dumps(pending_data, ensure_ascii=False)
                )

                return {
                    "status": "pending_confirm",
                    "cached_question": cached_question,
                    "similarity": best_similarity,
                    "confirmation_id": confirmation_id
                }

            else:
                # ❌ Similarity too low -> Miss
                logger.info(f"❌ Cache Miss (Max Similarity: {best_similarity:.2%} < {self.threshold_confirm})")
                return {"status": "miss"}

        except Exception as e:
            logger.error(f"❌ Cache query error: {e}", exc_info=True)
            return {"status": "miss"}

    async def confirm_cache(self, confirmation_id: str, user_confirmed: bool) -> Optional[str]:
        """
        Handle user cache confirmation

        Args:
            confirmation_id: Confirmation ID
            user_confirmed: Whether user confirmed to use cache

        Returns:
            Cached answer if confirmed, else None
        """
        if not self._available:
            return None

        try:
            # Get pending data
            pending_key = f"cache:pending:{confirmation_id}"
            pending_data_json = self.redis.get(pending_key)

            if not pending_data_json:
                logger.warning(f"⚠️ Confirmation ID expired or not found: {confirmation_id}")
                return None

            pending_data = json.loads(pending_data_json.decode('utf-8'))

            if user_confirmed:
                # User confirmed similarity -> Use cached answer
                logger.info(f"✅ User confirmed similarity, using cached answer")
                logger.info(f"   Original Question: {pending_data['cached_question'][:50]}...")
                logger.info(f"   New Question: {pending_data['question'][:50]}...")

                # Update stats
                self._update_hit_stats(pending_data['cached_id'], pending_data['cached_question'])

                # Add new question to similar questions group
                self._add_similar_question(
                    pending_data['cached_id'],
                    pending_data['question']
                )

                # Delete pending data
                self.redis.delete(pending_key)

                return pending_data['cached_answer']

            else:
                # User denied similarity -> Re-query
                logger.info(f"❌ User denied similarity, will re-query")
                self.redis.delete(pending_key)
                return None

        except Exception as e:
            logger.error(f"❌ Error handling cache confirmation: {e}", exc_info=True)
            return None

    def set(
        self,
        question: str,
        answer: str,
        cache_type: str = "auto",
        quality_score: int = 0,
        source_info: str = None
    ):
        """
        Add new cache entry

        Args:
            question: Question text
            answer: Answer text
            cache_type: Cache type ("auto" | "confirmed" | "manual")
            quality_score: Quality score (0-10, manual=10, confirmed=5, auto=0)
            source_info: Source info (JSON string of parent_hash list or manual text)
        """
        if not self._available:
            return

        try:
            # 1. Calculate question embedding
            question_embedding_list = self.embedding_engine.encode([question])
            embedding = np.array(question_embedding_list[0], dtype=np.float32)

            # 2. Check cache size limit
            cache_size = self.redis.zcard("cache:embeddings")
            if cache_size >= self.max_cache_size:
                # LRU Eviction: Remove oldest entry
                oldest_ids = self.redis.zrange("cache:embeddings", 0, 0)
                if oldest_ids:
                    oldest_id = oldest_ids[0]
                    self._evict_cache(oldest_id)
                    logger.info(f"🗑️ Cache full, LRU evicted oldest entry")

            # 3. Compute Hash ID
            cache_id = self._compute_hash(question)

            # 4. Store cache data
            cache_data = {
                "question": question.encode('utf-8'),
                "answer": answer.encode('utf-8'),
                "embedding": embedding.tobytes(),
                "timestamp": datetime.now().isoformat().encode('utf-8'),
                "hit_count": b"0",
                "last_hit": b"",
                "cache_type": cache_type.encode('utf-8'),
                "quality_score": str(quality_score).encode('utf-8'),
                "source_info": (source_info or "").encode('utf-8')
            }

            self.redis.hset(
                f"cache:question:{cache_id}",
                mapping=cache_data
            )

            # 5. Add to time index (for LRU)
            self.redis.zadd(
                "cache:embeddings",
                {cache_id: datetime.now().timestamp()}
            )

            # 6. Cache persists (no TTL set)
            # Note: Only admin can delete or update cache

            # 7. Initialize popular questions stats (first store counts as 1 view)
            self.redis.zincrby("cache:popular", 1, question)

            # 8. Store cache type marker
            self.redis.set(f"cache:type:{cache_id}", cache_type)

            logger.info(f"💾 Added to cache: {cache_id[:8]}... | Type: {cache_type} | Quality: {quality_score} | Question: {question[:50]}...")

        except Exception as e:
            logger.error(f"❌ Error adding cache: {e}", exc_info=True)

    def _update_hit_stats(self, cache_id: str, question: str):
        """
        Update cache hit statistics

        Args:
            cache_id: Cache entry ID
            question: Question text
        """
        try:
            # 1. Increment hit count
            self.redis.hincrby(f"cache:question:{cache_id}", "hit_count", 1)
            self.redis.hset(
                f"cache:question:{cache_id}",
                "last_hit",
                datetime.now().isoformat().encode('utf-8')
            )

            # 2. Update popular questions rank (Sorted Set, by hit count)
            self.redis.zincrby("cache:popular", 1, question)

            # 3. Update LRU timestamp (Most recently used moves to back)
            self.redis.zadd(
                "cache:embeddings",
                {cache_id: datetime.now().timestamp()}
            )

        except Exception as e:
            logger.error(f"❌ Error updating stats: {e}", exc_info=True)

    def _add_similar_question(self, canonical_id: str, new_question: str):
        """
        Add new question to similar questions group

        Args:
            canonical_id: Canonical question ID
            new_question: New similar question
        """
        try:
            self.redis.hincrby(
                f"cache:similar:{canonical_id}",
                new_question,
                1
            )
        except Exception as e:
            logger.error(f"❌ Error adding similar question: {e}", exc_info=True)

    def _evict_cache(self, cache_id):
        """
        Evict cache entry (LRU)

        Args:
            cache_id: Cache ID to remove
        """
        try:
            cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

            # Delete main data
            self.redis.delete(f"cache:question:{cache_id_str}")

            # Remove from index
            self.redis.zrem("cache:embeddings", cache_id_str)

            # Delete similar questions mapping
            self.redis.delete(f"cache:similar:{cache_id_str}")

        except Exception as e:
            logger.error(f"❌ Error evicting cache: {e}", exc_info=True)

    def get_popular_questions(self, top_n: int = 3) -> List[Dict]:
        """
        Get popular questions (for frontend display)

        Args:
            top_n: Number of popular questions to return

        Returns:
            List of popular questions, each containing:
            - question: Question text
            - count: Cumulative visit count
            - similar_count: Number of similar questions
        """
        if not self._available:
            return []

        try:
            # Get top N from popular rank (descending score)
            popular = self.redis.zrevrange("cache:popular", 0, top_n - 1, withscores=True)

            result = []
            for question_bytes, count in popular:
                question = question_bytes.decode('utf-8') if isinstance(question_bytes, bytes) else question_bytes

                # Get ID for this question
                cache_id = self._compute_hash(question)

                # Get similar question count
                similar_count = self.redis.hlen(f"cache:similar:{cache_id}")

                result.append({
                    "question": question,
                    "count": int(count),
                    "similar_count": similar_count
                })

            return result

        except Exception as e:
            logger.error(f"❌ Error getting popular questions: {e}", exc_info=True)
            return []

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Stats dictionary containing:
            - available: Is cache service available
            - total_entries: Total cache entries
            - total_hits: Total hit count
            - popular_questions: List of popular questions
            - cache_by_type: Cache count grouped by type
        """
        if not self._available:
            return {
                "available": False,
                "total_entries": 0,
                "total_hits": 0,
                "popular_questions": [],
                "cache_by_type": {}
            }

        try:
            total_entries = self.redis.zcard("cache:embeddings")

            # Calculate total hits and stats by type
            total_hits = 0
            cache_by_type = {"auto": 0, "confirmed": 0, "manual": 0}

            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)
            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

                # Count hits
                hit_count = self.redis.hget(f"cache:question:{cache_id_str}", "hit_count")
                if hit_count:
                    total_hits += int(hit_count.decode('utf-8') if isinstance(hit_count, bytes) else hit_count)

                # Count by type
                cache_type = self.redis.get(f"cache:type:{cache_id_str}")
                if cache_type:
                    cache_type_str = cache_type.decode('utf-8') if isinstance(cache_type, bytes) else cache_type
                    if cache_type_str in cache_by_type:
                        cache_by_type[cache_type_str] += 1

            return {
                "available": True,
                "total_entries": total_entries,
                "total_hits": total_hits,
                "popular_questions": self.get_popular_questions(10),
                "cache_by_type": cache_by_type
            }

        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}", exc_info=True)
            return {
                "available": False,
                "total_entries": 0,
                "total_hits": 0,
                "popular_questions": [],
                "cache_by_type": {}
            }

    def clear_cache(self, cache_types: List[str] = None) -> int:
        """
        Clear cache

        Args:
            cache_types: List of cache types to clear. None means clear all.

        Returns:
            Number of deleted cache entries
        """
        if not self._available:
            return 0

        try:
            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)
            deleted_count = 0

            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

                # Filter by type if specified
                if cache_types:
                    cache_type = self.redis.get(f"cache:type:{cache_id_str}")
                    if cache_type:
                        cache_type_str = cache_type.decode('utf-8') if isinstance(cache_type, bytes) else cache_type
                        if cache_type_str not in cache_types:
                            continue

                # Delete cache
                self._evict_cache(cache_id_str)
                deleted_count += 1

            # Clear popular questions if clearing all
            if not cache_types:
                self.redis.delete("cache:popular")

            logger.info(f"🗑️ Cleared cache: {deleted_count} entries")
            return deleted_count

        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}", exc_info=True)
            return 0

    def get_all_cached_questions(self, limit: int = 100) -> List[Dict]:
        """
        Get all cached questions (for admin view)

        Args:
            limit: Maximum number to return

        Returns:
            List of cached questions
        """
        if not self._available:
            return []

        try:
            cached_ids = self.redis.zrange("cache:embeddings", 0, limit - 1)
            result = []

            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id
                cached_data = self.redis.hgetall(f"cache:question:{cache_id_str}")

                if cached_data:
                    question = cached_data[b'question'].decode('utf-8')
                    answer = cached_data.get(b'answer', b'').decode('utf-8')
                    hit_count = int(cached_data.get(b'hit_count', b'0').decode('utf-8'))
                    timestamp = cached_data.get(b'timestamp', b'').decode('utf-8')
                    cache_type = cached_data.get(b'cache_type', b'auto').decode('utf-8')
                    quality_score = int(cached_data.get(b'quality_score', b'0').decode('utf-8'))
                    source_info = cached_data.get(b'source_info', b'').decode('utf-8')

                    result.append({
                        "cache_id": cache_id_str,
                        "question": question,
                        "answer": answer,
                        "hit_count": hit_count,
                        "timestamp": timestamp,
                        "cache_type": cache_type,
                        "quality_score": quality_score,
                        "source_info": source_info
                    })

            return result

        except Exception as e:
            logger.error(f"❌ Error getting cache list: {e}", exc_info=True)
            return []
