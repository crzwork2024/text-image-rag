"""
语义缓存模块 - 基于 Redis 和 Embedding 的智能缓存
作者: RAG 项目团队
描述: 使用向量相似度进行问题匹配，支持用户确认机制
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
    语义缓存类

    功能:
    - 基于 Embedding 的语义相似度匹配
    - 三层阈值策略（直接返回/用户确认/未命中）
    - LRU 缓存淘汰
    - 热门问题统计
    - 相似问题聚类
    """

    def __init__(self, embedding_engine):
        """
        初始化语义缓存

        参数:
            embedding_engine: 嵌入引擎实例（用于计算问题向量）
        """
        try:
            # Redis 连接配置
            redis_config = {
                'host': config.REDIS_HOST,
                'port': config.REDIS_PORT,
                'db': config.REDIS_DB,
                'decode_responses': False,  # 保留二进制数据（用于存储 embedding）
                'socket_timeout': 5,
                'socket_connect_timeout': 5
            }

            # 只有在密码非空时才添加密码参数
            if config.REDIS_PASSWORD:
                redis_config['password'] = config.REDIS_PASSWORD

            self.redis = redis.Redis(**redis_config)

            # 测试连接
            self.redis.ping()
            self._available = True
            logger.info(f"✅ Redis 连接成功: {config.REDIS_HOST}:{config.REDIS_PORT}")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"⚠️ Redis 连接失败，缓存功能已禁用: {e}")
            self._available = False
            self.redis = None

        # 嵌入引擎
        self.embedding_engine = embedding_engine

        # 相似度阈值（从配置读取）
        self.threshold_direct = config.CACHE_THRESHOLD_DIRECT      # 0.98 - 直接返回
        self.threshold_confirm = config.CACHE_THRESHOLD_CONFIRM    # 0.95 - 需要确认

        # 缓存配置
        self.cache_ttl = config.CACHE_TTL                         # 过期时间（秒）
        self.max_cache_size = config.CACHE_MAX_SIZE               # 最大缓存条目数

        if self._available:
            logger.info(f"📦 缓存配置 - 直接阈值: {self.threshold_direct}, "
                       f"确认阈值: {self.threshold_confirm}, "
                       f"TTL: {self.cache_ttl}s, "
                       f"最大容量: {self.max_cache_size}")

    def is_available(self) -> bool:
        """检查缓存服务是否可用"""
        return self._available

    def _compute_hash(self, text: str) -> str:
        """
        计算文本的哈希ID

        参数:
            text: 输入文本

        返回:
            16位哈希字符串
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        参数:
            emb1: 向量1
            emb2: 向量2

        返回:
            相似度分数 (0-1)
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
        查询缓存

        参数:
            question: 用户问题
            session_id: 会话ID

        返回:
            字典，包含以下键:
            - status: "hit" | "pending_confirm" | "miss"
            - answer: 答案内容（仅当 hit 时）
            - cached_question: 相似问题（hit 或 pending_confirm 时）
            - similarity: 相似度分数（hit 或 pending_confirm 时）
            - confirmation_id: 确认ID（仅当 pending_confirm 时）
        """
        if not self._available:
            return {"status": "miss"}

        try:
            # 1. 计算问题的embedding（使用embedding_engine）
            question_embedding_list = self.embedding_engine.encode([question])
            question_embedding = np.array(question_embedding_list[0], dtype=np.float32)

            # 2. 获取所有缓存的问题ID
            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)

            if not cached_ids or len(cached_ids) == 0:
                logger.debug("💭 缓存为空，首次查询")
                return {"status": "miss"}

            logger.info(f"🔍 开始语义缓存查询 - 当前缓存: {len(cached_ids)} 条")

            # 3. 遍历所有缓存条目，计算相似度
            best_match = None
            best_similarity = 0.0
            best_id = None

            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

                # 获取缓存的数据
                cached_data = self.redis.hgetall(f"cache:question:{cache_id_str}")
                if not cached_data:
                    continue

                # 反序列化 embedding
                cached_embedding = np.frombuffer(
                    cached_data[b'embedding'],
                    dtype=np.float32
                )

                # 计算余弦相似度
                similarity = self._cosine_similarity(question_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_data
                    best_id = cache_id_str

            # 4. 根据相似度分层处理
            logger.info(f"🎯 最高相似度: {best_similarity:.4f} "
                       f"(直接阈值: {self.threshold_direct}, 确认阈值: {self.threshold_confirm})")

            if best_similarity >= self.threshold_direct:
                # ✅ 高度相似 → 直接返回缓存
                cached_question = best_match[b'question'].decode('utf-8')
                cached_answer = best_match[b'answer'].decode('utf-8')

                logger.info(f"✅ 缓存直接命中! 相似度: {best_similarity:.2%}")
                logger.info(f"   缓存问题: {cached_question[:50]}...")
                logger.info(f"   当前问题: {question[:50]}...")

                # 更新统计信息
                self._update_hit_stats(best_id, cached_question)

                return {
                    "status": "hit",
                    "answer": cached_answer,
                    "cached_question": cached_question,
                    "similarity": best_similarity
                }

            elif best_similarity >= self.threshold_confirm:
                # ⚠️ 中等相似 → 需要用户确认
                cached_question = best_match[b'question'].decode('utf-8')
                cached_answer = best_match[b'answer'].decode('utf-8')

                logger.info(f"⚠️ 发现相似问题，等待用户确认 (相似度: {best_similarity:.2%})")
                logger.info(f"   缓存问题: {cached_question[:50]}...")
                logger.info(f"   当前问题: {question[:50]}...")

                # 生成唯一的确认ID
                confirmation_id = f"{session_id}_{int(datetime.now().timestamp() * 1000)}"

                # 存储待确认数据（5分钟过期）
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
                    300,  # 5分钟过期
                    json.dumps(pending_data, ensure_ascii=False)
                )

                return {
                    "status": "pending_confirm",
                    "cached_question": cached_question,
                    "similarity": best_similarity,
                    "confirmation_id": confirmation_id
                }

            else:
                # ❌ 相似度太低 → 缓存未命中
                logger.info(f"❌ 缓存未命中 (最高相似度: {best_similarity:.2%} < {self.threshold_confirm})")
                return {"status": "miss"}

        except Exception as e:
            logger.error(f"❌ 缓存查询出错: {e}", exc_info=True)
            return {"status": "miss"}

    async def confirm_cache(self, confirmation_id: str, user_confirmed: bool) -> Optional[str]:
        """
        处理用户的缓存确认

        参数:
            confirmation_id: 确认ID
            user_confirmed: 用户是否确认使用缓存

        返回:
            如果用户确认，返回缓存的答案；否则返回 None
        """
        if not self._available:
            return None

        try:
            # 获取待确认数据
            pending_key = f"cache:pending:{confirmation_id}"
            pending_data_json = self.redis.get(pending_key)

            if not pending_data_json:
                logger.warning(f"⚠️ 确认ID已过期或不存在: {confirmation_id}")
                return None

            pending_data = json.loads(pending_data_json.decode('utf-8'))

            if user_confirmed:
                # 用户确认是相同问题 → 使用缓存答案
                logger.info(f"✅ 用户确认相似，使用缓存答案")
                logger.info(f"   原问题: {pending_data['cached_question'][:50]}...")
                logger.info(f"   新问题: {pending_data['question'][:50]}...")

                # 更新统计信息
                self._update_hit_stats(pending_data['cached_id'], pending_data['cached_question'])

                # 将新问题添加到相似问题列表
                self._add_similar_question(
                    pending_data['cached_id'],
                    pending_data['question']
                )

                # 删除待确认数据
                self.redis.delete(pending_key)

                return pending_data['cached_answer']

            else:
                # 用户否认是相同问题 → 需要重新检索
                logger.info(f"❌ 用户否认相似，将重新检索")
                self.redis.delete(pending_key)
                return None

        except Exception as e:
            logger.error(f"❌ 处理缓存确认时出错: {e}", exc_info=True)
            return None

    def set(
        self,
        question: str,
        answer: str,
        cache_type: str = "auto",
        quality_score: int = 0
    ):
        """
        添加新的缓存条目

        参数:
            question: 问题文本
            answer: 答案文本
            cache_type: 缓存类型 ("auto" | "confirmed" | "manual")
            quality_score: 质量分数 (0-10, manual=10, confirmed=5, auto=0)
        """
        if not self._available:
            return

        try:
            # 1. 计算问题的embedding
            question_embedding_list = self.embedding_engine.encode([question])
            embedding = np.array(question_embedding_list[0], dtype=np.float32)

            # 2. 检查缓存大小限制
            cache_size = self.redis.zcard("cache:embeddings")
            if cache_size >= self.max_cache_size:
                # LRU 淘汰：删除最旧的条目
                oldest_ids = self.redis.zrange("cache:embeddings", 0, 0)
                if oldest_ids:
                    oldest_id = oldest_ids[0]
                    self._evict_cache(oldest_id)
                    logger.info(f"🗑️ 缓存已满，LRU淘汰最旧条目")

            # 3. 计算哈希ID
            cache_id = self._compute_hash(question)

            # 4. 存储缓存数据
            cache_data = {
                "question": question.encode('utf-8'),
                "answer": answer.encode('utf-8'),
                "embedding": embedding.tobytes(),
                "timestamp": datetime.now().isoformat().encode('utf-8'),
                "hit_count": b"0",
                "last_hit": b"",
                "cache_type": cache_type.encode('utf-8'),
                "quality_score": str(quality_score).encode('utf-8')
            }

            self.redis.hset(
                f"cache:question:{cache_id}",
                mapping=cache_data
            )

            # 5. 添加到时间索引（用于 LRU）
            self.redis.zadd(
                "cache:embeddings",
                {cache_id: datetime.now().timestamp()}
            )

            # 6. 设置 TTL
            self.redis.expire(f"cache:question:{cache_id}", self.cache_ttl)

            # 7. 初始化热门问题统计（首次存储也算作1次访问）
            self.redis.zincrby("cache:popular", 1, question)
            
            # 8. 存储缓存类型标记
            self.redis.set(f"cache:type:{cache_id}", cache_type)

            logger.info(f"💾 添加到缓存: {cache_id[:8]}... | 类型: {cache_type} | 质量: {quality_score} | 问题: {question[:50]}...")

        except Exception as e:
            logger.error(f"❌ 添加缓存时出错: {e}", exc_info=True)

    def _update_hit_stats(self, cache_id: str, question: str):
        """
        更新缓存命中统计

        参数:
            cache_id: 缓存条目ID
            question: 问题文本
        """
        try:
            # 1. 增加该缓存条目的命中次数
            self.redis.hincrby(f"cache:question:{cache_id}", "hit_count", 1)
            self.redis.hset(
                f"cache:question:{cache_id}",
                "last_hit",
                datetime.now().isoformat().encode('utf-8')
            )

            # 2. 更新热门问题排行（Sorted Set，按命中次数排序）
            self.redis.zincrby("cache:popular", 1, question)

            # 3. 更新 LRU 时间戳（最近使用的排到后面）
            self.redis.zadd(
                "cache:embeddings",
                {cache_id: datetime.now().timestamp()}
            )

        except Exception as e:
            logger.error(f"❌ 更新统计信息时出错: {e}", exc_info=True)

    def _add_similar_question(self, canonical_id: str, new_question: str):
        """
        将新问题添加到相似问题组

        参数:
            canonical_id: 代表性问题的ID
            new_question: 新的相似问题
        """
        try:
            self.redis.hincrby(
                f"cache:similar:{canonical_id}",
                new_question,
                1
            )
        except Exception as e:
            logger.error(f"❌ 添加相似问题时出错: {e}", exc_info=True)

    def _evict_cache(self, cache_id):
        """
        删除缓存条目（LRU 淘汰）

        参数:
            cache_id: 要删除的缓存ID
        """
        try:
            cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id

            # 删除主数据
            self.redis.delete(f"cache:question:{cache_id_str}")

            # 从索引中删除
            self.redis.zrem("cache:embeddings", cache_id_str)

            # 删除相似问题映射
            self.redis.delete(f"cache:similar:{cache_id_str}")

        except Exception as e:
            logger.error(f"❌ 删除缓存时出错: {e}", exc_info=True)

    def get_popular_questions(self, top_n: int = 3) -> List[Dict]:
        """
        获取最热门的问题（供前端显示）

        参数:
            top_n: 返回的热门问题数量

        返回:
            热门问题列表，每项包含:
            - question: 问题文本
            - count: 累计访问次数
            - similar_count: 相似问题数量
        """
        if not self._available:
            return []

        try:
            # 从热门排行中获取 top N（按分数降序）
            popular = self.redis.zrevrange("cache:popular", 0, top_n - 1, withscores=True)

            result = []
            for question_bytes, count in popular:
                question = question_bytes.decode('utf-8') if isinstance(question_bytes, bytes) else question_bytes

                # 获取这个问题的ID
                cache_id = self._compute_hash(question)

                # 获取相似问题数量
                similar_count = self.redis.hlen(f"cache:similar:{cache_id}")

                result.append({
                    "question": question,
                    "count": int(count),
                    "similar_count": similar_count
                })

            return result

        except Exception as e:
            logger.error(f"❌ 获取热门问题时出错: {e}", exc_info=True)
            return []

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        返回:
            统计信息字典，包含:
            - available: 缓存服务是否可用
            - total_entries: 缓存条目总数
            - total_hits: 总命中次数
            - popular_questions: 热门问题列表
            - cache_by_type: 按类型分组的缓存数量
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

            # 计算总命中次数和按类型统计
            total_hits = 0
            cache_by_type = {"auto": 0, "confirmed": 0, "manual": 0}
            
            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)
            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id
                
                # 统计命中次数
                hit_count = self.redis.hget(f"cache:question:{cache_id_str}", "hit_count")
                if hit_count:
                    total_hits += int(hit_count.decode('utf-8') if isinstance(hit_count, bytes) else hit_count)
                
                # 统计缓存类型
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
            logger.error(f"❌ 获取缓存统计时出错: {e}", exc_info=True)
            return {
                "available": False,
                "total_entries": 0,
                "total_hits": 0,
                "popular_questions": [],
                "cache_by_type": {}
            }
    
    def clear_cache(self, cache_types: List[str] = None) -> int:
        """
        清除缓存
        
        参数:
            cache_types: 要清除的缓存类型列表，None 表示清除所有
            
        返回:
            删除的缓存条目数
        """
        if not self._available:
            return 0
        
        try:
            cached_ids = self.redis.zrange("cache:embeddings", 0, -1)
            deleted_count = 0
            
            for cache_id in cached_ids:
                cache_id_str = cache_id.decode('utf-8') if isinstance(cache_id, bytes) else cache_id
                
                # 如果指定了类型过滤
                if cache_types:
                    cache_type = self.redis.get(f"cache:type:{cache_id_str}")
                    if cache_type:
                        cache_type_str = cache_type.decode('utf-8') if isinstance(cache_type, bytes) else cache_type
                        if cache_type_str not in cache_types:
                            continue
                
                # 删除缓存
                self._evict_cache(cache_id_str)
                deleted_count += 1
            
            # 如果清除所有，也清空热门问题
            if not cache_types:
                self.redis.delete("cache:popular")
            
            logger.info(f"🗑️ 清除缓存: {deleted_count} 条")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 清除缓存时出错: {e}", exc_info=True)
            return 0
    
    def get_all_cached_questions(self, limit: int = 100) -> List[Dict]:
        """
        获取所有缓存的问题列表（用于管理员查看）
        
        参数:
            limit: 返回的最大数量
            
        返回:
            缓存问题列表
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
                    
                    result.append({
                        "cache_id": cache_id_str,
                        "question": question,
                        "answer": answer,
                        "hit_count": hit_count,
                        "timestamp": timestamp,
                        "cache_type": cache_type,
                        "quality_score": quality_score
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 获取缓存列表时出错: {e}", exc_info=True)
            return []