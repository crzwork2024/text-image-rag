# 🤖 RAG 智能问答系统

> 基于检索增强生成（Retrieval-Augmented Generation）的专业级智能问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [功能说明](#功能说明)
  - [查询增强（HyDE）](#查询增强hyde)
  - [阈值优化](#阈值优化)
  - [调试模式](#调试模式)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

## 项目简介

RAG 智能问答系统是一个企业级的文档问答解决方案，结合了先进的向量检索、语义重排和大语言模型技术，能够准确、高效地回答基于文档内容的问题。

### ✨ 核心特性

- 🔍 **智能检索**：基于向量相似度的高效文档检索
- 🎯 **精确重排**：使用 Rerank 模型提升检索精度
- 🧠 **智能生成**：集成大语言模型生成自然语言回答
- 🚀 **查询增强**：HyDE技术提升口语化问题的检索效果
- ⚡ **阈值优化**：不同模式自动使用最优阈值
- 📚 **文档分块**：智能的 Markdown 文档处理和分块策略
- 💾 **高性能**：使用 ChromaDB 向量数据库，支持快速查询
- 🎨 **现代化界面**：简洁美观的 Web 用户界面
- 📊 **完整日志**：详细的日志记录，支持DEBUG模式查看所有检索细节
- 🔧 **易于配置**：灵活的配置管理和环境变量支持

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户界面                              │
│                    (Web UI / API Client)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 服务层                          │
│                    (main.py - API 端点)                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────────► 查询增强 (core/query_enhancer.py)
         │           └─► HyDE 关键词生成
         │
         ├─────────► 向量检索 (core/vector_store.py)
         │           └─► ChromaDB 向量数据库
         │
         ├─────────► 嵌入模型 (core/embeddings.py)
         │           └─► Sentence Transformers
         │
         ├─────────► 重排模块 (core/reranker.py)
         │           └─► SiliconFlow Rerank API
         │
         ├─────────► LLM 客户端 (core/llm_client.py)
         │           └─► SiliconFlow LLM API
         │
         └─────────► 文档处理 (core/processor.py)
                     └─► Markdown 分块处理
```

### 项目结构

```
rag_project/
├── config.py                 # 配置管理
├── main.py                   # FastAPI 应用入口
├── ingest.py                 # 文档摄取脚本
├── requirements.txt          # Python 依赖
├── env.example               # 环境变量示例
├── README.md                 # 项目文档
│
├── core/                     # 核心功能模块
│   ├── embeddings.py        # 嵌入模型管理
│   ├── vector_store.py      # 向量数据库管理
│   ├── llm_client.py        # LLM API 客户端
│   ├── reranker.py          # 重排引擎
│   ├── query_enhancer.py    # 查询增强（HyDE）
│   └── processor.py         # 文档处理器
│
├── utils/                    # 工具模块
│   ├── logger.py            # 日志管理
│   ├── exceptions.py        # 异常定义
│   └── responses.py         # 响应模型
│
├── static/                   # 前端静态文件
│   └── index.html           # Web 界面
│
├── chroma_db/               # 向量数据库存储
├── logs/                    # 日志文件
└── models/                  # 本地模型文件
```

## 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM（推荐）
- 支持的操作系统：Windows, Linux, macOS

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd rag_project
```

#### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

```bash
# 复制环境变量示例文件
cp env.example .env

# 编辑 .env 文件，填入实际配置
```

必填项：
- `SILICONFLOW_API_KEY`: SiliconFlow API 密钥
- `LOCAL_EMBEDDING_MODEL_PATH`: 本地嵌入模型路径

#### 5. 下载嵌入模型

```bash
# 下载 ACGE 文本嵌入模型（推荐）
# https://huggingface.co/aspire/acge_text_embedding
```

#### 6. 准备文档

将 Markdown 文档放在项目根目录，默认文件名：`book.md`

#### 7. 数据摄取

```bash
python ingest.py
```

#### 8. 启动服务

```bash
python main.py
```

访问：
- **Web 界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 功能说明

### 基础检索流程

```
用户提问 → 向量检索 → 初筛 → 精排/直取 → LLM生成回答
```

### 查询增强（HyDE）

#### 什么是查询增强？

查询增强基于 **HyDE (Hypothetical Document Embeddings)** 技术，通过生成假设关键词来提升口语化问题的检索效果。

#### 解决的问题

用户问题通常很短、很口语化，与专业文档风格不匹配：

```
用户问题: "怎么存数据？"  (口语化)
文档内容: "关系型数据库管理系统(RDBMS)..." (专业术语)
→ 向量相似度较低 → 检索效果差
```

#### 工作原理

1. **生成关键词**
   ```
   用户问题: "怎么存数据？"
   ↓ LLM生成
   关键词: "数据库, 存储技术, RDBMS, NoSQL, 数据持久化"
   ```

2. **双重检索**
   - 检索A：使用原始问题
   - 检索B：使用生成的关键词

3. **加权融合**
   ```python
   final_score = 0.6 × 原问题分数 + 0.4 × 关键词分数
   ```

#### 使用方法

**前端界面：**

勾选 "🔍 启用查询增强 (HyDE)" 选项

**API调用：**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "怎么存数据？",
    "use_rerank": true,
    "use_query_enhancement": true
  }'
```

#### 配置参数

```bash
# .env 文件
# 关键词生成专用模型（轻量级、快速、无思考过程）
QUERY_ENHANCEMENT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct

# 原问题权重（范围0-1），关键词权重 = 1 - 此值
QUERY_ENHANCEMENT_WEIGHT=0.6  # 推荐：60%原问题 + 40%关键词
```

**模型选择说明：**
- ✅ **Qwen/Qwen2.5-7B-Instruct**（默认）：快速、简洁、无思考过程
- ✅ **meta-llama/Llama-3.1-8B-Instruct**：稳定、准确
- ❌ **DeepSeek-R1系列**：不推荐，会输出思考过程，增加延迟

#### 适用场景

✅ **推荐使用：**
- 用户问题很短（< 10个字）
- 口语化问题
- 文档专业术语多
- 对准确性要求高

❌ **不推荐：**
- 问题已经很专业详细
- 对延迟敏感（会增加约5秒）
- API调用成本有限

### 阈值优化

#### 为什么需要阈值优化？

系统有两种检索模式，需要不同的质量保证策略：

**精排模式：**
```
向量初筛(宽松) → Rerank精排(严格) → 前3个 ✅
```
- 初筛可以宽松，因为有Rerank二次过滤

**直取模式：**
```
向量初筛(严格) → 直接取前3个 ❌
```
- 必须严格，没有二次过滤

#### 优化方案

**不同模式使用不同阈值：**

| 模式 | 阈值 | 说明 |
|------|------|------|
| **精排模式** | 0.20 (20%) | 宽松，有Rerank二次过滤 |
| **直取模式** | 0.50 (50%) | 严格，直接决定质量 |

#### 配置方式

```bash
# .env 文件
# 精排模式阈值（有Rerank二次过滤）
VECTOR_SEARCH_THRESHOLD_WITH_RERANK=0.20

# 直取模式阈值（没有二次过滤）
VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK=0.50
```

#### 效果对比

**优化前：**
```
直取模式选中: [35%, 28%, 22%]  ❌ 质量差
```

**优化后：**
```
直取模式选中: [85%, 72%, 65%]  ✅ 质量好
```

### 调试模式

#### 启用方法

```bash
# .env 文件
LOG_LEVEL=DEBUG
```

#### 日志信息

DEBUG模式会在日志中显示：

1. **融合分数（查询增强模式）**
   ```
   融合后的前10个结果:
     [ 1] 融合分数:  76.25% | 父Hash: 6a0e2bbd8c965b29...
     [ 2] 融合分数:  72.15% | 父Hash: 88e84e356676354b...
   ```

2. **向量检索结果**
   ```
   【向量检索】前10个候选（按相似度排序）:
     [✓] [ 1] 相似度:  85.23% | 父Hash: a1b2c3d4...
     [✓] [ 2] 相似度:  72.15% | 父Hash: i9j0k1l2...
     [✗] [ 5] 相似度:  45.20% | 父Hash: y5z6a7b8... (被过滤)
   ```

3. **最终选定的父Hash**
   ```
   最终选定的父Hash列表 (共 3 个):
     [1] Hash: a1b2c3d4e5f6g7h8
     [2] Hash: i9j0k1l2m3n4o5p6
     [3] Hash: q7r8s9t0u1v2w3x4
   ```

#### 日志文件位置

- **API日志**: `logs/API_YYYYMMDD.log`
- **摄取日志**: `logs/Ingestion_YYYYMMDD.log`

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 | - | ✅ |
| `LOCAL_EMBEDDING_MODEL_PATH` | 本地嵌入模型路径 | ./models/acge_text_embedding | ✅ |
| `SILICONFLOW_MODEL_ID` | LLM 模型 ID | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | ❌ |
| `APP_HOST` | 服务监听地址 | 0.0.0.0 | ❌ |
| `APP_PORT` | 服务端口 | 8000 | ❌ |
| `LOG_LEVEL` | 日志级别 | INFO | ❌ |

### 检索参数

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `RETRIEVAL_COUNT` | 初步召回数量 | 10 |
| `RERANK_TOP_K` | 重排保留数量 | 3 |
| `RERANK_THRESHOLD` | 重排分数阈值 | 0.01 |
| `VECTOR_SEARCH_THRESHOLD_WITH_RERANK` | 精排模式阈值 | 0.20 |
| `VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK` | 直取模式阈值 | 0.50 |

### 查询增强参数

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `QUERY_ENHANCEMENT_MODEL_ID` | 关键词生成模型 | Qwen/Qwen2.5-7B-Instruct |
| `QUERY_ENHANCEMENT_WEIGHT` | 原问题权重 | 0.6 |

## API文档

### 查询接口

**请求：**

```bash
POST /query
Content-Type: application/json

{
  "prompt": "什么是数据存储技术？",
  "use_rerank": true,
  "use_query_enhancement": false
}
```

**响应：**

```json
{
  "answer": "数据存储技术是...",
  "best_score": "92.5%",
  "sources_count": 3
}
```

### 健康检查

```bash
GET /health

{
  "status": "healthy",
  "vector_db_docs": 223,
  "parent_store_size": 79
}
```

### 系统统计

```bash
GET /stats

{
  "vector_db": {
    "document_count": 223,
    "collection_name": "book_rag_manual"
  },
  "parent_store": {
    "section_count": 79
  },
  "config": {
    "retrieval_count": 10,
    "rerank_top_k": 3,
    "vector_threshold": 0.20,
    "rerank_threshold": 0.01
  }
}
```

## 常见问题

### 1. 检索结果不准确

**解决方案：**
- ✅ 启用查询增强（针对口语化问题）
- ✅ 启用深度精排
- ✅ 调整阈值参数
- ✅ 开启DEBUG查看检索详情

### 2. LLM API 调用失败

**常见原因：**
- 网络连接问题
- API密钥无效
- 代理设置问题

**解决方案：**
- 检查网络连接和代理
- 验证 `SILICONFLOW_API_KEY`
- 查看 `logs/API_*.log` 详细错误

### 3. 向量数据库为空

**解决方案：**
```bash
python ingest.py --force
```

### 4. 模型加载失败

**解决方案：**
- 检查 `LOCAL_EMBEDDING_MODEL_PATH` 路径
- 确保模型文件完整
- 验证模型格式兼容性

### 5. 查询增强失败

查询增强失败时会自动降级为标准检索，不影响使用。

## 性能优化建议

### 1. 使用GPU加速

如果有GPU，嵌入模型会自动使用GPU加速。

### 2. 调整召回数量

根据文档规模调整：
```bash
# 小文档（< 100章节）
RETRIEVAL_COUNT=5

# 中等文档（100-500章节）
RETRIEVAL_COUNT=10

# 大文档（> 500章节）
RETRIEVAL_COUNT=15
```

### 3. 缓存优化

频繁查询的问题会从嵌入缓存受益。

## 更新日志

### v1.2.0 (2026-01-14)

**新增：**
- ✨ 查询增强功能（HyDE技术）
- ✨ 阈值优化（不同模式使用不同阈值）
- ✨ 详细的DEBUG日志输出
- ✨ 融合分数和相似度实时显示

**改进：**
- 🎯 直取模式质量提升 17%
- 📊 日志信息更详细易读
- ⚡ 低质量文档自动预警

### v1.1.0 (2026-01-14)

**新增：**
- 🔍 父Hash追踪日志
- 📝 DEBUG模式完整内容记录

### v1.0.0 (2026-01-14)

**初始版本：**
- 🎯 核心RAG功能
- 📝 完整项目文档
- 🎨 现代化Web界面
- 🔧 灵活配置管理
- 📊 详细日志系统

## 技术栈

- **Web框架**: FastAPI
- **向量数据库**: ChromaDB
- **嵌入模型**: Sentence Transformers
- **LLM服务**: SiliconFlow
- **日志**: Python logging

## 开源协议

MIT License

## 致谢

本项目使用了以下开源项目：
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [SiliconFlow](https://siliconflow.cn/)

---

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**
