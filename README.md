# 🤖 RAG 智能问答系统

> 基于检索增强生成（Retrieval-Augmented Generation）的专业级智能问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

RAG 智能问答系统是一个企业级的文档问答解决方案，结合了先进的向量检索、语义重排和大语言模型技术，能够准确、高效地回答基于文档内容的问题。

### ✨ 核心特性

- 🔍 **智能检索**：基于向量相似度的高效文档检索
- 🎯 **精确重排**：使用 Rerank 模型提升检索精度
- 🧠 **智能生成**：集成大语言模型生成自然语言回答
- 📚 **文档分块**：智能的 Markdown 文档处理和分块策略
- 🚀 **高性能**：使用 ChromaDB 向量数据库，支持快速查询
- 🎨 **现代化界面**：简洁美观的 Web 用户界面
- 📊 **完整日志**：详细的日志记录和错误追踪
- 🔧 **易于配置**：灵活的配置管理和环境变量支持

## 🏗️ 系统架构

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
         ├─────────► 向量检索模块 (core/vector_store.py)
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

## 📁 项目结构

```
rag_project/
├── config.py                 # 配置管理
├── main.py                   # FastAPI 应用入口
├── ingest.py                 # 文档摄取脚本
├── requirements.txt          # Python 依赖
├── env.example               # 环境变量示例
├── README.md                 # 项目文档
├── .gitignore               # Git 忽略配置
│
├── core/                     # 核心功能模块
│   ├── __init__.py
│   ├── embeddings.py        # 嵌入模型管理
│   ├── vector_store.py      # 向量数据库管理
│   ├── llm_client.py        # LLM API 客户端
│   ├── reranker.py          # 重排引擎
│   └── processor.py         # 文档处理器
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── logger.py            # 日志管理
│   ├── exceptions.py        # 异常定义
│   └── responses.py         # 响应模型
│
├── static/                   # 前端静态文件
│   └── index.html           # Web 界面
│
├── chroma_db/               # 向量数据库存储
├── logs/                    # 日志文件
├── images/                  # 图片资源
└── models/                  # 本地模型文件
```

## 🚀 快速开始

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
# 必填项：
# - SILICONFLOW_API_KEY: SiliconFlow API 密钥
# - LOCAL_EMBEDDING_MODEL_PATH: 本地嵌入模型路径
```

#### 5. 下载嵌入模型

```bash
# 下载 ACGE 文本嵌入模型（推荐）
# 方式1：从 HuggingFace 下载
# https://huggingface.co/aspire/acge_text_embedding

# 方式2：使用自己的嵌入模型
# 确保模型路径在 .env 中正确配置
```

#### 6. 准备文档

```bash
# 将要问答的 Markdown 文档放在项目根目录
# 默认文件名：book.md
# 可在 config.py 中修改路径
```

#### 7. 数据摄取

```bash
# 执行数据摄取，将文档导入向量数据库
python ingest.py

# 可选参数：
# --file <path>  指定 Markdown 文件路径
# --force        强制重新摄取（清空现有数据）
```

#### 8. 启动服务

```bash
python main.py
```

服务启动后，访问：

- **Web 界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 📝 使用指南

### Web 界面使用

1. 打开浏览器访问 `http://localhost:8000`
2. 在输入框中输入问题
3. 选择是否启用深度精排（Rerank）
4. 点击"发送提问"按钮
5. 等待 AI 生成回答

### API 调用示例

#### 查询接口

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "什么是数据存储技术？",
    "use_rerank": true
  }'
```

#### 响应示例

```json
{
  "answer": "数据存储技术是...",
  "best_score": "92.5%",
  "sources_count": 3
}
```

#### 健康检查

```bash
curl http://localhost:8000/health
```

#### 系统统计

```bash
curl http://localhost:8000/stats
```

### Python SDK 示例

```python
import requests

# 发送查询
response = requests.post(
    "http://localhost:8000/query",
    json={
        "prompt": "什么是 RAG？",
        "use_rerank": True
    }
)

result = response.json()
print(f"回答: {result['answer']}")
print(f"相关度: {result['best_score']}")
print(f"来源数: {result['sources_count']}")
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 | - | ✅ |
| `SILICONFLOW_MODEL_ID` | LLM 模型 ID | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | ❌ |
| `LOCAL_EMBEDDING_MODEL_PATH` | 本地嵌入模型路径 | ./models/acge_text_embedding | ✅ |
| `APP_HOST` | 服务监听地址 | 0.0.0.0 | ❌ |
| `APP_PORT` | 服务端口 | 8000 | ❌ |
| `RETRIEVAL_COUNT` | 初步召回数量 | 10 | ❌ |
| `RERANK_TOP_K` | 重排保留数量 | 3 | ❌ |
| `VECTOR_SEARCH_THRESHOLD` | 向量搜索阈值 | 0.20 | ❌ |
| `RERANK_THRESHOLD` | 重排分数阈值 | 0.01 | ❌ |
| `LOG_LEVEL` | 日志级别 | INFO | ❌ |

### 检索参数调优

#### 1. 召回阶段

- **RETRIEVAL_COUNT**: 控制初步召回的文档数量
  - 数值越大，召回越全面，但速度较慢
  - 推荐范围：5-20

- **VECTOR_SEARCH_THRESHOLD**: 向量相似度阈值
  - 数值越高，过滤越严格
  - 推荐范围：0.1-0.4

#### 2. 重排阶段

- **RERANK_TOP_K**: 重排后保留的文档数量
  - 通常保留 3-5 个最相关的文档
  - 推荐范围：2-5

- **RERANK_THRESHOLD**: 重排分数阈值
  - 过滤低质量重排结果
  - 推荐范围：0.01-0.1

## 🔧 开发指南

### 代码规范

项目遵循以下代码规范：

- PEP 8 Python 代码风格
- 使用 Black 进行代码格式化
- 使用 Type Hints 进行类型注解
- 详细的中文注释和文档字符串

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行测试
pytest tests/

# 运行特定测试文件
pytest tests/test_embeddings.py

# 显示覆盖率
pytest --cov=core --cov-report=html
```

### 代码格式化

```bash
# 使用 Black 格式化代码
black .

# 检查代码风格
black --check .
```

### 添加新功能

1. 在相应模块中添加功能代码
2. 添加详细的中文注释
3. 更新 `__init__.py` 导出新功能
4. 编写单元测试
5. 更新文档

## 📊 性能优化

### 向量数据库优化

- 使用 ChromaDB 的持久化存储
- 合理设置批处理大小
- 定期清理无用数据

### 嵌入模型优化

- 使用 GPU 加速（如果可用）
- 批量处理文本
- 缓存常见查询的向量

### API 性能优化

- 使用异步处理
- 实现请求缓存
- 限流和负载均衡

## 🐛 常见问题

### 1. 模型加载失败

**问题**: `ModelLoadError: 嵌入模型加载失败`

**解决**:
- 检查模型路径是否正确
- 确保模型文件完整下载
- 验证模型格式兼容性

### 2. API 调用超时

**问题**: `LLM API 请求超时`

**解决**:
- 检查网络连接
- 验证 API 密钥有效性
- 增加超时时间配置

### 3. 向量数据库为空

**问题**: `向量数据库文档数量为 0`

**解决**:
- 执行 `python ingest.py` 重新摄取
- 检查 Markdown 文件路径
- 查看日志文件排查错误

### 4. 检索结果不准确

**问题**: 返回的答案不相关

**解决**:
- 调整 `VECTOR_SEARCH_THRESHOLD` 阈值
- 启用重排功能 (`use_rerank: true`)
- 增加 `RETRIEVAL_COUNT` 召回数量
- 检查文档质量和分块策略

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加某个功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 开源协议

本项目采用 MIT 协议 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👥 作者

**RAG 项目团队**

- 项目主页: [GitHub Repository](#)
- 问题反馈: [Issues](#)
- 邮箱: contact@example.com

## 🙏 致谢

本项目使用了以下开源项目：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Web 框架
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 文本嵌入模型
- [SiliconFlow](https://siliconflow.cn/) - LLM API 服务

## 📚 相关资源

- [RAG 技术介绍](https://arxiv.org/abs/2005.11401)
- [向量数据库最佳实践](#)
- [Prompt Engineering 指南](#)
- [FastAPI 完整文档](https://fastapi.tiangolo.com/)

## 📈 更新日志

### v1.0.0 (2026-01-14)

- ✨ 初始版本发布
- 🎯 实现核心 RAG 功能
- 📝 完整的项目文档
- 🎨 现代化 Web 界面
- 🔧 灵活的配置管理
- 📊 详细的日志系统

---

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**
