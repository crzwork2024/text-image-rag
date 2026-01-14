# 📁 项目结构详解

## 完整目录树

```
rag_project/
│
├── 📄 __init__.py                    # 项目包初始化文件
├── 📄 config.py                      # 配置管理模块
├── 📄 main.py                        # FastAPI 主应用入口
├── 📄 ingest.py                      # 数据摄取脚本
├── 📄 run.py                         # 便捷启动脚本
│
├── 📋 requirements.txt               # Python 依赖清单
├── 📋 env.example                    # 环境变量模板
├── 📋 .gitignore                     # Git 忽略配置
├── 📋 LICENSE                        # 开源协议（MIT）
│
├── 📖 README.md                      # 项目主文档
├── 📖 QUICKSTART.md                  # 快速开始指南
├── 📖 ARCHITECTURE.md                # 架构设计文档
├── 📖 CHANGELOG.md                   # 更新日志
├── 📖 REFACTORING_SUMMARY.md         # 重构总结
├── 📖 PROJECT_STRUCTURE.md           # 项目结构说明（本文件）
│
├── 📂 core/                          # 核心功能模块
│   ├── __init__.py                  # 模块初始化
│   ├── embeddings.py                # 嵌入引擎（文本向量化）
│   ├── vector_store.py              # 向量存储管理（ChromaDB）
│   ├── llm_client.py                # LLM 客户端（SiliconFlow API）
│   ├── reranker.py                  # 重排引擎（精确排序）
│   └── processor.py                 # 文档处理器（Markdown 分块）
│
├── 📂 utils/                         # 工具模块
│   ├── __init__.py                  # 模块初始化
│   ├── logger.py                    # 日志管理（彩色日志）
│   ├── exceptions.py                # 异常定义（自定义异常）
│   └── responses.py                 # 响应模型（标准响应格式）
│
├── 📂 static/                        # 前端静态文件
│   └── index.html                   # Web 用户界面
│
├── 📂 .vscode/                       # VS Code 配置（可选）
│   ├── settings.json                # 编辑器设置
│   ├── extensions.json              # 推荐扩展
│   └── launch.json                  # 调试配置
│
├── 📂 chroma_db/                     # 向量数据库存储
│   └── (由 ChromaDB 自动管理)
│
├── 📂 logs/                          # 日志文件目录
│   ├── API_20260114.log            # API 服务日志
│   └── Ingestion_20260114.log      # 数据摄取日志
│
├── 📂 images/                        # 图片资源（可选）
│   └── (文档中引用的图片)
│
├── 📂 models/                        # 本地模型文件
│   └── acge_text_embedding/        # 嵌入模型
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
├── 📄 book.md                        # 待问答的文档（示例）
├── 📄 parent_store.json              # 父节点映射（自动生成）
└── 📄 vector_ingest.json             # 调试导出文件（自动生成）
```

---

## 📦 核心模块详解

### 1. 配置层

#### config.py
```python
# 职责：集中管理所有配置
class Config:
    # 路径配置
    BASE_DIR, MD_FILE_PATH, CHROMA_PATH, ...

    # API 配置
    SILICONFLOW_API_KEY, SILICONFLOW_MODEL_ID, ...

    # 参数配置
    RETRIEVAL_COUNT, RERANK_TOP_K, ...

    # 方法
    validate()           # 验证配置
    create_directories() # 创建目录
```

**依赖关系**: 被所有模块引用

---

### 2. 核心功能层 (core/)

#### embeddings.py
```python
# 职责：文本向量化
class EmbeddingEngine:
    encode(sentences)              # 文本转向量
    get_embedding_dimension()      # 获取维度
```
**技术栈**: Sentence Transformers, PyTorch
**模型**: ACGE Text Embedding (768 维)

#### vector_store.py
```python
# 职责：向量数据库管理
class VectorStoreManager:
    add_documents()     # 添加文档
    query()             # 查询相似文档
    count()             # 文档数量
    reset()             # 重置数据库
```
**技术栈**: ChromaDB
**度量方式**: 余弦相似度

#### llm_client.py
```python
# 职责：LLM API 调用
class LLMClient:
    generate()          # 生成回答
    is_available()      # 检查可用性
```
**技术栈**: SiliconFlow API
**模型**: DeepSeek-R1-Distill-Qwen-7B

#### reranker.py
```python
# 职责：检索结果重排
class RerankEngine:
    rerank()            # 重排文档
    is_available()      # 检查可用性
```
**技术栈**: SiliconFlow Rerank API
**模型**: BAAI/bge-reranker-v2-m3

#### processor.py
```python
# 职责：文档处理
class DocumentProcessor:
    process_markdown_to_chunks()  # 文档分块
    validate_markdown()           # 验证文档
```
**分块策略**: 父子节点结构（章节-段落）

---

### 3. 工具层 (utils/)

#### logger.py
```python
# 职责：日志管理
setup_logger()      # 创建日志器
get_logger()        # 获取日志器

class ColoredFormatter:
    # 彩色日志格式化
```
**特性**:
- 彩色控制台输出
- 文件持久化
- 按日期分割

#### exceptions.py
```python
# 职责：异常定义
class RAGBaseException          # 基础异常
class ModelLoadError            # 模型加载错误
class VectorStoreError          # 向量库错误
class LLMAPIError               # LLM API 错误
# ... 更多异常类
```

#### responses.py
```python
# 职责：响应格式标准化
class QueryResponse             # 查询响应
class StandardResponse          # 标准响应
class ErrorResponse             # 错误响应

success_response()              # 成功响应
error_response()                # 错误响应
```

---

### 4. 应用层

#### main.py
```python
# 职责：Web 服务主入口
app = FastAPI(...)

@app.post("/query")            # 问答接口
@app.get("/health")            # 健康检查
@app.get("/stats")             # 统计信息

# 生命周期管理
async def lifespan(app):
    # 启动初始化
    # 关闭清理
```

#### ingest.py
```python
# 职责：数据摄取
def run_ingestion(md_file_path, force_reingest):
    # 1. 读取文档
    # 2. 文档分块
    # 3. 生成向量
    # 4. 存储数据库
    # 5. 保存映射
```

#### run.py
```python
# 职责：便捷启动
def check_environment()        # 环境检查
def check_dependencies()       # 依赖检查
def main()                     # 启动主流程
```

---

## 🔄 数据流向

### 摄取流程
```
book.md
  ↓
processor.py (分块)
  ↓
embeddings.py (向量化)
  ↓
vector_store.py (存储)
  ↓
ChromaDB
```

### 查询流程
```
用户问题
  ↓
embeddings.py (向量化)
  ↓
vector_store.py (检索)
  ↓
reranker.py (重排，可选)
  ↓
processor.py (组装上下文)
  ↓
llm_client.py (生成回答)
  ↓
返回结果
```

---

## 📊 依赖关系图

```
main.py
  ├── config.py
  ├── ingest.py
  ├── core/
  │   ├── embeddings.py
  │   ├── vector_store.py
  │   ├── llm_client.py
  │   ├── reranker.py
  │   └── processor.py
  └── utils/
      ├── logger.py
      ├── exceptions.py
      └── responses.py

ingest.py
  ├── config.py
  ├── core/
  │   ├── embeddings.py
  │   ├── vector_store.py
  │   └── processor.py
  └── utils/
      ├── logger.py
      └── exceptions.py

run.py
  ├── config.py
  ├── main.py
  ├── ingest.py
  └── core/vector_store.py
```

---

## 🎯 文件职责分类

### 入口文件
- `main.py` - Web 服务入口
- `ingest.py` - 数据摄取入口
- `run.py` - 快速启动入口

### 核心业务
- `core/embeddings.py` - 向量化
- `core/vector_store.py` - 存储
- `core/llm_client.py` - 生成
- `core/reranker.py` - 重排
- `core/processor.py` - 处理

### 基础设施
- `config.py` - 配置
- `utils/logger.py` - 日志
- `utils/exceptions.py` - 异常
- `utils/responses.py` - 响应

### 文档
- `README.md` - 主文档
- `QUICKSTART.md` - 快速开始
- `ARCHITECTURE.md` - 架构设计
- `CHANGELOG.md` - 更新日志
- `REFACTORING_SUMMARY.md` - 重构总结
- `PROJECT_STRUCTURE.md` - 结构说明

### 配置
- `requirements.txt` - 依赖
- `env.example` - 环境变量
- `.gitignore` - Git 忽略
- `.vscode/` - VS Code 配置

---

## 📏 代码规模统计

| 模块 | 文件数 | 代码行数 | 注释行数 | 文档行数 |
|------|--------|---------|---------|---------|
| **核心模块** | 6 | ~800 | ~400 | ~200 |
| **工具模块** | 4 | ~300 | ~150 | ~100 |
| **应用层** | 3 | ~600 | ~300 | ~150 |
| **配置** | 1 | ~150 | ~80 | ~40 |
| **文档** | 6 | - | - | ~3000 |
| **总计** | 20 | ~1850 | ~930 | ~3490 |

---

## 🔍 模块使用指南

### 如何添加新功能？

#### 1. 添加新的核心功能
```python
# 在 core/ 目录创建新文件
# core/new_feature.py

from utils.logger import get_logger
from utils.exceptions import RAGBaseException

logger = get_logger(__name__)

class NewFeature:
    """新功能类"""
    def process(self):
        """处理逻辑"""
        pass

# 在 core/__init__.py 中导出
from .new_feature import NewFeature
```

#### 2. 添加新的 API 端点
```python
# 在 main.py 中添加
@app.get("/new-endpoint")
async def new_endpoint():
    """新端点"""
    return {"status": "ok"}
```

#### 3. 添加新的异常类型
```python
# 在 utils/exceptions.py 中添加
class NewException(RAGBaseException):
    """新异常"""
    pass

# 在 utils/__init__.py 中导出
```

---

## 🧪 测试结构（规划中）

```
tests/
├── __init__.py
├── test_embeddings.py      # 嵌入模块测试
├── test_vector_store.py    # 向量库测试
├── test_llm_client.py      # LLM 客户端测试
├── test_reranker.py        # 重排测试
├── test_processor.py       # 处理器测试
└── test_api.py             # API 测试
```

---

## 📝 命名规范

### 文件命名
- 模块文件：`snake_case.py`
- 配置文件：`UPPERCASE.md` 或 `lowercase.txt`
- 脚本文件：`snake_case.py`

### 类命名
- 类名：`PascalCase`
- 示例：`EmbeddingEngine`, `VectorStoreManager`

### 函数命名
- 函数名：`snake_case`
- 示例：`process_markdown`, `setup_logger`

### 常量命名
- 常量：`UPPER_SNAKE_CASE`
- 示例：`RETRIEVAL_COUNT`, `API_URL`

---

## 🎨 代码组织原则

### 1. 模块职责单一
每个模块只负责一个主要功能

### 2. 高内聚低耦合
模块内部紧密相关，模块间松散依赖

### 3. 配置集中管理
所有配置统一在 `config.py` 中

### 4. 错误统一处理
自定义异常体系，统一错误处理

### 5. 日志规范记录
使用标准化日志格式

---

**文档版本**: v1.0.0
**最后更新**: 2026-01-14
**维护者**: RAG 项目团队
