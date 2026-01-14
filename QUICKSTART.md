# ⚡ 快速开始指南

> 5 分钟启动你的 RAG 智能问答系统

## 📋 准备工作

### 系统要求
- ✅ Python 3.8 或更高版本
- ✅ 8GB+ RAM
- ✅ 10GB+ 可用磁盘空间
- ✅ 互联网连接（用于下载模型和调用 API）

### 获取 API 密钥
1. 访问 [SiliconFlow](https://siliconflow.cn/)
2. 注册账号并登录
3. 获取 API 密钥

---

## 🚀 5 步快速启动

### 步骤 1: 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

⏱️ **预计时间**: 2-3 分钟

---

### 步骤 2: 配置环境变量

```bash
# 1. 复制环境变量模板
cp env.example .env

# 2. 编辑 .env 文件（使用你喜欢的编辑器）
notepad .env  # Windows
# 或
nano .env     # Linux/macOS
```

**必须配置的内容**:

```ini
# SiliconFlow API 密钥（必填）
SILICONFLOW_API_KEY=sk-your-api-key-here

# 本地嵌入模型路径（必填）
LOCAL_EMBEDDING_MODEL_PATH=./models/acge_text_embedding
```

⏱️ **预计时间**: 1 分钟

---

### 步骤 3: 下载嵌入模型

#### 方案 A: 从 HuggingFace 下载（推荐）

```bash
# 创建模型目录
mkdir -p models

# 使用 git clone 下载
cd models
git clone https://huggingface.co/aspire/acge_text_embedding
cd ..
```

#### 方案 B: 使用已有模型

如果你已经有嵌入模型，只需在 `.env` 中指定路径：

```ini
LOCAL_EMBEDDING_MODEL_PATH=/path/to/your/embedding/model
```

⏱️ **预计时间**: 3-5 分钟（取决于网速）

---

### 步骤 4: 准备文档

```bash
# 将你的 Markdown 文档放在项目根目录
# 默认文件名为 book.md
cp /path/to/your/document.md book.md
```

**文档格式要求**:
- ✅ Markdown 格式（.md）
- ✅ 包含标题结构（# ## ### 等）
- ✅ UTF-8 编码

**示例文档结构**:
```markdown
# 第一章：简介

这是第一章的内容...

## 1.1 背景

详细的背景介绍...

# 第二章：核心概念

第二章的内容...
```

⏱️ **预计时间**: 1 分钟

---

### 步骤 5: 启动系统

```bash
# 使用便捷启动脚本（推荐）
python run.py
```

启动脚本会自动：
- ✅ 检查 Python 版本
- ✅ 验证依赖安装
- ✅ 检查配置完整性
- ✅ 检查向量数据库
- ✅ 提示是否需要数据摄取
- ✅ 启动 Web 服务

⏱️ **预计时间**: 10-30 秒（首次启动需要数据摄取，约 1-3 分钟）

---

## 🎉 启动成功！

看到以下信息表示启动成功：

```
============================================================
   RAG 智能问答系统 - 启动检查
============================================================

正在检查依赖...
✓ fastapi
✓ uvicorn
✓ chromadb
...

✅ 依赖检查通过

正在检查运行环境...
✓ Python 版本: 3.10.0
✓ SILICONFLOW_API_KEY 已配置
✓ 嵌入模型路径: ./models/acge_text_embedding
✓ 文档文件: book.md

✅ 环境检查通过

✓ 向量数据库已包含 150 个文档

============================================================
正在启动服务...
============================================================

📝 服务地址: http://0.0.0.0:8000
📚 API 文档: http://0.0.0.0:8000/docs
🌐 Web 界面: http://0.0.0.0:8000

按 Ctrl+C 停止服务

============================================================
```

---

## 🌐 使用系统

### Web 界面

1. 打开浏览器
2. 访问 `http://localhost:8000`
3. 在输入框中输入问题
4. 点击"发送提问"
5. 查看 AI 生成的回答

### API 调用

```bash
# 使用 curl 测试
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "这篇文档主要讲了什么？",
    "use_rerank": true
  }'
```

### Python 调用

```python
import requests

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
```

---

## 🔧 常见问题

### Q1: 启动时提示"模型路径不存在"？

**解决方案**:
1. 检查 `.env` 中的路径配置
2. 确保模型已下载到正确位置
3. 使用绝对路径代替相对路径

```ini
# 示例：使用绝对路径
LOCAL_EMBEDDING_MODEL_PATH=C:\Projects\models\acge_text_embedding
```

---

### Q2: 数据摄取失败？

**解决方案**:
1. 检查 `book.md` 文件是否存在
2. 确认文件编码为 UTF-8
3. 查看日志文件 `logs/Ingestion_*.log`

```bash
# 手动执行数据摄取
python ingest.py --file book.md
```

---

### Q3: API 调用超时？

**解决方案**:
1. 检查网络连接
2. 验证 API 密钥有效性
3. 查看日志 `logs/API_*.log`

---

### Q4: 向量数据库为空？

**解决方案**:
```bash
# 强制重新摄取数据
python ingest.py --force
```

---

### Q5: 查询结果不准确？

**解决方案**:
1. 启用重排功能（`use_rerank: true`）
2. 调整配置参数：

```ini
# 在 .env 中调整
RETRIEVAL_COUNT=15           # 增加召回数量
VECTOR_SEARCH_THRESHOLD=0.15 # 降低过滤阈值
```

---

## 📊 验证安装

运行以下命令验证系统是否正常工作：

```bash
# 检查健康状态
curl http://localhost:8000/health

# 预期输出
{
  "status": "healthy",
  "vector_db_docs": 150,
  "parent_store_size": 50
}
```

```bash
# 查看系统统计
curl http://localhost:8000/stats

# 预期输出
{
  "vector_db": {
    "document_count": 150,
    "collection_name": "book_rag_manual"
  },
  "parent_store": {
    "section_count": 50
  },
  ...
}
```

---

## 🎯 下一步

恭喜你成功启动了 RAG 系统！接下来可以：

### 1. 探索功能
- 📖 尝试不同类型的问题
- 🎛️ 测试启用/禁用重排的效果
- 📊 查看 API 文档：http://localhost:8000/docs

### 2. 优化配置
- ⚙️ 调整检索参数
- 🎨 自定义系统提示词
- 📝 修改日志级别

### 3. 阅读文档
- 📚 [完整文档](README.md)
- 🏗️ [架构设计](ARCHITECTURE.md)
- 📝 [更新日志](CHANGELOG.md)

### 4. 开发集成
- 🔌 集成到你的应用
- 🛠️ 自定义功能
- 🧪 编写测试

---

## 📞 获取帮助

### 查看日志
```bash
# 查看 API 日志
cat logs/API_*.log

# 查看摄取日志
cat logs/Ingestion_*.log
```

### 问题反馈
- 📧 Email: contact@example.com
- 🐛 Issues: [GitHub Issues](#)
- 💬 讨论: [GitHub Discussions](#)

---

## 🎊 享受使用！

现在你已经成功启动了 RAG 智能问答系统，开始探索 AI 的强大能力吧！

**需要更多帮助？** 查看 [README.md](README.md) 获取详细文档。

---

**最后更新**: 2026-01-14
**版本**: v1.0.0
