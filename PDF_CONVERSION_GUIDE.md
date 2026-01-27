# PDF 转换使用指南

本指南详细介绍如何在 RAG 智能问答系统中使用 PDF 到 Markdown 的转换功能。

## 🚀 快速开始

### 方法 1: 使用专用转换工具（推荐）

```bash
# 基本用法 - GPU 加速模式（默认）
python convert_pdf.py your_document.pdf

# 使用 CPU 模式（如果没有 GPU）
python convert_pdf.py your_document.pdf --no-gpu

# 指定 GPU 设备（多 GPU 环境）
python convert_pdf.py your_document.pdf --gpu-id 1
```

转换完成后，文件会自动复制到项目目录：
- `book.md` - 转换后的 Markdown 文件
- `images/` - 提取的图片目录

### 方法 2: 一键转换并导入

```bash
# 转换 PDF 并直接导入到向量数据库
python ingest.py --pdf your_document.pdf --force

# CPU 模式
python ingest.py --pdf your_document.pdf --force --no-gpu

# 指定 GPU
python ingest.py --pdf your_document.pdf --force --gpu-id 1
```

这个命令会：
1. 将 PDF 转换为 Markdown
2. 提取并组织图片
3. 复制文件到项目目录
4. 清空现有数据并重新导入

## 📁 输出结构

转换过程中，MinerU 会创建以下目录结构：

```
pdf_output/
└── your_document/          # 以 PDF 文件名命名
    └── auto/
        ├── your_document.md    # 转换后的 Markdown
        └── images/             # 提取的图片
            ├── image_1.png
            ├── image_2.jpg
            └── ...
```

使用 `--no-copy` 选项可以保留原始输出结构，不复制到项目目录：

```bash
python convert_pdf.py document.pdf --no-copy
```

## ⚙️ 配置选项

### 环境变量配置（.env 文件）

```bash
# 启用 GPU 加速
PDF_USE_GPU=True

# 指定 GPU 设备 ID
PDF_GPU_ID=0
```

### 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `pdf_path` | PDF 文件路径（必需） | - |
| `--output-dir` | 自定义输出目录 | `pdf_output/` |
| `--no-gpu` | 禁用 GPU，使用 CPU | False（使用 GPU）|
| `--gpu-id` | GPU 设备 ID | 0 |
| `--no-copy` | 不复制文件到项目目录 | False（会复制）|

## 🔧 高级用法

### 示例 1: 批量处理多个 PDF

```bash
# 创建批处理脚本 (batch_convert.sh)
for pdf in *.pdf; do
    echo "Converting $pdf..."
    python convert_pdf.py "$pdf" --no-copy --output-dir "outputs"
done
```

### 示例 2: 自定义输出目录

```bash
# 指定自定义输出目录
python convert_pdf.py document.pdf --output-dir ./my_conversions

# 转换后的文件在: ./my_conversions/document/auto/
```

### 示例 3: 在 Python 代码中使用

```python
from pdf_converter import PDFConverter

# 创建转换器（GPU 模式）
converter = PDFConverter(use_gpu=True)

# 方法 1: 仅转换，保留原始输出
md_file, images_dir = converter.convert(
    pdf_path="document.pdf",
    output_dir="pdf_output",
    gpu_id=0
)
print(f"Markdown: {md_file}")
print(f"Images: {images_dir}")

# 方法 2: 转换并准备文件（推荐）
final_md, final_images = converter.convert_and_prepare(
    pdf_path="document.pdf",
    copy_to_project=True,
    gpu_id=0
)
print(f"准备完成: {final_md}")
```

## 🐛 故障排除

### 问题 1: 找不到 mineru 命令

**原因**: 未安装 magic-pdf 包

**解决方案**:
```bash
pip install magic-pdf[full]
```

### 问题 2: GPU 相关错误

**原因**: CUDA 未正确配置或驱动问题

**解决方案**:
```bash
# 使用 CPU 模式
python convert_pdf.py document.pdf --no-gpu

# 或在 .env 中设置
PDF_USE_GPU=False
```

### 问题 3: 未找到图片目录

**原因**: PDF 中可能没有图片，或 MinerU 未能提取

**解决方案**: 这是正常情况，系统会记录警告但不会报错。纯文本 PDF 不会生成图片目录。

### 问题 4: 图片路径在 Markdown 中不正确

**原因**: 转换器会自动更新路径

**解决方案**: 如果使用 `--no-copy` 选项，图片路径可能需要手动调整。建议不使用该选项，让系统自动处理。

## 📊 性能建议

### GPU vs CPU

| 模式 | 适用场景 | 相对速度 |
|------|----------|----------|
| **GPU** | 大文件、频繁转换 | ⚡⚡⚡ 快 3-10 倍 |
| **CPU** | 小文件、偶尔使用 | 🐌 较慢但稳定 |

### 最佳实践

1. **大文件（>50MB）**: 推荐使用 GPU 模式
2. **批量处理**: 使用 GPU 可显著提升效率
3. **首次使用**: 建议使用 `--no-copy` 测试，确认输出正确后再正式使用

## 🔗 相关文档

- [MinerU 官方文档](https://github.com/opendatalab/MinerU)
- [RAG 系统主文档](README.md)
- [配置指南](README.md#configuration)

## 💡 提示与技巧

1. **检查 PDF 质量**: 扫描版 PDF 的转换效果可能不如原生 PDF
2. **图片清晰度**: 提取的图片质量取决于 PDF 中的原始图片质量
3. **中文支持**: MinerU 对中文文档支持良好
4. **复杂排版**: 表格、多栏排版等可能需要手动调整

## 📝 示例工作流程

完整的从 PDF 到可查询系统的流程：

```bash
# 1. 转换 PDF
python convert_pdf.py my_book.pdf

# 输出:
# ✓ 转换成功!
#   Markdown: c:\...\rag_project\book.md
#   图片目录: c:\...\rag_project\images
#
# 提示: 现在可以运行 'python ingest.py --force' 来导入文档

# 2. 导入文档
python ingest.py --force

# 输出:
# ✓ Generated 1234 text chunks
# ✓ Parent node mapping saved
# ✓ Embeddings generated
# ✓ Data stored in ChromaDB

# 3. 启动服务
python run.py

# 输出:
# ✓ Server running at http://localhost:8000
# ✓ API Docs at http://localhost:8000/docs
```

现在您可以在 Web UI 中查询文档内容了！

---

**遇到问题？** 请查看日志文件 `logs/PDFConverter.log` 获取详细信息。
