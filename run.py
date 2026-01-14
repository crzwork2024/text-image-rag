"""
快速启动脚本 - RAG 智能问答系统
作者：RAG 项目团队
描述：提供便捷的启动方式，自动检查配置和依赖
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("正在检查运行环境...")
    print("=" * 60)

    errors = []
    warnings = []

    # 检查 Python 版本
    if sys.version_info < (3, 8):
        errors.append(f"Python 版本过低: {sys.version}，需要 3.8+")
    else:
        print(f"✓ Python 版本: {sys.version.split()[0]}")

    # 检查必需的配置
    if not config.SILICONFLOW_API_KEY:
        errors.append("缺少 SILICONFLOW_API_KEY 环境变量")
    else:
        print("✓ SILICONFLOW_API_KEY 已配置")

    # 检查模型路径
    if not Path(config.LOCAL_MODEL_PATH).exists():
        errors.append(f"本地嵌入模型路径不存在: {config.LOCAL_MODEL_PATH}")
    else:
        print(f"✓ 嵌入模型路径: {config.LOCAL_MODEL_PATH}")

    # 检查文档文件
    if not config.MD_FILE_PATH.exists():
        warnings.append(f"文档文件不存在: {config.MD_FILE_PATH}")
    else:
        print(f"✓ 文档文件: {config.MD_FILE_PATH}")

    # 输出错误和警告
    if errors:
        print("\n" + "=" * 60)
        print("❌ 发现错误:")
        for error in errors:
            print(f"  - {error}")
        print("=" * 60)
        return False

    if warnings:
        print("\n" + "=" * 60)
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  - {warning}")
        print("=" * 60)

    print("\n✅ 环境检查通过\n")
    return True


def check_dependencies():
    """检查依赖是否安装"""
    print("=" * 60)
    print("正在检查依赖...")
    print("=" * 60)

    required_packages = [
        "fastapi",
        "uvicorn",
        "chromadb",
        "sentence_transformers",
        "requests",
        "pydantic",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} (未安装)")

    if missing:
        print("\n" + "=" * 60)
        print("❌ 缺少依赖，请运行:")
        print("pip install -r requirements.txt")
        print("=" * 60)
        return False

    print("\n✅ 依赖检查通过\n")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("   RAG 智能问答系统 - 启动检查")
    print("=" * 60 + "\n")

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查环境
    if not check_environment():
        sys.exit(1)

    # 询问是否需要执行数据摄取
    from core.vector_store import vector_db
    doc_count = vector_db.count()

    if doc_count == 0:
        print("=" * 60)
        print("⚠️  向量数据库为空")
        print("=" * 60)

        response = input("是否立即执行数据摄取？(y/n): ").lower().strip()
        if response == 'y':
            print("\n开始数据摄取...")
            from ingest import run_ingestion
            try:
                run_ingestion()
                print("\n✅ 数据摄取完成\n")
            except Exception as e:
                print(f"\n❌ 数据摄取失败: {e}\n")
                sys.exit(1)
        else:
            print("\n⚠️  跳过数据摄取，系统将在首次启动时自动执行\n")
    else:
        print(f"✓ 向量数据库已包含 {doc_count} 个文档\n")

    # 启动服务
    print("=" * 60)
    print("正在启动服务...")
    print("=" * 60)
    print(f"\n📝 服务地址: http://{config.APP_HOST}:{config.APP_PORT}")
    print(f"📚 API 文档: http://{config.APP_HOST}:{config.APP_PORT}/docs")
    print(f"🌐 Web 界面: http://{config.APP_HOST}:{config.APP_PORT}\n")
    print("按 Ctrl+C 停止服务\n")
    print("=" * 60 + "\n")

    # 导入并启动主应用
    from main import main as run_main
    run_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n服务已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)
