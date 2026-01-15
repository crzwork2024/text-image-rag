"""
Quick Start Script - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Provides a convenient way to start, automatically checking configuration and dependencies.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config


def check_environment():
    """Check runtime environment"""
    print("=" * 60)
    print("Checking runtime environment...")
    print("=" * 60)

    errors = []
    warnings = []

    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python version too old: {sys.version}, need 3.8+")
    else:
        print(f"✓ Python Version: {sys.version.split()[0]}")

    # Check required config
    if not config.SILICONFLOW_API_KEY:
        errors.append("Missing SILICONFLOW_API_KEY environment variable")
    else:
        print("✓ SILICONFLOW_API_KEY configured")

    # Check model path
    if not Path(config.LOCAL_MODEL_PATH).exists():
        errors.append(f"Local embedding model path not found: {config.LOCAL_MODEL_PATH}")
    else:
        print(f"✓ Embedding Model Path: {config.LOCAL_MODEL_PATH}")

    # Check document file
    if not config.MD_FILE_PATH.exists():
        warnings.append(f"Document file not found: {config.MD_FILE_PATH}")
    else:
        print(f"✓ Document File: {config.MD_FILE_PATH}")

    # Output errors and warnings
    if errors:
        print("\n" + "=" * 60)
        print("❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        print("=" * 60)
        return False

    if warnings:
        print("\n" + "=" * 60)
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("=" * 60)

    print("\n✅ Environment Check Passed\n")
    return True


def check_dependencies():
    """Check if dependencies are installed"""
    print("=" * 60)
    print("Checking dependencies...")
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
            print(f"✗ {package} (Not Installed)")

    if missing:
        print("\n" + "=" * 60)
        print("❌ Missing dependencies, please run:")
        print("pip install -r requirements.txt")
        print("=" * 60)
        return False

    print("\n✅ Dependency Check Passed\n")
    return True


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("   RAG Intelligent Q&A System - Startup Check")
    print("=" * 60 + "\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Ask if data ingestion is needed
    from core.vector_store import vector_db
    doc_count = vector_db.count()

    if doc_count == 0:
        print("=" * 60)
        print("⚠️  Vector Database is empty")
        print("=" * 60)

        response = input("Run data ingestion now? (y/n): ").lower().strip()
        if response == 'y':
            print("\nStarting data ingestion...")
            from ingest import run_ingestion
            try:
                run_ingestion()
                print("\n✅ Data ingestion complete\n")
            except Exception as e:
                print(f"\n❌ Data ingestion failed: {e}\n")
                sys.exit(1)
        else:
            print("\n⚠️  Skipping data ingestion, will run automatically on first start\n")
    else:
        print(f"✓ Vector database contains {doc_count} documents\n")

    # Start service
    print("=" * 60)
    print("Starting Service...")
    print("=" * 60)
    print(f"\n📝 Service URL: http://{config.APP_HOST}:{config.APP_PORT}")
    print(f"📚 API Docs: http://{config.APP_HOST}:{config.APP_PORT}/docs")
    print(f"🌐 Web UI: http://{config.APP_HOST}:{config.APP_PORT}\n")
    print("Press Ctrl+C to stop service\n")
    print("=" * 60 + "\n")

    # Import and run main app
    from main import main as run_main
    run_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nService stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)
