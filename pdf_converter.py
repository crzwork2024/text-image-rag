"""
PDF 转换模块 - RAG 智能问答系统
作者: RAG 项目团队
描述: 使用 MinerU 将 PDF 文件转换为 Markdown 格式，支持 GPU 加速
"""

import subprocess
import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
from config import config
from utils.logger import setup_logger

# 初始化日志
logger = setup_logger(
    "PDFConverter",
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_dir=config.LOG_DIR
)


class PDFConverter:
    """PDF 到 Markdown 转换器（基于 MinerU）"""

    def __init__(self, use_gpu: bool = True):
        """
        初始化 PDF 转换器

        Args:
            use_gpu: 是否使用 GPU 加速（默认 True）
        """
        self.use_gpu = use_gpu
        logger.info(f"PDF 转换器初始化 - GPU 模式: {'启用' if use_gpu else '禁用'}")

    def convert(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        gpu_id: int = 0
    ) -> Tuple[Path, Path]:
        """
        将 PDF 文件转换为 Markdown

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录（可选，默认使用配置中的路径）
            gpu_id: GPU 设备 ID（仅在 use_gpu=True 时有效）

        Returns:
            (md_file_path, images_dir_path): Markdown 文件路径和图片目录路径

        Raises:
            FileNotFoundError: PDF 文件不存在
            RuntimeError: 转换失败
        """
        # 验证 PDF 文件
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 确定输出目录
        if output_dir is None:
            output_dir = str(config.PDF_OUTPUT_DIR)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("开始 PDF 到 Markdown 转换")
        logger.info("=" * 60)
        logger.info(f"输入文件: {pdf_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"文件大小: {pdf_file.stat().st_size / (1024*1024):.2f} MB")

        # 准备环境变量
        env = os.environ.copy()
        if self.use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"GPU 模式: 使用 GPU {gpu_id}")
        else:
            # 禁用 GPU
            env["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("CPU 模式: GPU 已禁用")

        # 构建命令
        command = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(output_dir),
            "--format", "markdown"
        ]

        try:
            # 执行转换
            logger.info("执行 MinerU 转换...")
            logger.info(f"命令: {' '.join(command)}")

            result = subprocess.run(
                command,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )

            logger.info("✓ MinerU 转换成功")

            # 定位生成的文件
            # MinerU 默认结构: output_dir/pdf_filename/auto/
            pdf_stem = pdf_file.stem  # 不带扩展名的文件名
            auto_dir = output_path / pdf_stem / "auto"

            if not auto_dir.exists():
                raise RuntimeError(f"未找到生成的输出目录: {auto_dir}")

            # 查找生成的 Markdown 文件
            md_files = list(auto_dir.glob("*.md"))
            if not md_files:
                raise RuntimeError(f"未找到生成的 Markdown 文件: {auto_dir}")

            md_file = md_files[0]  # 通常只有一个 md 文件

            # 图片目录
            images_dir = auto_dir / "images"
            if not images_dir.exists():
                logger.warning(f"未找到图片目录: {images_dir}，可能文档中没有图片")
                images_dir = None

            logger.info(f"✓ Markdown 文件: {md_file}")
            logger.info(f"✓ 图片目录: {images_dir if images_dir else '无'}")

            # 统计信息
            md_size = md_file.stat().st_size / 1024
            logger.info(f"✓ Markdown 文件大小: {md_size:.2f} KB")

            if images_dir and images_dir.exists():
                image_files = list(images_dir.glob("*"))
                logger.info(f"✓ 图片数量: {len(image_files)}")

            logger.info("=" * 60)
            logger.info("PDF 转换完成!")
            logger.info("=" * 60)

            return md_file, images_dir

        except subprocess.CalledProcessError as e:
            error_msg = f"MinerU 转换失败: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"转换过程出错: {str(e)}"
            logger.error(error_msg)
            raise

    def convert_and_prepare(
        self,
        pdf_path: str,
        copy_to_project: bool = True,
        gpu_id: int = 0
    ) -> Tuple[Path, Path]:
        """
        转换 PDF 并准备文件供 RAG 系统使用

        此方法会：
        1. 将 PDF 转换为 Markdown
        2. 将生成的 MD 文件复制到项目根目录（可选）
        3. 将图片目录复制到项目 images 目录（可选）

        Args:
            pdf_path: PDF 文件路径
            copy_to_project: 是否复制文件到项目目录（默认 True）
            gpu_id: GPU 设备 ID

        Returns:
            (final_md_path, final_images_dir): 最终的 MD 文件路径和图片目录路径
        """
        # 执行转换
        md_file, images_dir = self.convert(pdf_path, gpu_id=gpu_id)

        if not copy_to_project:
            return md_file, images_dir

        # 复制到项目目录
        logger.info("-" * 60)
        logger.info("复制文件到项目目录")
        logger.info("-" * 60)

        # 复制 Markdown 文件
        final_md_path = config.MD_FILE_PATH
        shutil.copy2(md_file, final_md_path)
        logger.info(f"✓ Markdown 文件已复制到: {final_md_path}")

        # 复制图片目录
        final_images_dir = None
        if images_dir and images_dir.exists():
            final_images_dir = config.IMAGE_DIR
            
            # 清空并重新创建图片目录
            if final_images_dir.exists():
                shutil.rmtree(final_images_dir)
            final_images_dir.mkdir(parents=True, exist_ok=True)

            # 复制所有图片
            image_count = 0
            for img_file in images_dir.glob("*"):
                if img_file.is_file():
                    shutil.copy2(img_file, final_images_dir / img_file.name)
                    image_count += 1

            logger.info(f"✓ {image_count} 个图片已复制到: {final_images_dir}")

            # 更新 Markdown 中的图片路径
            self._update_image_paths(final_md_path, images_dir.name, "images")
            logger.info("✓ Markdown 中的图片路径已更新")

        logger.info("-" * 60)
        logger.info("文件准备完成!")
        logger.info("-" * 60)

        return final_md_path, final_images_dir

    @staticmethod
    def _update_image_paths(md_file: Path, old_dir: str, new_dir: str):
        """
        更新 Markdown 文件中的图片路径

        Args:
            md_file: Markdown 文件路径
            old_dir: 旧的图片目录名（例如 "images"）
            new_dir: 新的图片目录名（例如 "images"）
        """
        # 读取文件
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 替换路径（适配各种可能的路径格式）
        # 例如: ![](images/xxx.png) -> ![](images/xxx.png)
        # 或: ![](./images/xxx.png) -> ![](images/xxx.png)
        # 或: ![](auto/images/xxx.png) -> ![](images/xxx.png)
        
        import re
        # 匹配图片语法: ![...](path)
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        def replace_path(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 提取文件名
            img_filename = Path(img_path).name
            
            # 构建新路径
            new_path = f"{new_dir}/{img_filename}"
            
            return f"![{alt_text}]({new_path})"
        
        # 替换所有图片路径
        updated_content = re.sub(pattern, replace_path, content)

        # 写回文件
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(updated_content)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF 到 Markdown 转换工具（使用 MinerU）"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="PDF 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认使用配置中的路径）"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="禁用 GPU 加速（使用 CPU）"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU 设备 ID（默认 0）"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制文件到项目目录"
    )

    args = parser.parse_args()

    try:
        # 创建转换器
        converter = PDFConverter(use_gpu=not args.no_gpu)

        # 执行转换
        if args.no_copy:
            md_file, images_dir = converter.convert(
                args.pdf_path,
                output_dir=args.output_dir,
                gpu_id=args.gpu_id
            )
            print(f"\n✓ 转换成功!")
            print(f"  Markdown: {md_file}")
            print(f"  图片目录: {images_dir if images_dir else '无'}")
        else:
            final_md, final_images = converter.convert_and_prepare(
                args.pdf_path,
                copy_to_project=True,
                gpu_id=args.gpu_id
            )
            print(f"\n✓ 转换并准备完成!")
            print(f"  Markdown: {final_md}")
            print(f"  图片目录: {final_images if final_images else '无'}")
            print(f"\n提示: 现在可以运行 'python ingest.py --force' 来导入文档")

    except Exception as e:
        logger.error(f"转换失败: {e}")
        print(f"\n✗ 转换失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
