"""
统一的中文语音识别(ASR)模型测试框架
支持: WeNet, PaddleSpeech, FunASR
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRBenchmark:
    """ASR模型基准测试类"""
    
    def __init__(self, audio_file: str, device: str = "cpu"):
        """
        初始化测试框架
        
        Args:
            audio_file: 测试音频文件路径
            device: 推理设备 ("cpu" 或 "cuda:0")
        """
        self.audio_file = audio_file
        self.device = device
        self.results = {}
        
        if not os.path.exists(audio_file):
            logger.error(f"❌ 测试音频文件不存在: {audio_file}")
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        logger.info(f"✅ 使用测试音频: {audio_file}")
        logger.info(f"📊 推理设备: {device}\n")
    
    # ============ PaddleSpeech ============
    def test_paddlespeech(self) -> Dict:
        """测试 PaddleSpeech 模型"""
        logger.info("=" * 70)
        logger.info("1️⃣  测试 PaddleSpeech (百度)")
        logger.info("=" * 70)
        
        result = {
            "model": "PaddleSpeech",
            "status": "failed",
            "error": None,
            "result": None,
            "time": 0
        }
        
        try:
            from paddlespeech.cli.asr import ASRExecutor
            
            logger.info("📦 初始化 PaddleSpeech 模型...")
            start_time = time.time()
            
            asr = ASRExecutor()
            
            logger.info("🎤 开始语音识别...")
            rec_result = asr(
                audio_file=self.audio_file,
                model='conformer_wenetspeech',  # 中文模型
                lang='zh',
                device=self.device
            )
            
            elapsed_time = time.time() - start_time
            
            result["status"] = "success"
            result["result"] = rec_result
            result["time"] = elapsed_time
            
            logger.info(f"✅ 识别成功！")
            logger.info(f"📝 结果: {rec_result}")
            logger.info(f"⏱️  耗时: {elapsed_time:.2f}s\n")
            
        except ImportError:
            result["error"] = "PaddleSpeech 未安装"
            logger.warning(f"⚠️  {result['error']}")
            logger.info("📦 安装命令: pip install paddlespeech\n")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ 错误: {e}\n")
        
        self.results["paddlespeech"] = result
        return result
    
    # ============ FunASR ============
    def test_funasr(self) -> Dict:
        """测试 FunASR 模型"""
        logger.info("=" * 70)
        logger.info("2️⃣  测试 FunASR (阿里达摩院)")
        logger.info("=" * 70)
        
        result = {
            "model": "FunASR",
            "status": "failed",
            "error": None,
            "result": None,
            "time": 0
        }
        
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            
            logger.info("📦 初始化 FunASR 模型...")
            start_time = time.time()
            
            model = AutoModel(
                model="paraformer-zh",      # 中文模型
                vad_model="fsmn-vad",       # 语音活动检测
                punc_model="ct-punc",       # 标点符号
                device=self.device
            )
            
            logger.info("🎤 开始语音识别...")
            rec_result = model.generate(
                input=self.audio_file,
                language="zh"
            )
            
            # 后处理结果
            text = rich_transcription_postprocess(rec_result[0]["text"])
            
            elapsed_time = time.time() - start_time
            
            result["status"] = "success"
            result["result"] = {
                "text": text,
                "raw_result": rec_result[0]
            }
            result["time"] = elapsed_time
            
            logger.info(f"✅ 识别成功！")
            logger.info(f"📝 结果: {text}")
            logger.info(f"⏱️  耗时: {elapsed_time:.2f}s\n")
            
        except ImportError:
            result["error"] = "FunASR 未安装"
            logger.warning(f"⚠️  {result['error']}")
            logger.info("📦 安装命令: pip install funasr\n")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ 错误: {e}\n")
        
        self.results["funasr"] = result
        return result
    
    # ============ WeNet ============
    def test_wenet(self) -> Dict:
        """测试 WeNet 模型"""
        logger.info("=" * 70)
        logger.info("3️⃣  测试 WeNet (腾讯)")
        logger.info("=" * 70)
        
        result = {
            "model": "WeNet",
            "status": "failed",
            "error": None,
            "result": None,
            "time": 0,
            "note": "需要手动下载预训练模型"
        }
        
        try:
            import torch
            import torchaudio
            from wenet.utils.config import read_symbol_table, init_config_from_file
            from wenet.models.asr_model import init_asr_model
            
            logger.info("📦 初始化 WeNet 模型...")
            logger.warning("⚠️  WeNet 需要单独下载预训练模型")
            logger.info("📥 下载链接: https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md")
            logger.info("🔧 使用说明:")
            logger.info("   1. 下载中文模型 (e.g., AISHELL-1)")
            logger.info("   2. 修改下面的 config_path 和 model_path")
            logger.info("   3. 重新运行此函数\n")
            
            result["error"] = "WeNet 需要手动配置模型路径，见上方说明"
            
        except ImportError:
            result["error"] = "WeNet 未安装或依赖缺失"
            logger.warning(f"⚠️  {result['error']}")
            logger.info("📦 安装命令:")
            logger.info("   git clone https://github.com/wenet-e2e/wenet.git")
            logger.info("   cd wenet && pip install -r requirements.txt && pip install .\n")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ 错误: {e}\n")
        
        self.results["wenet"] = result
        return result
    
    def run_all_tests(self) -> Dict:
        """运行所有模型测试"""
        logger.info("\n" + "🧪 " * 35)
        logger.info("开始运行 ASR 模型基准测试")
        logger.info("🧪 " * 35 + "\n")
        
        self.test_paddlespeech()
        self.test_funasr()
        self.test_wenet()
        
        return self.results
    
    def print_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 测试摘要")
        logger.info("=" * 70)
        
        for model_name, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"\n{status_icon} {result['model']}")
            
            if result["status"] == "success":
                logger.info(f"   结果: {result['result']}")
                logger.info(f"   耗时: {result['time']:.2f}s")
            else:
                logger.info(f"   错误: {result['error']}")
                if "note" in result:
                    logger.info(f"   说明: {result['note']}")
        
        logger.info("\n" + "=" * 70 + "\n")
    
    def save_results(self, output_file: str = "asr_results.json"):
        """保存测试结果到JSON文件"""
        # 转换不可序列化的对象
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_result = result.copy()
            if isinstance(serializable_result["result"], dict):
                serializable_result["result"] = {
                    k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in serializable_result["result"].items()
                }
            serializable_results[model_name] = serializable_result
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 结果已保存到: {output_file}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR 模型基准测试")
    parser.add_argument("--audio", type=str, required=True, help="测试音频文件路径")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备 (cpu 或 cuda:0)")
    parser.add_argument("--output", type=str, default="asr_results.json", help="结果输出文件")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=["paddlespeech", "funasr", "wenet"],
                       help="要测试的模型 (paddlespeech, funasr, wenet)")
    
    args = parser.parse_args()
    
    # 初始化基准测试
    try:
        benchmark = ASRBenchmark(args.audio, device=args.device)
        
        # 选择性测试
        for model in args.models:
            if model.lower() == "paddlespeech":
                benchmark.test_paddlespeech()
            elif model.lower() == "funasr":
                benchmark.test_funasr()
            elif model.lower() == "wenet":
                benchmark.test_wenet()
        
        # 打印摘要和保存结果
        benchmark.print_summary()
        benchmark.save_results(args.output)
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
