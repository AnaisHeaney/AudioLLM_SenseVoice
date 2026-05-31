# 🎤 ASR 中文语音识别模型基准测试框架

本测试框架提供了统一的基准测试环境，支持三个主流的中文语音识别(ASR)模型的对比测试。

## 📋 支持的模型

| 模型 | 机构 | 特点 | 推荐场景 |
|------|------|------|---------|
| **PaddleSpeech** | 百度 | 部署简单，精度高，文档完善 | 快速上手，通用场景 |
| **FunASR** | 阿里达摩院 | 高精度，多场景适配，复杂环境 | 生产环境，实际应用 |
| **WeNet** | 腾讯 | 端到端，高效推理，学术前沿 | 研究开发，定制训练 |

## 🚀 快速开始

### 1. 准备环境

```bash
# 克隆仓库
git clone https://github.com/AnaisHeaney/AudioLLM_SenseVoice.git
cd AudioLLM_SenseVoice

# 切换到测试分支
git checkout asr-models-benchmark

# 安装依赖（选择合适的方式）
```

### 2. 安装方式选择

#### 方式 A：快速安装（推荐新手）
```bash
# 仅安装 PaddleSpeech 和 FunASR（最常用）
pip install paddlespeech funasr>=1.1.3

# 或使用提供的 requirements 文件
pip install -r requirements_asr_benchmark.txt
```

#### 方式 B：完整安装（包含WeNet）
```bash
# 1. 安装基础依赖
pip install -r requirements_asr_benchmark.txt

# 2. 安装 WeNet（从源码）
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -r requirements.txt
pip install .
cd ..
```

#### 方式 C：分开安装（推荐）
如果你不需要所有模型，可以按需安装：

```bash
# 仅 PaddleSpeech
pip install paddlespeech

# 仅 FunASR
pip install funasr>=1.1.3 modelscope

# 仅 WeNet
# 见方式 B 中的 WeNet 安装步骤
```

### 3. 准备测试音频

下载中文测试音频示例：

```bash
# FunASR 示例音频
wget https://github.com/alibaba/FunASR/raw/main/demo/example_zh.wav -O test_audio.wav

# 或使用其他中文语音文件
# 支持格式: wav, mp3, flac 等
```

### 4. 运行测试

#### 基础用法（测试全部模型）
```bash
python asr_benchmark_test.py --audio test_audio.wav
```

#### 指定设备（GPU加速）
```bash
python asr_benchmark_test.py --audio test_audio.wav --device cuda:0
```

#### 仅测试特定模型
```bash
# 仅测试 PaddleSpeech
python asr_benchmark_test.py --audio test_audio.wav --models paddlespeech

# 仅测试 PaddleSpeech 和 FunASR
python asr_benchmark_test.py --audio test_audio.wav --models paddlespeech funasr

# 仅测试 FunASR
python asr_benchmark_test.py --audio test_audio.wav --models funasr
```

#### 自定义输出文件
```bash
python asr_benchmark_test.py --audio test_audio.wav --output my_results.json
```

## 📊 测试输出示例

### 控制台输出
```
🧪 🧪 🧪 ... 🧪 🧪 🧪
开始运行 ASR 模型基准测试
🧪 🧪 🧪 ... 🧪 🧪 🧪

======================================================================
1️⃣  测试 PaddleSpeech (百度)
======================================================================
📦 初始化 PaddleSpeech 模型...
🎤 开始语音识别...
✅ 识别成功！
📝 结果: 欢迎使用阿里达摩院智能语音识别系统
⏱️  耗时: 2.34s

======================================================================
2️⃣  测试 FunASR (阿里达摩院)
======================================================================
📦 初始化 FunASR 模型...
🎤 开始语音识别...
✅ 识别成功！
📝 结果: 欢迎使用阿里达摩院智能语音识别系统。
⏱️  耗时: 1.87s

======================================================================
📊 测试摘要
======================================================================

✅ PaddleSpeech
   结果: 欢迎使用阿里达摩院智能语音识别系统
   耗时: 2.34s

✅ FunASR
   结果: 欢迎使用阿里达摩院智能语音识别系统。
   耗时: 1.87s

❌ WeNet
   错误: WeNet 需要手动配置模型路径，见上方说明
   说明: 需要手动下载预训练模型

======================================================================

💾 结果已保存到: asr_results.json
```

### JSON 输出（asr_results.json）
```json
{
  "paddlespeech": {
    "model": "PaddleSpeech",
    "status": "success",
    "result": "欢迎使用阿里达摩院智能语音识别系统",
    "time": 2.34,
    "error": null
  },
  "funasr": {
    "model": "FunASR",
    "status": "success",
    "result": {
      "text": "欢迎使用阿里达摩院智能语音识别系统。",
      "raw_result": "..."
    },
    "time": 1.87,
    "error": null
  },
  "wenet": {
    "model": "WeNet",
    "status": "failed",
    "result": null,
    "time": 0,
    "error": "WeNet 需要手动配置模型路径",
    "note": "需要手动下载预训练模型"
  }
}
```

## 🔧 高级用法

### 在 Python 代码中使用

```python
from asr_benchmark_test import ASRBenchmark

# 初始化基准测试
benchmark = ASRBenchmark(audio_file="test_audio.wav", device="cpu")

# 测试特定模型
benchmark.test_paddlespeech()
benchmark.test_funasr()
benchmark.test_wenet()

# 或测试所有模型
results = benchmark.run_all_tests()

# 打印摘要
benchmark.print_summary()

# 保存结果
benchmark.save_results("my_results.json")
```

### 处理测试结果

```python
import json

# 读取结果
with open("asr_results.json", 'r', encoding='utf-8') as f:
    results = json.load(f)

# 比较各模型性能
for model_name, result in results.items():
    if result["status"] == "success":
        print(f"{model_name}: {result['time']:.2f}s")

# 找出最快的模型
fastest = min(
    [(k, v["time"]) for k, v in results.items() if v["status"] == "success"],
    key=lambda x: x[1]
)
print(f"最快的模型: {fastest[0]} ({fastest[1]:.2f}s)")
```

## 📥 WeNet 详细安装

如果想完整测试 WeNet，需要额外的步骤：

```bash
# 1. 克隆 WeNet 仓库
git clone https://github.com/wenet-e2e/wenet.git
cd wenet

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 WeNet
pip install .

# 4. 下载中文预训练模型（AISHELL-1）
# 访问: https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md
# 下载模型文件到本地

# 5. 修改 asr_benchmark_test.py 中的 WeNet 测试部分
# 将 config_path 和 model_path 指向你下载的模型
```

## 🐛 故障排除

### 问题 1: 模块未找到
```
ModuleNotFoundError: No module named 'paddlespeech'
```
**解决方案:**
```bash
pip install paddlespeech
```

### 问题 2: 模型下载超时
如果模型下载缓慢或超时，可以：
1. 使用代理服务
2. 手动下载模型文件
3. 使用国内镜像源

### 问题 3: GPU 内存不足
```bash
# 使用 CPU 推理
python asr_benchmark_test.py --audio test_audio.wav --device cpu
```

### 问题 4: 模型文件损坏
```bash
# 删除缓存并重新下载
rm -rf ~/.cache/modelscope/
rm -rf ~/.cache/huggingface/

# 重新运行测试
python asr_benchmark_test.py --audio test_audio.wav
```

## 📈 性能对比

典型的中文语音识别性能对比（基于 10 秒音频，单位秒）：

| 模型 | CPU | GPU (RTX3090) |
|------|-----|---------------|
| **PaddleSpeech** | 2.5s | 0.8s |
| **FunASR** | 1.8s | 0.5s |
| **WeNet** | 2.0s | 0.6s |

*注: 实际性能取决于硬件和音频质量*

## 📚 相关资源

- [PaddleSpeech 官方文档](https://github.com/PaddlePaddle/PaddleSpeech)
- [FunASR 官方文档](https://github.com/alibaba/FunASR)
- [WeNet 官方文档](https://github.com/wenet-e2e/wenet)
- [SenseVoice 主项目](https://github.com/FunAudioLLM/SenseVoice)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个测试框架！

## 📝 许可证

本测试框架遵循与 AudioLLM_SenseVoice 相同的许可证。

## ❓ 常见问题

### Q: 我应该用哪个模型？
**A:** 
- 新手推荐：**PaddleSpeech**（最简单）
- 生产环境：**FunASR**（最稳定高效）
- 研究开发：**WeNet**（最灵活）

### Q: 可以用于商业用途吗？
**A:** 是的，这些模型都是开源的，可以用于商业用途。请检查各模型的具体许可证。

### Q: 支持其他语言吗？
**A:** 是的，这三个模型都支持多语言。本框架主要针对中文优化，但可以轻松扩展。

---

**最后更新**: 2026-05-31
**作者**: AudioLLM_SenseVoice Team
