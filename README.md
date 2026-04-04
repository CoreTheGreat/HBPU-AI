# HBPU-AI
This repository is built for students of Hubei Polytechnic Unersity to learning artificial intelligence.

Metierials link of Machine Learning 2024: https://pan.baidu.com/s/1Tv_VCZxnoBmxC-z9qimyiQ?pwd=2493

Metierials link of Artificial Intelligence 2025: https://pan.baidu.com/s/1IxueuU_ioVRQVJQhvtFxOg?pwd=c8nj

Metierials link of Artificial Intelligence 2026: https://pan.baidu.com/s/1gYJPvpX0N9OsuMgOXRK04A?pwd=dwt8

# 工程文件夹说明
- AI_2025: 2025年课件
- AI_2026: 2026年课件
- Machine_Learning_2024: 2024年《机器学习》课件及实验
- data：实验数据
- fonts：实验所需字体
- images：工程图片

# 参考资料与致敬
本课程参考以下文献以及多媒体资源，向原作者表达崇高敬意。

## 📚 参考教材
- 张奇，桂涛，郑锐，黄萱菁. 大语言模型从理论到实践（第二版）.电子工业出版社，2025.
- 教材代码仓库：
  - （国际）[https://github.com/intro-llm/intro-llm-code](https://github.com/intro-llm/intro-llm-code)
  - （国内）[https://gitee.com/CoreTheGreat/intro-llm-code](https://gitee.com/CoreTheGreat/intro-llm-code)

## 🎥 参考视频资源
- 李宏毅.智能体2026系列课程：[BV1o3wvzUEDD（哔哩哔哩）](https://www.bilibili.com/video/BV1o3wvzUEDD)
- 李宏毅.人工智能2025系列课程：[BV1oduvzPEk4（哔哩哔哩）](https://www.bilibili.com/video/BV1oduvzPEk4)
- 李宏毅.生成式AI导论2024：[BV1XS411w7qr（哔哩哔哩）](https://www.bilibili.com/video/BV1XS411w7qr)
- 李宏毅YouTube官网：[https://www.youtube.com/@HungyiLeeNTU](https://www.youtube.com/@HungyiLeeNTU)
- 吴恩达. 2026智能体系列课程：[BV1DfrdByE2H（哔哩哔哩）](https://www.bilibili.com/video/BV1DfrdByE2H)
- 吴恩达. 2025生成式人工智能：[BV1sMEyzhEM3（哔哩哔哩）](https://www.bilibili.com/video/BV1sMEyzhEM3)
- 吴恩达Agentic AI官网：[https://www.deeplearning.ai/courses/agentic-ai](https://www.deeplearning.ai/courses/agentic-ai)

## 💻 实验参考代码仓库
- **LLM & Agent 核心技术实战教程**
  - Greebie/LLM-Agent-Core_Concept_Code: [https://github.com/Greebbie/LLM-Agent-Core_Concept_Code](https://github.com/Greebbie/LLM-Agent-Core_Concept_Code)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/LLM-Agent-Core_Concept_Code](https://gitee.com/CoreTheGreat/LLM-Agent-Core_Concept_Code)

- **LLM Course**
  - Mlabonne/llm-course: [https://github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/llm-course](https://gitee.com/CoreTheGreat/llm-course)

- **从0开始训练自己的Phi2中文小模型**
  - charent/Phi2-mini-Chinese: [https://github.com/CoreTheGreat/Phi2-mini-Chinese](https://github.com/CoreTheGreat/Phi2-mini-Chinese)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/Phi2-mini-Chinese](https://gitee.com/CoreTheGreat/Phi2-mini-Chinese)

- **中文对话0.2B小模型**
  - charent/ChatLM-mini-Chinese: [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/ChatLM-mini-Chinese](https://gitee.com/CoreTheGreat/ChatLM-mini-Chinese)

- **OpenManus**
  - FoundationAgents/OpenManus: [https://github.com/FoundationAgents/OpenManus](https://github.com/FoundationAgents/OpenManus)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/OpenManus](https://gitee.com/CoreTheGreat/OpenManus)

- **System Prompts Leaks**
  - asgeirtj/system_prompts_leaks: [https://github.com/asgeirtj/system_prompts_leaks](https://github.com/asgeirtj/system_prompts_leaks)
  - (Gitee Mirror): [https://gitee.com/CoreTheGreat/system_prompts_leaks](https://gitee.com/CoreTheGreat/system_prompts_leaks)

# 实验环境（Anaconda + VS Code）安装参考博客
博客园：https://www.cnblogs.com/superhui/category/2492198.html

CSDN：https://blog.csdn.net/south_rosefinch/category_13120133.html

> 参考来自肖彭辉

## 🛠️ 环境准备（Conda + CPU 版本）

默认前提：
1. 已安装 Anaconda / Miniconda
2. `conda venv` 还未创建
3. 相关包尚未安装

### 1. 创建 Conda 虚拟环境

```bash
conda create -n hbpu-ai python=3.11 -y
conda activate hbpu-ai
```

### 2. 安装 PyTorch（CPU 版本）

```bash
# 仅安装 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 2.1 可选：配置国内镜像（更快）

#### Conda 镜像

可先临时使用：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --set show_channel_urls yes
```

#### pip 镜像

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 或者清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2.2 验证国内镜像是否生效

- Conda 频道验证：
  - `conda config --show channels`（查看已添加镜像优先级）
  - `conda search numpy`（如果来自镜像源即成功）

- pip 源验证：
  - `pip config get global.index-url`（如果配置了全局源）
  - `pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`（看下载源是否为阿里）

或直接在安装时观察输出：
- `Looking in indexes: https://mirrors.aliyun.com/pypi/simple/` 表示镜像成功。

### 3. 安装依赖
```bash
pwd
```
确保当前路径在"./HBPU-AI"

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### 4. 运行环境验证

1. 在环境中运行 `environment_validation.ipynb`
2. 或在终端中执行"python"进入python观景，并输出以下脚本检查版本

```bash
import torch, transformers, accelerate, sentence_transformers, datasets
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('sentence_transformers', sentence_transformers.__version__)
print('datasets', datasets.__version__)
```

### 5. 工程拉取补充说明

models/uer_gpt2_chinese/model.safetensors等模型文件过大无法上传，请自行下载

1. 可运行.ipynb中的相关代码下载模型

2. 从网盘链接手动下载至models目录下

网盘链接: https://pan.baidu.com/s/1gYJPvpX0N9OsuMgOXRK04A?pwd=dwt8