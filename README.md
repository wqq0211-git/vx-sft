# vx sft

`vx sft` 是一个可公开发布的对话风格 SFT / LoRA 微调模板，包含训练、合并、推理和 Gradio UI。

这个公开版只发布流程与代码，不发布任何私有聊天数据和私有权重。

## 包含内容

- `training/`：LoRA 训练、推理、合并脚本
- `configs/`：训练配置模板
- `ui/`：本地聊天 UI
- `docs/`：训练、部署、复线说明

## 不包含内容

- 原始聊天记录
- 清洗后的私有数据集
- 已训练好的私有模型权重
- 含个人信息的日志或分析文件

## 快速开始

### 1. 准备基础环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ui/requirements.txt
```

### 2. 准备基座模型

示例目录：

```bash
models/Qwen2.5-3B-Instruct
```

### 3. 准备训练数据

示例目录：

```bash
outputs/train.jsonl
outputs/valid.jsonl
```

### 4. 开始训练

```bash
python3 training/train_lora.py --config configs/lora_qwen3b.yaml
```

第二轮训练：

```bash
python3 training/train_lora.py --config configs/lora_qwen3b_round2.yaml
```

### 5. 合并 LoRA

```bash
python3 training/merge_lora.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora-round2 \
  --output-dir merged/vx-sft-merged
```

### 6. 启动 UI

merged 模式：

```bash
bash ui/run_ui_merged.sh
```

adapter 模式：

```bash
bash ui/run_ui_adapter.sh
```

## 文档

- `docs/How_It_Was_Done.md`：这次项目是怎么做的
- `docs/Train_Steps.md`：自己手工怎么复线训练
- `docs/Deployment.md`：怎么部署和迁移

## 隐私建议

公开发布时建议始终忽略：

- `texts/`
- `outputs/`
- `analysis/`
- `checkpoints/`
- `merged/`
- `logs/`
- `models/`

这样别人可以复用你的工程，但拿不到你的私有聊天数据。
