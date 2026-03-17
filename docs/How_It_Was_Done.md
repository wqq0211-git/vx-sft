# 这次是怎么做的

这次项目实际分成 6 步：

## 1. 整理数据

先把聊天语料整理成适合监督微调的 JSONL 格式，核心目标是：
- 保留自然口语感
- 控制回复长度
- 避免过度书面化
- 去掉无意义噪声样本

## 2. 做第一轮 LoRA 微调

使用第一版训练集跑基础风格对齐，让模型先学会整体说话方式。

典型命令：

```bash
python3 training/train_lora.py --config configs/lora_qwen3b.yaml
```

## 3. 批量推理抽检 badcase

训练完后，不是直接结束，而是先做批量推理：
- 看回复是不是太长
- 看是不是太书面
- 看是不是不够像聊天
- 找出需要二次修正的 badcase

## 4. 做第二轮 LoRA 微调

第二轮不是简单重复训练，而是针对 badcase 修正：
- 降低学习率
- 补更贴近日常聊天的数据
- 强化稳定性和一致性

典型命令：

```bash
python3 training/train_lora.py --config configs/lora_qwen3b_round2.yaml
```

或者双卡：

```bash
bash training/run_train_round2_ddp.sh
```

## 5. 合并成完整模型

当第二轮效果稳定后，把 LoRA adapter merge 到基座模型上，得到可直接使用的完整模型。

典型命令：

```bash
python3 training/merge_lora.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora-round2 \
  --output-dir merged/vx-sft-merged
```

## 6. 起本地 UI 做人工对话验证

最后用 Gradio UI 做真实问答验证，手工看风格、长度、稳定性是不是达到预期。

```bash
bash ui/run_ui_merged.sh
```

## 公开发布时做了什么处理

为了能发到 GitHub，上公开版时做了这些脱敏：
- 删除私有聊天数据
- 删除训练输出数据
- 删除权重目录
- 删除服务器绝对路径
- 删除个人化命名
- 改成可复用的通用模板说明
