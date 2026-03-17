# 手工复线训练步骤

如果你自己纯手工复线，可以按这个顺序做。

## 第 1 步：准备目录

```bash
mkdir -p models outputs checkpoints merged logs
```

## 第 2 步：准备环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ui/requirements.txt
```

如果训练还缺依赖，再补：

```bash
pip install datasets pyyaml tqdm accelerate peft trl
```

## 第 3 步：放入基座模型

把你的基础模型放到：

```bash
models/Qwen2.5-3B-Instruct
```

## 第 4 步：准备训练数据

你需要至少两份 JSONL：

```bash
outputs/train.jsonl
outputs/valid.jsonl
```

第二轮可再准备：

```bash
outputs/round2_train.jsonl
outputs/round2_valid.jsonl
```

## 第 5 步：跑第一轮训练

```bash
python3 training/train_lora.py --config configs/lora_qwen3b.yaml
```

输出通常会写到：

```bash
checkpoints/vx-sft-lora
```

## 第 6 步：做推理抽检

```bash
python3 training/infer_lora.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora \
  --prompt "今天忙完了吗？"
```

## 第 7 步：跑第二轮训练

单卡：

```bash
python3 training/train_lora.py --config configs/lora_qwen3b_round2.yaml
```

双卡：

```bash
bash training/run_train_round2_ddp.sh
```

查看训练进度：

```bash
bash training/check_progress.sh
```

## 第 8 步：合并模型

```bash
python3 training/merge_lora.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora-round2 \
  --output-dir merged/vx-sft-merged
```

## 第 9 步：启动 UI

merged 模式：

```bash
bash ui/run_ui_merged.sh
```

adapter 模式：

```bash
bash ui/run_ui_adapter.sh
```

## 第 10 步：人工验证

重点看这几项：
- 是否足够口语化
- 是否回复过长
- 是否容易跑偏
- 是否重复表达
- 是否在高温度下仍稳定

## 建议节奏

- 第一轮先学整体风格
- 第二轮只修 badcase
- 每轮训练后都做批量抽检
- 不要一开始就把轮数和数据量拉太大
