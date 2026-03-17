# 部署说明

## 项目定位

这是一个公开发布版模板：
- 可以公开上传到 GitHub
- 可以给别人复用训练流程和 UI
- 不包含任何私有聊天数据和训练权重

## 你需要自己准备的内容

- 基座模型目录，例如 `models/Qwen2.5-3B-Instruct`
- 你自己的训练数据，例如 `outputs/train.jsonl`
- 你自己的 LoRA 输出目录，例如 `checkpoints/vx-sft-lora`

## 本地启动 UI

### merged 模式

```bash
bash ui/run_ui_merged.sh
```

要求：
- `merged/vx-sft-merged` 已存在

### adapter 模式

```bash
bash ui/run_ui_adapter.sh
```

要求：
- `models/Qwen2.5-3B-Instruct` 已存在
- `checkpoints/vx-sft-lora-round2` 已存在

## 手工命令

### 启动 merged 模型

```bash
python3 ui/app.py --base-model merged/vx-sft-merged
```

### 启动 adapter 模型

```bash
python3 ui/app.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora-round2
```

## 本地迁移建议

迁移到另一台机器时，建议只同步：
- `ui/`
- `training/`
- `configs/`
- `docs/`
- 你自己的私有模型目录

不要公开同步：
- 聊天原始数据
- 中间清洗结果
- 训练日志中的敏感内容
