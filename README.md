# vx sft

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![LoRA](https://img.shields.io/badge/Finetune-LoRA-orange.svg)
![UI](https://img.shields.io/badge/UI-Gradio-pink.svg)

一个面向私有聊天风格微调的公开模板仓库，包含 SFT / LoRA 训练、二轮训练、LoRA 合并、批量推理检查和 Gradio UI。

这个仓库的目标不是直接公开私有模型，而是提供一套可以复用的工程骨架，让你在**不公开聊天数据**的前提下，自己完成训练、验证、合并和部署。

## 目录

- [项目特点](#项目特点)
- [目录结构](#目录结构)
- [界面预览](#界面预览)
- [隐私原则](#隐私原则)
- [快速开始](#快速开始)
- [推荐工作流](#推荐工作流)
- [文档说明](#文档说明)
- [注意事项](#注意事项)
- [许可证](#许可证)

## 项目特点

- 支持 LoRA / SFT 训练流程
- 支持第二轮微调配置和双卡训练脚本
- 支持 LoRA merge 成完整模型
- 支持本地 Gradio 聊天 UI
- 支持训练进度查看和手工复线
- 支持整理成 GitHub 可公开发布版

## 目录结构

- `training/`：训练、推理、合并、进度查看脚本
- `configs/`：训练配置模板
- `ui/`：本地对话 UI
- `docs/`：训练说明、部署说明、复线记录
- `assets/`：公开展示用素材
- `requirements-train.txt`：训练依赖清单

## 界面预览

你可以把后续的 UI 截图放到：

- `assets/ui-demo.png`

仓库当前已预留素材目录：`assets/README.md`

如果你要做 GitHub 首页展示，建议补两张图：

- 一张聊天 UI 截图
- 一张训练到部署的流程图

## 隐私原则

本仓库**不包含**以下内容：

- 原始聊天记录
- 清洗后的私有训练数据
- 已训练完成的私有权重
- 含个人信息的日志、样本、分析结果

如果你要继续公开自己的版本，建议始终忽略：

- `texts/`
- `outputs/`
- `analysis/`
- `checkpoints/`
- `merged/`
- `logs/`
- `models/`

## 快速开始

### 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r ui/requirements.txt
```

如果要训练，再安装训练依赖：

```bash
pip install -r requirements-train.txt
```

### 2. 准备基座模型

把你的基座模型放到：

```bash
models/Qwen2.5-3B-Instruct
```

### 3. 准备数据集

放入你自己的 JSONL 数据：

```bash
outputs/train.jsonl
outputs/valid.jsonl
```

第二轮数据可放到：

```bash
outputs/round2_train.jsonl
outputs/round2_valid.jsonl
```

### 4. 第一轮训练

```bash
python3 training/train_lora.py --config configs/lora_qwen3b.yaml
```

### 5. 第二轮训练

单卡：

```bash
python3 training/train_lora.py --config configs/lora_qwen3b_round2.yaml
```

双卡：

```bash
bash training/run_train_round2_ddp.sh
```

查看进度：

```bash
bash training/check_progress.sh
```

### 6. 合并 LoRA

```bash
python3 training/merge_lora.py \
  --base-model models/Qwen2.5-3B-Instruct \
  --adapter-path checkpoints/vx-sft-lora-round2 \
  --output-dir merged/vx-sft-merged
```

### 7. 启动 UI

merged 模式：

```bash
bash ui/run_ui_merged.sh
```

adapter 模式：

```bash
bash ui/run_ui_adapter.sh
```

## 推荐工作流

1. 用第一轮训练让模型先学整体风格
2. 做批量推理，收集 badcase
3. 用第二轮训练集中修 badcase
4. 效果稳定后 merge 成完整模型
5. 最后用 UI 做人工对话验证

## 文档说明

- `docs/How_It_Was_Done.md`：这次项目是怎么做的
- `docs/Train_Steps.md`：手工复线训练步骤
- `docs/Deployment.md`：部署与迁移说明
- `docs/双卡训练与进度查看.md`：双卡训练与进度查看
- `docs/第二轮训练方案.md`：第二轮训练策略建议
- `docs/训练与推理说明.md`：训练、推理、合并说明

## 注意事项

- 默认示例基于 `Qwen2.5-3B-Instruct`
- UI 更适合本地验证和内部演示
- 公开仓库建议只放代码、模板和文档
- 如果你要公开模型，请先确认训练数据没有隐私风险
- 如果你要迁移到本地电脑，优先同步 `ui/`、`training/`、`configs/`、`docs/`

## 许可证

本项目使用 `MIT` 许可证，见 `LICENSE`。
