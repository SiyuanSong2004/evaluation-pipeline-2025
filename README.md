# 2026 Chinese BabyLM Challenge — Evaluation Pipeline

<details open>
<summary><b>English</b></summary>

This repository contains the evaluation pipeline for the **2026 Chinese BabyLM Challenge**, adapted from the original BabyLM 2025 English evaluation pipeline. The pipeline covers three evaluation tracks: sentence zero-shot, fine-tuning, and cognitive benchmark (Cogbench).

## Setup

```bash
pip install -r requirements.txt
```

For gated HuggingFace datasets:

```bash
huggingface-cli login
```

## Evaluation Tracks

### 1. Sentence Zero-Shot

Logit-based scoring of minimal pairs for Chinese linguistic phenomena (ZhoBLiMP).

```bash
bash eval_zero_shot.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix>
```

For intermediate checkpoints:

```bash
bash eval_zero_shot_fast.sh <model_path> <revision_name> <backend>

# All checkpoints at once:
bash eval_zero_shot_fast_all_revisions.sh <model_path> <backend> <track>
```

### 2. Fine-Tuning

Sequence classification on Chinese CLUE tasks.

```bash
bash eval_finetuning.sh <model_path> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
```

**Tasks:** AFQMC, OCNLI, TNEWS, CLUEWSC2020

### 3. Cognitive Benchmark (Cogbench)

Evaluates models against human brain recording data (fMRI). Uses ridge regression to fit model representations to neural signals, evaluated by Pearson/Spearman correlation.

#### Data

Download the Cogbench fMRI data from HuggingFace:

```
https://huggingface.co/datasets/zhiheng-qian/cogbench
```

Extract `cogbench-fmri-0415.tar` under `evaluation_data/` so the default path becomes:

```
evaluation_data/cogbench-fmri-0415/
```

The `train` and `dev` splits are available for development. The `test` split will be released later.

#### Run

Sanity check (fast mode):

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --backend causal \
  --output_dir .
```

Full evaluation (default tasks: `word_fmri,fmri`):

```bash
bash eval_cogbench.sh \
  --model_path Qwen/Qwen3-0.6B \
  --output_dir .
```

Select specific tasks with `--task` (comma-separated):

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --task word_fmri \
  --output_dir .
```

**Available tasks:** `word_fmri`, `fmri` *(meg, eye_tracking: coming soon)*

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | HuggingFace model name or local path |
| `--backend` | `causal` | `causal`, `mlm`, `mntp`, `enc_dec_mask`, `enc_dec_prefix` |
| `--task` | `word_fmri,fmri` | Comma-separated task list |
| `--eval_dir` | `evaluation_data/cogbench-fmri-0415` | Path to data |
| `--output_dir` | current directory | Output directory |
| `--revision_name` | — | Model revision name |

#### Model Loading

The pipeline loads your model via `transformers.AutoModel`. If the config is encoder-decoder, it falls back to `AutoModelForSeq2SeqLM`. For unsupported architectures, update `get_model_and_tokenizer` in `evaluation_pipeline/cogbench/utils/utils.py`.

By default, last hidden states are used as features. For encoder-decoder models, decoder hidden states are used.

## Results Structure

```
results/<model_name>/<revision_or_"main">/
  finetune/<task>/predictions.jsonl + results.txt
  zero_shot/<backend>/<task>/<dataset>/predictions.jsonl + best_temperature_report.txt
  cogbench/<task>/predictions.jsonl + results.txt
```

## Collating Results

```bash
bash collate_preds.sh <model_name> <backend> <track>
```

## Evaluation Data

All datasets are placed under `evaluation_data/` in JSONL format.

| Track | Dataset | Source |
|---|---|---|
| Zero-shot | ZhoBLiMP | `Junrui1202/zhoblimp` |
| Fine-tuning | CLUE (AFQMC, OCNLI, TNEWS, CLUEWSC2020) | `clue` |
| Cogbench | fMRI recordings | `zhiheng-qian/cogbench` |

</details>

---

<details>
<summary><b>中文</b></summary>

本仓库为 **2026 中文 BabyLM 挑战赛** 评测流水线，基于原版英文 BabyLM 2025 评测代码改编。流水线涵盖三个评测方向：句子零样本、微调和认知基准（Cogbench）。

## 环境配置

```bash
pip install -r requirements.txt
```

如需访问受限 HuggingFace 数据集：

```bash
huggingface-cli login
```

## 评测方向

### 1. 句子零样本

基于对数概率对中文最小对进行语言现象评测（ZhoBLiMP）。

```bash
bash eval_zero_shot.sh <model_path> <causal|mlm|mntp|enc_dec_mask|enc_dec_prefix>
```

中间检查点评测：

```bash
bash eval_zero_shot_fast.sh <model_path> <revision_name> <backend>

# 批量评测所有检查点：
bash eval_zero_shot_fast_all_revisions.sh <model_path> <backend> <track>
```

### 2. 微调

在中文 CLUE 任务上进行序列分类微调。

```bash
bash eval_finetuning.sh <model_path> [lr] [batch_size] [max_epochs] [wsc_epochs] [seed]
```

**任务：** AFQMC, OCNLI, TNEWS, CLUEWSC2020

### 3. 认知基准（Cogbench）

将模型表征与人脑神经记录数据（fMRI）对齐，使用岭回归拟合，以 Pearson/Spearman 相关系数评测。

#### 数据

从 HuggingFace 下载 Cogbench fMRI 数据：

```
https://huggingface.co/datasets/zhiheng-qian/cogbench
```

将 `cogbench-fmri-0415.tar` 解压至 `evaluation_data/`，默认路径为：

```
evaluation_data/cogbench-fmri-0415/
```

`train` 和 `dev` 分片可用于开发，`test` 分片将于后续发布。

#### 运行

快速验证：

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --backend causal \
  --output_dir .
```

完整评测（默认任务：`word_fmri,fmri`）：

```bash
bash eval_cogbench.sh \
  --model_path Qwen/Qwen3-0.6B \
  --output_dir .
```

通过 `--task` 指定任务（逗号分隔）：

```bash
bash eval_cogbench_fast.sh \
  --model_path Qwen/Qwen3-0.6B \
  --task word_fmri \
  --output_dir .
```

**可用任务：** `word_fmri`, `fmri` *（meg, eye_tracking 即将支持）*

**完整参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | 必填 | HuggingFace 模型名或本地路径 |
| `--backend` | `causal` | `causal`, `mlm`, `mntp`, `enc_dec_mask`, `enc_dec_prefix` |
| `--task` | `word_fmri,fmri` | 逗号分隔的任务列表 |
| `--eval_dir` | `evaluation_data/cogbench-fmri-0415` | 数据路径 |
| `--output_dir` | 当前目录 | 输出目录 |
| `--revision_name` | — | 模型版本名 |

#### 模型加载

流水线通过 `transformers.AutoModel` 加载模型。如配置为编码器-解码器结构，则回退至 `AutoModelForSeq2SeqLM`。对于不支持的模型结构，请修改 `evaluation_pipeline/cogbench/utils/utils.py` 中的 `get_model_and_tokenizer`。

默认使用最后一层隐藏状态作为特征；编码器-解码器模型默认使用解码器隐藏状态。

## 结果目录结构

```
results/<model_name>/<revision_or_"main">/
  finetune/<task>/predictions.jsonl + results.txt
  zero_shot/<backend>/<task>/<dataset>/predictions.jsonl + best_temperature_report.txt
  cogbench/<task>/predictions.jsonl + results.txt
```

## 汇总结果

```bash
bash collate_preds.sh <model_name> <backend> <track>
```

## 评测数据

所有数据集以 JSONL 格式存放于 `evaluation_data/`。

| 方向 | 数据集 | 来源 |
|---|---|---|
| 零样本 | ZhoBLiMP | `Junrui1202/zhoblimp` |
| 微调 | CLUE (AFQMC, OCNLI, TNEWS, CLUEWSC2020) | `clue` |
| Cogbench | fMRI 神经记录 | `zhiheng-qian/cogbench` |

</details>
