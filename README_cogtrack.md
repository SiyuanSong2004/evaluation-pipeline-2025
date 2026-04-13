# 2026 Chinese BabyLM Challenge Evaluation Pipeline

## Cogbench

### Data
Download the Cogbench fMRI data from Hugging Face:
```
https://huggingface.co/datasets/zhiheng-qian/cogbench
```

Extract `cogbench-fmri-0415.tar` under `evaluation_data/` so that the default path becomes:
```
evaluation_data/cogbench-fmri-0415
```

You can use the `train` and `dev` splits for your own development. The test split will be released later.
By default, the eval module fits a ridge-regression model on the training split and evaluates it on the dev split. Correlation is used as the metric.


### Run

Install dependencies first:
```
pip install -r requirements.txt
```

Sanity check (fast mode):
```
bash eval_cogbench_fast.sh \
  --model_path Qwen3-0.6B \
  --output_dir .
```

Full eval (default tasks: `word_fmri,fmri`):
```
bash eval_cogbench.sh \
  --model_path Qwen3-0.6B \
  --output_dir .
```

Use `--task` to choose tasks. You can pass a comma-separated list.
By default, both `word_fmri` and `fmri` are run.
```
bash eval_cogbench_fast.sh \
  --model_path Qwen3-0.6B \
  --task word_fmri \
  --output_dir .
```

Model loading behavior:
- The pipeline loads your model from `--model_path`.
- It first tries `transformers.AutoModel`.
- If the config is encoder-decoder and unsupported by `AutoModel`, it falls back to `transformers.AutoModelForSeq2SeqLM`.
- If your model is unsupported, update `get_model_and_tokenizer` in `evaluation_pipeline/cogbench/utils/utils.py` to the exact class you used.


By default, inference extracts last hidden states as features for regression against human data.
For encoder-decoder models, decoder hidden states are used by default.
You can customize this behavior in the inference module if needed.
