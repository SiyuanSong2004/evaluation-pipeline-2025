import argparse
import json
import os

import numpy as np
import torch

from ..utils.utils import DEVICE, forward_for_representations, get_model_and_tokenizer, get_model_name_from_path



SAVE_PREDICTIONS = True
BATCH_SIZE = 64


def _resolve_cogbench_root(data_path: str) -> str:
	"""Accept either cogbench root or the word_fmri subdir as input."""
	if os.path.isdir(os.path.join(data_path, "word")) and os.path.isdir(os.path.join(data_path, "word_fmri")):
		return data_path

	if os.path.basename(os.path.normpath(data_path)) == "word_fmri":
		candidate_root = os.path.dirname(os.path.normpath(data_path))
		if os.path.isdir(os.path.join(candidate_root, "word")):
			return candidate_root

	return data_path


def _load_words(root_path: str) -> list[str]:
	default_word_path = os.path.join(root_path, "word", "word.txt")
	if os.path.exists(default_word_path):
		with open(default_word_path, "r", encoding="utf-8") as f:
			return [line.strip() for line in f if line.strip()]

	split_names = ("train", "dev", "test")
	split_paths = {split_name: os.path.join(root_path, split_name, "word", "word.txt") for split_name in split_names}
	available_splits = [name for name, path in split_paths.items() if os.path.exists(path)]
	if "train" in available_splits and "dev" in available_splits:
		seen = set()
		merged_words = []
		for split_name in split_names:
			if split_name not in available_splits:
				continue
			path = split_paths[split_name]
			with open(path, "r", encoding="utf-8") as f:
				for line in f:
					word = line.strip()
					if not word or word in seen:
						continue
					seen.add(word)
					merged_words.append(word)
		if merged_words:
			return merged_words

	raise FileNotFoundError(
		f"No word list found. Expected either {default_word_path} or split files with at least train/dev under {root_path}."
	)


def _mean_pool_last_hidden(last_hidden_state, attention_mask=None):
	if attention_mask is None:
		return last_hidden_state.mean(dim=1)

	mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
	masked_hidden = last_hidden_state * mask
	token_count = mask.sum(dim=1).clamp(min=1e-9)
	return masked_hidden.sum(dim=1) / token_count


def extract_word_features(words, model, tokenizer, batch_size=BATCH_SIZE, backend: str | None = None):
	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token

	word_features = {}
	for start in range(0, len(words), batch_size):
		batch_words = words[start:start + batch_size]
		inputs = tokenizer(
			batch_words,
			return_tensors="pt",
			padding=True,
			truncation=True,
		)
		inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

		with torch.inference_mode():
			outputs = forward_for_representations(model, inputs, backend=backend)

		pooled = _mean_pool_last_hidden(outputs.last_hidden_state, inputs.get("attention_mask"))
		pooled = pooled.to(dtype=torch.float32).detach().cpu().numpy()

		for index, word in enumerate(batch_words):
			word_features[word] = pooled[index]

	return word_features


def infer_word(
	model_path_or_name: str,
	datapath: str,
	output_root: str | None = None,
	save_predictions: bool = SAVE_PREDICTIONS,
	revision_name: str | None = None,
	backend: str | None = None,
):
	root_path = _resolve_cogbench_root(datapath)
	model_name = get_model_name_from_path(model_path_or_name)
	words = _load_words(root_path)

	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name, backend=backend)
	word_features = extract_word_features(words, model, tokenizer, backend=backend)

	if save_predictions:
		persist_root = output_root if output_root is not None else root_path
		save_path = os.path.join(persist_root, model_name, "word_feature.json")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		serializable_features = {word: feature.tolist() for word, feature in word_features.items()}
		with open(save_path, "w", encoding="utf-8") as f:
			json.dump(serializable_features, f, ensure_ascii=False)
		print(f"Saved features: {save_path}")
	return word_features


def main(args):
	infer_word(
		model_path_or_name=args.model_name,
		datapath=args.data_path,
		output_root=getattr(args, "output_root", None),
		save_predictions=not args.no_save_predictions,
		revision_name=args.revision_name,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extract and evaluate word features for word_fmri.")
	parser.add_argument("--data_path", type=str, required=True, help="Cogbench data root.")
	parser.add_argument(
		"--model_name",
		type=str,
		default="bert-base-chinese",
		help="Hugging Face model name or local model path.",
	)
	parser.add_argument(
		"--revision_name",
		type=str,
		default=None,
		help="Optional Hugging Face revision.",
	)
	parser.add_argument(
		"--no_save_predictions",
		action="store_true",
		help="Disable saving word_feature.json.",
	)

	main(parser.parse_args())
