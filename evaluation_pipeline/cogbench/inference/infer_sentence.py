import argparse
import glob
import os
import re
import time
from typing import List

import numpy as np
import scipy.io as scio
import torch
from ..utils.utils import DEVICE, forward_for_representations, get_model_and_tokenizer


BATCH_SIZE = 64
SENTENCE_FEATURE_PREFIX = "sentence_feature"

def parse_story_id(path: str) -> int:
	name = os.path.basename(path)
	match = re.search(r"story_(\d+)\.txt$", name)
	if not match:
		raise ValueError(f"Invalid story filename: {path}")
	return int(match.group(1))


def _collect_story_files(datapath: str) -> List[str]:
	root_story_dir = os.path.join(datapath, "story")
	root_files = sorted(glob.glob(os.path.join(root_story_dir, "story_*.txt")), key=parse_story_id)
	if root_files:
		return root_files

	collected_by_id = {}
	for split_name in ("train", "dev", "test"):
		split_story_dir = os.path.join(datapath, split_name, "story")
		for path in glob.glob(os.path.join(split_story_dir, "story_*.txt")):
			story_id = parse_story_id(path)
			if story_id not in collected_by_id:
				collected_by_id[story_id] = path

	return [collected_by_id[sid] for sid in sorted(collected_by_id.keys())]


def read_words_per_line(path: str) -> List[List[str]]:
	lines = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			words = line.split()
			if words:
				lines.append(words)
	return lines


def split_words_to_fit_model(words: List[str], tokenizer, max_content_tokens: int) -> List[List[str]]:
	if not words:
		return []

	chunks = []
	current = []
	for word in words:
		trial = current + [word]
		n_tokens = len(
			tokenizer(trial, is_split_into_words=True, add_special_tokens=False)["input_ids"]
		)
		if n_tokens <= max_content_tokens:
			current = trial
		else:
			if current:
				chunks.append(current)
			current = [word]
	if current:
		chunks.append(current)
	return chunks


def encode_words_mean_pool(
	words_per_line: List[List[str]],
	tokenizer,
	model,
	layer_index: int,
	backend: str | None = None,
) -> np.ndarray:
	all_word_reprs = []
	special = tokenizer.num_special_tokens_to_add(pair=False)
	tokenizer_max_len = int(getattr(tokenizer, "model_max_length", 512))
	if tokenizer_max_len > 100000:
		tokenizer_max_len = 512

	model_max_len = getattr(model.config, "max_position_embeddings", None)
	if model_max_len is None and (backend in {"enc_dec_mask", "enc_dec_prefix"} or getattr(model.config, "is_encoder_decoder", False)):
		max_candidates = []
		encoder = model.get_encoder() if hasattr(model, "get_encoder") else getattr(model, "encoder", None)
		decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "decoder", None)
		if encoder is not None:
			enc_max = getattr(encoder.config, "max_position_embeddings", None)
			if enc_max is not None:
				max_candidates.append(int(enc_max))
		if decoder is not None:
			dec_max = getattr(decoder.config, "max_position_embeddings", None)
			if dec_max is not None:
				max_candidates.append(int(dec_max))
		if max_candidates:
			model_max_len = min(max_candidates)

	if model_max_len is None:
		max_len = tokenizer_max_len
	else:
		max_len = int(min(tokenizer_max_len, int(model_max_len)))
	max_content_tokens = max(1, max_len - special)

	for line_words in words_per_line:
		for words in split_words_to_fit_model(line_words, tokenizer, max_content_tokens):
			encoded_cpu = tokenizer(
				words,
				is_split_into_words=True,
				return_tensors="pt",
				truncation=True,
				max_length=max_len,
			)
			word_ids = encoded_cpu.word_ids(batch_index=0)
			encoded = {key: value.to(DEVICE) for key, value in encoded_cpu.items()}

			with torch.inference_mode():
				outputs = forward_for_representations(model, encoded, backend=backend)
			hidden = outputs.hidden_states[layer_index][0]

			for word_idx in range(len(words)):
				token_positions = [idx for idx, wid in enumerate(word_ids) if wid == word_idx]
				if not token_positions:
					all_word_reprs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
					continue

				token_vecs = hidden[token_positions]
				word_vec = token_vecs.mean(dim=0)
				all_word_reprs.append(word_vec.to(dtype=torch.float32).detach().cpu().numpy())

	if not all_word_reprs:
		return np.zeros((0, model.config.hidden_size), dtype=np.float32)
	return np.stack(all_word_reprs, axis=0)


def infer_sentence(
	model_path_or_name: str,
	datapath: str,
	output_dir: str | None = None,
	save_predictions: bool = True,
	revision_name: str | None = None,
	layer_index: int = -1,
	backend: str | None = None,
):
	step_start = time.time()
	print(f"[STEP] Starting inference for model: {model_path_or_name}")

	model_name = os.path.basename(os.path.normpath(model_path_or_name))
	if output_dir is None:
		output_dir = os.path.join(datapath, model_name)

	os.makedirs(output_dir, exist_ok=True)

	story_files = _collect_story_files(datapath)
	if not story_files:
		raise FileNotFoundError(
			f"No story files found in: {os.path.join(datapath, 'story')} "
			f"or in split dirs under {datapath}/{{train,dev,test}}/story"
		)
	print(f"[INFO] Found {len(story_files)} story files to process")

	print(f"[STEP] Loading model and tokenizer...")
	model_load_start = time.time()
	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name, backend=backend)
	print(f"[TIME] Model loading completed in {time.time() - model_load_start:.2f}s")

	print(f"[STEP] Extracting hidden states from stories...")
	extraction_start = time.time()
	for i, story_file in enumerate(story_files):
		story_start = time.time()
		story_id = parse_story_id(story_file)
		words_per_line = read_words_per_line(story_file)
		data = encode_words_mean_pool(
			words_per_line=words_per_line,
			tokenizer=tokenizer,
			model=model,
			layer_index=layer_index,
			backend=backend,
		)

		if save_predictions:
			save_path = os.path.join(output_dir, f"{SENTENCE_FEATURE_PREFIX}_story_{story_id}.mat")
			scio.savemat(save_path, {"data": data})
			print(f"[PROGRESS] Story {i+1}/{len(story_files)} (ID={story_id}): saved {data.shape} in {time.time() - story_start:.2f}s")

	print(f"[TIME] Feature extraction completed in {time.time() - extraction_start:.2f}s ({(time.time() - extraction_start)/60:.2f}m)")
	total_time = time.time() - step_start
	print(f"[TIME] Total inference completed in {total_time:.2f}s ({total_time/60:.2f}m)")

	return output_dir

