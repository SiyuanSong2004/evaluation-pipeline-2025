import argparse
import glob
import os
import re
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
				return_special_tokens_mask=True
			)
			if getattr(tokenizer, "is_fast", False):
				word_ids = encoded_cpu.word_ids(batch_index=0)
			else:
				word_ids = _word_ids_slow_fallback(tokenizer, words, encoded_cpu)

			# -------------------------------------------------
			# Debug reconstructed/aligned word_ids
			# Enable with:
			#   export DEBUG_WORD_IDS=1
			#   export DEBUG_WORD_IDS_MAX=10
			if os.getenv("DEBUG_WORD_IDS", "0") == "1":
				max_print = int(os.getenv("DEBUG_WORD_IDS_MAX", "10"))
				printed = getattr(encode_words_mean_pool, "_debug_word_ids_printed", 0)
				if printed < max_print:
					input_ids_dbg = encoded_cpu["input_ids"][0].tolist()
					tokens_dbg = tokenizer.convert_ids_to_tokens(input_ids_dbg)

					# compact per-word coverage summary
					coverage = {}
					for i, wid in enumerate(word_ids):
						if wid is None:
							continue
						coverage.setdefault(wid, []).append(i)

					print("\n[DEBUG_WORD_IDS]")
					print(f"is_fast={getattr(tokenizer, 'is_fast', False)}")
					print(f"words={words}")
					print(f"tokens={tokens_dbg}")
					print(f"word_ids={word_ids}")
					print("coverage_by_word_index:")
					for wid in range(len(words)):
						pos = coverage.get(wid, [])
						toks = [tokens_dbg[p] for p in pos]
						print(f"  word[{wid}]='{words[wid]}' -> positions={pos}, tokens={toks}")

					setattr(encode_words_mean_pool, "_debug_word_ids_printed", printed + 1)
			# -------------------------------------------------

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


def _word_ids_slow_fallback(tokenizer, words, encoded_cpu):
	"""
	Reconstruct a word_ids-like list for slow tokenizers.
	Returns: list[Optional[int]] length == sequence length
	"""
	input_ids = encoded_cpu["input_ids"][0].tolist()

	if "special_tokens_mask" in encoded_cpu:
		special_mask = encoded_cpu["special_tokens_mask"][0].tolist()
	else:
		special_set = set(getattr(tokenizer, "all_special_ids", []))
		special_mask = [1 if tid in special_set else 0 for tid in input_ids]

	# Non-special token positions in final sequence
	content_positions = [i for i, m in enumerate(special_mask) if m == 0]

	# Token count per original word (without specials)
	piece_lens = []
	for w in words:
		ids = tokenizer(w, add_special_tokens=False)["input_ids"]
		piece_lens.append(len(ids))

	word_ids = [None] * len(input_ids)
	cursor = 0
	for wid, n_pieces in enumerate(piece_lens):
		for _ in range(n_pieces):
			if cursor >= len(content_positions):
				return word_ids  # truncated sequence
			pos = content_positions[cursor]
			word_ids[pos] = wid
			cursor += 1

	return word_ids


def infer_sentence(
	model_path_or_name: str,
	datapath: str,
	output_dir: str | None = None,
	save_predictions: bool = True,
	revision_name: str | None = None,
	layer_index: int = -1,
	backend: str | None = None,
):
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

	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name, backend=backend)

	for story_file in story_files:
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
			print(f"Saved {save_path}: data shape = {data.shape}")

	return output_dir

