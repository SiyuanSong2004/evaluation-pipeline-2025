import json
import os

import numpy as np
import torch
from tqdm import tqdm


USE_STANDARDIZATION = True
INFER_CACHE_FILENAME = "eye_tracking_infer_cache.npz"
DEFAULT_SAMPLE_SEED = 42
EPS = 1e-8


def standardize_matrix(feature_matrix, mean=None, std=None):
	mean = np.nanmean(feature_matrix, axis=0) if mean is None else mean
	std = np.nanstd(feature_matrix, axis=0) if std is None else std
	std = np.where(std == 0, 1.0, std)
	return (feature_matrix - mean) / std


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
	mean = np.mean(matrix, axis=1, keepdims=True)
	std = np.std(matrix, axis=1, keepdims=True)
	std = np.where(std == 0, 1.0, std)
	return (matrix - mean) / std


def _sample_indices(total_words: int, max_words: int | None, seed: int) -> np.ndarray | None:
	if max_words is None or max_words <= 0 or total_words <= max_words:
		return None

	rng = np.random.default_rng(seed)
	indices = np.sort(rng.choice(total_words, size=max_words, replace=False))
	return indices


def _normalize_rows_torch(matrix: torch.Tensor) -> torch.Tensor:
	mean = matrix.mean(dim=1, keepdim=True)
	std = matrix.std(dim=1, unbiased=False, keepdim=True)
	std = torch.where(std == 0, torch.ones_like(std), std)
	return (matrix - mean) / std


def _columnwise_pearson_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
	left_centered = left - left.mean(dim=0, keepdim=True)
	right_centered = right - right.mean(dim=0, keepdim=True)
	numerator = (left_centered * right_centered).sum(dim=0)
	left_norm = torch.sqrt((left_centered * left_centered).sum(dim=0).clamp_min(EPS))
	right_norm = torch.sqrt((right_centered * right_centered).sum(dim=0).clamp_min(EPS))
	return numerator / (left_norm * right_norm).clamp_min(EPS)


def get_layer_similarity(word_vectors, eye_tensor: torch.Tensor):
	word_tensor = torch.as_tensor(word_vectors, dtype=torch.float32, device=eye_tensor.device)
	normalized_words = _normalize_rows_torch(word_tensor)
	feature_dim = max(int(normalized_words.shape[1]), 1)

	# Equivalent to (R - I) @ E where R = normalized_words @ normalized_words.T / feature_dim,
	# but avoids constructing the huge n x n matrix explicitly.
	projected_eye = normalized_words @ (normalized_words.T @ eye_tensor)
	model_matrix = projected_eye / float(feature_dim) - eye_tensor

	similarities_tensor = _columnwise_pearson_torch(model_matrix, eye_tensor)
	average_similarity = float(similarities_tensor.mean().item())
	similarities = [float(x) for x in similarities_tensor.detach().cpu().tolist()]
	return average_similarity, similarities


def eval_eye_tracking(args):
	output_root = str(args.output_dir)
	model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
	revision_name = args.revision_name if args.revision_name is not None else "main"
	result_dir = os.path.join(output_root, model_name, revision_name, "cogbench", "eye_tracking")

	cache_path = os.path.join(result_dir, INFER_CACHE_FILENAME)
	if not os.path.exists(cache_path):
		raise FileNotFoundError(f"Eye-tracking inference cache not found: {cache_path}")

	cache = np.load(cache_path)
	eye_matrix = np.asarray(cache["eye_matrix"], dtype=np.float32)
	max_words = getattr(args, "eye_max_words", None)
	sample_seed = int(getattr(args, "eye_sample_seed", DEFAULT_SAMPLE_SEED))
	sample_indices = _sample_indices(total_words=int(eye_matrix.shape[0]), max_words=max_words, seed=sample_seed)
	if sample_indices is not None:
		eye_matrix = eye_matrix[sample_indices]
		print(f"eye_tracking subsample enabled: total_words={len(cache['eye_matrix'])}, sampled_words={len(sample_indices)}, seed={sample_seed}")

	layer_keys = sorted([key for key in cache.files if key.startswith("layer_")], key=lambda x: int(x.split("_")[1]))

	if eye_matrix.shape[0] == 0 or not layer_keys:
		raise ValueError(f"Invalid eye-tracking cache content: {cache_path}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"eye_tracking regression device: {device}")

	if USE_STANDARDIZATION:
		eye_matrix = standardize_matrix(eye_matrix)
	eye_tensor = torch.as_tensor(eye_matrix, dtype=torch.float32, device=device)

	layer_similarity = []
	layer_feature_similarity = {}

	for layer_key in tqdm(layer_keys, desc="eye_tracking regression", unit="layer"):
		layer_idx = int(layer_key.split("_")[1])
		word_vectors = np.asarray(cache[layer_key], dtype=np.float32)
		if sample_indices is not None:
			word_vectors = word_vectors[sample_indices]

		avg_sim, sims = get_layer_similarity(
			word_vectors=word_vectors,
			eye_tensor=eye_tensor,
		)

		print(f"layer={layer_idx + 1} avg={avg_sim:.6f} sims={sims}")
		layer_similarity.append(float(avg_sim))
		layer_feature_similarity[str(layer_idx)] = sims

	report = {
		"task": "eye_tracking",
		"model_name": model_name,
		"cache_path": cache_path,
		"num_layers": len(layer_keys),
		"eye_max_words": max_words,
		"eye_sample_seed": sample_seed,
		"eye_subsampled": sample_indices is not None,
		"total_content_words": int(eye_matrix.shape[0]),
		"layer_mean_similarity": layer_similarity,
		"layer_feature_similarity": layer_feature_similarity,
		"standardize": USE_STANDARDIZATION,
	}

	os.makedirs(result_dir, exist_ok=True)
	report_path = os.path.join(result_dir, f"cogbench_eye_tracking_{model_name}_report.json")
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	print(f"Saved eye-tracking report: {report_path}")
	return report_path
