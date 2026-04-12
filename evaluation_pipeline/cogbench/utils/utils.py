import os
import torch
from types import SimpleNamespace
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_DEC_BACKENDS = {"enc_dec_mask", "enc_dec_prefix"}


def get_model_name_from_path(model_path_or_name: str) -> str:
	"""Convert a model path to a usable name by joining path components with '--'.

	This preserves the full path structure instead of just using the basename,
	which helps distinguish models with the same folder name but different paths.
	"""
	# Normalize path and split into components
	normalized = os.path.normpath(model_path_or_name)
	# Remove leading/trailing slashes and split
	parts = normalized.strip(os.sep).split(os.sep)
	# Join with '--' to create a flat name
	return "--".join(parts)

def get_model_and_tokenizer(model_path_or_name: str, revision_name: str | None = None, backend: str | None = None):
	if backend in ENC_DEC_BACKENDS:
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_path_or_name,
			trust_remote_code=True,
			revision=revision_name,
		)
	else:
		try:
			model = AutoModel.from_pretrained(
				model_path_or_name,
				trust_remote_code=True,
				revision=revision_name,
			)
		except ValueError as exc:
			# EncoderDecoderConfig is not supported by AutoModel in some HF versions.
			if "EncoderDecoderConfig" not in str(exc):
				raise
			model = AutoModelForSeq2SeqLM.from_pretrained(
				model_path_or_name,
				trust_remote_code=True,
				revision=revision_name,
			)
	tokenizer = AutoTokenizer.from_pretrained(
		model_path_or_name,
		trust_remote_code=True,
		revision=revision_name,
		use_fast=True,
	)
	model = model.to(DEVICE)
	model.eval()

	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
		else:
			# For models without eos_token, add a new pad_token
			tokenizer.add_special_tokens({"pad_token": "[PAD]"})
			# Resize model embeddings to account for new token
			model.resize_token_embeddings(len(tokenizer))

	return model, tokenizer


def forward_for_representations(model, inputs: dict, backend: str | None = None):
	"""Run a model forward pass for hidden-state extraction across architectures.

	For encoder-decoder models, use decoder outputs as the representation source.
	"""
	if backend in ENC_DEC_BACKENDS or getattr(model.config, "is_encoder_decoder", False):
		forward_kwargs = dict(inputs)
		forward_kwargs["decoder_input_ids"] = inputs["input_ids"]
		if "attention_mask" in inputs:
			forward_kwargs["decoder_attention_mask"] = inputs["attention_mask"]

		outputs = model(**forward_kwargs, output_hidden_states=True, return_dict=True)

		decoder_hidden_states = getattr(outputs, "decoder_hidden_states", None)
		decoder_last_hidden_state = getattr(outputs, "decoder_last_hidden_state", None)
		if decoder_hidden_states is None:
			decoder_hidden_states = getattr(outputs, "hidden_states", None)
		if decoder_last_hidden_state is None and decoder_hidden_states is not None:
			decoder_last_hidden_state = decoder_hidden_states[-1]

		return SimpleNamespace(
			hidden_states=decoder_hidden_states,
			last_hidden_state=decoder_last_hidden_state,
		)

	return model(**inputs, output_hidden_states=True, return_dict=True)