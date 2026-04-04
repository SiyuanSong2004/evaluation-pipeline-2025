import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_DEC_BACKENDS = {"enc_dec_mask", "enc_dec_prefix"}

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

	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token

	return model, tokenizer


def forward_for_representations(model, inputs: dict, backend: str | None = None):
	"""Run a model forward pass for hidden-state extraction across architectures.

	For encoder-decoder models, use encoder outputs to avoid requiring decoder inputs.
	"""
	if backend in ENC_DEC_BACKENDS or getattr(model.config, "is_encoder_decoder", False):
		encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
		return encoder(**inputs, output_hidden_states=True, return_dict=True)

	return model(**inputs, output_hidden_states=True, return_dict=True)