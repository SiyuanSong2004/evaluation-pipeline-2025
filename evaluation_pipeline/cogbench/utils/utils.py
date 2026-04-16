import torch
from types import SimpleNamespace
import inspect
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENC_DEC_BACKENDS = {"enc_dec_mask", "enc_dec_prefix"}


def _filter_forward_inputs(model, inputs: dict):
	"""Keep only keyword arguments accepted by model.forward.

	Some architectures (e.g., Mamba) do not accept token_type_ids.
	"""
	forward_sig = inspect.signature(model.forward)
	accepted_keys = set(forward_sig.parameters.keys())
	accepts_var_kwargs = any(
		param.kind == inspect.Parameter.VAR_KEYWORD
		for param in forward_sig.parameters.values()
	)
	if accepts_var_kwargs:
		return dict(inputs)
	return {key: value for key, value in inputs.items() if key in accepted_keys}

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

	For encoder-decoder models, use decoder outputs as the representation source.
	"""
	if backend in ENC_DEC_BACKENDS or getattr(model.config, "is_encoder_decoder", False):
		forward_kwargs = _filter_forward_inputs(model, inputs)
		forward_kwargs["decoder_input_ids"] = inputs["input_ids"]
		if "attention_mask" in inputs:
			forward_kwargs["decoder_attention_mask"] = inputs["attention_mask"]
		forward_kwargs = _filter_forward_inputs(model, forward_kwargs)

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

	forward_kwargs = _filter_forward_inputs(model, inputs)
	return model(**forward_kwargs, output_hidden_states=True, return_dict=True)