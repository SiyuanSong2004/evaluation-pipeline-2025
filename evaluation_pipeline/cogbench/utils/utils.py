import torch
from transformers import AutoModel, AutoTokenizer 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_and_tokenizer(model_path_or_name: str, revision_name: str | None = None):
	model = AutoModel.from_pretrained(
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