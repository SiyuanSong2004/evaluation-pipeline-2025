"""
pipeline_util.py – helper functions for the ``detect`` subcommand.

Public API
----------
compute_token_lengths   – tokenise training data, report percentile lengths
find_max_batch_sizes    – binary-search the largest batch size per sequence length
"""

import json
import math
import sys

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal: read texts from CLUE-style JSONL
# ─────────────────────────────────────────────────────────────────────────────

def _read_texts_from_jsonl(path):
    """Yield ``(text,)`` or ``(text1, text2)`` tuples from a CLUE JSONL file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # sentence pair tasks (afqmc, ocnli)
            if "sentence1" in obj:
                s2 = obj.get("sentence2")
                yield (obj["sentence1"], s2) if s2 else (obj["sentence1"],)
            # single sentence tasks (tnews)
            elif "sentence" in obj:
                yield (obj["sentence"],)
            # text field (cluewsc2020)
            elif "text" in obj:
                yield (obj["text"],)
            # fallback: concatenate all string values
            else:
                yield (" ".join(v for v in obj.values() if isinstance(v, str)),)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_lengths(model_path, train_data_path,
                          percentiles=(95, 99, 100), round_to=8):
    """
    Tokenise every example in *train_data_path* with the tokeniser from
    *model_path*.

    Returns ``{percentile: rounded_token_count}``.  Lengths are rounded up
    to the nearest multiple of *round_to*.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )

    lengths = []
    for texts in _read_texts_from_jsonl(train_data_path):
        if len(texts) == 2:
            ids = tokenizer.encode(texts[0], texts[1],
                                   add_special_tokens=True)
        else:
            ids = tokenizer.encode(texts[0], add_special_tokens=True)
        lengths.append(len(ids))

    if not lengths:
        raise ValueError(f"No examples found in {train_data_path}")

    result = {}
    for p in percentiles:
        raw = float(np.percentile(lengths, p))
        rounded = int(math.ceil(raw / round_to) * round_to)
        result[p] = max(rounded, round_to)
    return result

def find_max_batch_sizes(model_path, backend, sequence_lengths, num_labels,
                         max_batch_size=128):
    """
    For each value in *sequence_lengths*, binary-search for the largest
    power-of-2 batch size (≤ *max_batch_size*) that survives a forward +
    backward pass on the current GPU.

    The model is loaded **once** and reused for every probe.  Longer
    sequences are probed first so that the heavier cases run while GPU
    memory is cleanest.

    Returns ``{sequence_length: batch_size}``.
    """
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — returning default "
              "batch_size=8", file=sys.stderr)
        return {sl: 8 for sl in sequence_lengths}

    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = torch.device("cuda")

    # Load tokenizer to resolve pad_token_id ──────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(
        model_path, num_labels=num_labels, trust_remote_code=True,
    )
    cfg.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=cfg, trust_remote_code=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    # Candidate batch sizes: powers of 2, largest first
    candidates = []
    bs = 1
    while bs <= max_batch_size:
        candidates.append(bs)
        bs *= 2
    candidates.reverse()

    results = {}
    for seq_len in sorted(sequence_lengths, reverse=True):
        best = None
        for bs in candidates:
            torch.cuda.empty_cache()
            try:
                dummy_ids = torch.ones(
                    bs, seq_len, dtype=torch.long, device=device,
                )
                dummy_labels = torch.zeros(
                    bs, dtype=torch.long, device=device,
                )
                # Use pad_token_id for the dummy input so attention masking
                # behaves normally; the actual token values don't matter for
                # a memory probe but this avoids edge-case complaints.
                attention_mask = torch.ones(
                    bs, seq_len, dtype=torch.long, device=device,
                )
                outputs = model(
                    input_ids=dummy_ids,
                    attention_mask=attention_mask,
                    labels=dummy_labels,
                )
                outputs.loss.backward()
                model.zero_grad(set_to_none=True)
                del dummy_ids, dummy_labels, attention_mask, outputs
                torch.cuda.empty_cache()
                best = bs
                break
            except RuntimeError as e:
                # Catch CUDA OOM (covers both torch.cuda.OutOfMemoryError
                # and older RuntimeError variants)
                if "out of memory" not in str(e).lower():
                    raise
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        if best is None:
            print(f"WARNING: batch_size=1 caused OOM at seq_len={seq_len} "
                  f"— returning 1", file=sys.stderr)
            best = 1
        results[seq_len] = best

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results
