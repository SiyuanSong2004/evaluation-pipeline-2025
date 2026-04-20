from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from transformers.modeling_outputs import ModelOutput

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


def get_logits(outputs: Any) -> torch.Tensor:
    """This helper function, checks the type passed outputs,
    and extracts the logits from them.

    Args:
        outputs(Any): The outputs of a HuggingFace model.

    Returns:
        torch.Tensor: The logits of the model.
    """
    if type(outputs) is tuple:
        encoding: torch.Tensor = outputs[0]
    elif isinstance(outputs, ModelOutput):
        if hasattr(outputs, "logits"):
            encoding = outputs.logits
        elif hasattr(outputs, "last_hidden_state"):
            encoding = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            encoding = outputs.hidden_states[-1]
        else:
            print("Unknown name for output of the model!")
            exit()
    else:
        print(f"Add support for output type: {type(outputs)}!")
        exit()

    return encoding


def sigmoid_function(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """Sigmoid function for fitting learning curves: f(x) = a / (1 + exp(-b*(x-c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d


# class AoAEvaluator:
#     """Evaluates Age of Acquisition based on Chang & Bergen 2022 methodology.
#     Not used by the Chinese evaluation pipeline.
#
#     def __init__(self, cdi_data_path: Path):
#         self.cdi_data = pd.read_csv(cdi_data_path)
#         self.prepare_cdi_data()
#
#     def prepare_cdi_data(self) -> None: ...
#     def compute_child_aoa(self, word_idx, threshold=0.5) -> float | None: ...
#     def compute_model_aoa(self, surprisal_data, training_steps, ...) -> float | None: ...
#     def extract_step_number(self, step_name) -> float | None: ...
#     def compute_curve_fitness(self, model_results, target_words=None) -> dict: ...
#     """
