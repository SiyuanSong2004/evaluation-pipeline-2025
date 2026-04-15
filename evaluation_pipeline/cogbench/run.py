import argparse
import glob
import json
import os
import pathlib

import numpy as np
import scipy.io as sio

from .infer import infer
from .eval import eval

BACKEND_CHOICES = ["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"]


def _model_name(args: argparse.Namespace) -> str:
    return os.path.basename(os.path.normpath(str(args.model_path_or_name)))


def _revision_name(args: argparse.Namespace) -> str:
    return args.revision_name if args.revision_name is not None else "main"


def _task_output_dir(args: argparse.Namespace) -> pathlib.Path:
    return pathlib.Path(args.output_dir) / _model_name(args) / _revision_name(args) / "cogbench" / args.task

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", required=True, type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--task", required=True, type=str, help="The task that is being evaluated.", choices=["word_fmri", "fmri", "meg", "eye_tracking"])
    parser.add_argument("--model_path_or_name", required=True, type=str, help="Path to the model to evaluate.")
    parser.add_argument(
        "--backend",
        default="causal",
        type=str,
        help="Model architecture backend label (kept consistent with zero-shot entry).",
        choices=BACKEND_CHOICES,
    )
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")

    parser.add_argument("--save_predictions", default=False, action="store_true", help="Whether or not to save predictions.")
    parser.add_argument("--fast", default=False, action="store_true", help="Enable fast evaluation mode.")
    parser.add_argument(
        "--eye_max_words",
        default=None,
        type=int,
        help="Optional cap for eye-tracking evaluation words to avoid O(n^2) blow-up.",
    )
    parser.add_argument(
        "--eye_sample_seed",
        default=42,
        type=int,
        help="Random seed used when eye-tracking word subsampling is enabled.",
    )

    return parser.parse_args()


def create_evaluation_report(args: argparse.ArgumentParser):
    model_name = _model_name(args)
    task_dir = _task_output_dir(args)
    task_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    if args.task == "word_fmri":
        pattern = str(task_dir / "*_score.mat")
        if args.fast:
            pattern = str(task_dir / "*_sanity_score.mat")

        for file_path in sorted(glob.glob(pattern)):
            mat = sio.loadmat(file_path)
            if "score" not in mat:
                continue
            score = float(np.asarray(mat["score"]).squeeze())
            metrics.append({"file": file_path, "value": score})

    elif args.task == "fmri":
        patterns = [
            str(task_dir / "*" / "*_average.mat"),
        ]
        fmri_files = []
        for pattern in patterns:
            fmri_files.extend(glob.glob(pattern))

        for file_path in sorted(set(fmri_files)):
            mat = sio.loadmat(file_path)
            if "test_corrs" not in mat:
                continue
            score = float(np.nanmean(np.asarray(mat["test_corrs"], dtype=float)))
            metrics.append({"file": file_path, "value": score})

    elif args.task == "meg":
        pattern = str(task_dir / "*_rsa_*.mat")
        for file_path in sorted(glob.glob(pattern)):
            mat = sio.loadmat(file_path)
            if "sess_avg" not in mat:
                continue

            sess_avg = mat["sess_avg"]
            score = None
            if sess_avg.dtype.names:
                row = sess_avg[0, 0]
                values = []
                for field in row.dtype.names:
                    values.append(np.asarray(row[field], dtype=float))
                if values:
                    score = float(np.nanmean(np.concatenate([v.reshape(-1) for v in values])))
            else:
                score = float(np.nanmean(np.asarray(sess_avg, dtype=float)))

            if score is not None:
                metrics.append({"file": file_path, "value": score})
    elif args.task == "eye_tracking":
        eye_report_path = str(task_dir / f"cogbench_eye_tracking_{model_name}_report.json")
        if os.path.exists(eye_report_path):
            with open(eye_report_path, "r", encoding="utf-8") as f:
                eye_report = json.load(f)
            for layer_idx, score in enumerate(eye_report.get("layer_mean_similarity", [])):
                metrics.append({"file": eye_report_path, "layer": layer_idx, "value": float(score)})

    values = [item["value"] for item in metrics]
    summary = {
        "task": args.task,
        "model_name": model_name,
        "output_root": str(task_dir),
        "fast": bool(args.fast),
        "n_result_files": len(metrics),
        "mean": float(np.nanmean(values)) if values else None,
        "min": float(np.nanmin(values)) if values else None,
        "max": float(np.nanmax(values)) if values else None,
        "files": metrics,
    }

    report_path = task_dir / f"cogbench_{args.task}_{model_name}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation report: {report_path}")


def main():
    args = _parse_arguments()
    infer(args)
    eval(args)
    create_evaluation_report(args)

if __name__ == "__main__":
    main()
    


