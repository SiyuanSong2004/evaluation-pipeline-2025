"""
Chinese BabyLM Evaluation Pipeline
===================================
Usage:
    python pipeline.py download [--eval_dir evaluation_data]
    python pipeline.py eval [--config config.yaml] [--results_dir results]
"""

import argparse
import json
import pathlib
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning task specs
# ─────────────────────────────────────────────────────────────────────────────

FINETUNE_SPECS = {
    "afqmc":       {"num_labels": 2,  "metrics": ["accuracy", "f1", "mcc"]},
    "ocnli":       {"num_labels": 3,  "metrics": ["accuracy"]},
    "tnews":       {"num_labels": 15, "metrics": ["accuracy"]},
    "cluewsc2020": {"num_labels": 2,  "metrics": ["accuracy", "f1", "mcc"], "use_wsc_epochs": True},
}

# Zero-shot task → data directory name
ZERO_SHOT_DATA_DIRS = {
    "zhoblimp":       "zhoblimp",
    "hanzi_structure": "hanzi_structure",
    "hanzi_pinyin":   "hanzi_pinyin",
}


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: download
# ─────────────────────────────────────────────────────────────────────────────

def cmd_download(args):
    from prepare_chinese_data import (
        prepare_zhoblimp,
        prepare_hanzi_structure,
        prepare_hanzi_pinyin,
        prepare_cogbench,
        prepare_afqmc,
        prepare_ocnli,
        prepare_tnews,
        prepare_cluewsc2020,
    )

    output_dir = pathlib.Path(args.eval_dir)
    print(f"Downloading / preparing evaluation data into: {output_dir}\n")

    prepare_functions = [
        ("zhoblimp",        prepare_zhoblimp),
        ("hanzi_structure", prepare_hanzi_structure),
        ("hanzi_pinyin",    prepare_hanzi_pinyin),
        ("cogbench",        prepare_cogbench),
        ("afqmc",           prepare_afqmc),
        ("ocnli",           prepare_ocnli),
        ("tnews",           prepare_tnews),
        ("cluewsc2020",     prepare_cluewsc2020),
    ]

    for name, fn in prepare_functions:
        print(f"  Preparing {name} ...")
        fn(output_dir=output_dir)

    print("\nAll datasets prepared.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build subprocess commands
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, task_label):
    print(f"\n=== Evaluating {task_label} ===")
    print("Command:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"WARNING: command for '{task_label}' exited with "
            f"returncode {result.returncode}",
            file=sys.stderr,
        )


def _build_zero_shot_cmd(model_path, backend, task, eval_dir, results_dir):
    data_dir = ZERO_SHOT_DATA_DIRS[task]
    return [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", model_path,
        "--backend", backend,
        "--task", task,
        "--data_path", str(pathlib.Path(eval_dir) / "full_eval" / data_dir),
        "--output_dir", results_dir,
        "--save_predictions",
    ]


def _build_cogbench_cmd(model_path, backend, task, eval_dir, results_dir):
    return [
        sys.executable, "-m", "evaluation_pipeline.cogbench.run",
        "--model_path_or_name", model_path,
        "--backend", backend,
        "--task", task,
        "--data_path", str(pathlib.Path(eval_dir) / "cogbench-fmri-0415"),
        "--output_dir", results_dir,
        "--save_predictions",
    ]


def _build_finetune_cmd(model_path, task, eval_dir, results_dir, hparams):
    spec = FINETUNE_SPECS[task]
    clue_dir = pathlib.Path(eval_dir) / "full_eval" / "clue"
    epochs = hparams["wsc_epochs"] if spec.get("use_wsc_epochs") else hparams["max_epochs"]

    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", model_path,
        "--train_data",   str(clue_dir / f"{task}.train.jsonl"),
        "--valid_data",   str(clue_dir / f"{task}.valid.jsonl"),
        "--predict_data", str(clue_dir / f"{task}.valid.jsonl"),
        "--task", task,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(hparams["batch_size"]),
        "--learning_rate", str(hparams["lr"]),
        "--num_epochs", str(epochs),
        "--sequence_length", "512",
        "--results_dir", results_dir,
        "--metrics", *spec["metrics"],
        "--metric_for_valid", "accuracy",
        "--seed", str(hparams["seed"]),
    ]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: collect results
# ─────────────────────────────────────────────────────────────────────────────

def _collect_zero_shot(results_dir, model_stem, backend, task):
    report = (
        pathlib.Path(results_dir)
        / model_stem / "main" / "zero_shot" / backend / task / task
        / "best_temperature_report.txt"
    )
    if not report.exists():
        return None
    lines = report.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "### AVERAGE ACCURACY" in line or "### AVERAGE SPEARMAN'S RHO" in line:
            # Value is on the next non-empty line
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate:
                    try:
                        return float(candidate)
                    except ValueError:
                        return None
    return None


def _collect_finetune(results_dir, model_stem, task):
    result_file = (
        pathlib.Path(results_dir)
        / model_stem / "main" / "finetune" / task / "results.txt"
    )
    if not result_file.exists():
        return None
    for line in result_file.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            if key.strip() == "accuracy":
                try:
                    score = float(val.strip())
                    return score * 100 if score <= 1.0 else score
                except ValueError:
                    return None
    return None


def _collect_cogbench(results_dir, model_stem, task):
    report = (
        pathlib.Path(results_dir)
        / model_stem / "results"
        / f"cogbench_{task}_{model_stem}_report.json"
    )
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        return float(data["mean"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_tasks, scores):
    """
    scores: dict[model_path] -> dict[task] -> float | None
    """
    model_paths = list(scores.keys())
    model_stems = [pathlib.Path(p).name for p in model_paths]

    col_width = 16
    stem_width = max(24, *(len(s) for s in model_stems), len("Model"))

    header_cells = [f"{'Model':<{stem_width}}"] + [
        f"{t:>{col_width}}" for t in all_tasks
    ]
    header = " " + " ".join(header_cells)
    divider = "=" * len(header)

    print()
    print(divider)
    print(header)
    print(divider)

    for model_path in model_paths:
        stem = pathlib.Path(model_path).name
        row_cells = [f"{stem:<{stem_width}}"]
        for task in all_tasks:
            val = scores[model_path].get(task)
            if val is None:
                cell = "-"
            else:
                cell = f"{val:.1f}"
            row_cells.append(f"{cell:>{col_width}}")
        print(" " + " ".join(row_cells))

    print(divider)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: eval
# ─────────────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    models      = cfg.get("models", [])
    tasks_cfg   = cfg.get("tasks", {})
    eval_dir    = cfg.get("eval_dir", "evaluation_data")
    results_dir = args.results_dir if args.results_dir else cfg.get("results_dir", "results")

    hparams = cfg.get("finetune_hparams", {})
    hparams.setdefault("lr",         3e-5)
    hparams.setdefault("batch_size", 32)
    hparams.setdefault("max_epochs", 10)
    hparams.setdefault("wsc_epochs", 30)
    hparams.setdefault("seed",       42)

    zero_shot_tasks = tasks_cfg.get("zero_shot", [])
    cogbench_tasks  = tasks_cfg.get("cogbench",  [])
    finetune_tasks  = tasks_cfg.get("finetune",  [])
    all_tasks       = zero_shot_tasks + cogbench_tasks + finetune_tasks

    # ── Run evaluations ──────────────────────────────────────────────────────
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name

        for task in zero_shot_tasks:
            cmd = _build_zero_shot_cmd(model_path, backend, task, eval_dir, results_dir)
            _run(cmd, f"{stem} on {task}")

        for task in cogbench_tasks:
            cmd = _build_cogbench_cmd(model_path, backend, task, eval_dir, results_dir)
            _run(cmd, f"{stem} on {task}")

        for task in finetune_tasks:
            cmd = _build_finetune_cmd(model_path, task, eval_dir, results_dir, hparams)
            _run(cmd, f"{stem} on {task}")

    # ── Collect results ──────────────────────────────────────────────────────
    print("\nCollecting results ...")
    scores = {}
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name
        scores[model_path] = {}

        for task in zero_shot_tasks:
            scores[model_path][task] = _collect_zero_shot(results_dir, stem, backend, task)

        for task in cogbench_tasks:
            scores[model_path][task] = _collect_cogbench(results_dir, stem, task)

        for task in finetune_tasks:
            scores[model_path][task] = _collect_finetune(results_dir, stem, task)

    _print_summary(all_tasks, scores)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chinese BabyLM Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── download ─────────────────────────────────────────────────────────────
    dl_parser = subparsers.add_parser(
        "download",
        help="Prepare / download all evaluation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dl_parser.add_argument(
        "--eval_dir",
        default="evaluation_data",
        help="Output directory for prepared evaluation data",
    )
    dl_parser.set_defaults(func=cmd_download)

    # ── eval ─────────────────────────────────────────────────────────────────
    ev_parser = subparsers.add_parser(
        "eval",
        help="Run evaluations according to a YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ev_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file",
    )
    ev_parser.add_argument(
        "--results_dir",
        default=None,
        help="Override the results directory from the config",
    )
    ev_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
