import os
import pathlib

from .inference.infer_sentence import infer_sentence
from .inference.infer_word import infer_word
from .inference.infer_eye_tracking import infer_eye_tracking

def infer(args):
    """
    forward inference to get the features
    """
    task = args.task
    backend = args.backend
    model_path_or_name = args.model_path_or_name
    datapath = args.data_path
    output_root = pathlib.Path(args.output_dir)
    model_name = os.path.basename(os.path.normpath(model_path_or_name))
    revision_name = args.revision_name if args.revision_name is not None else "main"
    task_output_dir = output_root / model_name / revision_name / "cogbench" / task
    task_output_dir_str = str(task_output_dir)

    match task:
        case "word_fmri":
            return infer_word(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_root=task_output_dir_str,
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
                backend=backend,
            )
        case "fmri" | "meg":
            return infer_sentence(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_dir=task_output_dir_str,
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
                backend=backend,
            )
        case "eye_tracking":
            return infer_eye_tracking(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_dir=task_output_dir_str,
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
                fast=args.fast,
                backend=backend,
            )
        case _:
            raise ValueError(f"Unsupported task: {task}")
