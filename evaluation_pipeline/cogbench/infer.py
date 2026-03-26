from .inference.infer_sentence import infer_sentence
from .inference.infer_word import infer_word


def infer(args):
    """
    forward inference to get the features
    """
    task = args.task
    model_path_or_name = args.model_path_or_name
    datapath = args.data_path

    match task:
        case "word_fmri":
            return infer_word(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
            )
        case "fmri" | "meg":
            return infer_sentence(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                revision_name=args.revision_name,
            )
        case _:
            raise ValueError(f"Unsupported task: {task}")
