import argparse
import pathlib
import torch
from .infer import infer
from .eval import eval
import json
from io import TextIOWrapper

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", required=True, type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--task", required=True, type=str, help="The task that is being evaluated.", choices=["word_fmri", "fmri", "meg"])
    parser.add_argument("--model_path_or_name", required=True, type=str, help="Path to the model to evaluate.")
    parser.add_argument("--backend", required=True, type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"])
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")

    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--save_predictions", default=False, action="store_true", help="Whether or not to save predictions.")
    parser.add_argument("--fast", default=False, action="store_true", help="Enable fast evaluation mode.")

    return parser.parse_args()


def process_results(args: argparse.ArgumentParser, results: dict):
    """This function computes accuracy metrics and, if necessary, other dataset-specific metrics
    given dataset sizes and numbers of correct predictions

    Args:
        args (argparse.ArgumentParser): ArgumentParser object used to determine task
        results (dict): Results obtained from running compute_results
    """
    # Compute accuracies
    accuracies = {temp : {} for temp in results}
    for temp, temp_results in results.items():
        for subdomain, count_dict in temp_results.items():
            keys = count_dict["total"].keys()
            subdomain_accs = {key : 100.0 * count_dict["correct"][key] / count_dict["total"][key] for key in keys}
            accuracies[temp][subdomain] = subdomain_accs

    # Average accuracies
    average_accuracies = {}
    if args.task != "entity_tracking":
        for temp, accuracy in accuracies.items():
            average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())
    else:
        splits = ["regular", "ambiref", "move_contents"]
        for temp, subdomain_dict in accuracies.items():
            split_accs = []
            split_dict = subdomain_dict["UID"]
            for split in splits:
                split_keys = [key for key in split_dict if key.startswith(split)]
                if not split_keys:
                    continue
                curr_acc = sum([split_dict[key] for key in split_keys]) / len(split_keys)

                split_dict[split] = curr_acc
                split_accs.append(curr_acc)
            average_accuracies[temp] = sum(split_accs) / len(split_accs)

    return accuracies, average_accuracies


def create_evaluation_report(temperature: float, avg_accuracy: torch.Tensor, accuracies: dict[str, list[dict[str, float]]], task: str | None = None, file: TextIOWrapper | None = None) -> None:
    """This function creates a report and either saves it to a file or prints it to the terminal.

    Args:
        temperature(float): The temperature at which the model is evaluated.
        temperature_pos(int): The position of the evaluated temperature.
        avg_accuracy(torch.Tensor): The average accuracy of the model at the given temperature.
        avg_accuracy(dict[str, list[dict[str, float]]]): The finegrained accuracies of the model
            at the given temperature.
        file(TextIOWrapper | None): The file to write to results to. (If None, it will printed
            printed to the terminal)
    """
    metric = "ACCURACY" if "wug" not in task else "SPEARMAN'S RHO"
    print(f"TEMPERATURE: {temperature:.2f}", file=file)
    print(file=file)

    for domain, accuracy in accuracies.items():
        print(f"### {domain.upper()} {metric}", file=file)
        for subdomain, acc in accuracy.items():
            print(f"{subdomain}: {acc:.2f}", file=file)
        print(file=file)

    print(f"### AVERAGE {metric}", file=file)
    print(f"{avg_accuracy:.2f}", file=file)
    print(file=file)


def save_predictions(args, predictions, best_temp):
    with (args.output_path / "predictions.json").open("w") as f:
        json.dump(predictions[best_temp], f)


def main():
    args = _parse_arguments()
    infer(args)
    eval(args)

if __name__ == "__main__":
    main()
    


