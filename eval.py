import os
import json
from datetime import datetime

from main_eval import main_eval
from globals import device
from tqdm import tqdm
from tabulate import tabulate


os.environ["OMP_NUM_THREADS"] = "1"


def full_eval(args, train_dir):
    assert args is not None
    assert train_dir is not None

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    # args.test_or_val = 'val'

    args.results_json = os.path.join(train_dir, 'result.json')

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.save_model_dir)]
    checkpoints = [
        (f, int(s[-3]))
        for (f, s) in checkpoints
        if len(s) >= 4 and f.startswith(args.title) and int(s[-3]) >= args.test_start_from
    ]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0

    # To find best model
    for (f, train_ep) in tqdm(checkpoints, desc="Checkpoints."):

        model = os.path.join(args.save_model_dir, f)
        # args.load_model = model

        # run eval on model
        # args.test_or_val = "val"
        main_eval(args, saved_model=model, outdir=train_dir, device=device)

        # check if best on val.
        with open(args.results_json, "r") as f:
            results = json.load(f)

        if results["success"] > best_performance_on_val:
            best_model_on_val = model
            best_performance_on_val = results["success"]

        print(f"{'=' * 20}\nModel: {model}")
        print(f"val/success: {results['success']}, {train_ep}")
        print(f"val/spl: results['spl'], {train_ep}")

    # Evaluate on the test dataset
    # args.test_or_val = "test"
    args.load_model = best_model_on_val
    main_eval(args, saved_model=model, outdir=train_dir, device=device)

    with open(args.results_json, "r") as f:
        results = json.load(f)

    print(
        tabulate(
            [
                ["SPL >= 1:", results["GreaterThan/1/spl"]],
                ["Success >= 1:", results["GreaterThan/1/success"]],
                ["SPL >= 5:", results["GreaterThan/5/spl"]],
                ["Success >= 5:", results["GreaterThan/5/success"]],
            ],
            headers=["Metric", "Result"],
            tablefmt="orgtbl",
        )
    )

    print("Best model:", args.load_model)


if __name__ == "__main__":
    full_eval()
