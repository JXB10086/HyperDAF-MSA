"""Single-seed wrapper around the released LNLN robust evaluation path."""
import argparse
import json
import os

import torch
import yaml

from core.dataset import MMDataEvaluationLoader
from core.metric import MetricsTop
from models.lnln import build_model


def evaluate(model, eval_loader, metrics, device):
    y_pred, y_true = [], []
    model.eval()
    for data in eval_loader:
        incomplete_input = (
            data["vision_m"].to(device),
            data["audio_m"].to(device),
            data["text_m"].to(device),
        )
        labels = data["labels"]["M"].to(device)
        with torch.no_grad():
            output = model((None, None, None), incomplete_input)
        y_pred.append(output["sentiment_preds"].cpu())
        y_true.append(labels.cpu())
    return metrics(torch.cat(y_pred), torch.cat(y_true))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config_file) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["dataset"]["dataPath"] = args.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = MetricsTop(train_mode=config["base"]["train_mode"]).getMetics(
        config["dataset"]["datasetName"]
    )

    rows = []
    for rate in [i / 10 for i in range(10)]:
        config["base"]["missing_rate_eval_test"] = rate
        loader = MMDataEvaluationLoader(config)
        result = evaluate(model, loader, metrics, device)
        row = {"missing_rate": rate, **{key: float(value) for key, value in result.items()}}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    with open(args.output, "w") as handle:
        json.dump(
            {
                "protocol": "LNLN native token/frame-level erase",
                "seed": args.seed,
                "checkpoint": os.path.abspath(args.checkpoint),
                "rows": rows,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
