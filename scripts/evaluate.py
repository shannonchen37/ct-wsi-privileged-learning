import argparse

import numpy as np
import torch

from src.models import CTModel, WSIModel
from src.training import classification_metrics, predict
from src.utils import load_config, save_json

from .common import checkpoint_path, experiment_context, imaging_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("ct_only", "ct_anchors", "wsi_teacher",
                                               "ct_self_training", "privileged_wsi"), required=True)
    parser.add_argument("--budget", type=int, choices=(25, 50, 100), required=True)
    args = parser.parse_args(); config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics, pooled_labels, pooled_probabilities = [], [], []
    for fold in range(5):
        context = experiment_context(config, fold, args.budget)
        sample_ids, labels, _, valid_indices, _, anchors, statistics, ct_store = context
        modality = "wsi" if args.method == "wsi_teacher" else "ct"
        model = WSIModel(True) if modality == "wsi" else CTModel(args.method != "ct_only")
        state = torch.load(checkpoint_path(config, args.method, args.budget, fold),
                           map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"]); model = model.to(device)
        valid_loader = imaging_loader(valid_indices, sample_ids, labels, modality, config, ct_store)
        output = predict(model, valid_loader, device); target = labels[output["indices"]]
        metric = classification_metrics(target, output["probability_lusc"]); metric["fold"] = fold
        fold_metrics.append(metric); pooled_labels.extend(target.tolist())
        pooled_probabilities.extend(output["probability_lusc"].tolist())
    pooled = classification_metrics(pooled_labels, pooled_probabilities)
    names = ("accuracy", "auc", "sensitivity", "specificity", "f1")
    mean_std = {name: {"mean": float(np.mean([x[name] for x in fold_metrics])),
                       "population_std": float(np.std([x[name] for x in fold_metrics], ddof=0))}
                for name in names}
    result = {"method": args.method, "diagnosis_label_budget": args.budget,
              "fold_metrics": fold_metrics, "mean_population_std": mean_std,
              "pooled_oof": pooled, "oof_count": len(pooled_labels)}
    save_json(config["output_dir"] + f"/metrics/{args.method}_budget_{args.budget}.json", result)
    print(result)


if __name__ == "__main__": main()

