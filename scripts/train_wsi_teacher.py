import argparse

from src.models import WSIModel
from src.training import train_multitask
from src.utils import device_from_environment, load_config, save_json, seed_everything

from .common import checkpoint_path, experiment_context, imaging_loader, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--budget", type=int, choices=(25, 50, 100), required=True)
    args = parser.parse_args(); config = load_config(args.config)
    seed = config.get("seed", 42) * 100 + args.budget + args.fold; seed_everything(seed)
    context = experiment_context(config, args.fold, args.budget)
    sample_ids, labels, train_indices, _, labeled_indices, anchors, statistics, ct_store = context
    device = device_from_environment(); model = WSIModel(with_anchors=True).to(device)
    anchor_loader = imaging_loader(train_indices, sample_ids, labels, "wsi", config, ct_store,
                                   anchors, statistics, shuffle=True, seed=seed)
    labeled_loader = imaging_loader(labeled_indices, sample_ids, labels, "wsi", config, ct_store,
                                    include_labels=True, shuffle=True, seed=seed + 17)
    history = train_multitask(model, anchor_loader, labeled_loader, statistics, device,
                              config["wsi"]["epochs"], config["wsi"]["learning_rate"],
                              config["optimizer"]["momentum"], config["optimizer"]["weight_decay"],
                              config["anchor_weight"])
    path = checkpoint_path(config, "wsi_teacher", args.budget, args.fold)
    save_checkpoint(path, model, {"fold": args.fold, "budget": args.budget,
                                  "diagnosis_labeled_count": len(labeled_indices),
                                  "deployment_model": False})
    save_json(path.with_suffix(".history.json"), history)


if __name__ == "__main__": main()

