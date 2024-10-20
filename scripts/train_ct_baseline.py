import argparse

from src.models import CTModel
from src.training import train_ct_only, train_multitask
from src.utils import device_from_environment, load_config, save_json, seed_everything

from .common import checkpoint_path, experiment_context, imaging_loader, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--budget", type=int, choices=(25, 50, 100), required=True)
    parser.add_argument("--with-anchors", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    seed = config.get("seed", 42) * 100 + args.budget + args.fold; seed_everything(seed)
    context = experiment_context(config, args.fold, args.budget)
    sample_ids, labels, train_indices, _, labeled_indices, anchors, statistics, ct_store = context
    device = device_from_environment(); model = CTModel(with_anchors=args.with_anchors).to(device)
    labeled_loader = imaging_loader(labeled_indices, sample_ids, labels, "ct", config, ct_store,
                                    include_labels=True, shuffle=True, seed=seed + 17)
    if args.with_anchors:
        anchor_loader = imaging_loader(train_indices, sample_ids, labels, "ct", config, ct_store,
                                       anchors, statistics, shuffle=True, seed=seed)
        history = train_multitask(model, anchor_loader, labeled_loader, statistics, device,
                                  config["ct"]["epochs"], config["ct"]["learning_rate"],
                                  config["optimizer"]["momentum"], config["optimizer"]["weight_decay"],
                                  config["anchor_weight"])
        method = "ct_anchors"
    else:
        epochs = int(config["ct"]["epochs"] * 100 / args.budget)
        history = train_ct_only(model, labeled_loader, device, epochs, config["ct"]["learning_rate"],
                                config["optimizer"]["momentum"], config["optimizer"]["weight_decay"])
        method = "ct_only"
    path = checkpoint_path(config, method, args.budget, args.fold)
    save_checkpoint(path, model, {"fold": args.fold, "budget": args.budget,
                                  "diagnosis_labeled_count": len(labeled_indices),
                                  "clinical_is_input": False})
    save_json(path.with_suffix(".history.json"), history)


if __name__ == "__main__": main()

