import argparse
import shutil

import numpy as np
import torch

from src.models import CTModel, WSIModel
from src.training import (calibrated_probability, choose_recipe, clinical_reliability,
                          predict, select_pseudo, train_multitask, train_student)
from src.utils import device_from_environment, inner_partition, load_config, save_json, seed_everything

from .common import checkpoint_path, experiment_context, imaging_loader, save_checkpoint


def load_model(path, model, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--budget", type=int, choices=(25, 50, 100), required=True)
    parser.add_argument("--teacher", choices=("wsi", "ct"), default="wsi")
    args = parser.parse_args(); config = load_config(args.config)
    seed = config.get("seed", 42) * 100 + args.budget + args.fold; seed_everything(seed)
    context = experiment_context(config, args.fold, args.budget)
    sample_ids, labels, train_indices, _, labeled_indices, anchors, statistics, ct_store = context
    method = "privileged_wsi" if args.teacher == "wsi" else "ct_self_training"
    base_path = checkpoint_path(config, "ct_anchors", args.budget, args.fold)
    destination = checkpoint_path(config, method, args.budget, args.fold)
    unlabeled_indices = sorted(set(train_indices) - set(labeled_indices))
    if not unlabeled_indices:
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(base_path, destination)
        save_json(destination.with_suffix(".recipe.json"), {"equivalent_to": "ct_anchors", "reason": "no unlabeled samples"})
        return

    device = device_from_environment(); inner_train, inner_valid = inner_partition(labeled_indices, labels)
    modality = args.teacher
    inner_model = WSIModel(True) if modality == "wsi" else CTModel(True)
    inner_model = inner_model.to(device)
    anchor_loader = imaging_loader(train_indices, sample_ids, labels, modality, config, ct_store,
                                   anchors, statistics, shuffle=True, seed=seed + 5000)
    inner_labeled_loader = imaging_loader(inner_train, sample_ids, labels, modality, config, ct_store,
                                          include_labels=True, shuffle=True, seed=seed + 5017)
    section = config[modality]
    train_multitask(inner_model, anchor_loader, inner_labeled_loader, statistics, device,
                    section["epochs"], section["learning_rate"], config["optimizer"]["momentum"],
                    config["optimizer"]["weight_decay"], config["anchor_weight"])
    inner_loader = imaging_loader(inner_valid, sample_ids, labels, modality, config, ct_store,
                                  anchors, statistics)
    heldout = predict(inner_model, inner_loader, device)
    from src.training import fit_temperature
    temperature, _ = fit_temperature(heldout["logits"], labels[heldout["indices"]])
    heldout_probability = calibrated_probability(heldout["logits"], temperature)
    heldout_reliability = clinical_reliability(heldout["indices"], heldout["anchors"],
                                               sample_ids, anchors, statistics)
    recipe = choose_recipe(heldout_probability, labels[heldout["indices"]], heldout_reliability)
    recipe["temperature"] = float(temperature)

    teacher_path = checkpoint_path(config, "wsi_teacher" if modality == "wsi" else "ct_anchors",
                                   args.budget, args.fold)
    teacher_model = WSIModel(True) if modality == "wsi" else CTModel(True)
    teacher_model = load_model(teacher_path, teacher_model, device)
    unlabeled_loader = imaging_loader(unlabeled_indices, sample_ids, labels, modality, config, ct_store,
                                      anchors, statistics)
    teacher_prediction = predict(teacher_model, unlabeled_loader, device)
    teacher_prediction["probability_lusc"] = calibrated_probability(teacher_prediction["logits"], temperature)
    reliability = clinical_reliability(teacher_prediction["indices"], teacher_prediction["anchors"],
                                       sample_ids, anchors, statistics)
    pseudo = select_pseudo(teacher_prediction, recipe, reliability)

    student = load_model(base_path, CTModel(True), device)
    all_loader = imaging_loader(train_indices, sample_ids, labels, "ct", config, ct_store,
                                anchors, statistics, shuffle=True, seed=seed + 9000)
    labeled_loader = imaging_loader(labeled_indices, sample_ids, labels, "ct", config, ct_store,
                                    include_labels=True, shuffle=True, seed=seed + 9017)
    student_config = config["student"]
    history = train_student(student, all_loader, labeled_loader, pseudo, statistics, device,
                            student_config["epochs"], student_config["learning_rate"],
                            config["optimizer"]["momentum"], config["optimizer"]["weight_decay"],
                            config["anchor_weight"], config["pseudo_weight"],
                            student_config["step_size"], student_config["gamma"])
    save_checkpoint(destination, student, {"fold": args.fold, "budget": args.budget,
                                           "teacher": modality, "pseudo_selected_count": len(pseudo),
                                           "deployment": "CT-only"})
    save_json(destination.with_suffix(".history.json"), history)
    save_json(destination.with_suffix(".recipe.json"), recipe)


if __name__ == "__main__": main()

