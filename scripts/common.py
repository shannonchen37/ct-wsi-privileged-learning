from pathlib import Path

import torch

from src.data import AnchorStore, CTStore, ImagingDataset, load_manifest
from src.training import make_loader
from src.utils import nested_labeled_subset, outer_folds


def experiment_context(config, fold, budget):
    sample_ids, labels = load_manifest(config["manifest"])
    folds = outer_folds(labels, config.get("seed", 42))
    train_indices, valid_indices = folds[fold]
    labeled_indices = nested_labeled_subset(train_indices, labels, budget, fold, config.get("seed", 42))
    anchors = AnchorStore(config["clinical_table"])
    train_ids = [sample_ids[i] for i in train_indices]
    statistics = anchors.statistics(train_ids)
    ct_store = CTStore(config["ct_features"])
    return sample_ids, labels, train_indices, valid_indices, labeled_indices, anchors, statistics, ct_store


def imaging_loader(indices, sample_ids, labels, modality, config, ct_store,
                   anchors=None, statistics=None, include_labels=False,
                   shuffle=False, seed=42):
    dataset = ImagingDataset(
        indices, sample_ids, labels, modality,
        ct_store=ct_store, wsi_dir=config["wsi_features"],
        anchors=anchors, statistics=statistics, include_labels=include_labels,
    )
    section = config[modality]
    return make_loader(dataset, section["batch_size"], shuffle, seed,
                       workers=0 if modality == "ct" else 2)


def checkpoint_path(config, method, budget, fold):
    return Path(config["output_dir"]) / "checkpoints" / method / f"budget_{budget}" / f"fold_{fold}.pt"


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)

