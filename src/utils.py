import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold


def load_config(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def outer_folds(labels, seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(train.tolist(), valid.tolist()) for train, valid in splitter.split(np.zeros(len(labels)), labels)]


def nested_labeled_subset(train_indices, labels, budget, fold, seed=42):
    """Nested, deterministic, class-stratified diagnosis-label subsets."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed * 1000 + fold)
    selected = []
    for class_index in (0, 1):
        candidates = np.asarray([i for i in train_indices if labels[i] == class_index])
        candidates.sort(); rng.shuffle(candidates)
        count = len(candidates) if budget == 100 else int(round(len(candidates) * budget / 100.0))
        selected.extend(candidates[:count].tolist())
    return sorted(selected)


def inner_partition(labeled_indices, labels):
    labels = np.asarray(labels)
    train, valid = [], []
    for class_index in (0, 1):
        values = [i for i in labeled_indices if labels[i] == class_index]
        n_valid = max(2, int(round(0.2 * len(values))))
        train.extend(values[:-n_valid]); valid.extend(values[-n_valid:])
    return sorted(train), sorted(valid)


def device_from_environment():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

