import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest must contain sample_id,label")
    sample_ids = [row["sample_id"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if len(sample_ids) != len(set(sample_ids)) or not set(labels).issubset({0, 1}):
        raise ValueError("manifest IDs must be unique and labels must be 0/1")
    return sample_ids, labels


class CTStore:
    def __init__(self, path):
        with Path(path).open("rb") as handle:
            raw = pickle.load(handle)
        self.features = {}
        for sample_id, value in raw.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (512,) or not np.isfinite(array).all():
                raise ValueError(f"invalid CT feature for {sample_id}")
            self.features[str(sample_id)] = array


class AnchorStore:
    REQUIRED = {
        "sample_id", "t_size_max_cm", "t_size_mask",
        "cyfra_abnormal", "cyfra_abnormal_mask",
        "scc_abnormal", "scc_abnormal_mask",
    }

    def __init__(self, path):
        with Path(path).open() as handle:
            reader = csv.DictReader(handle)
            if not self.REQUIRED.issubset(reader.fieldnames or []):
                raise ValueError(f"clinical table must contain {sorted(self.REQUIRED)}")
            self.rows = {}
            for raw in reader:
                sample_id = raw["sample_id"]
                masks = [int(raw["t_size_mask"]), int(raw["cyfra_abnormal_mask"]),
                         int(raw["scc_abnormal_mask"])]
                self.rows[sample_id] = {
                    "t_size": float(raw["t_size_max_cm"]) if masks[0] else 0.0,
                    "cyfra": float(raw["cyfra_abnormal"]) if masks[1] else 0.0,
                    "scc": float(raw["scc_abnormal"]) if masks[2] else 0.0,
                    "masks": masks,
                }

    def statistics(self, train_ids):
        t_values = np.asarray(
            [np.log1p(self.rows[x]["t_size"]) for x in train_ids if self.rows[x]["masks"][0]],
            dtype=np.float64,
        )
        mean, std = float(t_values.mean()), float(t_values.std(ddof=0))
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("invalid train-fold T-size statistics")
        result = {"t_mean": mean, "t_std": std}
        for name, mask_index in (("cyfra", 1), ("scc", 2)):
            values = [self.rows[x][name] for x in train_ids if self.rows[x]["masks"][mask_index]]
            positives = int(sum(values)); negatives = len(values) - positives
            if positives == 0 or negatives == 0:
                raise ValueError(f"{name} requires both classes in the train fold")
            result[f"{name}_pos_weight"] = float(negatives / positives)
        return result

    def targets(self, sample_id, statistics):
        row = self.rows[sample_id]; masks = np.asarray(row["masks"], dtype=np.float32)
        t_size = ((np.log1p(row["t_size"]) - statistics["t_mean"]) / statistics["t_std"]
                  if masks[0] else 0.0)
        values = np.asarray([t_size, row["cyfra"], row["scc"]], dtype=np.float32)
        return values, masks


class ImagingDataset(Dataset):
    def __init__(self, indices, sample_ids, labels, modality, ct_store=None, wsi_dir=None,
                 anchors=None, statistics=None, include_labels=False):
        self.indices = list(indices); self.sample_ids = sample_ids; self.labels = labels
        self.modality = modality; self.ct_store = ct_store; self.wsi_dir = Path(wsi_dir) if wsi_dir else None
        self.anchors = anchors; self.statistics = statistics; self.include_labels = include_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]; sample_id = self.sample_ids[index]
        if self.modality == "ct":
            image = torch.from_numpy(self.ct_store.features[sample_id])
        else:
            value = np.load(self.wsi_dir / f"{sample_id}_fts.npy")
            if value.ndim != 2 or value.shape[1] != 1792 or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid WSI patches for {sample_id}")
            image = torch.from_numpy(value)
        output = [image, index]
        if self.include_labels:
            output.append(int(self.labels[index]))
        if self.anchors is not None:
            target, mask = self.anchors.targets(sample_id, self.statistics)
            output.extend([torch.from_numpy(target), torch.from_numpy(mask)])
        return tuple(output)

