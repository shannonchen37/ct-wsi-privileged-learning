import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import ImagingDataset
from .losses import masked_anchor_losses, soft_cross_entropy


def make_loader(dataset, batch_size, shuffle=False, seed=42, workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, generator=generator if shuffle else None)


def _next(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def pos_weights(statistics, device):
    return [torch.tensor(statistics["cyfra_pos_weight"], device=device),
            torch.tensor(statistics["scc_pos_weight"], device=device)]


def train_multitask(model, anchor_loader, labeled_loader, statistics, device,
                    epochs, learning_rate, momentum=0.9, weight_decay=0.0005,
                    anchor_weight=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.6)), gamma=0.1)
    weights = pos_weights(statistics, device); history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0; labeled_iterator = iter(labeled_loader)
        for image_a, _, target_a, mask_a in anchor_loader:
            (image_l, _, label_l), labeled_iterator = _next(labeled_iterator, labeled_loader)
            image_a = image_a.to(device); target_a = target_a.to(device); mask_a = mask_a.to(device)
            image_l = image_l.to(device); label_l = label_l.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, anchor_prediction, _ = model(image_a)
            losses = masked_anchor_losses(anchor_prediction, target_a, mask_a, weights)
            anchor_total = anchor_weight * sum(losses); anchor_total.backward()
            logits, _ = model.forward_diagnosis(image_l)
            classification = F.cross_entropy(logits, label_l); classification.backward()
            optimizer.step(); value = classification.detach() + anchor_total.detach()
            if not torch.isfinite(value): raise RuntimeError("non-finite multitask loss")
            total += value.item() * len(label_l); count += len(label_l)
        scheduler.step(); history.append({"epoch": epoch, "loss": total / count})
    return history


def train_ct_only(model, data_loader, device, epochs, learning_rate, momentum=0.9,
                  weight_decay=0.0005):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.6)), gamma=0.1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for image, _, label in data_loader:
            image = image.to(device); label = label.to(device)
            optimizer.zero_grad(set_to_none=True); logits, _ = model.forward_diagnosis(image)
            loss = F.cross_entropy(logits, label)
            if not torch.isfinite(loss): raise RuntimeError("non-finite CT loss")
            loss.backward(); optimizer.step(); total += loss.item() * len(label); count += len(label)
        scheduler.step(); history.append({"epoch": epoch, "loss": total / count})
    return history


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval(); indices, logits, anchors = [], [], []
    for batch in data_loader:
        image = batch[0].to(device); indices.extend(torch.as_tensor(batch[1]).numpy().tolist())
        output, anchor_output, _ = model(image)
        logits.append(output.cpu().numpy())
        if anchor_output is not None: anchors.append(anchor_output.cpu().numpy())
    result = {"indices": np.asarray(indices), "logits": np.concatenate(logits)}
    result["probability_lusc"] = torch.softmax(torch.from_numpy(result["logits"]), 1)[:, 1].numpy()
    if anchors: result["anchors"] = np.concatenate(anchors)
    return result


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    prediction = (probabilities >= 0.5).astype(int)
    tp = int(((labels == 1) & (prediction == 1)).sum()); tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum()); fn = int(((labels == 1) & (prediction == 0)).sum())
    return {"accuracy": float(accuracy_score(labels, prediction)), "auc": float(roc_auc_score(labels, probabilities)),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "f1": float(f1_score(labels, prediction))}


def fit_temperature(logits, labels, grid=(0.5, 0.75, 1.0, 1.5, 2.0)):
    logits = torch.tensor(logits, dtype=torch.float64); labels = torch.tensor(labels, dtype=torch.long)
    losses = {float(t): float(F.cross_entropy(logits / t, labels)) for t in grid}
    return min(grid, key=lambda t: (losses[float(t)], t)), losses


def calibrated_probability(logits, temperature):
    return torch.softmax(torch.tensor(logits, dtype=torch.float64) / temperature, 1)[:, 1].numpy()


def clinical_reliability(indices, predictions, sample_ids, anchors, statistics):
    values = []
    for index, prediction in zip(indices, predictions):
        target, mask = anchors.targets(sample_ids[int(index)], statistics); scores = []
        if mask[0]: scores.append(math.exp(-abs(float(prediction[0] - target[0]))))
        for column in (1, 2):
            if mask[column]:
                probability = float(torch.sigmoid(torch.tensor(prediction[column])))
                scores.append(probability if target[column] else 1.0 - probability)
        values.append(float(np.mean(scores)) if scores else 0.5)
    return np.asarray(values)


def choose_recipe(probability, labels, reliability, thresholds=(0.6, 0.7, 0.8, 0.9)):
    confidence = np.maximum(probability, 1.0 - probability); pseudo = (probability >= 0.5).astype(int)
    minimum = max(4, int(math.ceil(0.2 * len(labels)))); candidates = []
    for threshold in thresholds:
        mask = confidence >= threshold; n = int(mask.sum())
        accuracy = float((pseudo[mask] == labels[mask]).mean()) if n else 0.0
        coverage = n / len(labels); utility = coverage * max(0.0, 2 * accuracy - 1) ** 2 if n >= minimum else -1.0
        candidates.append((utility, -threshold, threshold, accuracy))
    _, _, threshold, base_accuracy = max(candidates)
    base = confidence >= threshold
    cutoff = np.quantile(reliability[base], 0.25) if base.any() else 1.0
    clinical = base & (reliability >= cutoff)
    clinical_accuracy = float((pseudo[clinical] == labels[clinical]).mean()) if clinical.any() else 0.0
    return {"confidence_threshold": float(threshold),
            "clinical_filter": bool(clinical.sum() >= 4 and clinical_accuracy >= base_accuracy + 0.02),
            "clinical_keep_fraction": 0.75}


def select_pseudo(prediction, recipe, reliability):
    probability = prediction["probability_lusc"]; confidence = np.maximum(probability, 1.0 - probability)
    selected = confidence >= recipe["confidence_threshold"]
    if recipe["clinical_filter"] and selected.any():
        predicted = (probability >= 0.5).astype(int); keep = np.zeros_like(selected)
        for class_index in (0, 1):
            mask = selected & (predicted == class_index)
            if mask.any():
                cutoff = np.quantile(reliability[mask], 1.0 - recipe["clinical_keep_fraction"])
                keep |= mask & (reliability >= cutoff)
        selected = keep
    output = {}
    for index, p, c, r, keep in zip(prediction["indices"], probability, confidence, reliability, selected):
        if keep:
            scaled = (c - recipe["confidence_threshold"]) / max(1e-6, 1 - recipe["confidence_threshold"])
            weight = 0.5 + 0.5 * float(np.clip(scaled, 0, 1))
            if recipe["clinical_filter"]: weight *= 0.5 + 0.5 * r
            output[int(index)] = (float(p), float(weight))
    return output


def train_student(model, all_loader, labeled_loader, pseudo, statistics, device,
                  epochs=50, learning_rate=0.01, momentum=0.9, weight_decay=0.0005,
                  anchor_weight=0.1, pseudo_weight=1.0, step_size=30, gamma=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    weights = pos_weights(statistics, device); history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0; labeled_iterator = iter(labeled_loader)
        for image, indices, anchor_target, anchor_mask in all_loader:
            (image_l, _, label_l), labeled_iterator = _next(labeled_iterator, labeled_loader)
            image = image.to(device); anchor_target = anchor_target.to(device); anchor_mask = anchor_mask.to(device)
            image_l = image_l.to(device); label_l = label_l.to(device)
            optimizer.zero_grad(set_to_none=True); logits, anchor_prediction, _ = model(image)
            selected = [i for i, index in enumerate(indices.tolist()) if index in pseudo]
            if selected:
                selected_tensor = torch.tensor(selected, device=device)
                probabilities = torch.tensor([pseudo[int(indices[i])][0] for i in selected], device=device)
                pseudo_weights = torch.tensor([pseudo[int(indices[i])][1] for i in selected], device=device)
                pseudo_loss = soft_cross_entropy(logits[selected_tensor], probabilities, pseudo_weights)
            else:
                pseudo_loss = logits.sum() * 0.0
            anchors_loss = masked_anchor_losses(anchor_prediction, anchor_target, anchor_mask, weights)
            labeled_logits, _ = model.forward_diagnosis(image_l)
            supervised = F.cross_entropy(labeled_logits, label_l)
            loss = supervised + pseudo_weight * pseudo_loss + anchor_weight * sum(anchors_loss)
            if not torch.isfinite(loss): raise RuntimeError("non-finite student loss")
            loss.backward(); optimizer.step(); total += loss.item() * len(indices); count += len(indices)
        scheduler.step(); history.append({"epoch": epoch, "loss": total / count})
    return history

