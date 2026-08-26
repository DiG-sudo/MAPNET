import csv
import gc
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from data import iterate_batches, project_path
from model import MAPNet


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config):
    device_index = str(config["runtime"]["device"])
    os.environ["CUDA_VISIBLE_DEVICES"] = device_index
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model_config(config, num_experts):
    model_config = dict(config["model"])
    model_config["num_experts"] = int(num_experts)
    return model_config


def compute_metrics(labels, logits, loss):
    probabilities = F.softmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy().astype(int)
    y_pred = probabilities.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, probabilities[:, 1])
    except ValueError:
        auc = 0.0
    return {
        "loss": float(loss),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": auc,
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def train_epoch(model, dataset, batch_size, optimizer, scaler, device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in iterate_batches(dataset, batch_size, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * int(batch.y.numel())
        total_samples += int(batch.y.numel())
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total_samples = 0
    for batch in iterate_batches(dataset, batch_size, shuffle=False):
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(batch)
            loss = criterion(logits, batch.y.long())
        logits_all.append(logits.float())
        labels_all.append(batch.y.long())
        total_loss += float(loss.item()) * int(batch.y.numel())
        total_samples += int(batch.y.numel())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return compute_metrics(labels, logits, total_loss / max(total_samples, 1))


def train_one(config, protein, num_experts, splits, data_stats, device, evaluate_test=False):
    train_config = config["train"]
    seed = int(train_config["seed"])
    setup_seed(seed)
    model_config = build_model_config(config, num_experts)
    model = MAPNet(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    batch_size = int(data_stats["batch_size"])
    best_val_loss = float("inf")
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    patience = 0
    final_train_loss = 0.0

    for _epoch in range(1, int(train_config["epochs"]) + 1):
        final_train_loss = train_epoch(model, splits["train"], batch_size, optimizer, scaler, device)
        validation = evaluate(model, splits["val"], batch_size, device)
        if validation["loss"] < best_val_loss - float(train_config["loss_improve_threshold"]):
            best_val_loss = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(train_config["patience"]):
                break

    model.load_state_dict(best_state)
    validation = evaluate(model, splits["val"], batch_size, device)
    test = evaluate(model, splits["test"], batch_size, device) if evaluate_test else None
    row = {
        "protein": protein,
        "seed": seed,
        "batch_size": batch_size,
        "num_experts": int(num_experts),
        "num_scales": int(model_config["num_scales"]),
        "router_temperature": float(model_config["router_temperature"]),
        "train_loss": final_train_loss,
        "val_loss": validation["loss"],
        "val_acc": validation["acc"],
        "val_f1": validation["f1"],
        "val_auc": validation["auc"],
    }
    if test is not None:
        row.update(
            {
                "test_loss": test["loss"],
                "test_acc": test["acc"],
                "test_f1": test["f1"],
                "test_auc": test["auc"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
            }
        )
    return row, best_state, model_config


def select_best(rows):
    if not rows:
        raise ValueError("No search results were produced.")
    return max(
        rows,
        key=lambda row: (
            float(row["val_auc"]),
            -float(row["val_loss"]),
            float(row["val_f1"]),
            -int(row["num_scales"]),
            -int(row["num_experts"]),
        ),
    )


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows, path, replace_keys=None):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv(path)
    if replace_keys:
        incoming = {tuple(str(row[key]) for key in replace_keys) for row in rows}
        existing = [
            row
            for row in existing
            if tuple(str(row.get(key, "")) for key in replace_keys) not in incoming
        ]
    merged = existing + rows
    fieldnames = list(merged[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def save_checkpoint(config, protein, model_config, model_state, row):
    output_dir = project_path(config, config["paths"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"{protein}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_name": "MAPNet",
            "protein": protein,
            "model_config": model_config,
            "model_state": model_state,
            "seed": int(config["train"]["seed"]),
            "train_ratio": float(config["data"]["train_ratio"]),
            "val_ratio": float(config["data"]["val_ratio"]),
            "metrics": row,
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def cleanup(*objects):
    for item in objects:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
