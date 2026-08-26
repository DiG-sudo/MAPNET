from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


FEATURE_KEYS = ("kmer1", "kmer2", "kmer3", "ncp", "dpcp", "circ2vec_embed")


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project_dir"] = str(config_path.parent)
    return config


def project_path(config, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["project_dir"]) / path


def load_feature_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Run prepare_data.py first.")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in (*FEATURE_KEYS, "label") if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        bundle = {
            key: torch.from_numpy(np.asarray(archive[key], dtype=np.float32))
            for key in FEATURE_KEYS
        }
        bundle["y"] = torch.from_numpy(np.asarray(archive["label"], dtype=np.int64))
    return bundle


def split_indices(total_size, seed, train_ratio, val_ratio):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(total_size, generator=generator)
    train_size = int(float(train_ratio) * total_size)
    val_size = int(float(val_ratio) * total_size)
    return (
        order[:train_size],
        order[train_size : train_size + val_size],
        order[train_size + val_size :],
    )


def calculate_batch_size(test_size, data_config):
    from_test = max(int(test_size * float(data_config["test_batch_fraction"])), 8)
    batch_size = max(int(data_config["batch_size_base"]), from_test)
    batch_size = min(batch_size, int(data_config["batch_size_max"]))
    batch_size = (batch_size // 8) * 8
    return max(int(data_config["batch_size_base"]), batch_size)


def subset_to_device(bundle, indices, device):
    return {
        **{
            key: bundle[key].index_select(0, indices).to(device, non_blocking=True)
            for key in FEATURE_KEYS
        },
        "y": bundle["y"].index_select(0, indices).to(device, non_blocking=True),
        "size": int(indices.numel()),
    }


def prepare_protein_data(config, protein, device):
    feature_dir = project_path(config, config["paths"]["feature_dir"])
    bundle = load_feature_bundle(feature_dir / f"{protein}_features.npz")
    train_idx, val_idx, test_idx = split_indices(
        total_size=int(bundle["y"].shape[0]),
        seed=config["train"]["seed"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
    )
    batch_size = calculate_batch_size(int(test_idx.numel()), config["data"])
    splits = {
        "train": subset_to_device(bundle, train_idx, device),
        "val": subset_to_device(bundle, val_idx, device),
        "test": subset_to_device(bundle, test_idx, device),
    }
    labels = bundle["y"].numpy()
    stats = {
        "total": int(labels.size),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "batch_size": batch_size,
    }
    return splits, stats


def iterate_batches(dataset, batch_size, shuffle):
    total_size = int(dataset["size"])
    order = torch.randperm(total_size, device="cpu") if shuffle else torch.arange(total_size)
    for start in range(0, total_size, batch_size):
        cpu_indices = order[start : start + batch_size]
        indices = cpu_indices.to(dataset["y"].device, non_blocking=True)
        values = {
            key: dataset[key].index_select(0, indices)
            for key in (*FEATURE_KEYS, "y")
        }
        yield SimpleNamespace(**values)
