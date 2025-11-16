import argparse
import csv
import os
from collections import defaultdict
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None

from labelnoise.cifar10 import get_cifar10_loaders
from labelnoise.cifarN import get_cifar10n_loaders
from models.resnet import ResNet18Classifier
from utils import log, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Plot accuracy trends and embeddings")
    parser.add_argument("--metrics_csv", type=str, default=None, help="CSV with columns mode,noise_rate,accuracy")
    parser.add_argument("--history_files", nargs="*", default=None, help="History CSV files produced during training")
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for embeddings")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar10n"], default="cifar10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--noise", type=str, default="clean")
    parser.add_argument("--noise_rate", type=float, default=0.0)
    parser.add_argument("--subset", type=str, default="aggre")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples for embeddings")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_metric_records(metrics_csv: Optional[str], history_files: Optional[Sequence[str]]):
    records = []
    if metrics_csv and os.path.exists(metrics_csv):
        with open(metrics_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if {"mode", "noise_rate", "accuracy"}.issubset(row.keys()):
                    records.append(
                        {
                            "mode": row["mode"],
                            "noise_rate": float(row["noise_rate"]),
                            "accuracy": float(row["accuracy"]),
                        }
                    )
    if history_files:
        for file in history_files:
            if not os.path.exists(file):
                continue
            with open(file, "r") as f:
                reader = list(csv.DictReader(f))
                if not reader:
                    continue
                last = reader[-1]
                records.append(
                    {
                        "mode": last.get("mode", "unknown"),
                        "noise_rate": float(last.get("noise_rate", 0.0)),
                        "accuracy": float(last.get("test_acc", last.get("val_acc", 0.0))),
                    }
                )
    return records


def plot_accuracy_vs_noise(records: List[dict], output_dir: str):
    if not records:
        return
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["mode"]].append((rec["noise_rate"], rec["accuracy"]))
    plt.figure()
    for mode, series in grouped.items():
        series.sort(key=lambda x: x[0])
        xs, ys = zip(*series)
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xlabel("Noise rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs noise")
    plt.legend()
    path = os.path.join(output_dir, "accuracy_vs_noise.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved {path}")


def plot_mode_bars(records: List[dict], output_dir: str):
    if not records:
        return
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["mode"]].append(rec["accuracy"])
    modes = sorted(grouped.keys())
    means = [np.mean(grouped[m]) for m in modes]
    plt.figure()
    plt.bar(modes, means)
    plt.ylabel("Accuracy")
    plt.title("Agreement-aware performance")
    path = os.path.join(output_dir, "mode_bar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved {path}")


def build_loader(args, split: str):
    if args.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            noise_type=args.noise,
            noise_rate=args.noise_rate,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader = get_cifar10n_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            subset=args.subset,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    if split == "train":
        return train_loader
    if split == "val":
        if val_loader is None:
            raise ValueError("Validation loader not available for this dataset")
        return val_loader
    return test_loader


def extract_embeddings(model, loader, device, max_samples: int):
    features, labels = [], []
    model.eval()
    collected = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            feats = model.extract_features(inputs)
            features.append(feats.cpu())
            labels.append(targets.cpu())
            collected += inputs.size(0)
            if collected >= max_samples:
                break
    features = torch.cat(features)[:max_samples]
    labels = torch.cat(labels)[:max_samples]
    return features.numpy(), labels.numpy()


def plot_embeddings(embeddings, labels, output_dir: str):
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    tsne_emb = tsne.fit_transform(embeddings)
    plt.figure()
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, cmap="tab10", s=5)
    plt.title("t-SNE embeddings")
    path = os.path.join(output_dir, "tsne.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved {path}")

    if umap is None:
        log("UMAP not installed; skipping UMAP plot")
        return
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    umap_emb = reducer.fit_transform(embeddings)
    plt.figure()
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, cmap="tab10", s=5)
    plt.title("UMAP embeddings")
    path = os.path.join(output_dir, "umap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_metric_records(args.metrics_csv, args.history_files)
    if records:
        plot_accuracy_vs_noise(records, args.output_dir)
        plot_mode_bars(records, args.output_dir)
    else:
        log("No metric records found; skipping accuracy plots")

    if args.checkpoint:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        loader = build_loader(args, args.split)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model = ResNet18Classifier(num_classes=10).to(device)
        model.load_state_dict(state_dict, strict=False)
        embeddings, labels = extract_embeddings(model, loader, device, args.max_samples)
        plot_embeddings(embeddings, labels, args.output_dir)
    else:
        log("No checkpoint specified; skipping embedding visualization")


if __name__ == "__main__":
    main()
