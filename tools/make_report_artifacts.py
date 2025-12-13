#!/usr/bin/env python3

import csv
import glob
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

mpl_config_dir = os.path.join(os.getcwd(), "outputs", ".mplconfig")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
numba_cache = os.path.join(os.getcwd(), "outputs", ".numba_cache")
os.makedirs(numba_cache, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", numba_cache)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labelnoise.cifarN import get_cifar10n_loaders
from models.resnet import ResNet18Classifier
from utils import log


def find_latest_metrics(pattern: str) -> Tuple[str, List[str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return "", []
    paths = sorted(paths, key=lambda p: os.path.getmtime(p))
    return paths[-1], paths


def read_metrics_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "epoch": float(r["epoch"]),
                        "val_acc": float(r.get("val_acc", "nan")),
                        "val_ce": float(r.get("val_ce", "nan")),
                    }
                )
            except Exception:
                continue
    return rows


def plot_learning_curves():
    run_patterns = {
        "baseline": "outputs/baseline/aggre/*/metrics.csv",
        "structural": "outputs/structural/aggre/*/metrics.csv",
        "hybrid": "outputs/hybrid/aggre/*/metrics.csv",
    }
    data = {}
    for name, pat in run_patterns.items():
        latest, searched = find_latest_metrics(pat)
        if not latest:
            print(f"[ERROR] No metrics found for '{name}'. Searched: {pat}")
            continue
        data[name] = (latest, read_metrics_csv(latest))

    if not data:
        print("[ERROR] No metrics files found; skipping plots.")
        return

    def plot_metric(metric, ylabel, outpath):
        plt.figure(figsize=(6, 4))
        for name, (path, rows) in data.items():
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows if metric in r and not np.isnan(r[metric])]
            vals = [r[metric] for r in rows if metric in r and not np.isnan(r[metric])]
            plt.plot(epochs, vals, label=name.capitalize())
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Epoch (CIFAR-10N AGGRE)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"[OK] Saved {outpath}")

    plot_metric("val_acc", "Validation Accuracy", "outputs/fig_learning_curve_acc_cifar10n_aggre.png")
    plot_metric("val_ce", "Validation CE Loss", "outputs/fig_learning_curve_ce_cifar10n_aggre.png")


def load_best_checkpoint(run_dir: str):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    candidates = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not candidates:
        candidates = glob.glob(os.path.join(run_dir, "*.pth"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p))
    best = [c for c in candidates if "best" in os.path.basename(c).lower()]
    if best:
        return best[-1]
    return candidates[-1]


def select_device():
    if torch.backends.mps.is_available():
        print("[LOG] Using device: MPS (Apple Silicon GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[LOG] Using device: CUDA")
        return torch.device("cuda")
    print("[LOG] Using device: CPU")
    return torch.device("cpu")


def run_structural_eval():
    latest_run_dir = ""
    candidate_dirs = sorted(glob.glob("outputs/structural/aggre/*"))
    if candidate_dirs:
        latest_run_dir = sorted(candidate_dirs, key=lambda p: os.path.getmtime(p))[-1]
    if not latest_run_dir:
        print("[ERROR] No structural aggre run found.")
        return

    ckpt_path = load_best_checkpoint(latest_run_dir)
    if ckpt_path is None:
        print(f"[ERROR] No checkpoint found in {latest_run_dir}")
        return

    device = select_device()
    model = ResNet18Classifier(num_classes=10)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    _, _, test_loader = get_cifar10n_loaders(batch_size=128, subset="aggre", data_root="data")

    all_features = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            feats = model.extract_features(imgs)
            all_features.append(feats.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    features = np.concatenate(all_features, axis=0)

    os.makedirs("outputs", exist_ok=True);
    np.savez("outputs/cifar10n_structural_aggre_preds.npz", y_true=y_true, y_pred=y_pred)
    print("[OK] Saved outputs/cifar10n_structural_aggre_preds.npz")

    per_class = []
    for c in range(10):
        mask = y_true == c
        acc = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
        per_class.append((c, acc))

    csv_path = "outputs/table_per_class_acc_cifar10n_structural_aggre.csv"
    md_path = "outputs/table_per_class_acc_cifar10n_structural_aggre.md"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for c, acc in per_class:
            w.writerow([c, acc])
    with open(md_path, "w") as f:
        f.write("| class | accuracy |\n| --- | --- |\n")
        for c, acc in per_class:
            f.write(f"| {c} | {acc:.4f} |\n")
    print(f"[OK] Saved {csv_path}")
    print(f"[OK] Saved {md_path}")

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Structural, CIFAR-10N AGGRE)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/fig_confusion_cifar10n_structural_aggre.png", dpi=200)
    plt.close()
    print("[OK] Saved outputs/fig_confusion_cifar10n_structural_aggre.png")

    try:
        import umap

        n = min(2000, features.shape[0])
        subset_idx = np.random.choice(features.shape[0], n, replace=False)
        emb = umap.UMAP(random_state=42).fit_transform(features[subset_idx])
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            emb[:, 0],
            emb[:, 1],
            c=y_true[subset_idx],
            cmap="tab10",
            s=5,
            alpha=0.8,
        )
        plt.title("UMAP (Structural, CIFAR-10N AGGRE, test)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.colorbar(scatter, ticks=range(10))
        plt.tight_layout()
        plt.savefig("outputs/fig_umap_cifar10n_structural_aggre.png", dpi=200)
        plt.close()
        print("[OK] Saved outputs/fig_umap_cifar10n_structural_aggre.png")
    except Exception as e:
        print(f"[WARN] UMAP not generated (missing dependency?): {e}")


def summarize_metrics_rows(rows: List[Dict[str, float]]):
    if not rows:
        return {"peak_acc": np.nan, "peak_epoch": np.nan, "final_acc": np.nan}
    peak_idx = int(np.nanargmax([r["val_acc"] for r in rows]))
    peak_acc = rows[peak_idx]["val_acc"]
    peak_epoch = rows[peak_idx]["epoch"]
    final_row = None
    for r in rows:
        if int(r["epoch"]) == 100:
            final_row = r
            break
    if final_row is None:
        final_row = rows[-1]
    final_acc = final_row["val_acc"]
    return {"peak_acc": peak_acc, "peak_epoch": peak_epoch, "final_acc": final_acc}


def build_main_tables():
    cifar10_dir = "cifar10_outputs"
    models = ["baseline", "structural", "hybrid", "contrastive"]
    cifar10_rows = []
    for m in models:
        pattern = os.path.join(cifar10_dir, f"{m}*", "**", "metrics.csv")
        paths = sorted(glob.glob(pattern, recursive=True), key=lambda p: os.path.getmtime(p))
        if not paths:
            print(f"[ERROR] No metrics found for CIFAR-10 model {m}")
            cifar10_rows.append([m, np.nan, np.nan, np.nan])
            continue
        metrics = read_metrics_csv(paths[-1])
        stats = summarize_metrics_rows(metrics)
        cifar10_rows.append([m, stats["peak_acc"], stats["peak_epoch"], stats["final_acc"]])

    modes = ["baseline", "structural", "hybrid"]
    cifarn_rows = []
    for m in modes:
        pattern = f"outputs/{m}/aggre/*/metrics.csv"
        latest, searched = find_latest_metrics(pattern)
        if not latest:
            print(f"[ERROR] No metrics found for CIFAR-10N model {m}. Searched: {pattern}")
            cifarn_rows.append([m, np.nan, np.nan, np.nan])
            continue
        metrics = read_metrics_csv(latest)
        stats = summarize_metrics_rows(metrics)
        cifarn_rows.append([m, stats["peak_acc"], stats["peak_epoch"], stats["final_acc"]])

    header = ["model", "peak_val_acc", "peak_epoch", "final_val_acc_epoch100"]
    csv_path = "outputs/table_main_results.csv"
    md_path = "outputs/table_main_results.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Section", *header])
        for row in cifar10_rows:
            w.writerow(["cifar10_clean", *row])
        for row in cifarn_rows:
            w.writerow(["cifar10n_aggre", *row])

    with open(md_path, "w") as f:
        f.write("| Section | model | peak_val_acc | peak_epoch | final_val_acc_epoch100 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in cifar10_rows:
            f.write(f"| cifar10_clean | {row[0]} | {row[1]:.4f} | {int(row[2]) if not np.isnan(row[2]) else 'nan'} | {row[3]:.4f} |\n")
        for row in cifarn_rows:
            f.write(f"| cifar10n_aggre | {row[0]} | {row[1]:.4f} | {int(row[2]) if not np.isnan(row[2]) else 'nan'} | {row[3]:.4f} |\n")

    print(f"[OK] Saved {csv_path}")
    print(f"[OK] Saved {md_path}")


def main():
    plot_learning_curves()
    run_structural_eval()
    build_main_tables()


if __name__ == "__main__":
    main()
