#!/usr/bin/env python3

import csv
import glob
import os
import sys

mpl_config_dir = os.path.join(os.getcwd(), "outputs", ".mplconfig")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_PATTERNS = {
    "baseline": "outputs/baseline/aggre/*/metrics.csv",
    "structural": "outputs/structural/aggre/*/metrics.csv",
    "hybrid": "outputs/hybrid/aggre/*/metrics.csv",
}

OUT_ACC = "outputs/fig_learning_curve_acc_cifar10n_aggre.png"
OUT_CE = "outputs/fig_learning_curve_ce_cifar10n_aggre.png"


def find_latest_csv(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, []
    paths = sorted(paths, key=lambda p: os.path.getmtime(p))
    return paths[-1], paths


def load_runs():
    data = {}
    missing = {}
    for name, pattern in RUN_PATTERNS.items():
        latest, searched = find_latest_csv(pattern)
        if latest is None:
            missing[name] = searched
            continue
        try:
            with open(latest, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print(f"[WARN] {latest} is empty")
                continue
            data[name] = (latest, rows)
        except Exception as e:
            print(f"[ERROR] Failed to read {latest}: {e}")
    return data, missing


def plot_metric(data, metric, ylabel, outfile):
    plt.figure(figsize=(6, 4))
    for name, (path, rows) in data.items():
        if not rows or metric not in rows[0] or "epoch" not in rows[0]:
            print(f"[WARN] {path} missing required columns for {metric}")
            continue
        epochs = []
        vals = []
        for r in rows:
            try:
                epochs.append(float(r["epoch"]))
                vals.append(float(r[metric]))
            except Exception:
                continue
        plt.plot(epochs, vals, label=name.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch (CIFAR-10N AGGRE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"[OK] Saved {outfile}")
    plt.close()


def main():
    data, missing = load_runs()

    if missing:
        for name, _ in missing.items():
            print(f"[ERROR] No metrics found for '{name}'. Searched: {RUN_PATTERNS[name]}")
    if not data:
        sys.exit(1)

    plot_metric(data, "val_acc", "Validation Accuracy", OUT_ACC)
    plot_metric(data, "val_ce", "Validation CE Loss", OUT_CE)


if __name__ == "__main__":
    main()
