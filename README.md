# Hybrid Learning Against Label Noise
**Combining Structural Consistency and Contrastive Initialization**

A controlled comparison of practical robustness strategies for image classification under human label noise, evaluated under a fixed ResNet-18 training pipeline on CIFAR-10 (clean) and CIFAR-10N (human-annotated, AGGRE).

## 📄 Course Paper

**[Hybrid Learning Against Label Noise (PDF)](paper/hybrid-label-noise-cv.pdf)** — full writeup including methods, experimental setup, results, and analysis.

## Overview

Deep neural networks trained with standard cross-entropy tend to memorize noisy labels during late training, degrading generalization. This is especially problematic for human-annotated datasets, where label disagreement often reflects genuine ambiguity rather than random corruption.

This project performs a **controlled comparison of four training variants**, all sharing the same ResNet-18 backbone, optimizer, and augmentation pipeline — so any observed difference can be attributed to the robustness mechanism itself rather than confounding factors.

## Methods Compared

| Method | Description |
|---|---|
| **Baseline** | Standard cross-entropy treating observed labels as ground truth |
| **Structural** | Adds neighbor-consistency regularization in feature space, encouraging predictions of nearby samples to be similar |
| **Hybrid** | Combines structural regularization with agreement-based down-weighting of locally inconsistent samples |
| **Contrastive-init** | Contrastive-style encoder initialization followed by standard supervised training (clean-label diagnostic only) |

## Key Results

| Method | CIFAR-10N (AGGRE) Peak | Epoch | Final (Epoch 100) |
|---|---|---|---|
| Baseline | 86.91% | 79 | 84.52% |
| **Structural** | **87.53%** | 76 | 84.23% |
| **Hybrid** | 86.64% | 100 | **86.64%** (no decay) |

**Key findings:**
- Structural neighbor-consistency achieves the highest peak validation accuracy under human label noise
- Hybrid reweighting eliminates the late-epoch memorization decay observed in baseline training
- Remaining errors concentrate in visually ambiguous animal classes (dog / cat / horse)

## Repository Structure
Hybrid-Label-Noise-CV/
├── configs/         # Training and experiment configurations
├── labelnoise/      # Label noise handling and dataset utilities
├── models/          # Model definitions and backbone components
├── outputs/         # Experimental outputs (curves, figures, tables)
├── paper/           # Course paper PDF
├── tools/           # Plotting and analysis scripts
├── utils/           # Reproducibility and helper utilities
├── train.py         # Main training entry point
├── eval.py          # Evaluation utilities
├── pretrain.py      # Contrastive-style initialization
├── hybrid_loss.py   # Hybrid loss and agreement weighting
├── model.py         # ResNet-18 model wrapper
└── requirements.txt

Datasets, checkpoints, and large binaries are intentionally excluded from version control.

## Getting Started

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Quick test** (small subset, 1 epoch):
```bash
python3 train.py --epochs 1 --dataset cifar10n --subset aggre --subset_fraction 0.01 --model structural
```

**Full experiment**:
```bash
python3 train.py --epochs 100 --dataset cifar10n --subset aggre --model hybrid
```

Training produces learning curves, confusion matrices, UMAP / t-SNE embeddings, and per-class accuracy tables under `outputs/`.

## Evaluation Approach

Evaluation diagnostics are designed to interpret **why** certain strategies perform better, not just how well:

- **Learning curves** track peak and late-epoch behavior to surface memorization dynamics
- **Confusion matrices** highlight class-level ambiguity
- **UMAP / t-SNE visualizations** reveal representation geometry under noisy supervision

## Datasets

- **CIFAR-10** — clean labels, standard image classification benchmark
- **CIFAR-10N** — human-annotated noisy labels from the [UCSC-REAL CIFAR-N repository](https://github.com/UCSC-REAL/cifar-10-100n) (AGGRE label set used in this project)

The CIFAR-N protocol is followed: training on noisy labels, evaluating on clean CIFAR-10 test labels.

## Limitations

- Results reported for a single random seed
- Subset sampling is not class-stratified
- Contrastive-init evaluated only as initialization (not end-to-end contrastive training under noise)
- Full noise-rate sweep proposed as future work

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `Pandas` · `UMAP` · `t-SNE` · `Matplotlib`

---

*This repository accompanies a course project submission and is intended for academic and educational use.*
