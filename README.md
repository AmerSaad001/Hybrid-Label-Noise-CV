
Hybrid Learning Against Label Noise

Combining Structural Consistency and Contrastive Initialization

This repository contains the code, configurations, and experimental outputs for a course project studying robust learning under label noise in image classification. The project evaluates several practical robustness strategies under a fixed and controlled training pipeline, focusing on real human annotation noise rather than synthetic corruption.

The experiments are conducted on CIFAR-10 (clean labels) and CIFAR-10N (human noisy labels, AGGRE) using a shared ResNet-18 backbone and identical optimization settings to isolate the effect of the robustness mechanisms themselves.

⸻

📌 Project Overview

Deep neural networks trained with standard cross-entropy tend to memorize noisy labels during late training, which degrades generalization. This issue is especially important when labels come from humans, where disagreement often reflects genuine ambiguity rather than random corruption.

In this project, we perform a controlled comparison of four training variants:
	•	Baseline: Standard cross-entropy training
	•	Structural: Neighbor-consistency regularization in representation space
	•	Hybrid: Structural consistency + agreement-based reweighting
	•	Contrastive-init: Contrastive-style initialization followed by supervised training

All methods:
	•	Use ResNet-18
	•	Train for 100 epochs
	•	Use SGD
	•	Share the same augmentation, optimizer, and evaluation protocol
	•	Optionally run under a 30% subset mode for reproducibility under limited compute

⸻

🧠 Methods Implemented

Baseline

Standard cross-entropy loss treating observed labels as ground truth.

Structural

Adds a neighbor-consistency regularization term that encourages predictions of nearby samples in feature space to be similar. This reduces memorization of locally mislabeled examples.

Hybrid

Combines:
	•	Agreement-based down-weighting of suspicious samples
	•	The same neighbor-consistency regularization used in Structural

Contrastive-init

Uses contrastive-style initialization of the encoder, followed by standard supervised training. No contrastive objective is applied during noisy supervised training.

⸻

📂 Repository Structure

Hybrid-CV-Project/
│
├── train.py                # Main training entry point
├── eval.py                 # Evaluation utilities
├── model.py                # ResNet-18 and model components
├── hybrid_loss.py          # Hybrid loss and agreement weighting
├── pretrain.py             # Contrastive-style initialization
├── utils.py                # Shared helpers
│
├── models/
│   └── neighbor_consistency.py
│
├── tools/
│   ├── plot_learning_curves.py
│   ├── plot_embeddings.py
│   └── make_report_artifacts.py
│
├── outputs/
│   ├── baseline/
│   ├── structural/
│   ├── hybrid/
│   ├── figures (.png)
│   └── tables (.csv, .md)
│
├── data/                   # (ignored) datasets
├── datasets/               # (ignored) raw data
│
├── requirements.txt
├── README.md
└── .gitignore

Note: Datasets, checkpoints, and large binaries are intentionally excluded from version control.

⸻

▶️ How to Run

1. Environment Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


⸻

2. Training (Example)

Run a quick test using a small subset:

python3 train.py \
  --epochs 1 \
  --dataset cifar10n \
  --subset aggre \
  --subset_fraction 0.01 \
  --model structural

Run a full experiment (example):

python3 train.py \
  --epochs 100 \
  --dataset cifar10n \
  --subset aggre \
  --model hybrid


⸻

3. Outputs

Training produces:
	•	Learning curves (accuracy & cross-entropy)
	•	Confusion matrices
	•	UMAP / t-SNE embeddings
	•	CSV tables for aggregate and per-class accuracy

All outputs are saved under:

outputs/<method>/<subset>/<timestamp>/


⸻

📊 Evaluation & Visualization
	•	Learning curves track both peak and late-epoch behavior
	•	Confusion matrices highlight class-level ambiguity
	•	UMAP / t-SNE visualize representation geometry under noisy supervision

These diagnostics are used to interpret why certain robustness strategies perform better, not just how well they perform.

⸻

🔁 Reproducibility

To ensure fair comparison:
	•	All methods share the same architecture, optimizer, and schedule
	•	Subset mode enables deterministic training under limited compute
	•	CIFAR-10N evaluation follows the standard protocol using clean test labels

⸻

📎 Dataset References
	•	CIFAR-10: Standard image classification benchmark
	•	CIFAR-10N: Human-annotated noisy labels
	•	AGGRE label set used in experiments
	•	Provided by the UCSC-REAL CIFAR-N repository

⸻

⚠️ Notes & Limitations
	•	Results are primarily reported for a single random seed
	•	Subset sampling is not class-stratified
	•	Contrastive-init is evaluated as initialization, not end-to-end contrastive training under noise
	•	A full noise-rate sweep is proposed as future work

⸻

🔗 Important Links

Project repository:
https://github.com/AmerSaad001/Hybrid-Label-Noise-CV

⸻

📄 License & Usage

This repository accompanies a course project submission and is intended for academic and educational use.

