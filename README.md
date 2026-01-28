Hybrid Learning Against Label Noise  
Combining Structural Consistency and Contrastive Initialization

This repository contains the code, configurations, and experimental outputs for a course project studying robust learning under label noise in image classification. The project evaluates several practical robustness strategies under a fixed and controlled training pipeline, focusing on real human annotation noise rather than synthetic corruption.

The experiments are conducted on CIFAR-10 (clean labels) and CIFAR-10N (human noisy labels, AGGRE) using a shared ResNet-18 backbone and identical optimization settings to isolate the effect of the robustness mechanisms themselves.

Paper (PDF): paper/Hybrid_Learning_Against_Label_Noise.pdf

Deep neural networks trained with standard cross-entropy tend to memorize noisy labels during late training, which degrades generalization. This issue is especially important when labels come from humans, where disagreement often reflects genuine ambiguity rather than random corruption.

In this project, we perform a controlled comparison of four training variants: a supervised baseline using standard cross-entropy, a structural method using neighbor-consistency regularization in representation space, a hybrid method combining structural consistency with agreement-based reweighting, and a contrastive-init variant that uses contrastive-style initialization followed by supervised training.

All methods use a ResNet-18 backbone, are trained for 100 epochs using SGD, share the same data augmentation, optimizer, and evaluation protocol, and can optionally be run under a 30% subset mode for reproducibility under limited compute.

The baseline method applies standard cross-entropy loss treating observed labels as ground truth. The structural method adds a neighbor-consistency regularization term that encourages predictions of nearby samples in feature space to be similar, reducing memorization of locally mislabeled examples. The hybrid method combines agreement-based down-weighting of suspicious samples with the same neighbor-consistency regularization used in the structural approach. The contrastive-init method uses contrastive-style initialization of the encoder followed by standard supervised training; no contrastive objective is applied during noisy supervised training.

The repository structure is organized as follows:

Hybrid-CV-Project/
├── train.py (main training entry point)
├── eval.py (evaluation utilities)
├── model.py (ResNet-18 and model components)
├── hybrid_loss.py (hybrid loss and agreement weighting)
├── pretrain.py (contrastive-style initialization)
├── neighbor_consistency.py (structural regularization)
├── utils/ (shared helpers)
│   ├── __init__.py
│   └── reproducibility.py (set_seed utility)
├── tools/
│   ├── plot_learning_curves.py
│   ├── plot_embeddings.py
│   └── make_report_artifacts.py
├── outputs/
│   ├── baseline/
│   ├── structural/
│   ├── hybrid/
│   ├── figures/ (.png)
│   └── tables/ (.csv, .md)
├── data/ (ignored datasets)
├── datasets/ (ignored raw data)
├── requirements.txt
├── README.md
└── .gitignore

Datasets, checkpoints, and large binaries are intentionally excluded from version control.

To run the project, first create and activate a virtual environment and install dependencies using:  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt

A quick test using a small subset can be run with:  
python3 train.py --epochs 1 --dataset cifar10n --subset aggre --subset_fraction 0.01 --model structural

A full experiment example can be run with:  
python3 train.py --epochs 100 --dataset cifar10n --subset aggre --model hybrid

Training produces learning curves (accuracy and cross-entropy), confusion matrices, UMAP or t-SNE embeddings, and CSV tables for aggregate and per-class accuracy. All outputs are saved under the outputs/ directory.

Evaluation diagnostics are used to interpret why certain robustness strategies perform better, not just how well they perform. Learning curves track both peak and late-epoch behavior, confusion matrices highlight class-level ambiguity, and UMAP or t-SNE visualizations reveal representation geometry under noisy supervision.

For reproducibility, all methods share the same architecture, optimizer, and training schedule. Subset mode enables deterministic training under limited compute, and CIFAR-10N evaluation follows the standard protocol using clean test labels.

The project uses CIFAR-10 as the standard image classification benchmark and CIFAR-10N as the human-annotated noisy-label dataset, with the AGGRE label set provided by the UCSC-REAL CIFAR-N repository.

Results are primarily reported for a single random seed, subset sampling is not class-stratified, contrastive-init is evaluated strictly as initialization rather than end-to-end contrastive training under noise, and a full noise-rate sweep is proposed as future work.

Project repository: https://github.com/AmerSaad001/Hybrid-Label-Noise-CV

This repository accompanies a course project submission and is intended for academic and educational use.
