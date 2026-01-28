Hybrid Learning Against Label Noise

📄 Paper: [Hybrid Learning against Label Noise (PDF)](paper/Hybrid_Learning_Against_Label_Noise.pdf)

This repository trains image classifiers on noisy labels using four variants: a supervised baseline, a structural method with neighbor consistency, a hybrid method combining consistency and agreement-aware weighting, and a contrastive-init option that can reuse SimCLR/ImageNet-initialized backbones. Models use ResNet-18 adapted for CIFAR.

Methods
- Baseline
- Structural
- Hybrid
- Contrastive-init

Datasets
- CIFAR-10 (clean or synthetic symmetric/asymmetric noise)
- CIFAR-10N (AGGRE), UCSC-REAL

Train
- CIFAR-10 clean baseline (example): `python3 train.py --dataset cifar10 --mode baseline --noise clean`
- CIFAR-10N AGGRE hybrid (example): `python3 train.py --dataset cifar10n --subset aggre --mode hybrid`
- Key options: `--config <yaml>` to load a preset, `--batch_size`, `--epochs`, `--lr`, `--subset_fraction` (e.g., 0.3 for 30%).

Evaluate
- `python3 eval.py --checkpoint <path> --dataset {cifar10|cifar10n} --subset aggre --noise clean`
- Use `--save_dir` to write confusion matrix, predictions, and labels.

Tools
- `tools/plot_learning_curves.py` and `tools/make_report_artifacts.py` generate learning-curve figures and tables.
- `tools/plot_embeddings.py` plots UMAP/t-SNE embeddings from checkpoints (UMAP requires `umap-learn`).

Repository Structure
```text
.
├── configs/
├── labelnoise/
├── models/
├── paper/
├── tools/
├── train.py
├── eval.py
├── plots.py
├── requirements.txt
└── README.md
```
