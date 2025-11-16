# Hybrid Learning Against Label Noise

This repository implements the Hybrid Learning Against Label Noise (HLALN) recipe that mixes four ingredients:

1. **Baseline supervised classifier** (ResNet-18 tailored for CIFAR).
2. **Contrastive pretraining** with SimCLR-style augmentations.
3. **Structural consistency** enforced with a KNN memory bank.
4. **Hybrid agreement-aware weighting** to down-weight uncertain labels.

The codebase supports CIFAR-10 with synthetic noise (symmetric/asymmetric) and CIFAR-10N real noise variants such as `aggre`, `worst`, and the `random*` splits.

## Project structure

```
labelnoise_hybrid/
├── configs/                # Ready-to-use experiment presets
├── data/                   # CIFAR data + CIFAR-10N labels (downloaded automatically)
├── labelnoise/             # Dataset + noise helpers
├── models/                 # ResNet, contrastive pretrainer, neighbor bank
├── outputs/                # Checkpoints, logs, plots
├── train.py                # Main training entrypoint
├── eval.py                 # Checkpoint evaluation (accuracy + confusion matrix)
├── plots.py                # Aggregate metrics + embedding visualizations
└── utils.py                # Common utilities (seed, logging, metrics)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Example: baseline clean CIFAR-10 for a single epoch (quick smoke test):

```bash
python3 train.py --epochs 1 --dataset cifar10 --noise clean --mode baseline
```

Key arguments:

- `--dataset {cifar10,cifar10n}` and `--noise {clean,symmetric,asymmetric}` (noise rate via `--noise_rate`).
- `--mode {baseline,contrastive,structural,hybrid}` toggles the learning recipe.
- `--config configs/hybrid.yaml` loads a YAML preset (CLI overrides still apply).
- `--contrastive_checkpoint path/to/backbone.pt` reuses a pre-trained encoder.

During training the script writes `outputs/<exp_name>/history.csv`, `best.ckpt`, `last.ckpt`, and (for contrastive modes) `contrastive_backbone.pt`.

## Evaluation

Evaluate a checkpoint and dump predictions/confusion matrix:

```bash
python3 eval.py --checkpoint outputs/baseline_clean/best.ckpt --dataset cifar10 --noise clean
```

Use `--save_dir` to persist `confusion_matrix.npy`, `preds.npy`, and `labels.npy`.

## Plotting + embeddings

Combine multiple history files (or a summary CSV) and optionally visualize embeddings:

```bash
python3 plots.py --history_files outputs/*/history.csv \
                 --checkpoint outputs/hybrid_run/best.ckpt \
                 --dataset cifar10n --subset aggre --split test
```

Generated figures (saved in `outputs/plots/` by default):

- `accuracy_vs_noise.png`: accuracy curves per mode vs noise rate.
- `mode_bar.png`: bar chart comparing modes.
- `tsne.png` and `umap.png`: 2-D embeddings (UMAP requires `umap-learn`).

## Expected outputs

- **Training**: Console logs every `log_interval` steps, plus CSV metrics and checkpoints.
- **Evaluation**: Overall accuracy + 10×10 confusion matrix in stdout (and `.npy` files if requested).
- **Plots**: PNG figures summarizing accuracy trends and embedding spaces.

## Data requirements

- CIFAR-10 downloads automatically (`torchvision.datasets`).
- CIFAR-10N labels must exist under `data/CIFAR-10N/*.npy` (see official repo). The scripts detect missing files and raise a clear error.

## Notes

- The ResNet-18 backbone automatically adapts to 32×32 inputs and falls back to random init if ImageNet weights cannot be retrieved (e.g., offline).
- Contrastive pretraining uses SimCLR-style InfoNCE with temperature scheduling, storing the resulting backbone for reuse.
- Structural regularization keeps a KNN memory bank (top-k cosine neighbors) and supports agreement-aware weighting via `agreement_min_weight` / `agreement_max_weight`.
- Warm starts: `--warmup_epochs` keeps a low LR (`--warmup_lr`) before switching to the main LR, and `--freeze_backbone_epochs` freezes the encoder for the requested number of epochs.
# Hybrid-Label-Noise-CV
