import os
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import (
    set_seed,
    load_config,
    accuracy,
    log,
    AverageMeter,
    save_checkpoint,
)

# datasets / noise
from labelnoise.cifar10 import get_cifar10_loaders
from labelnoise.cifarN import get_cifar10n_loaders

# models
from models.resnet import ResNet18Classifier
from models.contrastive_init import build_contrastive_resnet
from models.neighbor_consistency import (
    knn_neighbors,
    neighbor_consistency_loss,
    compute_agreement_weights,
)

# ------------------------------------------------------
# Arg parser
# ------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Learning against Label Noise (CIFAR-10 / CIFAR-10N)"
    )

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (optional)")

    # basic stuff
    parser.add_argument("--mode", type=str, default=None,
                        choices=["baseline", "contrastive", "structural", "hybrid"],
                        help="Training mode (if set, overrides YAML)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "cifar10n"],
                        help="cifar10 = synthetic noise, cifar10n = human noise")
    parser.add_argument("--noise", type=str, default=None,
                        choices=["clean", "symmetric", "asymmetric"],
                        help="Only used for CIFAR-10 synthetic noise")
    parser.add_argument("--noise_rate", type=float, default=None,
                        help="e.g. 0.2 for 20% symmetric noise")

    parser.add_argument("--cifarn_subset", type=str, default=None,
                        choices=["aggre", "worst", "random1", "random2", "random3"],
                        help="Which CIFAR-10N label set to use")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    # structural / hybrid bits
    parser.add_argument("--K", type=int, default=None, help="K for KNN neighbors")
    parser.add_argument("--lambda_consistency", type=float, default=None,
                        help="weight for consistency loss")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="agreement weighting threshold (for hybrid)")

    # contrastive init options
    parser.add_argument("--simclr_ckpt", type=str, default=None,
                        help="SimCLR-style checkpoint (if we have one)")
    parser.add_argument("--contrastive_schedule", type=str, default=None,
                        choices=["frozen-warm", "low-lr"],
                        help="How to fine-tune after contrastive pretrain")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="for frozen-warm, how long to freeze backbone")
    parser.add_argument("--backbone_lr_factor", type=float, default=None,
                        help="for low-lr schedule, scale for backbone LR")

    parser.add_argument("--use_cosine_lr", action="store_true",
                        help="turn on cosine LR schedule")

    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="where to drop logs/checkpoints")

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from"
    )

    return parser.parse_args()


# ------------------------------------------------------
# Config: YAML + CLI overrides
# ------------------------------------------------------
def build_config(args):
    # default baseline-ish config
    cfg = {
        "mode": "hybrid",
        "dataset": "cifar10",
        "noise": "symmetric",
        "noise_rate": 0.2,
        "cifarn_subset": "aggre",
        "seed": 42,
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.1,
        "K": 5,
        "lambda_consistency": 0.3,
        "confidence_threshold": 0.6,
        "simclr_ckpt": None,
        "contrastive_schedule": "frozen-warm",
        "warmup_epochs": 10,
        "backbone_lr_factor": 0.1,
        "use_cosine_lr": False,
    }

    # load YAML if user gave one
    if args.config is not None:
        yaml_cfg = load_config(args.config)
        cfg.update(yaml_cfg)

    # manual CLI overrides (I just overwrite if not None)
    override_keys = [
        "mode", "dataset", "noise", "noise_rate", "cifarn_subset",
        "seed", "epochs", "batch_size", "lr",
        "K", "lambda_consistency", "confidence_threshold",
        "simclr_ckpt", "contrastive_schedule",
        "warmup_epochs", "backbone_lr_factor",
    ]

    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    # flag for cosine
    if args.use_cosine_lr:
        cfg["use_cosine_lr"] = True

    return cfg


# ------------------------------------------------------
# Model + optimizer (and optional scheduler)
# ------------------------------------------------------
def build_model_and_optim(cfg, device):
    mode = cfg["mode"]

    # choose base model depending on mode
    if mode in ["baseline", "structural"]:
        log(f"Building plain ResNet18 classifier (mode={mode})")
        net = ResNet18Classifier(
            num_classes=10,
            pretrained=False,
            freeze_backbone=False,
        )
    elif mode in ["contrastive", "hybrid"]:
        log(f"Building contrastive-initialized ResNet18 (mode={mode})")
        net = build_contrastive_resnet(
            num_classes=10,
            simclr_ckpt=cfg.get("simclr_ckpt", None),
            freeze_backbone=False,  # I'll handle freezing outside
        )
    else:
        raise ValueError(f"Unknown mode = {mode}")

    net = net.to(device)

    # default optimizer parameters
    params = net.parameters()
    base_lr = cfg["lr"]

    # if using low-lr schedule, we separate backbone and head LR
    if mode in ["contrastive", "hybrid"] and cfg.get("contrastive_schedule") == "low-lr":
        backbone_params, head_params = [], []
        for name, p in net.named_parameters():
            if "fc" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

        factor = cfg.get("backbone_lr_factor", 0.1)
        optimizer = optim.SGD(
            [
                {"params": backbone_params, "lr": base_lr * factor},
                {"params": head_params, "lr": base_lr},
            ],
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        # normal single-LR setup
        optimizer = optim.SGD(
            params,
            lr=base_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    # cosine schedule is optional (I use it only if flag is set)
    if cfg.get("use_cosine_lr", False):
        sched = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    else:
        sched = None

    return net, optimizer, sched


# ------------------------------------------------------
# One training epoch
# ------------------------------------------------------
def train_one_epoch(epoch, model, train_loader, optimizer, device, cfg, weights=None):
    model.train()

    mode = cfg["mode"]
    K = cfg["K"]
    lam_cons = cfg["lambda_consistency"]
    conf_th = cfg["confidence_threshold"]

    ce_meter = AverageMeter()
    cons_meter = AverageMeter()
    acc_meter = AverageMeter()

    # keep CE per sample so we can re-weight it
    ce_crit = nn.CrossEntropyLoss(reduction="none")

    for batch in train_loader:
        if len(batch) == 2:
            imgs, labels = batch
        elif len(batch) == 3:
            imgs, labels, indices = batch
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)}")

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)
        batch_size = logits.size(0)

        # cross-entropy for each sample
        ce_per_sample = ce_crit(logits, labels)
        # default weights = 1 (no agreement weighting) unless provided
        if weights is None:
            sample_weights = torch.ones_like(ce_per_sample, device=device)
        else:
            sample_weights = torch.as_tensor(weights, device=device, dtype=ce_per_sample.dtype)

        # structural / hybrid extra loss
        cons_loss = torch.tensor(0.0, device=device)

        if mode in ["structural", "hybrid"]:
            # get some representation for neighbor search
            if hasattr(model, "extract_features"):
                feats = model.extract_features(imgs)
            else:
                # worst case, reuse logits (not ideal but works)
                feats = logits.detach()

            # KNN indexes in the batch
            neighbor_idx = knn_neighbors(feats, K=K)
            cons_loss = neighbor_consistency_loss(logits, neighbor_idx)

            # agreement-aware weighting only for hybrid
            if mode == "hybrid" and conf_th is not None and conf_th > 0:
                agreement_weights = compute_agreement_weights(
                    logits.detach(), labels, neighbor_idx,
                    confidence_threshold=conf_th,
                )
                sample_weights = sample_weights * agreement_weights

        # weighted CE
        ce_loss = (ce_per_sample * sample_weights).mean()
        total_loss = ce_loss + lam_cons * cons_loss

        total_loss.backward()
        optimizer.step()

        # quick metrics
        acc = accuracy(logits, labels)
        ce_meter.update(ce_loss.item(), batch_size)
        cons_meter.update(cons_loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

    log(
        f"[Train] Epoch {epoch} | CE={ce_meter.avg:.4f} "
        f"Cons={cons_meter.avg:.4f} Acc={acc_meter.avg*100:.2f}%"
    )

    return ce_meter.avg, acc_meter.avg


# ------------------------------------------------------
# Eval on val / test
# ------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, split_name="Val"):
    model.eval()

    crit = nn.CrossEntropyLoss(reduction="mean")
    ce_meter = AverageMeter()
    acc_meter = AverageMeter()

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = crit(logits, labels)
        acc = accuracy(logits, labels)

        bs = labels.size(0)
        ce_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

    log(f"[{split_name}] CE={ce_meter.avg:.4f} Acc={acc_meter.avg*100:.2f}%")
    return ce_meter.avg, acc_meter.avg


# ------------------------------------------------------
# Main script
# ------------------------------------------------------
def main():
    args = get_args()
    cfg = build_config(args)

    # ---- setup random seed / device ----
    seed = cfg["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    mode = cfg["mode"]
    dataset_name = cfg["dataset"]

    # run name just so I don't overwrite files
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{mode}_{dataset_name}_seed{seed}_{time_str}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log(f"Saving stuff under: {out_dir}")

    # ---- dataset loaders ----
    if dataset_name == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            data_root="./data",
            batch_size=cfg["batch_size"],
            noise_type=cfg["noise"],
            noise_rate=cfg["noise_rate"],
            seed=seed,
        )
    elif dataset_name == "cifar10n":
        train_loader, val_loader, test_loader = get_cifar10n_loaders(
            data_root="./data",
            batch_size=cfg["batch_size"],
            subset=cfg["cifarn_subset"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset = {dataset_name}")

    # ---- model / optimizer / scheduler ----
    model, optimizer, scheduler = build_model_and_optim(cfg, device)

    # contrastive schedule (if we actually use it)
    frozen_warm = (
        mode in ["contrastive", "hybrid"]
        and cfg.get("contrastive_schedule") == "frozen-warm"
    )
    warmup_epochs = cfg.get("warmup_epochs", 0)

    # initially freeze backbone if using frozen-warm
    if frozen_warm:
        log(f"[Schedule] Frozen-warm: freezing backbone for {warmup_epochs} epochs.")
        for name, p in model.named_parameters():
            if "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # re-create optimizer only for head params
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            head_params,
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=5e-4,
        )

    # ---- resume from checkpoint (optional) ----
    start_epoch = 1
    best_val_acc = 0.0

    if args.resume_ckpt is not None and os.path.isfile(args.resume_ckpt):
        log(f"Resuming from checkpoint: {args.resume_ckpt}")
        state = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])

        if "epoch" in state:
            start_epoch = state["epoch"] + 1
        if "best_val_acc" in state:
            best_val_acc = state["best_val_acc"]

    # if warmup already done, make sure backbone is unfrozen and optimizer covers all params
    if frozen_warm and start_epoch > warmup_epochs + 1:
        log("[Schedule] Warmup already completed in checkpoint, unfreezing backbone.")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=5e-4,
        )

    # ---- training loop ----
    metrics_path = os.path.join(out_dir, "metrics.csv")

    # little CSV log, easier to plot later
    if start_epoch == 1 or not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_ce",
                "train_acc",
                "val_ce",
                "val_acc",
            ])
    else:
        log(f"Resuming run, appending to existing metrics file: {metrics_path}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # unfreeze after warmup if using frozen-warm
        if frozen_warm and epoch == warmup_epochs + 1:
            log("[Schedule] Warmup done, unfreezing backbone.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg["lr"],
                momentum=0.9,
                weight_decay=5e-4,
            )

        train_ce, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, device, cfg
        )

        val_ce, val_acc = (0.0, 0.0)
        if val_loader is not None:
            val_ce, val_acc = evaluate(model, val_loader, device, split_name="Val")

        # step LR scheduler if we enabled it
        if scheduler is not None:
            scheduler.step()

        # append row to CSV
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_ce,
                train_acc,
                val_ce,
                val_acc,
            ])

        # use val_acc if available, otherwise fallback to train_acc
        metric_acc = val_acc if val_loader is not None else train_acc
        if metric_acc >= best_val_acc:
            best_val_acc = metric_acc
            ckpt_path = os.path.join(ckpt_dir, "best.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "best_val_acc": best_val_acc,
                },
                ckpt_path,
            )
            log(f"[Checkpoint] new best at epoch {epoch} (acc={metric_acc*100:.2f}%)")

    log(f"Training done. Best val acc = {best_val_acc*100:.2f}%")

    # ---- final test on best checkpoint ----
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    if os.path.isfile(best_ckpt):
        log("Loading best checkpoint for final test run...")
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        _, test_acc = evaluate(model, test_loader, device, split_name="Test")
        log(f"[FINAL TEST] Acc = {test_acc*100:.2f}%")
    else:
        log("No best checkpoint found, so I skipped the final test eval.")


if __name__ == "__main__":
    main()
