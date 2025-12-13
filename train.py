
import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch.serialization

try:
    from numpy.core.multiarray import _reconstruct
    import torch.serialization

    torch.serialization.add_safe_globals({_reconstruct: _reconstruct})

    print("Patched numpy._reconstruct for PyTorch safe loading.")
except Exception as e:
    print("Failed to patch numpy reconstruct:", e)
from utils import (
    set_seed,
    load_config,
    accuracy,
    log,
    AverageMeter,
    save_checkpoint,
)

from labelnoise.cifar10 import get_cifar10_loaders
from labelnoise.cifarN import get_cifar10n_loaders

from models.resnet import ResNet18Classifier
from models.contrastive_init import build_contrastive_resnet
from models.neighbor_consistency import (
    knn_neighbors,
    neighbor_consistency_loss,
    compute_agreement_weights,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Learning Against Label Noise"
    )

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--mode", type=str, default=None,
                        choices=["baseline", "contrastive", "structural", "hybrid"])
    parser.add_argument("--model", type=str, default=None,
                        choices=["baseline", "contrastive", "structural", "hybrid"])
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "cifar10n"])
    parser.add_argument("--noise", type=str, default=None,
                        choices=["clean", "symmetric", "asymmetric"])
    parser.add_argument("--noise_rate", type=float, default=None)

    parser.add_argument("--cifarn_subset", type=str, default=None,
                        choices=["aggre", "worst", "random1", "random2", "random3"])
    parser.add_argument("--subset", type=str, default=None,
                        choices=["clean", "aggre", "worst", "random1", "random2", "random3"])

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    parser.add_argument("--simclr_ckpt", type=str, default=None)
    parser.add_argument("--contrastive_schedule", type=str, default=None,
                        choices=["frozen-warm", "low-lr"])
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--backbone_lr_factor", type=float, default=None)

    parser.add_argument("--use_cosine_lr", action="store_true")

    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--load_pretrained", type=str, default=None)

    parser.add_argument("--subset_fraction", type=float, default=None)

    return parser.parse_args()

def build_config(args):
    cfg = {
        "mode": "hybrid",
        "dataset": "cifar10",
        "subset": "clean",
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
        "subset_fraction": 1.0,
        "load_pretrained": None,
    }

    if args.config is not None:
        cfg.update(load_config(args.config))

    override_keys = [
        "mode", "model", "dataset", "subset", "noise", "noise_rate",
        "cifarn_subset", "seed", "epochs", "batch_size", "lr",
        "K", "lambda_consistency", "confidence_threshold",
        "simclr_ckpt", "contrastive_schedule",
        "warmup_epochs", "backbone_lr_factor",
        "subset_fraction", "load_pretrained"
    ]

    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if cfg.get("model") is not None:
        cfg["mode"] = cfg["model"]

    if args.use_cosine_lr:
        cfg["use_cosine_lr"] = True

    return cfg

def build_model_and_optim(cfg, device):
    mode = cfg["mode"]

    if mode in ["baseline", "structural"]:
        net = ResNet18Classifier(num_classes=10)
    elif mode in ["contrastive", "hybrid"]:
        net = build_contrastive_resnet(
            num_classes=10,
            simclr_ckpt=cfg.get("simclr_ckpt"),
            freeze_backbone=False
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    net = net.to(device)

    base_lr = cfg["lr"]

    optimizer = optim.SGD(
        net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4
    )

    scheduler = None
    if cfg.get("use_cosine_lr"):
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    return net, optimizer, scheduler

def train_one_epoch(epoch, model, loader, optimizer, device, cfg, weights=None):
    model.train()

    K = cfg["K"]
    lam = cfg["lambda_consistency"]
    conf_th = cfg["confidence_threshold"]
    mode = cfg["mode"]

    ce_meter = AverageMeter()
    cons_meter = AverageMeter()
    acc_meter = AverageMeter()

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    for batch in loader:
        if len(batch) == 2:
            imgs, labels = batch
        else:
            imgs, labels, _ = batch

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)
        ce_per_sample = ce_loss_fn(logits, labels)

        sample_weights = torch.ones_like(ce_per_sample, device=device)
        cons_loss = torch.tensor(0.0, device=device)

        if mode in ["structural", "hybrid"]:
            if hasattr(model, "extract_features"):
                feats = model.extract_features(imgs)
            else:
                feats = logits.detach()

            neigh = knn_neighbors(feats, K=K)
            cons_loss = neighbor_consistency_loss(logits, neigh)

            if mode == "hybrid":
                w = compute_agreement_weights(
                    logits.detach(), labels, neigh, confidence_threshold=conf_th
                )
                sample_weights *= w

        ce_loss = (ce_per_sample * sample_weights).mean()
        loss = ce_loss + lam * cons_loss

        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)

        ce_meter.update(ce_loss.item(), imgs.size(0))
        cons_meter.update(cons_loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))

    log(f"[Train] Epoch {epoch} | CE={ce_meter.avg:.4f} Cons={cons_meter.avg:.4f} Acc={acc_meter.avg*100:.2f}%")

    return ce_meter.avg, acc_meter.avg

@torch.no_grad()
def evaluate(model, loader, device, split="Val"):
    model.eval()

    ce_meter = AverageMeter()
    acc_meter = AverageMeter()

    crit = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss = crit(logits, labels)
        acc = accuracy(logits, labels)

        ce_meter.update(loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))

    log(f"[{split}] CE={ce_meter.avg:.4f} Acc={acc_meter.avg*100:.2f}%")

    return ce_meter.avg, acc_meter.avg

def main():
    args = get_args()
    cfg = build_config(args)

    set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    import torch

    if torch.backends.mps.is_available():
        device = "mps"
        print("[LOG] Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("[LOG] Using device: CUDA GPU")
    else:
        device = "cpu"
        print("[LOG] Using device: CPU")

    log(f"[LOG] Using device: {device}")

    dataset_name = cfg["dataset"]
    subset_name = cfg.get("subset", "clean")

    if dataset_name == "cifar10n":
        if args.subset is not None:
            subset_name = args.subset
        else:
            subset_name = cfg.get("cifarn_subset")
    else:
        if subset_name == "clean":
            cfg["noise"] = "clean"

    os.makedirs("data", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, cfg["mode"], subset_name, time_str)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config_path = os.path.join(out_dir, "config.yaml")
    log(f"Saving run to: {out_dir}")

    data_root = "data"

    if dataset_name == "cifar10n":
        train_loader, val_loader, test_loader = get_cifar10n_loaders(
            root=data_root,
            batch_size=cfg["batch_size"],
            noise_type=subset_name
        )
    elif dataset_name == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            data_root=data_root,
            batch_size=cfg["batch_size"],
            noise_type=cfg["noise"],
            noise_rate=cfg["noise_rate"],
            seed=cfg["seed"]
        )

    def make_loader(ds, shuffle, base):
        return DataLoader(
            ds,
            batch_size=base.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=base.drop_last
        )

    frac = cfg["subset_fraction"]
    if frac < 1.0:
        torch.manual_seed(cfg["seed"])
        n = len(train_loader.dataset)
        idx = torch.randperm(n)[: int(n * frac)]
        train_loader = make_loader(Subset(train_loader.dataset, idx), True, train_loader)

        if val_loader is not None:
            n = len(val_loader.dataset)
            idx = torch.randperm(n)[: int(n * frac)]
            val_loader = make_loader(Subset(val_loader.dataset, idx), False, val_loader)
    else:
        train_loader = make_loader(train_loader.dataset, True, train_loader)
        val_loader = make_loader(val_loader.dataset, False, val_loader)
        test_loader = make_loader(test_loader.dataset, False, test_loader)

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    model, optimizer, scheduler = build_model_and_optim(cfg, device)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_ce", "train_acc", "val_ce", "val_acc"])

    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):

        train_ce, train_acc = train_one_epoch(epoch, model, train_loader, optimizer, device, cfg)

        if val_loader is not None:
            val_ce, val_acc = evaluate(model, val_loader, device, split="Val")
        else:
            val_ce, val_acc = 0.0, train_acc

        if scheduler is not None:
            scheduler.step()

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_ce, train_acc, val_ce, val_acc])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, "best.pth")
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "best_val_acc": best_val_acc},
                ckpt_path
            )
            log(f"[Checkpoint] New best: {best_val_acc*100:.2f}%")

    log(f"Training complete. Best Val Acc = {best_val_acc*100:.2f}%")

    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    if os.path.isfile(best_ckpt):
        log("Loading best checkpoint for final test…")
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        _, test_acc = evaluate(model, test_loader, device, split="Test")
        log(f"[FINAL TEST] Acc = {test_acc*100:.2f}%")
    else:
        log("No checkpoint found — skipping final test.")


if __name__ == "__main__":
    main()
