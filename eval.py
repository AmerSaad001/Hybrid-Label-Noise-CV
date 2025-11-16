import argparse
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from labelnoise.cifar10 import get_cifar10_loaders
from labelnoise.cifarN import get_cifar10n_loaders
from models.resnet import ResNet18Classifier
from utils import log, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar10n"], default="cifar10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--noise", type=str, default="clean")
    parser.add_argument("--noise_rate", type=float, default=0.0)
    parser.add_argument("--subset", type=str, default="aggre")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save metrics")
    return parser.parse_args()


def build_test_loader(args):
    if args.dataset == "cifar10":
        _, _, test_loader = get_cifar10_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            noise_type=args.noise,
            noise_rate=args.noise_rate,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        num_classes = 10
    else:
        _, _, test_loader = get_cifar10n_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            subset=args.subset,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        num_classes = 10
    return test_loader, num_classes


def evaluate_model(model, loader, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds.append(torch.argmax(logits, dim=1).cpu())
            labels.append(targets.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean().item()
    cm = confusion_matrix(labels, preds)
    return acc, cm, preds, labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    test_loader, num_classes = build_test_loader(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = ResNet18Classifier(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    acc, cm, preds, labels = evaluate_model(model, test_loader, device)

    log(f"Checkpoint: {args.checkpoint}")
    log(f"Accuracy: {acc * 100:.2f}%")
    log(f"Confusion matrix:\n{cm}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(os.path.join(args.save_dir, "confusion_matrix.npy"), cm)
        np.save(os.path.join(args.save_dir, "preds.npy"), preds)
        np.save(os.path.join(args.save_dir, "labels.npy"), labels)
        log(f"Saved evaluation artifacts to {args.save_dir}")


if __name__ == "__main__":
    main()
