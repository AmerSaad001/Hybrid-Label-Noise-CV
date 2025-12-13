#!/usr/bin/env python3

import argparse
import os
import random
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_repo_loaders():
    get_cifar10 = None
    get_cifar10n = None
    try:
        from labelnoise.cifar10 import get_cifar10_loaders
        get_cifar10 = get_cifar10_loaders
    except Exception:
        pass
    try:
        from labelnoise.cifarN import get_cifar10n_loaders
        get_cifar10n = get_cifar10n_loaders
    except Exception:
        pass
    return get_cifar10, get_cifar10n


def torchvision_cifar10_loader(
    data_root: str,
    split: str,
    seed: int,
    batch_size: int,
    subset_fraction: float,
    num_workers: int = 0,
) -> DataLoader:
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2430, 0.2610)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if split in ("train", "val"):
        base = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tfm)
        n = len(base)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:5000]
        train_idx = idx[5000:]

        if split == "val":
            chosen = val_idx
        else:
            chosen = train_idx

        if subset_fraction < 1.0:
            k = max(1, int(len(chosen) * subset_fraction))
            chosen = chosen[:k]

        ds = Subset(base, chosen.tolist())
    else:
        base = datasets.CIFAR10(root=data_root, train=False, download=False, transform=tfm)
        idx = np.arange(len(base))
        if subset_fraction < 1.0:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            k = max(1, int(len(idx) * subset_fraction))
            idx = idx[:k]
        ds = Subset(base, idx.tolist())

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)


def build_loader(
    dataset: str,
    data_root: str,
    split: str,
    seed: int,
    batch_size: int,
    subset_fraction: float,
    noise: str,
    subset: str,
) -> DataLoader:
    get_cifar10, get_cifar10n = try_repo_loaders()

    if dataset.lower() == "cifar10":
        if get_cifar10 is not None:
            try:
                loaders = get_cifar10(data_root=data_root, noise_type=noise, seed=seed, batch_size=batch_size)
            except TypeError:
                loaders = get_cifar10(data_root, noise, seed, batch_size)
            if isinstance(loaders, dict):
                if split == "train":
                    loader = loaders.get("train") or loaders.get("train_loader")
                elif split == "val":
                    loader = loaders.get("val") or loaders.get("val_loader")
                else:
                    loader = loaders.get("test") or loaders.get("test_loader")
                if loader is None:
                    raise RuntimeError("Repo CIFAR-10 loader returned dict but missing expected keys.")
                return loader
            else:
                if split == "train":
                    return loaders[0]
                if split == "val":
                    return loaders[1]
                return loaders[2]

        return torchvision_cifar10_loader(
            data_root=data_root,
            split=split,
            seed=seed,
            batch_size=batch_size,
            subset_fraction=subset_fraction,
        )

    if dataset.lower() == "cifar10n":
        if get_cifar10n is not None:
            try:
                loaders = get_cifar10n(data_root=data_root, subset=subset, seed=seed, batch_size=batch_size)
            except TypeError:
                loaders = get_cifar10n(data_root, subset, seed, batch_size)

            if isinstance(loaders, dict):
                key = "test" if split in ("test", "val") else "train"
                loader = loaders.get(key) or loaders.get(f"{key}_loader")
                if loader is None:
                    raise RuntimeError("Repo CIFAR-10N loader returned dict but missing expected keys.")
                return loader
            else:
                if split in ("test", "val"):
                    return loaders[1]
                return loaders[0]

        return torchvision_cifar10_loader(
            data_root=data_root,
            split="test",
            seed=seed,
            batch_size=batch_size,
            subset_fraction=subset_fraction,
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def build_model(mode: str) -> nn.Module:
    mode = mode.lower()
    if mode in ("baseline", "structural"):
        from models.resnet import ResNet18Classifier
        return ResNet18Classifier(num_classes=10)

    if mode in ("contrastive", "hybrid"):
        try:
            from models.contrastive_init import build_contrastive_resnet
            return build_contrastive_resnet(num_classes=10)
        except Exception:
            from models.resnet import ResNet18Classifier
            return ResNet18Classifier(num_classes=10)

    from models.resnet import ResNet18Classifier
    return ResNet18Classifier(num_classes=10)


def load_checkpoint_weights(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys (ok if head differs): {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")


def find_avgpool_module(model: nn.Module) -> Optional[nn.Module]:
    candidates = [
        "backbone.avgpool",
        "avgpool",
        "encoder.avgpool",
        "net.avgpool",
    ]

    for path in candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.Module):
            return cur

    for name, m in model.named_modules():
        if name.endswith("avgpool"):
            return m
    return None


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    feats_list = []
    labels_list = []

    avgpool = find_avgpool_module(model)
    if avgpool is None:
        raise RuntimeError(
            "Couldn't find avgpool module to hook. "
            "Tell me your model class structure (or share models/resnet.py forward), and I’ll adjust the hook."
        )

    buf = {"feat": None}

    def hook_fn(_module, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach()
        if out.ndim == 4:
            out = torch.flatten(out, 1)
        buf["feat"] = out

    handle = avgpool.register_forward_hook(hook_fn)

    seen = 0
    for x, y in loader:
        x = x.to(device)
        _ = model(x)

        f = buf["feat"]
        if f is None:
            raise RuntimeError("Hook did not capture features. Hooked the wrong module.")
        f = f.cpu().numpy()
        y = y.cpu().numpy()

        feats_list.append(f)
        labels_list.append(y)

        seen += f.shape[0]
        if seen >= max_samples:
            break

    handle.remove()

    X = np.concatenate(feats_list, axis=0)[:max_samples]
    Y = np.concatenate(labels_list, axis=0)[:max_samples]
    return X, Y


def run_umap(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap
    except Exception as e:
        raise RuntimeError("UMAP not installed. Run: pip install umap-learn") from e

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def scatter_plot(Z: np.ndarray, y: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[ok] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pth checkpoint")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar10n"])
    ap.add_argument("--mode", default="structural", help="baseline|structural|hybrid|contrastive (used to build model)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--subset_fraction", type=float, default=1.0, help="Optional subsample fraction for faster plotting")
    ap.add_argument("--noise", default="clean", help="For CIFAR-10 repo loader: clean/symmetric/asymmetric (if supported)")
    ap.add_argument("--subset", default="aggre", help="For CIFAR-10N repo loader: aggre/worse/worst/random1/random2/clean")
    ap.add_argument("--n_samples", type=int, default=2000, help="Max samples to embed (TSNE/UMAP gets slow above ~5k)")
    ap.add_argument("--pca_dim", type=int, default=50, help="PCA dims before TSNE/UMAP (speed + denoise)")
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--out", required=True, help="Output image path (.png)")
    args = ap.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    loader = build_loader(
        dataset=args.dataset,
        data_root=args.data_root,
        split=args.split,
        seed=args.seed,
        batch_size=args.batch_size,
        subset_fraction=args.subset_fraction,
        noise=args.noise,
        subset=args.subset,
    )

    model = build_model(args.mode)
    load_checkpoint_weights(model, args.ckpt)

    X, y = extract_features(
        model=model,
        loader=loader,
        device=args.device,
        max_samples=args.n_samples,
    )

    pca_dim = min(args.pca_dim, X.shape[1], X.shape[0])
    Xp = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X)

    if args.method == "tsne":
        Z = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(Xp)
        title = f"t-SNE (penultimate) | {args.dataset} {args.split} | mode={args.mode}"
    else:
        Z = run_umap(Xp, seed=args.seed, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
        title = f"UMAP (penultimate) | {args.dataset} {args.split} | mode={args.mode}"

    scatter_plot(Z, y, title=title, out_path=args.out)


if __name__ == "__main__":
    main()
