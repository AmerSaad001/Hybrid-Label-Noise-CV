import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_loss import train_step_hybrid


class DummyModel(nn.Module):
    """Simple model that returns (logits, features)"""
    def __init__(self, in_ch=3, img_size=32, feat_dim=128, num_classes=10):
        super().__init__()
        # small conv encoder -> flatten -> features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, feat_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits, feats


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = 4
    N = 100  # bank size
    D = 128
    C = 10

    model = DummyModel(feat_dim=D, num_classes=C).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # random banks
    feature_bank = torch.randn(N, D)
    pred_bank = torch.randn(N, C)

    # random batch
    images = torch.randn(B, 3, 32, 32)
    noisy_labels = torch.randint(0, C, (B,))
    # sample random indices into the bank for this batch
    indices = torch.randint(0, N, (B,))

    batch = (images, noisy_labels, indices)

    out = train_step_hybrid(
        model,
        optimizer,
        batch,
        feature_bank,
        pred_bank,
        criterion,
        device,
        K=5,
        confidence_thresh=0.8,
        reduce_weight=0.2,
        lambda_nc=1.0,
    )

    print("Run result:")
    for k, v in out.items():
        if k in ("features", "logits", "indices"):
            print(f"{k}:", v.shape)
        else:
            print(f"{k}:", v)


if __name__ == '__main__':
    main()
