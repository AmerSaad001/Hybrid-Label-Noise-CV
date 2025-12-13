import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Encoder, ProjectionHead, SimCLR_Model

class ContrastiveTransformations:
    def __init__(self, size, s=1.0, p_grayscale=0.2):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            ], p=0.8),
            transforms.RandomGrayscale(p=p_grayscale),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        labels = torch.cat((
            torch.arange(self.batch_size, 2 * self.batch_size),
            torch.arange(self.batch_size)
        )).to(self.device)

        mask = torch.eye(2 * self.batch_size, dtype=torch.bool).to(self.device)
        sim[mask] = -5e4
        loss = self.criterion(sim, labels)
        return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 3e-4
    TEMPERATURE = 0.5
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=ContrastiveTransformations(size=32)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    encoder = Encoder()
    projector = ProjectionHead(in_features=2048, hidden_dim=512, out_features=128)
    simclr_model = SimCLR_Model(encoder, projector).to(device)
    criterion = NTXentLoss(batch_size=BATCH_SIZE, temperature=TEMPERATURE, device=device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=LR, weight_decay=1e-6)

    print("Starting a short test pretraining loop...")
    simclr_model.train()
    for step, (views, _) in enumerate(train_loader):
        view1, view2 = views[0].to(device), views[1].to(device)
        _, z1 = simclr_model(view1)
        _, z2 = simclr_model(view2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"Step [1/10], Loss: {loss.item():.4f}")
        
        if step >= 9:
            break
            
    print("\n--- TEST PRETRAINING SUCCESSFUL! ---")
    print("The model is training with the contrastive loss.")
    encoder_path = 'encoder_pretrained.pth'
    torch.save(simclr_model.encoder.state_dict(), encoder_path)
    print(f"Successfully saved a test encoder to {encoder_path}")
    print("--------------------------------------")
