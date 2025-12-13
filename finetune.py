import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Encoder
from main import Cifar10Noisy

class FinetuneModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(FinetuneModel, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head = nn.Linear(2048, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        output = self.classifier_head(features)
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-3
    
    ENCODER_PATH = 'encoder_pretrained.pth'
    DATA_ROOT = './data'
    NOISY_LABELS_FILE = './data/CIFAR-10_human.pt'

    print(f"Loading pre-trained encoder from {ENCODER_PATH}...")
    base_encoder = Encoder()
    base_encoder.load_state_dict(torch.load(ENCODER_PATH))
    
    model = FinetuneModel(encoder=base_encoder, num_classes=10).to(device)
    print("Successfully loaded encoder and built fine-tune model.")

    transform_finetune = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    noisy_train_dataset = Cifar10Noisy(
        root=DATA_ROOT, 
        train=True, 
        transform=transform_finetune,
        noisy_labels_path=NOISY_LABELS_FILE
    )
    
    train_loader = DataLoader(
        noisy_train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    print("Successfully loaded noisy training data.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier_head.parameters(), lr=LR)

    print("Starting a short test fine-tuning loop (1 epoch)...")
    model.train()
    for epoch in range(1):
        for i, (images, noisy_labels) in enumerate(train_loader):
            
            images = images.to(device)
            noisy_labels = noisy_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, noisy_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [1/1], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("\n--- TEST FINE-TUNING SUCCESSFUL! ---")
    print("The classifier head was trained on the noisy labels.")
    print("--------------------------------------")
