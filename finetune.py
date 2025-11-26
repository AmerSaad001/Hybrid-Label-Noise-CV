import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# --- 1. Import our custom modules ---
from model import Encoder # Import the Encoder architecture
from main import Cifar10Noisy # Import the noisy dataset class

# --- 2. Define the new Fine-tuning Model ---
# This model combines our frozen, pre-trained encoder
# with a new, simple linear classifier.
class FinetuneModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(FinetuneModel, self).__init__()
        
        # 1. Load the pre-trained encoder
        self.encoder = encoder
        
        # 2. Freeze the encoder's parameters
        # We do this so the backpropagation only trains
        # the new classifier head.
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 3. Define the new classifier head
        # The ResNet-50 encoder outputs 2048 features
        self.classifier_head = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 1. Get features from the frozen encoder
        # We call .detach() to ensure no gradients
        # flow back into the encoder.
        with torch.no_grad():
            features = self.encoder(x)
        
        # 2. Pass features through the new classifier
        output = self.classifier_head(features)
        return output

# --- This block runs only when you execute `python finetune.py` ---
if __name__ == "__main__":
    
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 50 # Fine-tuning is usually faster
    LR = 1e-3
    
    # Paths
    ENCODER_PATH = 'encoder_pretrained.pth' # The file we saved
    DATA_ROOT = './data'
    NOISY_LABELS_FILE = './data/CIFAR-10_human.pt'

    # --- 2. Load the Pre-trained Encoder ---
    print(f"Loading pre-trained encoder from {ENCODER_PATH}...")
    # First, instantiate the encoder architecture
    base_encoder = Encoder()
    
    # Second, load the saved weights into it
    base_encoder.load_state_dict(torch.load(ENCODER_PATH))
    
    # --- 3. Create the Full Fine-tuning Model ---
    model = FinetuneModel(encoder=base_encoder, num_classes=10).to(device)
    print("Successfully loaded encoder and built fine-tune model.")

    # --- 4. Load the NOISY Dataset ---
    # We use basic transforms for fine-tuning
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

    # --- 5. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    
    # IMPORTANT: We only pass the parameters of the
    # classifier head to the optimizer!
    optimizer = optim.Adam(model.classifier_head.parameters(), lr=LR)

    # --- 6. Training Loop (Example: 1 Epoch) ---
    print("Starting a short test fine-tuning loop (1 epoch)...")
    model.train() # Set model to training mode
    
    # We'll just run 1 epoch to prove it works
    for epoch in range(1): # Normally this would be EPOCHS
        for i, (images, noisy_labels) in enumerate(train_loader):
            
            images = images.to(device)
            noisy_labels = noisy_labels.to(device)
            
            # --- Forward pass ---
            outputs = model(images)
            loss = criterion(outputs, noisy_labels)
            
            # --- Backward pass ---
            # This will ONLY update weights for 'classifier_head'
            # because that's all we gave the optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0: # Print every 100 batches
                print(f"Epoch [1/1], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("\n--- TEST FINE-TUNING SUCCESSFUL! ---")
    print("The classifier head was trained on the noisy labels.")
    print("--------------------------------------")