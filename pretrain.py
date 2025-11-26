import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. Import your model from model.py ---
from model import Encoder, ProjectionHead, SimCLR_Model

# --- 2. Define the Augmentations for SimCLR ---
# SimCLR needs two different, heavily augmented "views" of the same image
class ContrastiveTransformations:
    def __init__(self, size, s=1.0, p_grayscale=0.2):
        # Base transform with heavy augmentations
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
        # Return two different augmented versions of the same image
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

# --- 3. Define the NT-Xent Loss Function ---
# This is the "contrastive" loss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
        # We need this to compute similarity
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # This will compute the cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i and z_j are the two augmented "views" of the projections.
        Shape: [batch_size, projection_dim]
        """
        # 1. Concatenate all projections: (2*N, D)
        z = torch.cat((z_i, z_j), dim=0)
        
        # 2. Calculate cosine similarity matrix: (2*N, 2*N)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # 3. Create labels for positive pairs
        # The positive pair for z_i[k] is z_j[k], which is at index k + batch_size
        # The positive pair for z_j[k] is z_i[k], which is at index k
        
        # Create labels [N, N+1, ..., 2N-1] and [0, 1, ..., N-1]
        labels = torch.cat((
            torch.arange(self.batch_size, 2 * self.batch_size),
            torch.arange(self.batch_size)
        )).to(self.device)

        # 4. Create a mask to remove self-comparisons (diagonal elements)
        # We want to compare each image only with others, not itself
        mask = torch.eye(2 * self.batch_size, dtype=torch.bool).to(self.device)
        # Set diagonal (self-comparisons) to a very small number so they don't count
        sim[mask] = -5e4
        
        # 5. Calculate the loss
        # sim is (2N, 2N), labels is (2N)
        # This calculates cross-entropy for each row of 'sim'
        # The 'labels' tell it which *column* is the correct "positive" match
        loss = self.criterion(sim, labels)
        
        return loss

# --- This block runs only when you execute `python pretrain.py` ---
if __name__ == "__main__":
    
    # --- 1. Setup ---
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128 # SimCLR needs large batches
    EPOCHS = 100     # For a real run, you'd want 800-1000
    LR = 3e-4
    TEMPERATURE = 0.5
    
    # --- 2. Load Data ---
    # For pretraining, we use the standard CIFAR-10 but with our
    # special contrastive augmentations. We ignore the labels.
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
        drop_last=True # Important for contrastive loss
    )

    # --- 3. Initialize Model, Loss, and Optimizer ---
    # Model
    encoder = Encoder() # Using default 128-dim
    projector = ProjectionHead(in_features=2048, hidden_dim=512, out_features=128)
    simclr_model = SimCLR_Model(encoder, projector).to(device)
    
    # Loss
    criterion = NTXentLoss(batch_size=BATCH_SIZE, temperature=TEMPERATURE, device=device)
    
    # Optimizer
    optimizer = optim.Adam(simclr_model.parameters(), lr=LR, weight_decay=1e-6)

    # --- 4. Training Loop (Example: 10 steps) ---
    print("Starting a short test pretraining loop...")
    simclr_model.train() # Set model to training mode
    
    # We'll just run 10 steps to prove it works
    for step, (views, _) in enumerate(train_loader):
        # 'views' is a list [view1, view2]
        # We ignore the labels with '_'
        
        # Get the two views
        view1, view2 = views[0].to(device), views[1].to(device)
        
        # --- Forward pass ---
        # Get projections (z) for both views
        # We ignore the features (h) for now
        _, z1 = simclr_model(view1)
        _, z2 = simclr_model(view2)
        
        # Calculate loss
        loss = criterion(z1, z2)
        
        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0: # Print every step for this test
            print(f"Step [1/10], Loss: {loss.item():.4f}")
        
        if step >= 9: # Stop after 10 steps
            break
            
    print("\n--- TEST PRETRAINING SUCCESSFUL! ---")
    print("The model is training with the contrastive loss.")
    
    # --- 5. Save the ENCODER ---
    # After real training, you ONLY save the encoder's weights
    # The projector is thrown away
    encoder_path = 'encoder_pretrained.pth'
    torch.save(simclr_model.encoder.state_dict(), encoder_path)
    print(f"Successfully saved a test encoder to {encoder_path}")
    print("--------------------------------------")