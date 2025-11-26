import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# --- 1. Define your Custom Noisy Dataset ---
# This class will load the standard CIFAR-10 images
# but swap the clean labels for the noisy ones.

class Cifar10Noisy(Dataset):
    def __init__(self, root, train=True, transform=None, noisy_labels_path=None):
        
        # 1. Load the standard CIFAR-10 dataset
        self.cifar10_dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True, # Will download to ./data if not present
            transform=transform
        )
        
        # 2. Load the noisy labels
        if not noisy_labels_path:
            raise ValueError("You must provide the path to the noisy labels .pt file")
        
        noisy_data = torch.load(noisy_labels_path)
        self.noisy_labels = noisy_data['aggre_label'] # Using 'aggre_label'
        
        # 3. (Optional) Load clean labels for verification
        self.clean_labels = noisy_data['clean_label'] #

        # 4. Security check to ensure our dataset aligns with the noisy labels
        if train:
            # Convert list of targets to a tensor for comparison
            cifar_targets_tensor = torch.tensor(self.cifar10_dataset.targets)
            assert torch.equal(cifar_targets_tensor, self.clean_labels), \
                "Label mismatch! The clean labels from CIFAR-10N.pt do not match the torchvision CIFAR-10 targets."
            print("Label check passed: CIFAR-10 dataset aligns with noisy label file.")

    def __len__(self):
        # Return the total size of the dataset
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        # 1. Get the original image from the standard CIFAR-10 dataset
        # We ignore its original (clean) label by using '_'
        image, _ = self.cifar10_dataset[idx]
        
        # 2. Get the new NOISY label for that index
        noisy_label = self.noisy_labels[idx]
        
        # 3. Return the image and its corresponding NOISY label
        return image, noisy_label

# --- This block runs only when you execute `python main.py` ---
if __name__ == "__main__":
    
    print("Starting script...")
    
    # --- 1. Define Transforms ---
    # Basic transforms for the fine-tuning/testing phase
    # (We'll need different, heavier ones for pretraining later)
    transform_finetune = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # --- 2. Set Paths ---
    DATA_ROOT = './data'
    NOISY_LABELS_FILE = './data/CIFAR-10_human.pt'

    # --- 3. Instantiate the Dataset ---
    print(f"Loading noisy dataset from {DATA_ROOT}...")
    try:
        noisy_train_dataset = Cifar10Noisy(
            root=DATA_ROOT, 
            train=True, 
            transform=transform_finetune,
            noisy_labels_path=NOISY_LABELS_FILE
        )
        print("Successfully created noisy training dataset.")

        # --- 4. Create the DataLoader ---
        # A DataLoader batches and shuffles the data for us
        noisy_train_loader = DataLoader(
            noisy_train_dataset, 
            batch_size=64, # You can change this
            shuffle=True,
            num_workers=2  # Speeds up loading
        )
        print("Successfully created DataLoader.")

        # --- 5. Test the DataLoader ---
        print("Loading one batch from the DataLoader to test...")
        images, labels = next(iter(noisy_train_loader))
        
        print("\n--- TEST SUCCESSFUL! ---")
        print(f"Loaded one batch of {images.shape[0]} images.")
        print(f"Image tensor shape (Batch, Channels, H, W): {images.shape}")
        print(f"Label tensor shape (Batch): {labels.shape}")
        print(f"\nFirst 10 NOISY labels in this batch:")
        print(labels[:10])
        print("------------------------")

    except FileNotFoundError:
        print(f"\n*** ERROR ***")
        print(f"Could not find the file: {NOISY_LABELS_FILE}")
        print("Please make sure you have downloaded it and placed it in the 'data' folder.")
    except AssertionError as e:
        print(f"\n*** ERROR ***")
        print(f"{e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")