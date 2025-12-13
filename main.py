import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class Cifar10Noisy(Dataset):
    def __init__(self, root, train=True, transform=None, noisy_labels_path=None):
        self.cifar10_dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True,
            transform=transform
        )
        if not noisy_labels_path:
            raise ValueError("You must provide the path to the noisy labels .pt file")
        noisy_data = torch.load(noisy_labels_path)
        self.noisy_labels = noisy_data['aggre_label']
        self.clean_labels = noisy_data['clean_label']
        if train:
            cifar_targets_tensor = torch.tensor(self.cifar10_dataset.targets)
            assert torch.equal(cifar_targets_tensor, self.clean_labels), \
                "Label mismatch! The clean labels from CIFAR-10N.pt do not match the torchvision CIFAR-10 targets."
            print("Label check passed: CIFAR-10 dataset aligns with noisy label file.")

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        image, _ = self.cifar10_dataset[idx]
        noisy_label = self.noisy_labels[idx]
        return image, noisy_label

if __name__ == "__main__":
    print("Starting script...")
    transform_finetune = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    DATA_ROOT = './data'
    NOISY_LABELS_FILE = './data/CIFAR-10_human.pt'

    print(f"Loading noisy dataset from {DATA_ROOT}...")
    try:
        noisy_train_dataset = Cifar10Noisy(
            root=DATA_ROOT, 
            train=True, 
            transform=transform_finetune,
            noisy_labels_path=NOISY_LABELS_FILE
        )
        print("Successfully created noisy training dataset.")

        noisy_train_loader = DataLoader(
            noisy_train_dataset, 
            batch_size=64,
            shuffle=True,
            num_workers=2
        )
        print("Successfully created DataLoader.")

        print("Loading one batch from the DataLoader to test...")
        images, labels = next(iter(noisy_train_loader))
        
        print(f"Loaded one batch of {images.shape[0]} images.")
        print(f"Image tensor shape (Batch, Channels, H, W): {images.shape}")
        print(f"Label tensor shape (Batch): {labels.shape}")
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
