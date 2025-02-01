import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

img_size = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])

def train_loader_fn(batch_size):
    # Change this path to the location of your CelebA dataset on Kaggle
    celeba_path = "/kaggle/input/celeba-dataset/img_align_celeba"

    # Using CelebA dataset from torchvision.datasets
    celeba_data = datasets.CelebA(root=celeba_path, split="train", transform=transform)

    # Create the DataLoader for training
    train_loader = DataLoader(celeba_data, batch_size=batch_size, shuffle=True)

    return train_loader
