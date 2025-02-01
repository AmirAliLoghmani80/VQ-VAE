import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

img_size=64
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),])
#torchvision.transforms.Normalize([ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ])])
'''invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5],
                                                     std = [ 1., 1., 1. ]),
                               ])'''


def train_loader_fn(batch_size):

    # train_dataset = torchvision.datasets.ImageFolder(root="~/torch_datasets/img_align_celeba", transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    celeba_data = datasets.CelebA(root="/content/celeba", split="train", transform=transform)

    # You can access the data with a DataLoader

    train_loader = DataLoader(celeba_data, batch_size=batch_size, shuffle=True)
    return train_loader
