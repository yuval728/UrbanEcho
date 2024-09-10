import torch
# from torch.utils.data import DataLoader

from torchvision.datasets import DatasetFolder
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def get_dataset(data_dir, transform=None):
    dataset = DatasetFolder(data_dir, loader=npy_loader, extensions=tuple('.npy'), transform=transform)
    return dataset
