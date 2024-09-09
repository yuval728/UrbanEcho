import torch
from torch.utils.data import DataLoader

from torchvision.datasets import DatasetFolder
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def get_dataloader(data_dir, batch_size, num_workers, transform=None, shuffle=True, pin_memory=True):
    dataset = DatasetFolder(data_dir, loader=npy_loader, extensions=('.npy'), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader , dataset.classes
