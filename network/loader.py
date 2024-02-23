import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Callable
import pandas as pd

class MnistDataset(Dataset):    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.X = self.X.reshape(-1,28, 28)/255
        self.y = y

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def prepare_data(X, y, batch_size):
    dataset = MnistDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
