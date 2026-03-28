import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

LOOKBACK = 8
BITS = 16

class MemDataset(Dataset):
    def __init__(self, csv_path, vocab_size, split='train'):
        df = pd.read_csv(csv_path)
        n = len(df)
        cut = int(n * 0.8)
        df = df.iloc[:cut] if split == 'train' else df.iloc[cut:]
        self.delta_ids = torch.tensor(df['delta_id'].values, dtype=torch.long)
        self.pcs = torch.tensor(df['pc'].values % 65536, dtype=torch.long)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.delta_ids) - LOOKBACK

    def __getitem__(self, idx):
        x_delta = self.delta_ids[idx: idx + LOOKBACK]
        x_pc    = self.pcs[idx: idx + LOOKBACK]
        target_id = self.delta_ids[idx + LOOKBACK].item()
        target_bin = torch.tensor(
            [(target_id >> b) & 1 for b in range(BITS)],
            dtype=torch.float32)
        return x_delta, x_pc, target_bin, target_id
