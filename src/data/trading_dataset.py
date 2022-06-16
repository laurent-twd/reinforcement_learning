import torch
from torch.utils.data import Dataset

# import pandas as pd
from typing import List, Tuple
import numpy as np


class TradingState(torch.Tensor):
    pass


class TradingLabel(torch.Tensor):
    pass


def preprocess(
    df: np.ndarray, n_assets: int, n_channels: int, window: int
) -> Tuple[TradingState, TradingLabel, TradingLabel]:
    assert df.shape[-1] == n_assets * n_channels
    closing_index = [n_channels * k + (n_channels - 1) for k in range(n_assets)]
    prices = torch.FloatTensor(df)
    index = torch.arange(0, len(prices)).unsqueeze(1) - torch.arange(
        0, window
    ).unsqueeze(0)
    index = index.flip(-1)
    index = index[(window - 1) :]
    features = prices[index]
    inputs, labels = features[:-1, :], (
        features[:-1, -1, closing_index],
        features[1:, -1, closing_index],
    )
    return TradingState(inputs), TradingLabel(labels[0]), TradingLabel(labels[1])


class TradingDataset(Dataset):
    def __init__(
        self, df: np.ndarray, n_assets: int, n_channels: int, window: int = 30
    ):
        self.data = preprocess(
            df, n_assets=n_assets, n_channels=n_channels, window=window
        )

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx: int) -> Tuple[TradingState, TradingLabel, TradingLabel]:
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]
