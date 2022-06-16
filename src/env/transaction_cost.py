from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch


@dataclass
class TransactionCostConfig:
    c_s = 0.05
    c_b = 0.05


class TransactionCost(ABC):
    def __init__(self, config: TransactionCostConfig):
        self.config = config

    @abstractmethod
    def get_transaction_factor(self, *kwargs):
        return NotImplementedError


class DynamicTransactionCost(TransactionCost):
    def __init__(self, config):
        super().__init__(config)

    def get_transaction_factor(self, action, weights, steps=10):
        c_b, c_s = self.config.c_b, self.config.c_s
        mu = 1.0 - 0.5 * (c_b + c_s)
        for _ in range(steps):
            x = torch.relu(weights[1:] - mu * action[1:]).sum()
            x = 1.0 - c_b * weights[0] - (c_b + c_s - c_b * c_s) * x
            mu = x / (1.0 - c_b * action[0])
        return mu