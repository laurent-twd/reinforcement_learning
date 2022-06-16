from dataclasses import dataclass
from abc import ABC, abstractmethod


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

    def get_transaction_factor(self, weights, new_weights):
        return 1.0
