from dataclasses import dataclass
import torch
from typing import Union, Optional
from pathlib import Path
from upath import UPath


@dataclass
class AgentConfig:
    n_assets: int = 10


class TradingAgent:
    def __init__(self, config: AgentConfig, network: Optional[torch.nn.Module] = None):
        self.config = config
        self.network = network

    def __call__(self, state: torch.Tensor):
        if self.network is not None:
            return self.network(state)
        else:
            logits = torch.distributions.Normal(0.0, 1.0).sample(
                [1 + self.config.n_assets]
            )
            return torch.softmax(logits, dim=-1)
