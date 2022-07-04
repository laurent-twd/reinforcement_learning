from dataclasses import dataclass
import torch
from typing import Union, Optional
from pathlib import Path
from upath import UPath


def initialize(model):
    def init(x):
        try:
            return torch.nn.init.xavier_uniform(x)
        except:
            return torch.nn.init.zeros_(x)

    _ = [init(x) for x in model.network.parameters()]
    return model


@dataclass
class AgentConfig:
    n_assets: int = 10


class Agent:
    def __init__(self, config: AgentConfig, network: Optional[torch.nn.Module] = None):
        self.config = config
        self.network = network


class TradingAgent(Agent):
    def __init__(self, config: AgentConfig, network: Optional[torch.nn.Module] = None):
        super().__init__(config, network)

    def __call__(self, state: torch.Tensor):
        if self.network is not None:
            return self.network(state)
        else:
            logits = torch.distributions.Normal(0.0, 1.0).sample(
                [state.shape[0], 2 * (1 + self.config.n_assets)]
            )
            return torch.softmax(logits, dim=-1)


class TradingCritic(Agent):
    def __init__(self, config: AgentConfig, network: Optional[torch.nn.Module] = None):
        super().__init__(config, network)

    def __call__(self, state: torch.Tensor):
        if self.network is not None:
            return self.network(state)
        else:
            logits = torch.distributions.Normal(0.0, 1.0).sample([state.shape[0], 1])
            return logits
