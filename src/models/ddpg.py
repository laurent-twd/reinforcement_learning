from dataclasses import dataclass
from data.trading_dataset import TradingState
from src.models.agent import Agent
from typing import Union
from src.env.env import Environment
import torch
from copy import deepcopy


@dataclass
class DDPGConfig:
    eps_start: float = 1.0
    eps_end: float = 0.02
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_size: int = 100000
    seed: int = 123


class DDPG:
    def __init__(self, config: DDPGConfig, environment: Environment, agent: Agent):
        self.config = config
        self.primary_agent = agent
        self.target_agent = deepcopy(agent)
        self.env = environment

    def train_step(
        self,
        primary_agent: Agent,
        target_agent: Agent,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal_state: torch.Tensor,
    ):
        return NotImplementedError

    def train(self):
        return NotImplementedError
