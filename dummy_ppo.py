from sqlite3 import paramstyle
import numpy as np
from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent, TradingCritic
from src.env.portfolio import Portfolio
from src.data.replay_buffer import ReplayBuffer
import random
from dataclasses import astuple
from src.env.transaction_cost import (
    TransactionCost,
    TransactionCostConfig,
    DynamicTransactionCost,
)
import torch
from copy import deepcopy
from src.models.ppo import PPOConfig, PPO
from src.models.networks import RecurrentNetwork

n_assets = 5
T = 10000
C = 3
X = np.random.normal(100.0, 1.0, size=(T, n_assets * C))
window = 30
steps_per_episode = 500

dataset = TradingDataset(X, n_assets, C, 30)
cost = DynamicTransactionCost(TransactionCostConfig())

agent = TradingAgent(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 2 * (n_assets + 1), 2),
)

critic = TradingCritic(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 1, 2),
)

ppo = PPO(config=PPOConfig(lr=1e-5), agent=agent, critic=critic, transaction_cost=cost)
ppo.fit(dataset, n_episodes=100)

batch_size = 16
steps_per_episode = 500
self = ppo
