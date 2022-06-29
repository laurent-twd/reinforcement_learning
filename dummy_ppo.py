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
import pandas as pd


M = 365 * 2 * (60 * 24)
path_to_data = "/Users/laurentthanwerdas/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
df = pd.read_csv(path_to_data).tail(M)
columns = ["High", "Low", "Close", "Open"]
df = df.dropna(subset=columns)
norm = df["Open"].values
prices = df[["High", "Low", "Close"]].values

n_assets = 1
C = 3
window = 30
steps_per_episode = 500

dataset = TradingDataset(
    df=prices, n_assets=n_assets, n_channels=C, window=30, norm=norm
)
cost = DynamicTransactionCost(TransactionCostConfig())

agent = TradingAgent(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 2 * (n_assets + 1), 2),
)

critic = TradingCritic(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 1, 2),
)

ppo = PPO(
    config=PPOConfig(lr=1e-3, lambda_=1.0),
    agent=agent,
    critic=critic,
    transaction_cost=cost,
)
ppo.fit(dataset, n_episodes=1000)
