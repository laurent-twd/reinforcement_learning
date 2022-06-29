from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent, TradingCritic
from dataclasses import astuple
from src.env.transaction_cost import (
    TransactionCostConfig,
    DynamicTransactionCost,
)
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
cost = DynamicTransactionCost(TransactionCostConfig(c_b=0.05, c_s=0.05))

agent = TradingAgent(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 2 * (n_assets + 1), 2),
)

critic = TradingCritic(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * C, 32, 1, 2),
)

ppo = PPO(
    config=PPOConfig(lr=1e-2, lambda_=1.0),
    agent=agent,
    critic=critic,
    transaction_cost=cost,
)
batch_size = 16
self = ppo
ppo.fit(dataset, n_episodes=50)

from src.env.portfolio import Portfolio
from src.data.replay_buffer import ReplayBuffer
import torch

env = Portfolio(
    dataset=dataset,
    transaction_cost=ppo.transaction_cost,
    steps_per_episode=steps_per_episode,
    name="portfolio",
)

starting_step = torch.randint(0, len(dataset) - steps_per_episode, [1])
env.reset(starting_step)
buffer = ReplayBuffer(steps_per_episode)
rewards, values, ratios = ppo.run_episode(env, starting_step, buffer)

portfolio_value = rewards.add(1.0).exp().prod()
_, l0, l1 = dataset[starting_step + torch.arange(0, rewards.shape[0])]
call = (l1 / l0).prod()
portfolio_value, call

weights = buffer.get_all().action
action = weights.diff(dim=0)[:, -1] > 0
labels = (l1 - l0 > 0)[1:, 0]
assert labels.shape == action.shape

print(f"Edge: {(action == labels).float().mean()}")
