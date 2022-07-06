from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent, TradingCritic
from dataclasses import astuple
from src.env.transaction_cost import (
    TransactionCostConfig,
    DynamicTransactionCost,
    ConstantTransactionCost,
)
from src.models.ppo.ppo import PPOConfig, PPO
from src.models.networks import RecurrentNetwork, MLP
import pandas as pd
from src.env.portfolio import Portfolio
import torch
import numpy as np

from src.models.networks import TransformerEncoderModel


M = 365 * 2 * (60 * 24)
path_to_data = "/Users/laurentthanwerdas/Downloads/CryptoCurrency Historical Prices/BTC Historical Prices/Binance_BTCUSDT_1hour.csv"
df = pd.read_csv(path_to_data).tail(M)
columns = ["High", "Low", "Close", "Open"]
columns = [c.lower() for c in columns]
df = df.dropna(subset=columns)


# generated_prices = np.cos(0.1 * np.arange(0, df.shape[0] + 1)) + 10.0
# prices = generated_prices[:-1][:, np.newaxis]
# norm = generated_prices[1:]


norm = df[columns[-1]].values
prices = df[columns[:3]].values

n_assets = 1
n_channels = 3
window = 30

dataset = TradingDataset(
    df=prices, n_assets=n_assets, n_channels=n_channels, window=window, norm=norm
)
cost = ConstantTransactionCost(TransactionCostConfig(c_b=0.0, c_s=0.0))

actor = TradingAgent(
    AgentConfig(n_assets=n_assets),
    network=TransformerEncoderModel(
        input_dim=n_channels, output_dim=2, d_model=32, dim_feedforward=128
    ),
)

critic = TradingCritic(
    AgentConfig(n_assets=n_assets),
    network=TransformerEncoderModel(
        input_dim=n_channels, output_dim=1, d_model=32, dim_feedforward=128
    ),
)

ppo = PPO(
    config=PPOConfig(
        lr_actor=1e-5,
        lr_critic=1e-5,
        c_1=1e-3,
        c_2=0.0,
        steps_per_trajectory=90,
        n_trajectories=32,
        n_epochs=5,
        batch_size=16,
        lambda_=0.0,
        gamma=0.99,
        normalize_advantage=False,
    ),
    actor=actor,
    critic=critic,
    transaction_cost=cost,
)

ppo.fit(dataset, n_episodes=100)


batch_size = 64

env = Portfolio(
    dataset=dataset,
    transaction_cost=ppo.transaction_cost,
    steps_per_episode=ppo.config.steps_per_trajectory,
    name="portfolio",
)

starting_step = torch.randint(0, len(dataset) - ppo.config.steps_per_trajectory, [1])

starting_step = starting_step + torch.zeros(batch_size).long()

env.reset(starting_step)
ppo.buffer.reset()
log_probs, values = ppo.run_episode(env)
trajectories = ppo.buffer.collate([ppo.buffer[i] for i in range(len(ppo.buffer))])
rewards = trajectories[1].reward
actions = trajectories[1].action
portfolio_value = rewards.exp().prod(-1)
_, l0, l1 = dataset[starting_step[0] + torch.arange(0, rewards.shape[0])]
call = (l1 / l0)[:, 0].prod(-1)
(portfolio_value > call).float().mean()


import matplotlib.pyplot as plt

y1 = prices[starting_step[0] + np.arange(0, ppo.config.steps_per_trajectory), 0]
y2 = actions.mean(0)[:, 1]

plt.plot(((y1 - 10) + 1.0) / 2)
plt.plot(y2)
plt.show()
plt.savefig("test.png")
