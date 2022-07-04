from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent, TradingCritic
from dataclasses import astuple
from src.env.transaction_cost import (
    TransactionCostConfig,
    DynamicTransactionCost,
    ConstantTransactionCost,
)
from src.models.ppo.ppo import PPOConfig, PPO
from src.models.networks import RecurrentNetwork
import pandas as pd
from src.env.portfolio import Portfolio
import torch

M = 365 * 2 * (60 * 24)
path_to_data = "/Users/laurentthanwerdas/Downloads/CryptoCurrency Historical Prices/BTC Historical Prices/Binance_BTCUSDT_1hour.csv"
df = pd.read_csv(path_to_data).tail(M)
columns = ["High", "Low", "Close", "Open"]
columns = [c.lower() for c in columns]
df = df.dropna(subset=columns)
norm = df[columns[-1]].values
prices = df[columns[:3]].values

n_assets = 1
n_channels = 3
window = 48
steps_per_episode = 500

dataset = TradingDataset(
    df=prices, n_assets=n_assets, n_channels=n_channels, window=30, norm=norm
)
cost = ConstantTransactionCost(TransactionCostConfig(c_b=0.0, c_s=0.0))

actor = TradingAgent(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * n_channels, 32, n_assets + 1, 2),
)

critic = TradingCritic(
    AgentConfig(n_assets=n_assets),
    network=RecurrentNetwork(n_assets * n_channels, 32, 1, 1),
)

self = PPO(
    config=PPOConfig(),
    actor=actor,
    critic=critic,
    transaction_cost=cost,
)

env = Portfolio(
    dataset=dataset,
    transaction_cost=self.transaction_cost,
    steps_per_episode=self.config.steps_per_trajectory,
    name="portfolio",
)


starting_step = torch.randint(
    0,
    len(dataset) - self.config.steps_per_trajectory,
    [self.config.n_trajectories],
)


# batch_size = 100
# from src.env.portfolio import Portfolio
# from src.data.replay_buffer import ReplayBuffer
# import torch

# env = Portfolio(
#     dataset=dataset,
#     transaction_cost=ppo.transaction_cost,
#     steps_per_episode=steps_per_episode,
#     name="portfolio",
# )

# starting_step = torch.randint(0, len(dataset) - steps_per_episode, [batch_size])
# env.reset(starting_step)
# buffer = ReplayBuffer(steps_per_episode)
# rewards, values, ratios = ppo.run_episode(env, starting_step, buffer)
# self = ppo
# gamma = ppo.config.gamma
# lambda_ = ppo.config.lambda_
# epsilon = ppo.config.epsilon
# c_1 = ppo.config.c_1
# values = values.squeeze()

# rewards.exp().prod(-1).mean()

# portfolio_value = rewards.exp().prod(-1)
# _, l0, l1 = dataset[starting_step + torch.arange(0, rewards.shape[0])]
# call = (l1 / l0).prod(-1)
# (portfolio_value > call).float().mean()

# # weights = buffer.get_all().action
# # action = weights.diff(dim=0)[:, -1] > 0
# # labels = (l1 - l0 > 0)[1:, 0]
# # assert labels.shape == action.shape

# # print(f"Edge: {(action == labels).float().mean()}")
