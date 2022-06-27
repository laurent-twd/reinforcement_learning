import numpy as np
from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent, TradingCritic
from src.env.portfolio import Portfolio
from src.data.replay_buffer import ReplayBuffer
import random
from dataclasses import astuple
from src.env.transaction_cost import TransactionCostConfig, DynamicTransactionCost
import torch

n_assets = 5
T = 10000
C = 3
X = np.random.normal(100.0, 1.0, size=(T, n_assets * C))
window = 30
steps_per_episode = 500

dataset = TradingDataset(X, n_assets, C, 30)
cost = DynamicTransactionCost(TransactionCostConfig())
agent = TradingAgent(AgentConfig(n_assets=n_assets))
critic = TradingCritic(AgentConfig())

env = Portfolio(
    dataset=dataset,
    transaction_cost=cost,
    steps_per_episode=steps_per_episode,
    name="portfolio",
)
n_trajectories = 16
n_episodes = 10
counter = 0
gamma = 0.99

episode = 0
print(episode)
rewards = []
is_terminal = False
buffer = ReplayBuffer(steps_per_episode + 10)
starting_step = torch.randint(0, len(dataset), [n_trajectories])
env.reset(starting_step)
values = []
while not is_terminal:
    state = env.get_current_state()
    values.append(critic(state))
    action = agent(state)
    reward, next_state, terminal_state = env.step(action)
    buffer.add_experience(state, action, reward, next_state, terminal_state)
    is_terminal = terminal_state.bool().all()
values = torch.cat(values, dim=-1).T
trajectories = buffer.get_all()
terminal_states = trajectories.terminal_state
rewards = trajectories.reward
powers = torch.arange(0, rewards.shape[0])
discounted_rewards = rewards * torch.pow(gamma, powers).unsqueeze(-1)
mask = 1.0 - torch.concat(
    [torch.zeros(1, terminal_states.shape[-1]), terminal_states[:-1]]
)
mean_reward = (rewards.sum(0) / mask.sum(0)).mean(0)

values
