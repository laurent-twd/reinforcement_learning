import numpy as np
from src.data.trading_dataset import TradingDataset
from src.models.agent import AgentConfig, TradingAgent
from env.single_portfolio import Portfolio
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
env = Portfolio(
    dataset=dataset,
    transaction_cost=cost,
    steps_per_episode=steps_per_episode,
    name="portfolio",
)
buffer = ReplayBuffer(1000)
batch_size = 16
n_episodes = 10

counter = 0
for episode in range(n_episodes):
    print(episode)
    terminal_state = False
    while not terminal_state:
        state = env.get_current_state()
        action = agent(state)
        reward, next_state, terminal_state = env.step(action)
        buffer.add_experience(state, action, reward, next_state, terminal_state)
        batch = buffer.get_batch(batch_size)
        if batch is not None:
            counter += 1
            states, actions, rewards, next_states, terminale_states = astuple(batch)
            # perform batch gradient descent
        if terminal_state.bool():
            starting_step = random.randint(0, len(dataset))
            env.reset(starting_step)
