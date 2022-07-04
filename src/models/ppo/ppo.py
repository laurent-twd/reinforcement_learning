import torch
from src.models.agent import TradingAgent, TradingCritic
from src.env.portfolio import Portfolio
from dataclasses import dataclass
from src.data.trading_dataset import TradingDataset
from src.env.transaction_cost import DynamicTransactionCost
from typing import Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.models.ppo.losses import clipped_loss, easier_loss
from src.models.agent import initialize
from src.data.buffer import Buffer
from torch.utils.data import DataLoader
from dataclasses import astuple


@dataclass
class PPOConfig:
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    n_trajectories: int = 16
    steps_per_trajectory: float = 512
    n_epochs: int = 10
    batch_size: int = 128
    gamma: float = 0.99
    lambda_: float = 0.5
    epsilon: float = 0.2
    c_1: float = 1e-2
    c_2: float = 1e-2
    max_grad_norm: float = 0.5


class PPO:
    def __init__(
        self,
        config: PPOConfig,
        actor: TradingAgent,
        critic: TradingCritic,
        transaction_cost: DynamicTransactionCost,
    ):
        self.config = config

        self.actor = initialize(actor)
        self.critic = initialize(critic)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.network.parameters(), "lr": self.config.lr_actor},
                {
                    "params": self.critic.network.parameters(),
                    "lr": self.config.lr_critic,
                },
            ]
        )
        self.transaction_cost = transaction_cost
        self.writer = SummaryWriter()
        self.buffer = Buffer(
            memory_size=self.config.n_trajectories * self.config.steps_per_trajectory,
            batch=True,
        )

    def run_episode(self, env):

        values = []
        log_probs = []
        for _ in range(self.config.steps_per_trajectory):
            state = env.get_current_state()
            with torch.no_grad():
                log_alphas = self.actor(state)
                value = self.critic(state)
            policy = torch.distributions.Dirichlet(concentration=log_alphas.exp())
            action = policy.sample()
            reward, next_state, terminal_state = env.step(action)
            log_prob = policy.log_prob(action)
            self.buffer.add_experience(
                state, action, reward, next_state, terminal_state
            )
            log_probs.append(log_prob.unsqueeze(1))
            values.append(value)
        with torch.no_grad():
            values.append(self.critic(next_state))

        return torch.cat(log_probs, dim=1), torch.cat(values, dim=1).squeeze()

    def fit(
        self,
        dataset: TradingDataset,
        n_episodes: int = 1000,
    ):
        env = Portfolio(
            dataset=dataset,
            transaction_cost=self.transaction_cost,
            steps_per_episode=self.config.steps_per_trajectory,
            name="portfolio",
        )

        progbar = tqdm(range(n_episodes), desc="Episode ")
        for episode in progbar:
            starting_step = torch.randint(
                0,
                len(dataset) - self.config.steps_per_trajectory,
                [self.config.n_trajectories],
            )
            env.reset(starting_step)
            self.buffer.reset()
            self.optimizer.zero_grad()
            old_log_probs, values = self.run_episode(env)
            dataloader = DataLoader(
                dataset=self.buffer,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=self.buffer.collate,
            )

            for epoch in range(self.config.n_epochs):
                for idx, batch in iter(dataloader):
                    (
                        states,
                        actions,
                        rewards,
                        next_states,
                        terminal_state,
                    ) = astuple(batch)

                    log_alphas = self.actor(states.flatten(0, 1)).reshape(
                        states.shape[0], states.shape[1], -1
                    )
                    log_probs = torch.distributions.Dirichlet(
                        log_alphas.exp()
                    ).log_prob(actions)
                    ratios = (log_probs - old_log_probs[idx, :]).exp()
