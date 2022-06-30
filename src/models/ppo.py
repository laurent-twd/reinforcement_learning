from copy import deepcopy
import torch
from src.models.agent import TradingAgent, TradingCritic
from src.env.portfolio import Portfolio
from dataclasses import dataclass
from src.data.trading_dataset import TradingDataset
from src.data.replay_buffer import ReplayBuffer
from src.env.transaction_cost import DynamicTransactionCost
from typing import Union
from tqdm import tqdm


def clipped_loss(rewards, values, ratios, gamma, lambda_, epsilon, c_1):
    delta = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
    mask = torch.triu(torch.ones((delta.shape[-1], delta.shape[-1]))).unsqueeze(0)
    discounting = torch.pow(gamma * lambda_, mask.cumsum(-1) - 1.0) * mask
    adjustment_factor = 1.0 - torch.pow(lambda_, mask.sum(-1))
    adjustment_factor = adjustment_factor.unsqueeze(1) * mask
    if lambda_ == 1.0:
        advantage = (discounting * delta.unsqueeze(-1)).sum(-1)
    else:
        advantage = (adjustment_factor * discounting * delta.unsqueeze(-1)).sum(-1)
    # Actor Loss
    p1 = ratios[:, :-1] * advantage
    p2 = torch.where(
        advantage >= 0, (1.0 + epsilon) * advantage, (1.0 - epsilon) * advantage
    )
    actor_loss = torch.minimum(p1, p2)
    # Critic Loss
    mask = torch.triu(torch.ones((rewards.shape[-1], rewards.shape[-1]))).unsqueeze(0)
    forward_rewards = rewards.unsqueeze(-1) * mask
    discounting = torch.pow(gamma, mask.cumsum(-1) - 1.0) * mask
    target_values = (forward_rewards * discounting).sum(-1)
    critic_loss = (values - target_values).square().mean()
    return -(actor_loss.mean() - c_1 * critic_loss)


@dataclass
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    lambda_: float = 0.5
    epsilon: float = 0.2
    steps_per_episode: float = 500
    c_1: float = 1e-2


class PPO:
    def __init__(
        self,
        config: PPOConfig,
        agent: TradingAgent,
        critic: TradingCritic,
        transaction_cost: DynamicTransactionCost,
    ):
        self.config = config

        def initialize(model):
            def init(x):
                try:
                    return torch.nn.init.xavier_uniform(x)
                except:
                    return torch.nn.init.zeros_(x)

            _ = [init(x) for x in model.network.parameters()]
            return model

        self.agent = initialize(agent)
        self.critic = initialize(critic)
        self.old_agent = initialize(deepcopy(agent))
        self.old_critic = initialize(deepcopy(critic))
        self.agent_optimizer = torch.optim.Adam(
            self.agent.network.parameters(), self.config.lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.network.parameters(), self.config.lr
        )
        self.transaction_cost = transaction_cost

    def run_episode(
        self, env, starting_step: Union[int, torch.IntTensor], buffer: ReplayBuffer
    ):
        env.reset(starting_step)
        is_terminal = False
        values = []
        ratios = []

        while not is_terminal:
            state = env.get_current_state()
            inputs = state  # .reshape(state.shape[0], -1)
            with torch.no_grad():
                values.append(self.old_critic(inputs))
                old_log_alphas = self.old_agent(inputs)
                old_policy = torch.distributions.Dirichlet(
                    concentration=old_log_alphas.exp()
                )
                action = old_policy.sample()
            log_alphas = self.agent(inputs)
            policy = torch.distributions.Dirichlet(concentration=log_alphas.exp())
            ratio = (
                (policy.log_prob(action) - old_policy.log_prob(action))
                .exp()
                .unsqueeze(-1)
            )
            ratios.append(ratio)
            reward, next_state, terminal_state = env.step(action)
            buffer.add_experience(state, action, reward, next_state, terminal_state)
            is_terminal = terminal_state.bool().all()
        values = torch.cat(values, dim=-1)
        ratios = torch.cat(ratios, dim=-1)
        trajectories = buffer.get_all()
        rewards = trajectories.reward.T

        return rewards, values, ratios

    def fit(
        self,
        dataset: TradingDataset,
        batch_size: int = 16,
        steps_per_episode: int = 500,
        n_episodes: int = 1000,
    ):
        env = Portfolio(
            dataset=dataset,
            transaction_cost=self.transaction_cost,
            steps_per_episode=steps_per_episode,
            name="portfolio",
        )

        progbar = tqdm(range(n_episodes), desc="Episode ")
        for _ in progbar:
            starting_step = torch.randint(
                0, len(dataset) - steps_per_episode, [batch_size]
            )
            buffer = ReplayBuffer(steps_per_episode)

            self.agent_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            rewards, values, ratios = self.run_episode(env, starting_step, buffer)
            loss = clipped_loss(
                rewards,
                values.squeeze(),
                ratios,
                self.config.gamma,
                self.config.lambda_,
                self.config.epsilon,
                self.config.c_1,
            )

            loss.backward()

            self.old_agent.network.load_state_dict(self.agent.network.state_dict())
            self.old_critic.network.load_state_dict(self.critic.network.state_dict())

            self.agent_optimizer.step()
            self.critic_optimizer.step()

            progbar.set_description("Loss: {:e}".format(loss.detach().cpu()))
