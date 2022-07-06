import torch
from src.models.agent import TradingAgent, TradingCritic
from src.env.portfolio import Portfolio
from dataclasses import dataclass
from src.data.trading_dataset import TradingDataset
from src.env.transaction_cost import DynamicTransactionCost
from typing import Union
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.models.ppo.losses import GAE
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
    normalize_advantage: bool = True


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

    def run_episode(self, env, device):

        values = []
        log_probs = []
        for _ in range(self.config.steps_per_trajectory):
            state = env.get_current_state().to(device)
            with torch.no_grad():
                log_alphas = self.actor(state)
                value = self.critic(state)
            policy = torch.distributions.Dirichlet(concentration=log_alphas.exp())
            action = policy.sample()
            reward, next_state, terminal_state = env.step(action, device)
            log_prob = policy.log_prob(action)
            self.buffer.add_experience(
                state.cpu(),
                action.cpu(),
                reward.cpu(),
                next_state.cpu(),
                terminal_state.cpu(),
            )
            log_probs.append(log_prob.unsqueeze(1))
            values.append(value)
        with torch.no_grad():
            values.append(self.critic(next_state))

        return torch.cat(log_probs, dim=1), torch.cat(values, dim=1).squeeze()

    def fit(
        self, dataset: TradingDataset, n_episodes: int = 1000, use_gpu: bool = True
    ):

        self.actor.network.train()
        self.critic.network.train()
        use_gpu = use_gpu and torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        self.actor.network.to(device)
        self.critic.network.to(device)

        env = Portfolio(
            dataset=dataset,
            transaction_cost=self.transaction_cost,
            steps_per_episode=self.config.steps_per_trajectory,
            name="portfolio",
        )

        progbar = tqdm(range(n_episodes), desc="Episode ")
        for _ in progbar:
            starting_step = torch.randint(
                0,
                len(dataset) - self.config.steps_per_trajectory,
                [self.config.n_trajectories],
            )
            # starting_step = (
            #     torch.zeros(self.config.n_trajectories).long() + starting_step[0]
            # )
            env.reset(starting_step)
            self.buffer.reset()
            old_log_probs, old_values = self.run_episode(env, device)
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
                        _,
                        _,
                    ) = astuple(batch.to(device))

                    self.optimizer.zero_grad()

                    log_alphas = self.actor(states.flatten(0, 1)).reshape(
                        states.shape[0], states.shape[1], -1
                    )
                    distribution = torch.distributions.Dirichlet(log_alphas.exp())
                    log_probs = distribution.log_prob(actions)
                    ratios = (log_probs - old_log_probs[idx, :]).exp()
                    advantages = GAE(
                        rewards,
                        old_values[idx, :],
                        self.config.gamma,
                        self.config.lambda_,
                        device,
                    )
                    rewards_to_go = advantages + old_values[idx, :-1]
                    if self.config.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-5
                        )
                    p1 = ratios * advantages
                    p2 = (
                        torch.clamp(
                            ratios,
                            1 - self.config.epsilon,
                            1 + self.config.epsilon,
                        )
                        * advantages
                    )
                    actor_loss = -torch.minimum(p1, p2).mean()
                    entropy_loss = -distribution.entropy().mean()
                    with torch.no_grad():
                        values = (
                            self.critic(states.flatten(0, 1))
                            .reshape(states.shape[0], states.shape[1], -1)
                            .squeeze()
                        )

                    value_pred_clipped = old_values[idx, :-1] + (
                        values - old_values[idx, :-1]
                    ).clamp(-self.config.epsilon, self.config.epsilon)
                    value_losses = (values - rewards_to_go) ** 2
                    value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                    value_loss = value_loss.mean()
                    # value_loss = (values - rewards_to_go).square().mean()

                    loss = (
                        actor_loss
                        + self.config.c_1 * value_loss
                        + self.config.c_2 * entropy_loss
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.network.parameters(), self.config.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.network.parameters(), self.config.max_grad_norm
                    )

                    self.optimizer.step()

                    step = self.optimizer.state[
                        self.optimizer.param_groups[0]["params"][-1]
                    ]["step"]
                    self.writer.add_scalar("Loss", loss.detach().cpu(), step)
                    self.writer.add_scalar(
                        "Actor Loss", actor_loss.detach().cpu(), step
                    )
                    self.writer.add_scalar(
                        "Critic Loss", value_loss.detach().cpu(), step
                    )

                    progbar.set_description(f"Loss: {loss.detach().cpu():e}")
