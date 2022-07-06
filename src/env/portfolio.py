from src.env.env import Environment
from typing import Optional, Tuple
import torch
from src.data.trading_dataset import TradingDataset
from src.env.transaction_cost import TransactionCost


class TradingAction(torch.Tensor):
    pass


class Portfolio(Environment):
    def __init__(
        self,
        dataset: TradingDataset,
        transaction_cost: TransactionCost,
        steps_per_episode: int = 500,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.dataset = dataset
        self.transaction_cost = transaction_cost
        self.n_features = dataset[0][0].shape[-1]
        self.n_assets = dataset[0][1].shape[0]
        self.counter = 0
        self.steps_per_episode = steps_per_episode

    def reset(self, step: torch.Tensor):
        self.current_step = step.long()
        self.counter = 0
        self.weights = (
            torch.ones((self.current_step.shape[0], 1 + self.n_assets)).cumsum(dim=-1)
            <= 1.0
        ).float()

    def get_current_state(self) -> torch.Tensor:
        return self.dataset[self.current_step][0]

    def reward(
        self,
        action: torch.Tensor,
        device,
        reduce_reward: bool = True,
    ) -> float:
        _, current_price, next_price = self.dataset[self.current_step]
        y = next_price / current_price
        if len(y.shape) < 2:
            y = y.unsqueeze(0)
        while len(action.shape) < len(y.shape):
            action = action.unsqueeze(0)
        cash_return = torch.ones(size=(y.shape[0], 1))
        y = torch.cat([cash_return, y], dim=-1).to(device)
        self.weights.to(device)
        mu = self.transaction_cost.get_transaction_factor(action, self.weights)
        self.weights = (
            action * y / (action * y).sum(dim=-1, keepdim=True)
        )  # used to compute transaction costs
        r = (action * y).sum(dim=-1).mul(mu).log()
        if reduce_reward:
            return r.mean(dim=0)
        else:
            return r

    def step(
        self, action: torch.Tensor, device, reduce_reward: bool = False
    ) -> Tuple[float, torch.Tensor, bool]:

        current_state = self.get_current_state().to(device)
        reward = self.reward(action, device, reduce_reward)
        terminal_state = (self.current_step + 1 >= len(self.dataset)) | (
            self.counter >= (self.steps_per_episode)  # do not add -1 for now
        )
        self.current_step = torch.clip(self.current_step + 1, max=len(self.dataset) - 1)
        self.counter += 1

        new_state = self.get_current_state().to(device)
        terminal_state = terminal_state.to(device)
        new_state = torch.where(
            terminal_state.unsqueeze(-1).unsqueeze(-1), current_state, new_state
        )

        return reward, new_state, terminal_state.float()
