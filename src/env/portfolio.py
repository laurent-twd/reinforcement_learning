from src.env.env import Environment
from typing import Optional, Tuple
import torch
from src.data.trading_dataset import TradingState, TradingDataset
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

    def get_current_state(self) -> TradingState:
        return self.dataset[self.current_step][0]

    def reward(self, action: TradingAction, reduce_reward: bool = True) -> float:
        _, current_price, next_price = self.dataset[self.current_step]
        y = next_price / current_price
        if len(y.shape) < 2:
            y = y.unsqueeze(0)
        while len(action.shape) < len(y.shape):
            action = action.unsqueeze(0)
        cash_return = torch.ones(size=(y.shape[0], 1))
        y = torch.cat([cash_return, y], dim=-1)
        weights = (
            action * y / (action * y).sum(dim=-1, keepdim=True)
        )  # used to compute transaction costs
        mu = self.transaction_cost.get_transaction_factor(action, weights)
        r = (action * y).sum(dim=-1).mul(mu).log() - 1.0
        if reduce_reward:
            return r.mean(dim=0)
        else:
            return r

    def step(
        self, action: TradingAction, reduce_reward: bool = False
    ) -> Tuple[float, TradingState, bool]:

        current_state = self.get_current_state()
        reward = self.reward(action, reduce_reward)
        terminal_state = (self.current_step + 1 >= len(self.dataset)) | (
            self.counter >= (self.steps_per_episode - 1)
        )
        self.current_step = torch.clip(self.current_step + 1, max=len(self.dataset) - 1)
        self.counter += 1

        new_state = self.get_current_state()
        new_state = torch.where(
            terminal_state.unsqueeze(-1).unsqueeze(-1), current_state, new_state
        )

        return reward, new_state, terminal_state.float()
