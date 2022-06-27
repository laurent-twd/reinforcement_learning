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
        self.current_step = 0
        self.dataset = dataset
        self.transaction_cost = transaction_cost
        self.n_features = dataset[0][0].shape[-1]
        self.n_assets = dataset[0][1].shape[0]
        self.counter = 0
        self.steps_per_episode = steps_per_episode

    def reset(self, step: int = 0):
        self.current_step = step
        self.counter = 0

    def get_current_state(self) -> TradingState:
        return self.dataset[self.current_step][0]

    def reward(self, action: TradingAction, reduce_reward: bool = True) -> float:
        _, current_price, next_price = self.dataset[self.current_step]
        y = next_price / current_price
        y = torch.cat([torch.Tensor([1.0]), y], dim=0).unsqueeze(0)
        weights = (
            action * y / (action * y).sum(dim=-1, keepdim=True)
        )  # used to compute transaction costs
        mu = self.transaction_cost.get_transaction_factor(action, weights)
        r = (action * y).sum().mul(mu).log() - 1.0
        if reduce_reward:
            return r.mean(dim=0)
        else:
            return r

    def step(
        self, action: TradingAction, reduce_reward: bool = True
    ) -> Tuple[float, TradingState, bool]:

        if self.current_step >= len(self.dataset):
            raise ValueError("The environment must be reset at the end of the episode.")
        else:
            reward = self.reward(action, reduce_reward)
            self.current_step += 1
            self.counter += 1
            if (self.current_step >= len(self.dataset)) or (
                self.counter >= self.steps_per_episode
            ):
                return reward, None, torch.FloatTensor([1.0])
            else:
                return reward, self.get_current_state(), torch.FloatTensor([0.0])
