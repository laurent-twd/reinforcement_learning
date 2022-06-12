from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BatchInput:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    terminal_state: torch.Tensor

    @property
    def bs(self):
        return self.state.shape[0]

    def to(self, device: torch.device):
        return BatchInput(
            self.state.to(device) if self.state is not None else None,
            self.action.to(device) if self.action is not None else None,
            self.reward.to(device) if self.reward is not None else None,
            self.next_state.to(device) if self.next_state is not None else None,
            self.terminal_state.to(device) if self.terminal_state is not None else None,
        )


class ReplayBuffer:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.buffer = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "terminal_state": [],
        }

    def buffer_size(self):
        return len(self.buffer["state"])

    def get_batch(self, batch_size: int):
        if self.buffer_size() >= batch_size:
            idx = np.random.choice(self.buffer_size(), size=batch_size, replace=False)
            state, action, reward, next_state, terminal_state = list(
                map(
                    lambda k: torch.cat(
                        [self.buffer[k][i].unsqueeze(0) for i in idx], dim=0
                    ),
                    self.buffer.keys(),
                )
            )
            return BatchInput(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminal_state=terminal_state,
            )

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal_state: torch.Tensor,
    ):
        if self.buffer_size() >= self.memory_size:
            _ = [
                self.buffer[k].pop() for k in self.buffer.keys()
            ]  # dropping oldest sample

        self.buffer["state"].append(state.squeeze())
        self.buffer["action"].append(action.squeeze())
        self.buffer["reward"].append(reward.squeeze())
        self.buffer["next_state"].append(next_state.squeeze())
        self.buffer["terminal_state"].append(terminal_state.squeeze())

        assert self.buffer_size() <= self.memory_size
