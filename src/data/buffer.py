from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


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
            self.state.to(device),
            self.action.to(device),
            self.reward.to(device),
            self.next_state.to(device),
            self.terminal_state.to(device),
        )


class Buffer(Dataset):
    def __init__(self, memory_size: int, batch: bool = True):
        self.memory_size = memory_size
        self.data = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "terminal_state": [],
        }
        self.batch = batch

    def reset(self):
        self.data = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "terminal_state": [],
        }

    def __len__(self):
        if self.batch:
            return self.data["state"][0].shape[1]
        else:
            return len(self.data["state"])

    def collate(self, data):

        indexes, experiences = zip(*data)
        state, action, reward, next_state, terminal_state = [
            torch.cat(x, dim=0) for x in zip(*experiences)
        ]

        return indexes, BatchInput(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal_state=terminal_state,
        )

    def __getitem__(self, index) -> List:
        if self.batch:
            return index, [
                torch.cat([x[:, index] for x in self.data[k]], dim=0).unsqueeze(0)
                for k in self.data.keys()
            ]
        else:
            return index, [self.data[k][index] for k in self.data.keys()]

    def get_all(self):
        idx = np.arange(0, len(self))
        return self.collate(idx)

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Optional[torch.Tensor],
        terminal_state: torch.Tensor,
    ):

        if self.batch:
            data = [state, action, reward, next_state, terminal_state]
            for i, key in enumerate(self.data.keys()):
                self.data[key].append(data[i].unsqueeze(0))

        else:
            [self.data["state"].append(x) for x in torch.split(state, 1)]
            [self.data["action"].append(x) for x in torch.split(action, 1)]
            [self.data["reward"].append(x) for x in torch.split(reward, 1)]
            [
                self.data["terminal_state"].append(x)
                for x in torch.split(terminal_state, 1)
            ]

            if next_state is not None:
                [self.data["next_state"].append(x) for x in torch.split(next_state, 1)]
            else:
                [
                    self.data["next_state"].append((torch.zeros_like(s)))
                    for s in torch.split(state, 1)
                ]

            assert len(self) <= self.memory_size
