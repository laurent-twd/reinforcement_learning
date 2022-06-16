from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class Action:
    pass


class Environment(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    @abstractmethod
    def reward(self, action: Action, *wargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action, *wargs):
        raise NotImplementedError
