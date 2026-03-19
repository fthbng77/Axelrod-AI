"""Base class for all RL agents."""

from abc import ABC, abstractmethod
import numpy as np
from src.environments.ipd import Action


class BaseAgent(ABC):
    """Abstract base for IPD RL agents."""

    def __init__(self, name: str, memory_depth: int = 3):
        self.name = name
        self.memory_depth = memory_depth

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Action:
        """Choose an action given the current state."""

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update the agent after observing a transition."""

    @abstractmethod
    def get_policy(self) -> dict:
        """Return a serializable representation of the learned policy."""

    def reset(self):
        """Reset episode-specific state (optional override)."""
        pass
