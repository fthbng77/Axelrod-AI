"""Base class for all RL agents."""

from abc import ABC, abstractmethod
import numpy as np
from src.environments.ipd import Action, NUM_FEATURES


class BaseAgent(ABC):
    """Abstract base for IPD RL agents."""

    def __init__(self, name: str, state_dim: int = NUM_FEATURES):
        self.name = name
        self.state_dim = state_dim

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
