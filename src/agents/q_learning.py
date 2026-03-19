"""Tabular Q-Learning agent for IPD.

Based on: "Self-Play Q-learners Can Provably Collude in the IPD" (2023)
Key insight: epsilon-greedy Q-learning agents naturally discover cooperation
when discount factor and exploration are tuned correctly.
"""

import numpy as np
from collections import defaultdict
from typing import Optional
from src.agents.base import BaseAgent
from src.environments.ipd import Action


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning with state discretization for IPD."""

    def __init__(
        self,
        name: str = "Q-Learner",
        memory_depth: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ):
        super().__init__(name, memory_depth)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # Q-table: state_key -> [Q(cooperate), Q(defect)]
        self.q_table: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(2)
        )
        self.training = True

    def _state_key(self, state: np.ndarray) -> str:
        """Discretize continuous state into a hashable key."""
        return tuple(int(s) for s in state).__repr__()

    def select_action(self, state: np.ndarray) -> Action:
        if self.training and self.rng.random() < self.epsilon:
            return Action(self.rng.integers(0, 2))

        key = self._state_key(state)
        q_values = self.q_table[key]

        if q_values[0] == q_values[1]:
            return Action(self.rng.integers(0, 2))
        return Action(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        if not self.training:
            return

        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Q-learning update
        best_next = np.max(self.q_table[next_key]) if not done else 0.0
        target = reward + self.gamma * best_next
        self.q_table[key][int(action)] += self.lr * (
            target - self.q_table[key][int(action)]
        )

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> dict:
        policy = {}
        for key, q_values in self.q_table.items():
            policy[key] = {
                "cooperate_q": float(q_values[0]),
                "defect_q": float(q_values[1]),
                "action": "C" if q_values[0] >= q_values[1] else "D",
            }
        return policy

    def set_eval_mode(self):
        self.training = False

    def set_train_mode(self):
        self.training = True
