"""Iterated Prisoner's Dilemma environment for training RL agents.

Payoff matrix (row player perspective):
              Cooperate    Defect
Cooperate      (R, R)      (S, T)
Defect         (T, S)      (P, P)

Default: R=3, T=5, S=0, P=1
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional


class Action(IntEnum):
    COOPERATE = 0
    DEFECT = 1


@dataclass
class PayoffMatrix:
    R: float = 3.0  # Reward (mutual cooperation)
    T: float = 5.0  # Temptation (defect vs cooperate)
    S: float = 0.0  # Sucker (cooperate vs defect)
    P: float = 1.0  # Punishment (mutual defection)

    def get_payoffs(self, action1: Action, action2: Action) -> tuple[float, float]:
        matrix = {
            (Action.COOPERATE, Action.COOPERATE): (self.R, self.R),
            (Action.COOPERATE, Action.DEFECT): (self.S, self.T),
            (Action.DEFECT, Action.COOPERATE): (self.T, self.S),
            (Action.DEFECT, Action.DEFECT): (self.P, self.P),
        }
        return matrix[(action1, action2)]


class IPDEnvironment:
    """Multi-round IPD environment supporting configurable memory depth."""

    def __init__(
        self,
        memory_depth: int = 3,
        num_rounds: int = 200,
        noise: float = 0.0,
        payoff: Optional[PayoffMatrix] = None,
        seed: Optional[int] = None,
    ):
        self.memory_depth = memory_depth
        self.num_rounds = num_rounds
        self.noise = noise
        self.payoff = payoff or PayoffMatrix()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset environment for a new episode."""
        self.round = 0
        self.history1: list[Action] = []
        self.history2: list[Action] = []
        self.scores = [0.0, 0.0]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Encode state as the last `memory_depth` actions of both players.

        State vector: [own_actions..., opponent_actions...]
        Each action encoded as: 0=cooperate, 1=defect, -1=no history yet
        """
        state = np.full(self.memory_depth * 2, -1.0)

        for i in range(min(len(self.history1), self.memory_depth)):
            idx = self.memory_depth - 1 - i
            state[idx] = float(self.history1[-(i + 1)])

        for i in range(min(len(self.history2), self.memory_depth)):
            idx = self.memory_depth * 2 - 1 - i
            state[self.memory_depth + (self.memory_depth - 1 - i)] = float(
                self.history2[-(i + 1)]
            )

        return state

    def _apply_noise(self, action: Action) -> Action:
        """Flip action with probability `noise`."""
        if self.noise > 0 and self.rng.random() < self.noise:
            return Action(1 - action)
        return action

    def step(
        self, action1: Action, action2: Action
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
        """Execute one round.

        Returns: (state1, state2, reward1, reward2, done)
        """
        # Apply noise
        actual1 = self._apply_noise(action1)
        actual2 = self._apply_noise(action2)

        # Get payoffs
        r1, r2 = self.payoff.get_payoffs(actual1, actual2)
        self.scores[0] += r1
        self.scores[1] += r2

        # Update histories
        self.history1.append(actual1)
        self.history2.append(actual2)
        self.round += 1

        done = self.round >= self.num_rounds

        # Return states from each player's perspective
        state1 = self._get_state()
        # Player 2's state is mirrored
        state2 = np.concatenate(
            [state1[self.memory_depth :], state1[: self.memory_depth]]
        )

        return state1, state2, r1, r2, done

    def get_cooperation_rate(self, player: int = 0) -> float:
        """Calculate cooperation rate for a player."""
        history = self.history1 if player == 0 else self.history2
        if not history:
            return 0.0
        return sum(1 for a in history if a == Action.COOPERATE) / len(history)
