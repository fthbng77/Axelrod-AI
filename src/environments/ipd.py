"""Iterated Prisoner's Dilemma environment for training RL agents.

Payoff matrix (row player perspective):
              Cooperate    Defect
Cooperate      (R, R)      (S, T)
Defect         (T, S)      (P, P)

Default: R=3, T=5, S=0, P=1

State representation matches Harper 2017's 17-feature compute_features():
  [0]  opponent_first_c     [1]  opponent_first_d
  [2]  opponent_second_c    [3]  opponent_second_d
  [4]  my_prev_c            [5]  my_prev_d
  [6]  my_prev2_c           [7]  my_prev2_d
  [8]  opp_prev_c           [9]  opp_prev_d
  [10] opp_prev2_c          [11] opp_prev2_d
  [12] total_opp_c          [13] total_opp_d
  [14] total_my_c           [15] total_my_d
  [16] round_number
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

NUM_FEATURES = 17


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


def compute_features(
    my_history: list[Action], opp_history: list[Action], round_num: int
) -> np.ndarray:
    """Compute 17-feature state vector matching Harper 2017's ANN input.

    This is the exact same feature set used by Axelrod's ann.py compute_features(),
    ensuring our RL agents see the same information as Harper's EvolvedANN.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    n = len(opp_history)

    if n >= 1:
        features[0] = 1.0 if opp_history[0] == Action.COOPERATE else 0.0
        features[1] = 1.0 if opp_history[0] == Action.DEFECT else 0.0
        features[4] = 1.0 if my_history[-1] == Action.COOPERATE else 0.0
        features[5] = 1.0 if my_history[-1] == Action.DEFECT else 0.0
        features[8] = 1.0 if opp_history[-1] == Action.COOPERATE else 0.0
        features[9] = 1.0 if opp_history[-1] == Action.DEFECT else 0.0

    if n >= 2:
        features[2] = 1.0 if opp_history[1] == Action.COOPERATE else 0.0
        features[3] = 1.0 if opp_history[1] == Action.DEFECT else 0.0
        features[6] = 1.0 if my_history[-2] == Action.COOPERATE else 0.0
        features[7] = 1.0 if my_history[-2] == Action.DEFECT else 0.0
        features[10] = 1.0 if opp_history[-2] == Action.COOPERATE else 0.0
        features[11] = 1.0 if opp_history[-2] == Action.DEFECT else 0.0

    # Counts
    opp_c = sum(1 for a in opp_history if a == Action.COOPERATE)
    features[12] = float(opp_c)
    features[13] = float(n - opp_c)
    my_c = sum(1 for a in my_history if a == Action.COOPERATE)
    features[14] = float(my_c)
    features[15] = float(len(my_history) - my_c)
    features[16] = float(round_num)

    return features


class IPDEnvironment:
    """Multi-round IPD environment with Harper 2017 compatible 17-feature state."""

    def __init__(
        self,
        num_rounds: int = 200,
        noise: float = 0.0,
        payoff: Optional[PayoffMatrix] = None,
        seed: Optional[int] = None,
    ):
        self.num_rounds = num_rounds
        self.noise = noise
        self.payoff = payoff or PayoffMatrix()
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def state_dim(self) -> int:
        return NUM_FEATURES

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset environment for a new episode. Returns (state1, state2)."""
        self.round = 0
        self.history1: list[Action] = []
        self.history2: list[Action] = []
        self.scores = [0.0, 0.0]
        s1 = compute_features(self.history1, self.history2, self.round)
        s2 = compute_features(self.history2, self.history1, self.round)
        return s1, s2

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
        actual1 = self._apply_noise(action1)
        actual2 = self._apply_noise(action2)

        r1, r2 = self.payoff.get_payoffs(actual1, actual2)
        self.scores[0] += r1
        self.scores[1] += r2

        self.history1.append(actual1)
        self.history2.append(actual2)
        self.round += 1

        done = self.round >= self.num_rounds

        state1 = compute_features(self.history1, self.history2, self.round)
        state2 = compute_features(self.history2, self.history1, self.round)

        return state1, state2, r1, r2, done

    def get_cooperation_rate(self, player: int = 0) -> float:
        """Calculate cooperation rate for a player."""
        history = self.history1 if player == 0 else self.history2
        if not history:
            return 0.0
        return sum(1 for a in history if a == Action.COOPERATE) / len(history)
