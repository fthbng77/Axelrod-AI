"""LOLA: Learning with Opponent-Learning Awareness (Foerster et al. 2018).

Key idea: Instead of treating the opponent as stationary, LOLA agents
account for how their own policy update will affect the opponent's
learning, and vice versa. This leads to emergent cooperation in IPD.

Reference: arXiv:1709.04326
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from src.agents.base import BaseAgent
from src.environments.ipd import Action, NUM_FEATURES


class LOLAPolicy(nn.Module):
    """Differentiable policy for LOLA — needs gradient access to opponent."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, 2)
        # Bias toward cooperation
        with torch.no_grad():
            self.output_layer.bias[0] = 1.0   # C bias
            self.output_layer.bias[1] = -1.0   # D bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)

    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)


class LOLAAgent(BaseAgent):
    """LOLA agent that models and anticipates opponent learning.

    The agent maintains its own policy AND a differentiable model of
    the opponent's policy. During updates, it computes:
    1. Its own gradient (standard policy gradient)
    2. The opponent's expected gradient
    3. How the opponent's gradient affects its own future reward
    4. Adjusts its update to account for (3)
    """

    def __init__(
        self,
        name: str = "LOLA",
        state_dim: int = NUM_FEATURES,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        opponent_lr: float = 1e-3,
        lola_lr: float = 0.5,
        discount_factor: float = 0.95,
        use_opponent_model: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(name, state_dim)
        self.gamma = discount_factor
        self.lola_lr = lola_lr
        self.opponent_lr = opponent_lr
        self.use_opponent_model = use_opponent_model
        self.training = True

        self.device = torch.device("cpu")

        self.policy = LOLAPolicy(state_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.opponent_model = LOLAPolicy(state_dim, hidden_dim).to(self.device)
        self.opponent_optimizer = optim.Adam(
            self.opponent_model.parameters(), lr=opponent_lr
        )

        self.states: list[np.ndarray] = []
        self.opp_states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.opponent_actions: list[int] = []
        self.rewards: list[float] = []

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> Action:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_t).squeeze()

        if self.training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return Action(action.item())
        else:
            return Action(probs.argmax().item())

    def observe_opponent(self, own_state: np.ndarray, opp_state: np.ndarray, opponent_action: Action):
        """Record opponent's action and their state for opponent modeling."""
        self.opponent_actions.append(int(opponent_action))
        self.opp_states.append(opp_state)

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

        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(reward)

        if done:
            self._lola_update()

    def _lola_update(self):
        """LOLA update: standard PG + opponent-aware correction."""
        if not self.rewards:
            return

        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        opp_actions_t = torch.LongTensor(self.opponent_actions).to(self.device)

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns).to(self.device)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # === Step 1: Update opponent model ===
        has_opp_data = (
            self.use_opponent_model
            and len(self.opponent_actions) == len(self.states)
            and len(self.opp_states) == len(self.states)
        )
        if has_opp_data:
            opp_states_t = torch.FloatTensor(np.array(self.opp_states)).to(self.device)
            opp_log_probs = self.opponent_model.get_log_probs(
                opp_states_t, opp_actions_t
            )
            opp_loss = -(opp_log_probs * returns_t.detach()).mean()

            self.opponent_optimizer.zero_grad()
            opp_loss.backward()
            self.opponent_optimizer.step()

        # === Step 2: Standard policy gradient ===
        log_probs = self.policy.get_log_probs(states_t, actions_t)
        pg_loss = -(log_probs * returns_t.detach()).mean()

        # === Step 3: LOLA correction ===
        if has_opp_data:
            lola_correction = self._compute_lola_correction(
                states_t, actions_t, returns_t
            )
            total_loss = pg_loss + self.lola_lr * lola_correction
        else:
            total_loss = pg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.states.clear()
        self.opp_states.clear()
        self.actions.clear()
        self.opponent_actions.clear()
        self.rewards.clear()

    def _compute_lola_correction(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the LOLA correction term."""
        opp_states = torch.FloatTensor(np.array(self.opp_states)).to(self.device)
        opp_probs = self.opponent_model(opp_states)
        our_probs = self.policy(states)
        joint_prob = our_probs.gather(1, actions.unsqueeze(1)).squeeze()
        weighted = (joint_prob * returns).mean()
        return -weighted

    def reset(self):
        self.states.clear()
        self.opp_states.clear()
        self.actions.clear()
        self.opponent_actions.clear()
        self.rewards.clear()

    def get_policy(self) -> dict:
        return {"type": "lola", "lola_lr": self.lola_lr}

    def set_eval_mode(self):
        self.training = False
        self.policy.eval()

    def set_train_mode(self):
        self.training = True
        self.policy.train()
