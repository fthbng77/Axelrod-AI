"""Policy Gradient (REINFORCE) agent for IPD.

Direct policy optimization approach — learns a stochastic policy
mapping states to action probabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from src.agents.base import BaseAgent
from src.environments.ipd import Action, NUM_FEATURES


class PolicyNetwork(nn.Module):
    """Policy network outputting action probabilities."""

    def __init__(self, state_dim: int, hidden_dims: list[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.output_layer = nn.Linear(prev_dim, 2)
        self.network = nn.Sequential(*layers)
        # Bias toward cooperation: C logit starts higher
        with torch.no_grad():
            self.output_layer.bias[0] = 1.0   # C bias
            self.output_layer.bias[1] = -1.0   # D bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.network(x)
        logits = self.output_layer(h)
        return torch.softmax(logits, dim=-1)


class PolicyGradientAgent(BaseAgent):
    """REINFORCE agent with baseline for variance reduction."""

    def __init__(
        self,
        name: str = "PolicyGradient",
        state_dim: int = NUM_FEATURES,
        hidden_dims: list[int] = [64, 32],
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        entropy_coef: float = 0.01,
        seed: Optional[int] = None,
    ):
        super().__init__(name, state_dim)
        self.gamma = discount_factor
        self.entropy_coef = entropy_coef
        self.training = True

        self.device = torch.device("cpu")
        self.policy_net = PolicyNetwork(state_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.entropies: list[torch.Tensor] = []

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> Action:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_t).squeeze()

        if self.training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action))
            self.entropies.append(dist.entropy())
            return Action(action.item())
        else:
            return Action(probs.argmax().item())

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

        self.rewards.append(reward)

        if done:
            self._optimize()

    def _optimize(self):
        """Run REINFORCE update at end of episode."""
        if not self.rewards:
            return

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, entropy, R in zip(self.log_probs, self.entropies, returns):
            policy_loss.append(-log_prob * R - self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def reset(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def get_policy(self) -> dict:
        return {"type": "policy_gradient"}

    def set_eval_mode(self):
        self.training = False
        self.policy_net.eval()

    def set_train_mode(self):
        self.training = True
        self.policy_net.train()
