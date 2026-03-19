"""Deep Q-Network agent for IPD.

Extends Harper 2017's EvolvedANN approach with modern DQN techniques:
- Experience replay
- Target network
- Configurable architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional
from src.agents.base import BaseAgent
from src.environments.ipd import Action


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_dim: int, hidden_dims: list[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # 2 actions: C or D
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DeepQAgent(BaseAgent):
    """DQN agent for IPD with experience replay and target network."""

    def __init__(
        self,
        name: str = "Deep-Q",
        memory_depth: int = 3,
        hidden_dims: list[int] = [64, 32],
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(name, memory_depth)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.step_count = 0

        state_dim = memory_depth * 2
        self.device = torch.device("cpu")

        # Networks
        self.q_net = QNetwork(state_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray) -> Action:
        if self.training and self.rng.random() < self.epsilon:
            return Action(self.rng.integers(0, 2))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return Action(q_values.argmax(dim=1).item())

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

        self.replay_buffer.push(state, int(action), reward, next_state, done)
        self.step_count += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample and train
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def get_policy(self) -> dict:
        return {
            "type": "deep_q",
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "model_state": {
                k: v.tolist() for k, v in self.q_net.state_dict().items()
            },
        }

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def set_eval_mode(self):
        self.training = False
        self.q_net.eval()

    def set_train_mode(self):
        self.training = True
        self.q_net.train()
