"""Self-play training pipeline.

Trains RL agents against each other and/or fixed Axelrod strategies.
Based on Harper 2017 methodology + modern improvements.
"""

import numpy as np
from typing import Optional
from src.environments.ipd import IPDEnvironment, Action
from src.agents.base import BaseAgent


class SelfPlayTrainer:
    """Train two agents via self-play in IPD."""

    def __init__(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        num_rounds: int = 200,
        noise: float = 0.0,
        memory_depth: int = 3,
        seed: Optional[int] = None,
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.env = IPDEnvironment(
            memory_depth=memory_depth,
            num_rounds=num_rounds,
            noise=noise,
            seed=seed,
        )
        self.history: list[dict] = []

    def train(
        self,
        num_episodes: int = 1000,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> list[dict]:
        """Run self-play training for specified episodes."""
        for episode in range(num_episodes):
            stats = self._run_episode()
            self.history.append(stats)

            if verbose and (episode + 1) % log_interval == 0:
                recent = self.history[-log_interval:]
                avg_s1 = np.mean([s["score1"] for s in recent])
                avg_s2 = np.mean([s["score2"] for s in recent])
                avg_c1 = np.mean([s["coop_rate1"] for s in recent])
                avg_c2 = np.mean([s["coop_rate2"] for s in recent])
                print(
                    f"Episode {episode+1:>5}/{num_episodes} | "
                    f"{self.agent1.name}: {avg_s1:.1f} (coop: {avg_c1:.2f}) | "
                    f"{self.agent2.name}: {avg_s2:.1f} (coop: {avg_c2:.2f})"
                )

        return self.history

    def _run_episode(self) -> dict:
        """Run a single episode of self-play."""
        state = self.env.reset()
        self.agent1.reset()
        self.agent2.reset()

        # Mirror state for agent2
        md = self.env.memory_depth
        state2 = np.concatenate([state[md:], state[:md]])

        for _ in range(self.env.num_rounds):
            action1 = self.agent1.select_action(state)
            action2 = self.agent2.select_action(state2)

            state1_next, state2_next, r1, r2, done = self.env.step(action1, action2)

            self.agent1.update(state, action1, r1, state1_next, done)
            self.agent2.update(state2, action2, r2, state2_next, done)

            # For LOLA agents: observe opponent actions
            if hasattr(self.agent1, "observe_opponent"):
                self.agent1.observe_opponent(state, action2)
            if hasattr(self.agent2, "observe_opponent"):
                self.agent2.observe_opponent(state2, action1)

            state = state1_next
            state2 = state2_next

        return {
            "score1": self.env.scores[0],
            "score2": self.env.scores[1],
            "coop_rate1": self.env.get_cooperation_rate(0),
            "coop_rate2": self.env.get_cooperation_rate(1),
        }


class PopulationTrainer:
    """Train agents against a population of opponents (like Harper 2017).

    Each agent plays against every strategy in the population,
    creating selection pressure for general-purpose strategies.
    """

    def __init__(
        self,
        agent: BaseAgent,
        opponents: list,
        num_rounds: int = 200,
        noise: float = 0.0,
        memory_depth: int = 3,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.opponents = opponents  # List of Axelrod player instances
        self.num_rounds = num_rounds
        self.noise = noise
        self.memory_depth = memory_depth
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def train(
        self,
        num_generations: int = 100,
        episodes_per_opponent: int = 5,
        verbose: bool = True,
    ) -> list[dict]:
        """Train agent against the full population."""
        for gen in range(num_generations):
            gen_scores = []

            for opp in self.opponents:
                for _ in range(episodes_per_opponent):
                    score = self._play_against_axelrod(opp)
                    gen_scores.append(score)

            stats = {
                "generation": gen,
                "avg_score": np.mean(gen_scores),
                "min_score": np.min(gen_scores),
                "max_score": np.max(gen_scores),
            }
            self.history.append(stats)

            if verbose and (gen + 1) % 10 == 0:
                print(
                    f"Gen {gen+1:>4}/{num_generations} | "
                    f"Avg: {stats['avg_score']:.1f} | "
                    f"Range: [{stats['min_score']:.0f}, {stats['max_score']:.0f}]"
                )

        return self.history

    def _play_against_axelrod(self, opponent) -> float:
        """Play our RL agent against an Axelrod strategy."""
        import axelrod

        env = IPDEnvironment(
            memory_depth=self.memory_depth,
            num_rounds=self.num_rounds,
            noise=self.noise,
            seed=int(self.rng.integers(0, 2**31)),
        )
        state = env.reset()
        self.agent.reset()
        opponent.reset()

        opp_history = []
        agent_history = []

        for _ in range(self.num_rounds):
            # Our agent chooses
            action1 = self.agent.select_action(state)

            # Axelrod opponent chooses
            if not agent_history:
                opp_action = opponent.strategy(axelrod.Player())
            else:
                # Build a mock opponent from our agent's perspective
                opp_action = self._get_axelrod_action(
                    opponent, agent_history, opp_history
                )

            action2 = Action.COOPERATE if opp_action == axelrod.Action.C else Action.DEFECT

            state1, _, r1, _, done = env.step(action1, action2)

            self.agent.update(state, action1, r1, state1, done)

            agent_history.append(
                axelrod.Action.C if action1 == Action.COOPERATE else axelrod.Action.D
            )
            opp_history.append(opp_action)
            state = state1

        return env.scores[0]

    def _get_axelrod_action(self, opponent, agent_history, opp_history):
        """Get action from an Axelrod strategy given histories."""
        import axelrod

        # Create mock players with the correct history
        mock_self = axelrod.Cooperator()
        mock_other = axelrod.Cooperator()
        mock_self.history = axelrod.History(opp_history)
        mock_other.history = axelrod.History(agent_history)

        opponent.history = axelrod.History(opp_history)
        return opponent.strategy(mock_other)
