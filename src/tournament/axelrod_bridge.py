"""Bridge between our RL agents and the Axelrod tournament framework.

Wraps RL agents as Axelrod-compatible players so they can compete
directly in standard Axelrod tournaments against 200+ strategies.
"""

import numpy as np
import axelrod
from src.environments.ipd import Action
from src.agents.base import BaseAgent


class RLPlayer(axelrod.Player):
    """Wraps any BaseAgent as an Axelrod-compatible player.

    This is how our trained RL agents enter Axelrod tournaments.
    """

    name = "RL Agent"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, agent: BaseAgent, name: str = None):
        super().__init__()
        self.agent = agent
        self.agent.set_eval_mode()
        self.memory_depth_val = agent.memory_depth
        if name:
            self.name = name

    def strategy(self, opponent: axelrod.Player) -> axelrod.Action:
        """Convert state from Axelrod histories → RL state → action."""
        state = self._build_state(opponent)
        action = self.agent.select_action(state)
        return axelrod.Action.C if action == Action.COOPERATE else axelrod.Action.D

    def _build_state(self, opponent: axelrod.Player) -> np.ndarray:
        """Build RL state vector from Axelrod history objects."""
        md = self.memory_depth_val
        state = np.full(md * 2, -1.0)

        # Own recent actions
        for i in range(min(len(self.history), md)):
            action = self.history[-(i + 1)]
            state[md - 1 - i] = 0.0 if action == axelrod.Action.C else 1.0

        # Opponent's recent actions
        for i in range(min(len(opponent.history), md)):
            action = opponent.history[-(i + 1)]
            state[md + md - 1 - i] = 0.0 if action == axelrod.Action.C else 1.0

        return state

    def reset(self):
        super().reset()
        self.agent.reset()


def create_rl_player(agent: BaseAgent, name: str = None) -> RLPlayer:
    """Factory function to create Axelrod-compatible RL players.

    Creates a unique subclass per agent so Axelrod sees distinct names.
    """
    display_name = name or agent.name
    # Axelrod uses class-level `name`, so create a dynamic subclass
    player_cls = type(
        f"RLPlayer_{display_name}",
        (RLPlayer,),
        {"name": display_name},
    )
    player = player_cls(agent, display_name)
    return player


def run_tournament(
    rl_agents: list[BaseAgent],
    include_classics: bool = True,
    include_evolved: bool = True,
    turns: int = 200,
    repetitions: int = 5,
    seed: int = 42,
) -> axelrod.ResultSet:
    """Run an Axelrod tournament with RL agents + optional classic/evolved strategies."""

    players = []

    # Add our RL agents
    for agent in rl_agents:
        players.append(create_rl_player(agent, name=agent.name))

    # Add classic strategies
    if include_classics:
        classics = [
            axelrod.TitForTat(),
            axelrod.Cooperator(),
            axelrod.Defector(),
            axelrod.Grudger(),
            axelrod.GTFT(),
            axelrod.WinStayLoseShift(),
            axelrod.SuspiciousTitForTat(),
            axelrod.TitFor2Tats(),
            axelrod.Random(),
            axelrod.HardGoByMajority(),
        ]
        players.extend(classics)

    # Add Harper 2017 evolved strategies
    if include_evolved:
        evolved = [
            axelrod.EvolvedANN(),
            axelrod.EvolvedANN5(),
            axelrod.EvolvedANNNoise05(),
            axelrod.EvolvedFSM16(),
            axelrod.EvolvedFSM16Noise05(),
            axelrod.EvolvedFSM4(),
            axelrod.EvolvedFSM6(),
            axelrod.EvolvedHMM5(),
            axelrod.EvolvedLookerUp1_1_1(),
            axelrod.EvolvedLookerUp2_2_2(),
            axelrod.EvolvedAttention(),
            axelrod.PSOGambler1_1_1(),
            axelrod.PSOGambler2_2_2(),
            axelrod.PSOGambler2_2_2_Noise05(),
            axelrod.PSOGamblerMem1(),
        ]
        players.extend(evolved)

    tournament = axelrod.Tournament(
        players, turns=turns, repetitions=repetitions, seed=seed
    )
    results = tournament.play(progress_bar=False)
    return results


def print_results(results: axelrod.ResultSet, top_n: int = 25):
    """Pretty-print tournament results."""
    print(f"\n{'Rank':>4} {'Strategy':<35} {'Score/Turn':>10}")
    print("-" * 55)

    for i, (name, score) in enumerate(
        zip(results.ranked_names[:top_n], results.normalised_scores[:top_n])
    ):
        avg = sum(score) / len(score)
        print(f"{i+1:>4}. {name:<33} {avg:>10.4f}")
