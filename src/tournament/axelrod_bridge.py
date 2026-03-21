"""Bridge between our RL agents and the Axelrod tournament framework.

Wraps RL agents as Axelrod-compatible players so they can compete
directly in standard Axelrod tournaments against 200+ strategies.

Uses Axelrod's native compute_features from ann.py for state encoding,
ensuring our agents see the exact same 17-feature input as Harper's EvolvedANN.
"""

import numpy as np
import axelrod
from axelrod.strategies.ann import compute_features as axelrod_compute_features
from src.environments.ipd import Action
from src.agents.base import BaseAgent


class RLPlayer(axelrod.Player):
    """Wraps any BaseAgent as an Axelrod-compatible player."""

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
        if name:
            self.name = name

    def __repr__(self):
        return self.name

    def strategy(self, opponent: axelrod.Player) -> axelrod.Action:
        """Use Axelrod's native compute_features for state encoding."""
        features = axelrod_compute_features(self, opponent)
        state = features.astype(np.float32)
        action = self.agent.select_action(state)
        return axelrod.Action.C if action == Action.COOPERATE else axelrod.Action.D

    def reset(self):
        super().reset()
        self.agent.reset()


def create_rl_player(agent: BaseAgent, name: str = None) -> RLPlayer:
    """Factory function to create Axelrod-compatible RL players.

    Creates a unique subclass per agent so Axelrod sees distinct names.
    """
    display_name = name or agent.name
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

    for agent in rl_agents:
        players.append(create_rl_player(agent, name=agent.name))

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


def print_results(results: axelrod.ResultSet, top_n: int = 30):
    """Pretty-print tournament results."""
    print(f"\n{'Rank':>4} {'Strategy':<35} {'Score/Turn':>10}")
    print("-" * 55)

    for i, (name, score) in enumerate(
        zip(results.ranked_names[:top_n], results.normalised_scores[:top_n])
    ):
        avg = sum(score) / len(score)
        print(f"{i+1:>4}. {name:<33} {avg:>10.4f}")
