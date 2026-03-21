"""Main training + evaluation pipeline.

Phase 1: Self-play training (Q-Learning, DQN, Policy Gradient, LOLA)
Phase 2: Tournament evaluation against Harper 2017 baselines

Goal: Beat Harper 2017's evolved strategies in Axelrod tournaments.
Uses Harper-compatible 17-feature state representation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from src.agents.q_learning import QLearningAgent
from src.agents.deep_q import DeepQAgent
from src.agents.policy_gradient import PolicyGradientAgent
from src.agents.lola import LOLAAgent
from src.training.self_play import SelfPlayTrainer
from src.tournament.axelrod_bridge import run_tournament, print_results


def phase1_self_play(num_episodes: int = 2000) -> dict:
    """Phase 1: Train agents via self-play."""
    print("=" * 60)
    print("PHASE 1: Self-Play Training (17-feature Harper state)")
    print("=" * 60)

    agents = {}

    # --- Q-Learning Self-Play ---
    print("\n--- Q-Learning Self-Play ---")
    q1 = QLearningAgent("Q-Learning", learning_rate=0.15,
                         discount_factor=0.95, epsilon=0.3, seed=42)
    q2 = QLearningAgent("Q-Learning-2", learning_rate=0.15,
                         discount_factor=0.95, epsilon=0.3, seed=123)
    trainer = SelfPlayTrainer(q1, q2, num_rounds=200, seed=42)
    trainer.train(num_episodes=num_episodes, log_interval=500)
    agents["Q-Learning"] = q1

    # --- Deep Q Self-Play ---
    print("\n--- Deep Q-Network Self-Play ---")
    dq1 = DeepQAgent("Deep-Q", hidden_dims=[128, 64],
                      learning_rate=5e-4, epsilon=1.0, seed=42)
    dq2 = DeepQAgent("Deep-Q-2", hidden_dims=[128, 64],
                      learning_rate=5e-4, epsilon=1.0, seed=123)
    trainer = SelfPlayTrainer(dq1, dq2, num_rounds=200, seed=42)
    trainer.train(num_episodes=num_episodes, log_interval=500)
    agents["Deep-Q"] = dq1

    # --- Policy Gradient Self-Play ---
    print("\n--- Policy Gradient Self-Play ---")
    pg1 = PolicyGradientAgent("PolicyGrad", hidden_dims=[128, 64],
                               learning_rate=3e-4, entropy_coef=0.05, seed=42)
    pg2 = PolicyGradientAgent("PolicyGrad-2", hidden_dims=[128, 64],
                               learning_rate=3e-4, entropy_coef=0.05, seed=123)
    trainer = SelfPlayTrainer(pg1, pg2, num_rounds=200, seed=42)
    trainer.train(num_episodes=num_episodes, log_interval=500)
    agents["PolicyGrad"] = pg1

    # --- LOLA Self-Play ---
    print("\n--- LOLA Self-Play ---")
    lola1 = LOLAAgent("LOLA", hidden_dim=128,
                       learning_rate=1e-3, lola_lr=0.3, seed=42)
    lola2 = LOLAAgent("LOLA-2", hidden_dim=128,
                       learning_rate=1e-3, lola_lr=0.3, seed=123)
    trainer = SelfPlayTrainer(lola1, lola2, num_rounds=200, seed=42)
    trainer.train(num_episodes=num_episodes, log_interval=500)
    agents["LOLA"] = lola1

    return agents


def phase2_tournament(agents: dict) -> None:
    """Phase 2: Tournament against Harper 2017 + classics + EvolvedAttention."""
    print("\n" + "=" * 60)
    print("PHASE 2: Tournament Evaluation (incl. EvolvedAttention)")
    print("=" * 60)

    rl_agents = list(agents.values())
    results = run_tournament(
        rl_agents,
        include_classics=True,
        include_evolved=True,
        turns=200,
        repetitions=5,
        seed=42,
    )

    print_results(results)

    # Summary
    print("\n--- Our RL Agents' Rankings ---")
    our_names = {a.name for a in rl_agents}
    for i, name in enumerate(results.ranked_names):
        if name in our_names:
            avg = sum(results.normalised_scores[i]) / len(results.normalised_scores[i])
            print(f"  #{i+1}: {name} (score/turn: {avg:.4f})")

    harper_names = {
        "Evolved ANN", "Evolved ANN 5", "Evolved ANN 5 Noise 05",
        "Evolved FSM 16", "Evolved FSM 16 Noise 05", "Evolved FSM 4",
        "Evolved FSM 6", "Evolved HMM 5", "EvolvedLookerUp1_1_1",
        "EvolvedLookerUp2_2_2", "EvolvedAttention",
        "PSO Gambler 1_1_1", "PSO Gambler 2_2_2",
        "PSO Gambler 2_2_2 Noise 05", "PSO Gambler Mem1",
    }

    harper_scores = []
    our_scores = []
    for i, (name, score) in enumerate(
        zip(results.ranked_names, results.normalised_scores)
    ):
        avg = sum(score) / len(score)
        if name in harper_names:
            harper_scores.append(avg)
        if name in our_names:
            our_scores.append(avg)

    if harper_scores and our_scores:
        print(f"\n  Harper 2017 avg: {np.mean(harper_scores):.4f}")
        print(f"  Our RL avg:      {np.mean(our_scores):.4f}")
        diff = np.mean(our_scores) - np.mean(harper_scores)
        print(f"  Difference:      {'+' if diff > 0 else ''}{diff:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate RL agents for IPD")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    parser.add_argument("--skip-training", action="store_true", help="Skip to tournament")
    args = parser.parse_args()

    if not args.skip_training:
        agents = phase1_self_play(num_episodes=args.episodes)
    else:
        agents = {
            "Q-Learning": QLearningAgent("Q-Learning", seed=42),
            "Deep-Q": DeepQAgent("Deep-Q", seed=42),
            "PolicyGrad": PolicyGradientAgent("PolicyGrad", seed=42),
            "LOLA": LOLAAgent("LOLA", seed=42),
        }

    phase2_tournament(agents)
