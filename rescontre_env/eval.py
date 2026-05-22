"""Evaluation loop with common-random-number paired comparison.

Every policy is run on the SAME set of episode seeds. Because the seed fully
determines the world (arrivals, types, regime path, fees), policy A and policy B
face identical worlds on seed k. That turns the comparison into a paired test
and kills most of the variance that would otherwise drown the signal.

The output is the gradient table: random < greedy < ... < oracle. That table is
the first figure of the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import baselines
from .dynamics import DynamicsConfig
from .env import make


@dataclass
class EpisodeResult:
    reward: float
    fee_cost: float
    default_loss: float
    missed_penalty: float
    conservation_violations: int


def run_episode(env, policy, seed: int) -> EpisodeResult:
    obs_vec, info = env.reset(seed=seed)
    obs = env.observation_dict()
    total = fee = dflt = missed = 0.0
    violations = 0
    done = False
    while not done:
        action = policy(obs, info)
        obs_vec, reward, terminated, truncated, info = env.step(action)
        obs = env.observation_dict()
        total += reward
        fee += info.get("fee_cost", 0)
        dflt += info.get("default_loss", 0)
        missed += info.get("missed_penalty", 0)
        if not info.get("conservation_ok", True):
            violations += 1
        done = terminated or truncated
    return EpisodeResult(total, fee, dflt, missed, violations)


def evaluate(policy, episodes: int = 200, config: DynamicsConfig | None = None,
             seed_base: int = 0) -> dict:
    env = make(config)
    results = [run_episode(env, policy, seed_base + k) for k in range(episodes)]
    rewards = np.array([r.reward for r in results])
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "sem": float(rewards.std() / np.sqrt(len(rewards))),
        "mean_fee": float(np.mean([r.fee_cost for r in results])),
        "mean_default": float(np.mean([r.default_loss for r in results])),
        "mean_missed": float(np.mean([r.missed_penalty for r in results])),
        "conservation_violations": int(sum(r.conservation_violations for r in results)),
        "rewards": rewards,
    }


def standard_suite(episodes: int = 200, config: DynamicsConfig | None = None,
                   extra: dict | None = None) -> dict:
    """Run the standard baseline gradient. `extra` adds named policies (e.g. an LLM)."""
    rng = np.random.default_rng(0)
    policies = {
        "always_wait": baselines.always_wait,
        "random": baselines.random_policy(rng),
        "always_collect": baselines.always_collect,
        "greedy": baselines.greedy_policy(),
        "oracle (sees latents)": baselines.oracle_greedy(),
    }
    if extra:
        policies.update(extra)
    return {
        name: evaluate(p, episodes=episodes, config=config)
        for name, p in policies.items()
    }


def print_table(report: dict) -> None:
    print(f"\n{'policy':<26} {'mean reward':>14} {'± sem':>9} "
          f"{'fee':>9} {'default':>9} {'missed':>9} {'cons.viol':>10}")
    print("-" * 92)
    ranked = sorted(report.items(), key=lambda kv: kv[1]["mean_reward"])
    for name, r in ranked:
        print(f"{name:<26} {r['mean_reward']:>14.1f} {r['sem']:>9.1f} "
              f"{r['mean_fee']:>9.1f} {r['mean_default']:>9.1f} "
              f"{r['mean_missed']:>9.1f} {r['conservation_violations']:>10}")
    print("\n(reward is negative cost; higher = better. higher fee means more "
          "settling; higher default means more was lost to defaults.)")


def main() -> None:  # console entry point: rescontre-eval
    import argparse

    ap = argparse.ArgumentParser(description="SettlementTiming-v0 baseline eval")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--counterparties", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--llm", action="store_true", help="also evaluate an LLM agent")
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--llm-episodes", type=int, default=20,
                    help="episodes for the LLM (kept small; inference is slow)")
    args = ap.parse_args()

    cfg = DynamicsConfig(n_counterparties=args.counterparties, horizon=args.horizon)

    if args.llm:
        # Run baselines AND the LLM on the SAME (small) seed set so the
        # comparison is paired. Inference is slow, so the whole suite uses the
        # smaller llm_episodes count here.
        from .llm_agent import LLMAgent
        agent = LLMAgent(model=args.model, base_url=args.base_url, api_key=args.api_key)
        print(f"Evaluating baselines + LLM '{args.model}' on {args.llm_episodes} "
              f"episodes (model is called up to "
              f"{args.llm_episodes * args.horizon} times)...")
        report = standard_suite(
            episodes=args.llm_episodes, config=cfg,
            extra={f"LLM:{args.model}": agent},
        )
        print_table(report)
        print(f"\nLLM '{args.model}' parse-failure rate: "
              f"{agent.parse_failure_rate:.1%}")
    else:
        report = standard_suite(episodes=args.episodes, config=cfg)
        print_table(report)


if __name__ == "__main__":
    main()
