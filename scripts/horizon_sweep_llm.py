"""Horizon sweep for an LLM agent.

The thesis: a model's gap to the simple `greedy` heuristic should WIDEN as the
horizon grows, because longer horizons demand more credit assignment and belief
maintenance. If a mid-size model falls further behind at horizon 48 than at
12, it is failing specifically on long-horizon coherence.

We report the gap as a FRACTION of greedy's cost so horizons are comparable
(absolute cost grows with horizon trivially).

Usage:
    python scripts/horizon_sweep_llm.py --model qwen2.5:7b \
        --horizons 12 24 48 --episodes 4
"""

from __future__ import annotations

import argparse

from rescontre_env import baselines
from rescontre_env.dynamics import DynamicsConfig
from rescontre_env.eval import evaluate
from rescontre_env.llm_agent import LLMAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--counterparties", type=int, default=6)
    ap.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48])
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    print(f"Horizon sweep: model={args.model}, episodes={args.episodes}, "
          f"counterparties={args.counterparties}")
    # Primary metric is gap-to-ORACLE: the oracle is the stable ceiling. greedy
    # is shown for reference but is NOT a stable yardstick (it degrades at long
    # horizons, where the value of the hidden information kicks in).
    print(f"{'horizon':>8} {'LLM':>10} {'greedy':>10} {'oracle':>10} "
          f"{'gap_to_oracle':>14} {'gap %':>8} {'parse_fail':>11}")
    print("-" * 78)

    rows = []
    for h in args.horizons:
        cfg = DynamicsConfig(n_counterparties=args.counterparties, horizon=h)
        agent = LLMAgent(model=args.model, base_url=args.base_url,
                         api_key=args.api_key)
        # All evaluated on the SAME seed set (seed_base=0), so paired per horizon.
        llm = evaluate(agent, episodes=args.episodes, config=cfg)["mean_reward"]
        greedy = evaluate(baselines.greedy_policy(), episodes=args.episodes,
                          config=cfg)["mean_reward"]
        oracle = evaluate(baselines.oracle_greedy(), episodes=args.episodes,
                          config=cfg)["mean_reward"]
        gap = oracle - llm                       # >0 means LLM costs more than the ceiling
        gap_pct = 100.0 * gap / abs(oracle) if oracle != 0 else 0.0
        pf = agent.parse_failure_rate
        print(f"{h:>8} {llm:>10.1f} {greedy:>10.1f} {oracle:>10.1f} "
              f"{gap:>14.1f} {gap_pct:>7.1f}% {pf:>10.1%}")
        rows.append((h, llm, greedy, oracle, gap, gap_pct, pf))

    print("\nIf 'gap %' (LLM below the oracle ceiling) rises with horizon, the "
          "model falls further behind as the horizon grows: it is failing on "
          "long-horizon coherence specifically.")


if __name__ == "__main__":
    main()
