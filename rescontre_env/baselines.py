"""Baseline policies.

A policy is a callable `(obs_dict, info) -> action`, where action is a length-N
array of 0 (WAIT) / 1 (COLLECT). Honest policies ignore `info`; only the oracle
reads the hidden latents, and it is labelled as such.

The point of these baselines is the GRADIENT. A single policy's score is
meaningless; the benchmark's signal is the spread:

    random  <  greedy  <  [ a good learned policy ]  <  oracle (sees latents)

If a learned agent cannot beat greedy, it has learned nothing. The gap between
the best history-only policy and the latent-peeking oracle is the (directional)
value of information: the cost of having to infer the hidden state.
"""

from __future__ import annotations

import numpy as np

from .dynamics import PAY_PROB, TYPES
from .env import COLLECT, WAIT


def random_policy(rng: np.random.Generator):
    """COLLECT each counterparty with probability 0.5."""

    def policy(obs, info=None):
        n = len(obs["counterparties"])
        return (rng.random(n) < 0.5).astype(int)

    return policy


def always_collect(obs, info=None):
    n = len(obs["counterparties"])
    return np.full(n, COLLECT, dtype=int)


def always_wait(obs, info=None):
    n = len(obs["counterparties"])
    return np.full(n, WAIT, dtype=int)


def greedy_policy(default_rate_threshold: float = 0.20, max_age: int = 6,
                  cheap_gas: float = 50.0):
    """Myopic, history-only heuristic. Two observable signals: history + gas.

    COLLECT a counterparty's receivables when any of:
      - its observed default rate is high (looks risky -> grab before default),
      - gas is currently cheap (good moment to clear safe balances),
      - it has a payable due now (collecting nets against it),
      - the receivable has aged out (don't hold forever -> terminal write-off).
    Otherwise WAIT (hold a safe counterparty through an expensive gas spike).
    Uses only observable signal, no latents.
    """

    def policy(obs, info=None):
        gas = obs["gas_price"]
        action = []
        for cp in obs["counterparties"]:
            if cp["receivable"] <= 0:
                action.append(WAIT)
                continue
            seen = cp["times_paid"] + cp["times_defaulted"]
            default_rate = cp["times_defaulted"] / seen if seen > 0 else 0.0
            collect = (
                default_rate >= default_rate_threshold
                or gas <= cheap_gas
                or cp["payable_due_now"] > 0
                or cp["oldest_age"] >= max_age
            )
            action.append(COLLECT if collect else WAIT)
        return np.asarray(action, dtype=int)

    return policy


RELIABLE = TYPES.index("reliable")


def oracle_greedy(cheap_gas: float = 50.0):
    """Latent-peeking reference (NOT an honest policy).

    Reads the true types from `info` and uses them the way a perfectly-informed
    agent would: dump every NON-reliable counterparty early (grab it before a
    regime shift can default it), and hold only the genuinely reliable ones,
    settling them when gas is cheap or a payable nets against them. The honest
    agent has to *infer* which is which from default history, and pays for that
    lesson; this oracle never does. It is a directional upper-bound reference,
    not the true POMDP optimum.
    """

    def policy(obs, info=None):
        assert info is not None, "oracle needs info (true latents)"
        types = info["true_types"]
        gas = obs["gas_price"]
        action = []
        for cp in obs["counterparties"]:
            if cp["receivable"] <= 0:
                action.append(WAIT)
                continue
            is_reliable = types[cp["id"]] == RELIABLE
            collect = (
                (not is_reliable)            # known-risky: dump before it defaults
                or gas <= cheap_gas          # reliable: clear it when gas is cheap
                or cp["payable_due_now"] > 0
            )
            action.append(COLLECT if collect else WAIT)
        return np.asarray(action, dtype=int)

    return policy
