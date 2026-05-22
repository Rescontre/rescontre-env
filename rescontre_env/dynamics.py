"""The latent generative model.

This is the research heart of the environment. Everything that makes
SettlementTiming a *long-horizon-agency* problem rather than a solved
optimization problem lives here:

  - latent counterparty TYPES the agent cannot see and must infer,
  - a latent MACRO REGIME (an HMM) that makes defaults CORRELATED,
  - NON-STATIONARITY (the regime drifts, so a type learned once is not learned
    forever),
  - a fee process whose volatility creates a genuine settle-now-vs-defer timing
    decision.

Every latent is discrete on purpose. That keeps exact Bayesian filtering
tractable (see `filter.py`), which is what lets us compute a principled
"value of information" gap rather than hand-waving about difficulty.

Design rule: this module decides *who exists, who defaults when, what fees are*.
It never touches the netting core. The stochastic world wraps the deterministic
settlement engine; it does not contaminate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- Latent spaces (hidden from the agent) ---------------------------------

REGIMES = ("calm", "stressed", "crisis")
TYPES = ("reliable", "flaky", "adversarial")

# P(counterparty pays on time | type, regime).
#
# This single table does the heavy lifting. The columns (regime) are the shared
# macro factor: slide z_t toward "crisis" and EVERY counterparty's pay
# probability drops together. That correlation is what stops the problem from
# decomposing into N independent bandits. The "adversarial" row builds trust in
# calm times then collapses under stress, a strategic-default pattern.
PAY_PROB = {
    #             calm   stressed  crisis
    "reliable":    (0.999, 0.995, 0.980),   # safe to hold across gas spikes
    "flaky":       (0.850, 0.600, 0.300),   # bleeds if you wait
    "adversarial": (0.950, 0.450, 0.080),   # fine until stress, then collapses
}


@dataclass
class DynamicsConfig:
    """All the knobs. These are inputs, never hardcoded in the env."""

    n_counterparties: int = 8
    horizon: int = 40                       # cycles per episode (the long-horizon dial)

    # Prior over counterparty types (reliable, flaky, adversarial).
    type_prior: tuple[float, float, float] = (0.5, 0.3, 0.2)

    # Macro regime Markov chain. Rows = from-regime, cols = to-regime.
    # Tuned so calm is sticky, crisis is rare but reachable: this is the
    # correlated tail-risk that punishes over-extension in good times.
    regime_transition: tuple = (
        (0.80, 0.15, 0.05),   # from calm
        (0.30, 0.45, 0.25),   # from stressed
        (0.15, 0.35, 0.50),   # from crisis  (sticky: crises last)
    )
    initial_regime: int = 0                 # start calm

    # Obligation arrivals per cycle ~ Poisson(arrival_rate). Each is a receivable
    # the agent is owed by a counterparty (amount in integer minor units).
    arrival_rate: float = 3.0
    amount_low: int = 50
    amount_high: int = 350

    # Settlement cost is GAS PER NET TRANSFER, not a rate on volume. This is what
    # on-chain settlement actually costs and what the netting core minimizes
    # (fewest transfers). Gas is VOLATILE and far worse in bad regimes, so the
    # core decision is timing: settle the safe counterparties when gas is cheap,
    # defer them through a spike, but never hold a shaky counterparty long enough
    # to default. Mean-reverting (OU); gas and defaults spike together.
    gas_mean_by_regime: tuple[float, float, float] = (30.0, 80.0, 200.0)
    gas_reversion: float = 0.25             # kappa: how fast gas pulls to its mean
    gas_vol: float = 20.0                   # sigma: gas noise
    gas_init: float = 30.0
    gas_floor: float = 8.0

    # Penalty (per unit) for a payable the agent fails to cover from cash.
    missed_penalty_rate: float = 0.5
    initial_cash: int = 2000


@dataclass
class WorldState:
    """The full latent + observable state. The agent sees only part of this."""

    rng: np.random.Generator
    cfg: DynamicsConfig
    cycle: int = 0
    regime: int = 0                          # latent z_t
    types: np.ndarray = field(default=None)  # latent c_i, shape (N,)
    gas_price: float = 0.0                    # per-net-transfer settlement cost
    cash: int = 0

    @classmethod
    def initial(cls, cfg: DynamicsConfig, rng: np.random.Generator) -> "WorldState":
        types = rng.choice(len(TYPES), size=cfg.n_counterparties, p=cfg.type_prior)
        return cls(
            rng=rng,
            cfg=cfg,
            cycle=0,
            regime=cfg.initial_regime,
            types=types,
            gas_price=cfg.gas_init,
            cash=cfg.initial_cash,
        )

    def pay_prob(self, counterparty: int) -> float:
        """Ground-truth P(pays) for a counterparty in the current regime.

        Hidden from the agent. Used to realize defaults and to compute the
        full-information ceiling baseline.
        """
        return PAY_PROB[TYPES[self.types[counterparty]]][self.regime]

    def step_macro(self) -> None:
        """Advance the latent regime and the fee process by one cycle.

        This is the non-stationarity: the regime can drift under the agent's
        feet, so beliefs about counterparty reliability go stale and must be
        re-estimated. Without this step, a planner with the current state wins
        and there is no agency problem.
        """
        P = np.asarray(self.cfg.regime_transition)
        self.regime = int(self.rng.choice(len(REGIMES), p=P[self.regime]))

        # OU gas update toward the regime-dependent mean.
        mu = self.cfg.gas_mean_by_regime[self.regime]
        shock = self.cfg.gas_vol * self.rng.standard_normal()
        self.gas_price += self.cfg.gas_reversion * (mu - self.gas_price) + shock
        self.gas_price = max(self.cfg.gas_floor, self.gas_price)
        self.cycle += 1

    def realize_default(self, counterparty: int) -> bool:
        """Sample whether a counterparty pays (True) or defaults (False) now."""
        return bool(self.rng.random() < self.pay_prob(counterparty))
