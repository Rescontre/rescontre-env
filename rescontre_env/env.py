"""SettlementTiming-v0 — a verifiable-reward, long-horizon-agency environment.

The agent is a clearing member with a book of obligations against N
counterparties. Each cycle, for every counterparty, it chooses:

    COLLECT  settle this counterparty's outstanding receivables now. They are
             batched through the real netting core (so they can cancel against
             a payable due this cycle, lowering the fee), the agent pays a fee
             on the netted volume, and the money becomes safe cash.

    WAIT     do nothing this cycle. No fee. But each waited receivable's
             counterparty may DEFAULT (correlated across counterparties via the
             hidden macro regime), writing the receivable off as a loss.

Reward is negative economic cost:  -(fees + default losses + missed-payable
penalties). At episode end every uncollected receivable is written off, so
"wait forever" is not a strategy. Every settlement is gated on the netting
conservation check; a step that fails conservation scores a hard penalty. That
gate is the reward-integrity guarantee: you cannot score well by cheating the
books.

Why this is long-horizon agency and not a solved optimization:
  - The right COLLECT/WAIT call depends on a counterparty's hidden TYPE, which
    the agent can only infer from its observed pay/default history.
  - Defaults are CORRELATED through a hidden regime that DRIFTS, so a type
    learned in calm times goes stale in a crisis.
  - Fees move (worse in crisis), so "settle now vs defer and net" is a live
    decision, not a constant.

The agent never observes the regime, the types, or the pay probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import netting
from .dynamics import DynamicsConfig, WorldState

# Optional Gymnasium base. The env follows the Gymnasium API regardless; if the
# package is installed we inherit from it (and register the id), otherwise we
# fall back to a tiny shim so the package runs with numpy alone.
try:  # pragma: no cover - import shim
    import gymnasium as gym
    from gymnasium import spaces

    _GYM = True
    _Base = gym.Env
except Exception:  # pragma: no cover
    _GYM = False
    _Base = object
    spaces = None


WAIT, COLLECT = 0, 1
CONSERVATION_VIOLATION_PENALTY = 10_000  # should never trigger; it is a tripwire


@dataclass
class _Receivable:
    cp: int
    amount: int
    age: int = 0


@dataclass
class _Payable:
    cp: int
    amount: int
    due_in: int


@dataclass
class _Book:
    """The agent's outstanding obligations, plus what it has observed."""

    receivables: list[_Receivable] = field(default_factory=list)
    payables: list[_Payable] = field(default_factory=list)
    paid_count: np.ndarray = None      # observed: times each cp paid on collect
    default_count: np.ndarray = None   # observed: times each cp defaulted


class SettlementTimingEnv(_Base):
    """Gymnasium-compatible SettlementTiming-v0."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: DynamicsConfig | None = None):
        super().__init__()
        self.cfg = config or DynamicsConfig()
        n = self.cfg.n_counterparties

        if _GYM:
            # Action: per counterparty, WAIT(0) or COLLECT(1).
            self.action_space = spaces.MultiBinary(n)
            # Observation is exposed as a flat vector for RL; the dict form (for
            # LLM agents and humans) is available via `observation_dict`.
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32
            )

        self._world: WorldState | None = None
        self._book: _Book | None = None
        self._last_obs_dict: dict | None = None

    # ---- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        rng = np.random.default_rng(seed)
        self._world = WorldState.initial(self.cfg, rng)
        n = self.cfg.n_counterparties
        self._book = _Book(
            receivables=[],
            payables=[],
            paid_count=np.zeros(n, dtype=np.int64),
            default_count=np.zeros(n, dtype=np.int64),
        )
        self._arrivals()  # seed cycle 0 with some obligations
        obs_dict = self._build_obs()
        return self._flatten(obs_dict), self._info()

    def step(self, action):
        assert self._world is not None and self._book is not None, "call reset() first"
        action = np.asarray(action).astype(int).reshape(-1)
        w, book = self._world, self._book
        n = self.cfg.n_counterparties

        fee_cost = 0
        default_loss = 0
        missed_penalty = 0
        conservation_ok = True

        # 1. Build this cycle's settlement graph from COLLECT decisions and any
        #    payables that are due now. agent is node "agent"; counterparties
        #    are "cp{i}". COLLECT => cp -> agent edge; due payable => agent -> cp.
        graph: netting.Graph = {}

        def add_edge(frm: str, to: str, amt: int):
            if amt <= 0:
                return
            graph.setdefault(frm, {})
            graph.setdefault(to, {})
            graph[frm][to] = graph[frm].get(to, 0) + amt

        collected: list[_Receivable] = []
        kept_recv: list[_Receivable] = []
        for r in book.receivables:
            if action[r.cp] == COLLECT:
                add_edge(f"cp{r.cp}", "agent", r.amount)
                collected.append(r)
            else:
                kept_recv.append(r)

        due_payables: list[_Payable] = []
        kept_pay: list[_Payable] = []
        for p in book.payables:
            if p.due_in <= 0:
                add_edge("agent", f"cp{p.cp}", p.amount)
                due_payables.append(p)
            else:
                kept_pay.append(p)

        # 2. Net the graph through the production core and verify conservation.
        transfers = netting.compute_net_transfers(graph)
        if not netting.conservation_holds(graph, transfers):
            conservation_ok = False

        settled = netting.settled_volume(transfers)
        # Cost is GAS PER NET TRANSFER. The netting core already minimized the
        # transfer count; the agent's lever is timing (batch a counterparty's
        # receivables into one later settlement = one transfer, not one each
        # cycle) and netting receivables against payables to the same party.
        fee_cost = int(round(w.gas_price * len(transfers)))

        # 3. Realize cash flows. Collecting pulls the asset into safe cash;
        #    counterparties that settle now pay (no default on a live settle).
        for r in collected:
            w.cash += r.amount
            book.paid_count[r.cp] += 1
        for p in due_payables:
            if w.cash >= p.amount:
                w.cash -= p.amount
            else:
                shortfall = p.amount - max(0, w.cash)
                missed_penalty += int(round(self.cfg.missed_penalty_rate * shortfall))
                w.cash = max(0, w.cash - p.amount)
        w.cash -= fee_cost

        # 4. Defaults strike the receivables the agent chose to WAIT on.
        survivors: list[_Receivable] = []
        for r in kept_recv:
            if w.realize_default(r.cp):
                r.age += 1
                survivors.append(r)
            else:
                default_loss += r.amount
                book.default_count[r.cp] += 1
        book.receivables = survivors
        book.payables = kept_pay

        # 5. Advance the latent world (regime drift + fee process) and add new
        #    obligations for the next cycle.
        w.step_macro()
        for p in book.payables:
            p.due_in -= 1
        self._arrivals()

        terminated = False
        truncated = w.cycle >= self.cfg.horizon
        if truncated:
            # Episode end: write off everything still uncollected.
            for r in book.receivables:
                default_loss += r.amount
            book.receivables = []

        reward = -float(fee_cost + default_loss + missed_penalty)
        if not conservation_ok:
            reward -= CONSERVATION_VIOLATION_PENALTY

        obs_dict = self._build_obs()
        info = self._info()
        info.update(
            fee_cost=fee_cost,
            default_loss=default_loss,
            missed_penalty=missed_penalty,
            settled_volume=settled,
            gross_volume=netting.gross_volume(graph),
            conservation_ok=conservation_ok,
        )
        return self._flatten(obs_dict), reward, terminated, truncated, info

    # ---- Observation -------------------------------------------------------

    # Per-counterparty observed features (NOT type / regime / pay prob):
    #   receivable_total, oldest_age, paid_count, default_count,
    #   payable_due_now, payable_outstanding
    _PER_CP = 6

    def _obs_dim(self) -> int:
        # 3 globals (cycle_frac, cash, fee) + per-counterparty block.
        return 3 + self._PER_CP * self.cfg.n_counterparties

    def _build_obs(self) -> dict:
        w, book = self._world, self._book
        n = self.cfg.n_counterparties

        recv_total = np.zeros(n)
        oldest_age = np.zeros(n)
        for r in book.receivables:
            recv_total[r.cp] += r.amount
            oldest_age[r.cp] = max(oldest_age[r.cp], r.age)

        pay_due = np.zeros(n)
        pay_out = np.zeros(n)
        for p in book.payables:
            pay_out[p.cp] += p.amount
            if p.due_in <= 0:
                pay_due[p.cp] += p.amount

        obs = {
            "cycle": w.cycle,
            "horizon": self.cfg.horizon,
            "cash": w.cash,
            "gas_price": w.gas_price,
            "counterparties": [
                {
                    "id": i,
                    "receivable": int(recv_total[i]),
                    "oldest_age": int(oldest_age[i]),
                    "times_paid": int(book.paid_count[i]),
                    "times_defaulted": int(book.default_count[i]),
                    "payable_due_now": int(pay_due[i]),
                    "payable_outstanding": int(pay_out[i]),
                }
                for i in range(n)
            ],
        }
        self._last_obs_dict = obs
        return obs

    def observation_dict(self) -> dict:
        """The structured observation (for LLM agents and humans)."""
        return self._last_obs_dict

    def _flatten(self, obs: dict) -> np.ndarray:
        n = self.cfg.n_counterparties
        vec = [obs["cycle"] / max(1, obs["horizon"]), obs["cash"], obs["gas_price"]]
        for cp in obs["counterparties"]:
            vec += [
                cp["receivable"],
                cp["oldest_age"],
                cp["times_paid"],
                cp["times_defaulted"],
                cp["payable_due_now"],
                cp["payable_outstanding"],
            ]
        return np.asarray(vec, dtype=np.float32)

    # ---- Internals ---------------------------------------------------------

    def _arrivals(self) -> None:
        """Add new obligations for the upcoming cycle (Poisson arrivals)."""
        w, book, cfg = self._world, self._book, self.cfg
        rng = w.rng
        k = rng.poisson(cfg.arrival_rate)
        for _ in range(int(k)):
            cp = int(rng.integers(cfg.n_counterparties))
            amount = int(rng.integers(cfg.amount_low, cfg.amount_high + 1))
            book.receivables.append(_Receivable(cp=cp, amount=amount))
        # Occasionally the agent owes a counterparty (creates netting offsets +
        # liquidity pressure). ~half a payable per cycle on average.
        if rng.random() < 0.5:
            cp = int(rng.integers(cfg.n_counterparties))
            amount = int(rng.integers(cfg.amount_low, cfg.amount_high + 1))
            due_in = int(rng.integers(0, 4))
            book.payables.append(_Payable(cp=cp, amount=amount, due_in=due_in))

    def _info(self) -> dict:
        w = self._world
        # Ground-truth latents, exposed in info ONLY (never in the observation).
        # Used by the full-information baseline and for the belief-vs-truth
        # ablation; an honest policy must not read these.
        return {
            "regime": w.regime,
            "true_types": w.types.copy(),
            "cash": w.cash,
            "cycle": w.cycle,
        }


def make(config: DynamicsConfig | None = None) -> SettlementTimingEnv:
    """Construct the env directly (works with or without Gymnasium installed)."""
    return SettlementTimingEnv(config)


def register() -> None:
    """Register SettlementTiming-v0 with Gymnasium if it is available."""
    if not _GYM:
        return
    try:
        gym.register(
            id="SettlementTiming-v0",
            entry_point="rescontre_env.env:SettlementTimingEnv",
        )
    except Exception:
        pass  # already registered
