"""LLM-as-agent harness.

Serialize the observation to a prompt, ask a model for an action, parse it back.
This is the experiment you actually want first: drop a model in zero-shot and
see whether it can manage obligations over a long horizon under hidden,
correlated default risk. No training, just inference.

Uses an OpenAI-compatible client, which covers:
  - local Qwen via Ollama   (base_url="http://localhost:11434/v1")
  - local Qwen via vLLM      (base_url="http://localhost:8000/v1")
  - LM Studio                (base_url="http://localhost:1234/v1")
  - hosted APIs              (set base_url + api_key for the provider)

Two things this harness takes seriously, because they are findings, not bugs:
  - ROBUST PARSING with a defined fallback (WAIT on parse failure).
  - It tracks the PARSE-FAILURE RATE. "The model could not emit a valid action
    30% of the time" is a result about the model, and it is reported, not hidden.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .env import COLLECT, WAIT

SYSTEM_PROMPT = """You are a clearing member managing a book of obligations over many cycles.

Each cycle, for every counterparty, you choose:
  COLLECT (1): settle that counterparty's receivables now. You pay GAS for each
               net transfer the settlement requires (gas per transfer, times the
               current gas price), and the money becomes safe cash. Settling many
               of one counterparty's receivables together is ONE transfer, and
               collecting nets against any payable you owe that counterparty this
               cycle, cutting transfers further. Settling the same counterparty a
               little every cycle wastes gas.
  WAIT    (0): do nothing for that counterparty this cycle. No fee. But the
               counterparty may DEFAULT before you collect, and you lose the
               whole receivable. Default risk is higher for unreliable
               counterparties and rises for everyone in bad market regimes.

You do NOT see counterparty types or the market regime. Infer reliability from
each counterparty's paid/defaulted history. Your goal is to MINIMIZE total cost
over the episode: fees + default losses + missed-payment penalties. Uncollected
receivables are written off at the end, so do not wait forever; but collecting
everything immediately wastes fees on counterparties that would have paid anyway.

Respond with ONLY a JSON array of 0/1 of length equal to the number of
counterparties, in id order. Example for 4 counterparties: [0,1,0,1]
No prose, no explanation, just the array."""


def render_observation(obs: dict) -> str:
    """Human/LLM-readable observation. Deliberately excludes hidden latents."""
    lines = [
        f"Cycle {obs['cycle']}/{obs['horizon']} | cash={obs['cash']} | "
        f"gas_per_transfer={obs['gas_price']:.1f}",
        "Counterparties (id: receivable, oldest_age, paid, defaulted, "
        "payable_due_now, payable_outstanding):",
    ]
    for cp in obs["counterparties"]:
        lines.append(
            f"  {cp['id']}: recv={cp['receivable']}, age={cp['oldest_age']}, "
            f"paid={cp['times_paid']}, defaulted={cp['times_defaulted']}, "
            f"due_now={cp['payable_due_now']}, payable={cp['payable_outstanding']}"
        )
    lines.append(
        f"Return a JSON array of {len(obs['counterparties'])} values "
        "(0=WAIT, 1=COLLECT)."
    )
    return "\n".join(lines)


def parse_action(text: str, n: int) -> tuple[np.ndarray, bool]:
    """Parse a model reply into a length-n 0/1 action.

    Returns (action, ok). On any failure, action defaults to all-WAIT and
    ok=False so the caller can track the parse-failure rate.
    """
    match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list) and len(arr) == n:
                action = np.array([COLLECT if int(v) else WAIT for v in arr], dtype=int)
                return action, True
        except (ValueError, TypeError):
            pass
    return np.full(n, WAIT, dtype=int), False


class LLMAgent:
    """Wraps an OpenAI-compatible chat model as a SettlementTiming policy."""

    def __init__(
        self,
        model: str = "qwen2.5",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "LLMAgent needs the openai client. Install with: "
                "pip install 'rescontre-env[llm]'"
            ) from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.parse_failures = 0

    def __call__(self, obs: dict, info=None) -> np.ndarray:
        n = len(obs["counterparties"])
        self.calls += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_observation(obs)},
                ],
            )
            text = resp.choices[0].message.content or ""
        except Exception:
            self.parse_failures += 1
            return np.full(n, WAIT, dtype=int)
        action, ok = parse_action(text, n)
        if not ok:
            self.parse_failures += 1
        return action

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0
