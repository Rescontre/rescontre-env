# rescontre-env

**Verifiable-reward, long-horizon-agency environments grounded in clearing semantics.**

An RL environment where an agent manages a book of financial obligations over
many cycles under hidden, correlated default risk and volatile settlement costs.
Two things make it unusual:

1. **The reward is verifiable.** Settlement runs through a real multilateral
   netting algorithm, and every step is gated on a conservation check. A policy
   cannot score well by cheating the books — reward integrity is a checkable
   property, not a soft signal.
2. **It is a clean long-horizon-agency testbed.** The right action depends on a
   counterparty's *hidden type*, which the agent can only infer from observed
   pay/default history; defaults are *correlated* through a latent macro regime
   that *drifts*; and settlement gas is *volatile*. Sparse, delayed reward over a
   long horizon — the regime where current agents are weak.

> Research environment. Fully synthetic, testnet-shaped, no real funds, no chain
> calls. Nothing here touches production settlement.

## Quickstart (under 5 minutes)

```bash
git clone https://github.com/Rescontre/rescontre-env
cd rescontre-env
uv venv --python 3.12 && uv pip install -e ".[dev]"

# Run the baseline gradient (fully offline, no model needed):
uv run rescontre-eval --episodes 300
```

You should see a gradient like:

```
policy                        mean reward     ± sem       fee   default    missed
--------------------------------------------------------------------------------
always_wait                      -27216.1     144.8    1360.1   24712.7    1143.3
random                           -10129.0     150.8    6126.6    4002.4       0.0
always_collect                    -8774.0     125.9    8177.3     596.7       0.0
greedy                            -8571.8     134.4    4814.2    3757.6       0.0
oracle (sees latents)             -7388.6     127.5    5560.5    1828.1       0.0
```

Reward is negative cost (higher = better). The gap between `greedy` (the best
history-only heuristic) and `oracle` (which sees the hidden types) is the
**value of information**: what an agent could gain by inferring the latent state.
A learned policy is interesting exactly to the degree it beats `greedy` and
climbs toward `oracle`.

## Drop an LLM in (the experiment)

Test a model zero-shot as the policy. No training, just inference. Easiest path
is a local Qwen via [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5
uv pip install -e ".[llm]"

# Evaluate Qwen alongside the baselines on the same seed set (paired comparison):
uv run rescontre-eval --llm --model qwen2.5 --llm-episodes 20
```

The harness reports the model's score next to the baselines **and its
parse-failure rate** — "the model could not emit a valid action X% of the time"
is a result, not a bug to hide.

Point it at any OpenAI-compatible endpoint (vLLM, LM Studio, or a hosted API):

```bash
uv run rescontre-eval --llm --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 --api-key "$OPENAI_API_KEY"
```

## The environment

`SettlementTiming-v0`. Each cycle, for every counterparty, the agent chooses
**COLLECT** (settle now via netting, pay gas per net transfer, move the asset to
safe cash) or **WAIT** (no gas, but the counterparty may default — correlated
across counterparties when the hidden regime turns). Reward is negative cost:
gas + default losses + missed-payable penalties, gated on conservation.

```python
from rescontre_env import DynamicsConfig
from rescontre_env import eval

# Sweep the horizon to see where long-horizon coherence breaks down:
for h in (10, 20, 40, 80):
    cfg = DynamicsConfig(horizon=h)
    rep = eval.standard_suite(episodes=200, config=cfg)
    voi = rep["oracle (sees latents)"]["mean_reward"] - rep["greedy"]["mean_reward"]
    print(f"horizon={h:>3}  value-of-information gap = {voi:8.1f}")
```

Every knob (horizon, counterparty count, type prior, regime transitions, gas
process, arrival rate) is a config field, never hardcoded — so the env is a
curriculum, not a fixed task.

### What's deliberately not here yet

- PPO / training scripts and the exact-Bayes belief filter (the principled
  ceiling) — the design exists; this v0 is the env + baselines + LLM harness.
- The transfer experiment (does competence here lift a non-financial
  long-horizon task) — the result that would turn this from a benchmark into a
  training environment.

## License

MIT.
