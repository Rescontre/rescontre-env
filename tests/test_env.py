"""Environment sanity: API shape, determinism under seed, conservation always
holds, hidden latents never leak into the observation, and the baseline
gradient comes out in the right order.
"""

import numpy as np

from rescontre_env import DynamicsConfig, baselines, make
from rescontre_env.eval import evaluate, standard_suite


def test_reset_step_shapes():
    env = make(DynamicsConfig(n_counterparties=5, horizon=10))
    obs, info = env.reset(seed=0)
    assert obs.shape[0] == 3 + 6 * 5
    action = np.zeros(5, dtype=int)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)


def test_episode_runs_to_horizon():
    env = make(DynamicsConfig(n_counterparties=4, horizon=12))
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(4, dtype=int))
        steps += 1
        done = term or trunc
    assert steps == 12


def test_determinism_same_seed():
    cfg = DynamicsConfig(n_counterparties=6, horizon=15)
    r1 = evaluate(baselines.greedy_policy(), episodes=5, config=cfg)
    r2 = evaluate(baselines.greedy_policy(), episodes=5, config=cfg)
    assert np.allclose(r1["rewards"], r2["rewards"]), "same seeds must reproduce"


def test_conservation_never_violated():
    # Across a full random rollout the tripwire penalty must never fire.
    cfg = DynamicsConfig(n_counterparties=8, horizon=40)
    rep = evaluate(baselines.random_policy(np.random.default_rng(0)),
                   episodes=30, config=cfg)
    assert rep["conservation_violations"] == 0


def test_latents_absent_from_observation():
    # The observation dict must not contain regime / types / pay probabilities.
    env = make(DynamicsConfig(n_counterparties=4, horizon=5))
    env.reset(seed=0)
    obs = env.observation_dict()
    flat = str(obs).lower()
    for leak in ("regime", "type", "pay_prob", "payprob"):
        assert leak not in flat, f"latent '{leak}' leaked into observation"


def test_baseline_gradient_ordering():
    # The benchmark's core claim: greedy beats random, and the latent-peeking
    # oracle beats greedy. If this ordering breaks, the env has no signal.
    cfg = DynamicsConfig(n_counterparties=8, horizon=40)
    rep = standard_suite(episodes=120, config=cfg)
    assert rep["greedy"]["mean_reward"] > rep["random"]["mean_reward"]
    assert rep["oracle (sees latents)"]["mean_reward"] > rep["greedy"]["mean_reward"]
