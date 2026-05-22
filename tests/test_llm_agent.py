"""LLM harness parsing (no model calls). Robust parsing + failure accounting are
findings, not bugs, so they are tested directly.
"""

import numpy as np

from rescontre_env.llm_agent import parse_action, render_observation


def test_parse_clean_array():
    action, ok = parse_action("[0,1,0,1]", 4)
    assert ok
    assert list(action) == [0, 1, 0, 1]


def test_parse_with_surrounding_prose():
    action, ok = parse_action("Sure! My answer is [1, 0, 1] because...", 3)
    assert ok
    assert list(action) == [1, 0, 1]


def test_parse_wrong_length_fails_safe():
    action, ok = parse_action("[1,0]", 4)
    assert not ok
    assert list(action) == [0, 0, 0, 0]  # all-WAIT fallback


def test_parse_garbage_fails_safe():
    action, ok = parse_action("I refuse to answer.", 5)
    assert not ok
    assert list(action) == [0, 0, 0, 0, 0]


def test_render_excludes_latents():
    obs = {
        "cycle": 2, "horizon": 40, "cash": 1000, "gas_price": 35.2,
        "counterparties": [
            {"id": 0, "receivable": 100, "oldest_age": 1, "times_paid": 3,
             "times_defaulted": 0, "payable_due_now": 0, "payable_outstanding": 0},
        ],
    }
    text = render_observation(obs).lower()
    assert "regime" not in text and "type" not in text
    assert "gas_per_transfer" in text
