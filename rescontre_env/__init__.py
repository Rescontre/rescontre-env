"""rescontre-env: verifiable-reward, long-horizon-agency environments
grounded in clearing semantics.

Quick start:

    from rescontre_env import make, DynamicsConfig
    from rescontre_env import baselines, eval

    eval.print_table(eval.standard_suite(episodes=200))
"""

from . import baselines, dynamics, eval, netting
from .dynamics import DynamicsConfig
from .env import SettlementTimingEnv, make, register

__all__ = [
    "make",
    "register",
    "SettlementTimingEnv",
    "DynamicsConfig",
    "baselines",
    "dynamics",
    "eval",
    "netting",
]

__version__ = "0.1.0"

# Register SettlementTiming-v0 with Gymnasium if it is installed (no-op otherwise).
register()
