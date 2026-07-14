"""Long-range cooperative predator-prey environments."""

from gymnasium.envs.registration import register

from mat.envs.long_range_predator_prey.continuous import (
    LongRangePredatorPreyContinuousEnv,
    LongRangePredatorPreyConfig,
    LongRangePredatorPreyTorchCore,
)
from mat.envs.long_range_predator_prey.vec_env import LongRangePredatorPreyTorchVecEnv


register(
    id="LongRangePredatorPreyContinuous-v0",
    entry_point="mat.envs.long_range_predator_prey.continuous:LongRangePredatorPreyContinuousEnv",
)


__all__ = [
    "LongRangePredatorPreyConfig",
    "LongRangePredatorPreyContinuousEnv",
    "LongRangePredatorPreyTorchVecEnv",
    "LongRangePredatorPreyTorchCore",
]
