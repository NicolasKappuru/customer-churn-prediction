"""Environment setup for the FrozenLake reinforcement-learning example."""

import gymnasium as gym


DEFAULT_MAP_NAME = "4x4"
DEFAULT_SLIPPERY = True


def make_env(
    render_mode: str | None = None,
    map_name: str = DEFAULT_MAP_NAME,
    slippery: bool = DEFAULT_SLIPPERY,
) -> gym.Env:
    """Create a FrozenLake environment with the requested rendering mode."""
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=slippery,
        render_mode=render_mode,
    )