import numpy as np
from typing import Optional, SupportsFloat, Any

from crafter.env import Env as CrafterBaseEnv
from gymnasium.core import Env
from gymnasium import spaces
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


# 1. Crafter configuration dataclass
class CrafterConfig(BaseModel):
    area: tuple[int, int] = (64, 64)
    view: tuple[int, int] = (9, 9)
    size: tuple[int, int] = (64, 64)
    reward: bool = True
    length: int = 10000
    seed: Optional[int] = None


# 2. Inventory dataclass (fields match data.yaml/constants.py)
class Inventory(BaseModel):
    health: int
    food: int
    drink: int
    energy: int
    sapling: int
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int


# 3. Achievements dataclass (fields match data.yaml)
class Achievements(BaseModel):
    collect_coal: int
    collect_diamond: int
    collect_drink: int
    collect_iron: int
    collect_sapling: int
    collect_stone: int
    collect_wood: int
    defeat_skeleton: int
    defeat_zombie: int
    eat_cow: int
    eat_plant: int
    make_iron_pickaxe: int
    make_iron_sword: int
    make_stone_pickaxe: int
    make_stone_sword: int
    make_wood_pickaxe: int
    make_wood_sword: int
    place_furnace: int
    place_plant: int
    place_stone: int
    place_table: int
    wake_up: int


class CrafterStepInfo(BaseModel):
    inventory: Inventory
    achievements: Achievements
    discount: float
    semantic: npt.NDArray[
        np.uint8
    ]  # int values, shape (area_x, area_y), dtype usually uint8
    player_pos: tuple[int, int]
    reward: float
    truncated: bool = False
    terminated: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


# 5. The wrapper
class CrafterEnv(Env[np.ndarray, np.int64]):
    """A typed Gymnasium environment wrapper for Crafter."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: CrafterConfig = CrafterConfig()):
        """Initialize the Crafter environment with the given configuration."""
        super().__init__()

        self._env = CrafterBaseEnv(
            area=config.area,
            view=config.view,
            size=config.size,
            reward=config.reward,
            length=config.length,
            seed=config.seed,
        )
        self._max_steps = config.length
        self._step_count = 0

        # Set up spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self._env.action_space.n)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)  # Important: call super().reset() to properly seed
        obs = self._env.reset()
        info = self._make_info_from_game_state()
        return np.asarray(obs), info.model_dump()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        self._step_count += 1
        obs, reward, done, info_dict = self._env.step(action)

        # Determine truncation/termination like gymnasium does
        over = self._max_steps and self._step_count >= self._max_steps
        assert self._env._player is not None
        dead = self._env._player.health <= 0

        info = CrafterStepInfo(
            inventory=Inventory.model_validate(info_dict["inventory"]),
            achievements=Achievements.model_validate(info_dict["achievements"]),
            discount=float(info_dict["discount"]),
            semantic=np.asarray(info_dict["semantic"]),
            player_pos=tuple(info_dict["player_pos"]),
            reward=float(info_dict["reward"]),
            truncated=bool(over and not dead),
            terminated=bool(dead),
        )

        return (
            np.asarray(obs),
            float(reward),
            info.terminated,
            info.truncated,
            info.model_dump(),
        )

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Clean up resources."""
        return self._env.close()

    @property
    def action_names(self) -> list[str]:
        """Return the list of action names."""
        return list(self._env.action_names)

    def _make_info_from_game_state(self) -> CrafterStepInfo:
        """Construct info dictionary after reset."""
        assert self._env._player is not None
        player = self._env._player
        info = CrafterStepInfo(
            inventory=Inventory.model_validate(player.inventory),
            achievements=Achievements.model_validate(player.achievements),
            discount=1.0,
            semantic=np.asarray(self._env._sem_view()),
            player_pos=tuple(player.pos),
            reward=0.0,
            truncated=False,
            terminated=False,
        )
        return info


if __name__ == "__main__":
    config = CrafterConfig(area=(64, 64), reward=False, length=2)
    env = CrafterEnv(config)
    obs, info = env.reset()
    while True:
        # For Discrete action space, we can access the size with action_space.n
        assert isinstance(env.action_space, spaces.Discrete)
        action = int(np.random.randint(env.action_space.n))  # Convert to Python int
        obs, reward, terminated, truncated, info = env.step(action)
        print(info["inventory"]["food"])  # Access food from info dict
        if terminated or truncated:  # Check terminated or truncated
            break
