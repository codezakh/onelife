from dataclasses import dataclass
import numpy as np
from typing import Optional

from crafter.env import Env


# 1. Crafter configuration dataclass
@dataclass
class CrafterConfig:
    area: tuple[int, int] = (64, 64)
    view: tuple[int, int] = (9, 9)
    size: tuple[int, int] = (64, 64)
    reward: bool = True
    length: int = 10000
    seed: Optional[int] = None


# 2. Inventory dataclass (fields match data.yaml/constants.py)
@dataclass
class Inventory:
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

    @staticmethod
    def from_dict(d: dict) -> "Inventory":
        return Inventory(
            health=int(d["health"]),
            food=int(d["food"]),
            drink=int(d["drink"]),
            energy=int(d["energy"]),
            sapling=int(d["sapling"]),
            wood=int(d["wood"]),
            stone=int(d["stone"]),
            coal=int(d["coal"]),
            iron=int(d["iron"]),
            diamond=int(d["diamond"]),
            wood_pickaxe=int(d["wood_pickaxe"]),
            stone_pickaxe=int(d["stone_pickaxe"]),
            iron_pickaxe=int(d["iron_pickaxe"]),
            wood_sword=int(d["wood_sword"]),
            stone_sword=int(d["stone_sword"]),
            iron_sword=int(d["iron_sword"]),
        )


# 3. Achievements dataclass (fields match data.yaml)
@dataclass
class Achievements:
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

    @staticmethod
    def from_dict(d: dict) -> "Achievements":
        return Achievements(
            collect_coal=int(d["collect_coal"]),
            collect_diamond=int(d["collect_diamond"]),
            collect_drink=int(d["collect_drink"]),
            collect_iron=int(d["collect_iron"]),
            collect_sapling=int(d["collect_sapling"]),
            collect_stone=int(d["collect_stone"]),
            collect_wood=int(d["collect_wood"]),
            defeat_skeleton=int(d["defeat_skeleton"]),
            defeat_zombie=int(d["defeat_zombie"]),
            eat_cow=int(d["eat_cow"]),
            eat_plant=int(d["eat_plant"]),
            make_iron_pickaxe=int(d["make_iron_pickaxe"]),
            make_iron_sword=int(d["make_iron_sword"]),
            make_stone_pickaxe=int(d["make_stone_pickaxe"]),
            make_stone_sword=int(d["make_stone_sword"]),
            make_wood_pickaxe=int(d["make_wood_pickaxe"]),
            make_wood_sword=int(d["make_wood_sword"]),
            place_furnace=int(d["place_furnace"]),
            place_plant=int(d["place_plant"]),
            place_stone=int(d["place_stone"]),
            place_table=int(d["place_table"]),
            wake_up=int(d["wake_up"]),
        )


# 4. Step info and output dataclasses
@dataclass
class CrafterStepInfo:
    inventory: Inventory
    achievements: Achievements
    discount: float
    semantic: np.ndarray  # int values, shape (area_x, area_y), dtype usually uint8
    player_pos: tuple[int, int]
    reward: float
    truncated: bool = False
    terminated: bool = False


@dataclass
class CrafterStepOutput:
    obs: np.ndarray  # shape (size_x, size_y, 3), dtype uint8
    reward: float
    done: bool
    info: CrafterStepInfo


# 5. The wrapper
class CrafterEnv:
    def __init__(self, config: CrafterConfig = CrafterConfig()):
        self._env = Env(
            area=config.area,
            view=config.view,
            size=config.size,
            reward=config.reward,
            length=config.length,
            seed=config.seed,
        )
        self._max_steps = config.length
        self._step_count = 0

    def reset(self) -> tuple[np.ndarray, CrafterStepInfo]:
        obs = self._env.reset()
        info = self._make_info()
        return obs, info

    def step(self, action: int) -> CrafterStepOutput:
        self._step_count += 1
        obs, reward, done, info_dict = self._env.step(action)
        # Determine truncation/termination like gymnasium does
        over = self._max_steps and self._step_count >= self._max_steps
        assert self._env._player is not None
        dead = self._env._player.health <= 0
        info = CrafterStepInfo(
            inventory=Inventory.from_dict(info_dict["inventory"]),
            achievements=Achievements.from_dict(info_dict["achievements"]),
            discount=float(info_dict["discount"]),
            semantic=np.asarray(info_dict["semantic"]),
            player_pos=tuple(info_dict["player_pos"]),
            reward=float(info_dict["reward"]),
            truncated=bool(over and not dead),
            terminated=bool(dead),
        )
        return CrafterStepOutput(
            obs=np.asarray(obs), reward=reward, done=bool(done), info=info
        )

    def render(self, size: Optional[tuple[int, int]] = None) -> np.ndarray:
        return self._env.render(size)

    @property
    def action_names(self) -> list[str]:
        return list(self._env.action_names)

    @property
    def action_space_n(self) -> int:
        return self._env.action_space.n

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self._env.observation_space.shape

    def _make_info(self) -> CrafterStepInfo:
        # After reset, info isn't returned. We need to construct from env.
        assert self._env._player is not None
        player = self._env._player
        info = CrafterStepInfo(
            inventory=Inventory.from_dict(player.inventory),
            achievements=Achievements.from_dict(player.achievements),
            discount=1.0,
            semantic=np.asarray(self._env._sem_view()),
            player_pos=tuple(player.pos),
            reward=0.0,
            truncated=False,
            terminated=False,
        )
        return info


if __name__ == "__main__":
    config = CrafterConfig(area=(64, 64), reward=False, length=100)
    env = CrafterEnv(config)
    obs, info = env.reset()
    while True:
        action = np.random.randint(env.action_space_n)
        step = env.step(action)
        print(step.info.inventory.food)
        if step.done:
            break
