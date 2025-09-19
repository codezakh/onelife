import itertools
from dataclasses import dataclass

import crafter.constants
import numpy as np
from crafter.env import Env as BaseCrafterEnv
from crafter.state_export import WorldState, export_world_state
from PIL import Image

from distant_sunburn.balrog_components import EnvironmentConfig
from pydantic import Field

from .balrog_interfaces import (
    EnvironmentProtocol,
    Experience,
    Observation,
    OnResetExperience,
    Text,
)
from .typing_utils import implements
from typing import Optional
from typing import NamedTuple

MAP_SLUG_TO_ENGINE_ACTION = {
    slug: idx for idx, slug in enumerate(crafter.constants.actions)
}


MAP_DISPLAY_ACTION_TO_ENGINE_ACTION = {
    "Noop": MAP_SLUG_TO_ENGINE_ACTION["noop"],
    "Move West": MAP_SLUG_TO_ENGINE_ACTION["move_left"],
    "Move East": MAP_SLUG_TO_ENGINE_ACTION["move_right"],
    "Move North": MAP_SLUG_TO_ENGINE_ACTION["move_up"],
    "Move South": MAP_SLUG_TO_ENGINE_ACTION["move_down"],
    "Do": MAP_SLUG_TO_ENGINE_ACTION["do"],
    "Sleep": MAP_SLUG_TO_ENGINE_ACTION["sleep"],
    "Place Stone": MAP_SLUG_TO_ENGINE_ACTION["place_stone"],
    "Place Table": MAP_SLUG_TO_ENGINE_ACTION["place_table"],
    "Place Furnace": MAP_SLUG_TO_ENGINE_ACTION["place_furnace"],
    "Place Plant": MAP_SLUG_TO_ENGINE_ACTION["place_plant"],
    "Make Wood Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_wood_pickaxe"],
    "Make Stone Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_stone_pickaxe"],
    "Make Iron Pickaxe": MAP_SLUG_TO_ENGINE_ACTION["make_iron_pickaxe"],
    "Make Wood Sword": MAP_SLUG_TO_ENGINE_ACTION["make_wood_sword"],
    "Make Stone Sword": MAP_SLUG_TO_ENGINE_ACTION["make_stone_sword"],
    "Make Iron Sword": MAP_SLUG_TO_ENGINE_ACTION["make_iron_sword"],
}

MAP_DISPLAY_ACTION_TO_DESCRIPTION = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "Multiuse action to collect material, drink from lake and hit creature in front",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
}


def get_instruction_prompt():
    action_strings = ",\n".join(
        f"{action}: {MAP_DISPLAY_ACTION_TO_DESCRIPTION[action]}"
        for action in MAP_DISPLAY_ACTION_TO_ENGINE_ACTION
    )
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}.

These are the game achievements you can get:
1. Collect Wood
2. Place Table
3. Eat Cow
4. Collect Sampling
5. Collect Drink
6. Make Wood Pickaxe
7. Make Wood Sword
8. Place Plant
9. Defeat Zombie
10. Collect Stone
11. Place Stone
12. Eat Plant
13. Defeat Skeleton
14. Make Stone Pickaxe
15. Make Stone Sword
16. Wake Up
17. Place Furnace
18. Collect Coal
19. Collect Iron
20. Make Iron Pickaxe
21. Make Iron Sword
22. Collect Diamond

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

PLAY!
""".strip()

    return instruction_prompt


class CrafterEnvironmentConfig(EnvironmentConfig):
    area: tuple[int, int]
    view: tuple[int, int]
    size: tuple[int, int]
    reward: bool
    seed: Optional[int] = None
    name: str = "crafter"
    task: str = "open_ended"
    max_episode_steps: int = Field(default=2000)
    render_image: bool = Field(default=False)
    instruction_prompt: str = Field(default_factory=get_instruction_prompt)


def build_base_environment(config: CrafterEnvironmentConfig) -> BaseCrafterEnv:
    return BaseCrafterEnv(
        area=config.area,
        view=config.view,
        size=config.size,
        reward=config.reward,
        length=config.max_episode_steps,
        seed=config.seed,
    )


@dataclass
class TextRendererOutput:
    long_term_context: str
    short_term_context: str


class TextRenderer:
    """
    Renders a language observation for a Crafter environment.

    This code is adapted from the `CrafterLanguageWrapper` class in the BALROG repo.
    """

    vitals = [
        "health",
        "food",
        "drink",
        "energy",
    ]
    rot = np.array([[0, -1], [1, 0]])
    directions = ["front", "right", "back", "left"]

    def __init__(self, base_env: BaseCrafterEnv):
        self.id_to_item: dict[int, str] = dict()
        self.player_idx = None
        for name, ind in itertools.chain(
            base_env._world._mat_ids.items(), base_env._sem_view._obj_ids.items()
        ):
            name = (
                str(name)[str(name).find("objects.") + len("objects.") : -2].lower()
                if "objects." in str(name)
                else str(name)
            )
            self.id_to_item[ind] = name
            if name == "player":
                self.player_idx = ind

        assert self.player_idx is not None
        self.base_env = base_env

    def _augment_info(self, info: dict) -> dict:
        aug_info = info.copy()
        assert self.base_env._player is not None
        aug_info["sleeping"] = self.base_env._player.sleeping
        aug_info["player_facing"] = self.base_env._player.facing
        aug_info["dead"] = self.base_env._player.health <= 0
        assert self.base_env._unlocked is not None
        aug_info["unlocked"] = {
            name
            for name, count in self.base_env._player.achievements.items()
            if count > 0 and name not in self.base_env._unlocked
        }
        aug_info["view"] = self.base_env._view
        return aug_info

    def describe_inventory(self, info: dict):
        result = ""

        status_str = "Your status:\n{}".format(
            "\n".join(
                ["- {}: {}/9".format(v, info["inventory"][v]) for v in self.vitals]
            )
        )
        result += status_str + "\n\n"

        inventory_str = "\n".join(
            [
                "- {}: {}".format(i, num)
                for i, num in info["inventory"].items()
                if i not in self.vitals
            ]
        )
        inventory_str = "Your inventory:\n{}".format(inventory_str)
        result += inventory_str

        return result.strip()

    REF = np.array([0, 1])

    @staticmethod
    def rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        dot = np.dot(v1, v2)
        cross = np.cross(v1, v2)
        rotation_matrix = np.array([[dot, -cross], [cross, dot]])
        return rotation_matrix

    @staticmethod
    def describe_loc(ref: np.ndarray, P: np.ndarray) -> str:
        desc = []
        if ref[1] > P[1]:
            desc.append("north")
        elif ref[1] < P[1]:
            desc.append("south")
        if ref[0] > P[0]:
            desc.append("west")
        elif ref[0] < P[0]:
            desc.append("east")

        return "-".join(desc)

    def describe_env(self, info: dict):
        assert (
            info["semantic"][info["player_pos"][0], info["player_pos"][1]]
            == self.player_idx
        )
        semantic = info["semantic"][
            info["player_pos"][0]
            - info["view"][0] // 2 : info["player_pos"][0]
            + info["view"][0] // 2
            + 1,
            info["player_pos"][1]
            - info["view"][1] // 2
            + 1 : info["player_pos"][1]
            + info["view"][1] // 2,
        ]
        center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
        result = ""
        x = np.arange(semantic.shape[1])
        y = np.arange(semantic.shape[0])
        x1, y1 = np.meshgrid(x, y)
        loc = np.stack((y1, x1), axis=-1)
        dist = np.absolute(center - loc).sum(axis=-1)
        obj_info_list = []

        facing = info["player_facing"]
        max_y, max_x = semantic.shape
        target_x = center[0] + facing[0]
        target_y = center[1] + facing[1]

        if 0 <= target_x < max_x and 0 <= target_y < max_y:
            target_id = semantic[int(target_x), int(target_y)]
            target_item = self.id_to_item[target_id]
            obs = "You face {} at your front.".format(target_item)
        else:
            obs = "You face nothing at your front."

        for idx in np.unique(semantic):
            if idx == self.player_idx:
                continue

            smallest = np.unravel_index(
                np.argmin(np.where(semantic == idx, dist, np.inf)), semantic.shape
            )
            obj_info_list.append(
                (
                    self.id_to_item[idx],
                    dist[smallest],
                    self.describe_loc(np.array([0, 0]), smallest - center),
                )
            )

        if len(obj_info_list) > 0:
            status_str = "You see:\n{}".format(
                "\n".join(
                    [
                        "- {} {} steps to your {}".format(name, dist, loc)
                        for name, dist, loc in obj_info_list
                    ]
                )
            )
        else:
            status_str = "You see nothing away from you."
        result += status_str + "\n\n"
        result += obs.strip()

        return result.strip()

    @staticmethod
    def describe_act(action: str) -> str:
        result = ""

        action_str = action.replace("do_", "interact_")
        action_str = action_str.replace("move_up", "move_north")
        action_str = action_str.replace("move_down", "move_south")
        action_str = action_str.replace("move_left", "move_west")
        action_str = action_str.replace("move_right", "move_east")

        act = "You took action {}.".format(action_str)
        result += act

        return result.strip()

    @staticmethod
    def describe_status(info: dict) -> str:
        if info["sleeping"]:
            return "You are sleeping, and will not be able take actions until energy is full.\n\n"
        elif info["dead"]:
            return "You died.\n\n"
        else:
            return ""

    def describe_frame(self, info: dict) -> TextRendererOutput:
        result = ""

        result += self.describe_status(info)
        result += "\n\n"
        result += self.describe_env(info)
        result += "\n\n"

        return TextRendererOutput(
            long_term_context=result.strip(),
            short_term_context=self.describe_inventory(info),
        )

    def __call__(self, base_env_step_info: dict) -> TextRendererOutput:
        augmented_info = self._augment_info(base_env_step_info)
        return self.describe_frame(augmented_info)


class LanguageSymbolicWrapper:
    def __init__(self, config: CrafterEnvironmentConfig):
        self.config = config
        self.base_env = build_base_environment(config)
        self.renderer = TextRenderer(self.base_env)
        self.step_count = 0
        self.score_tracker = 0
        self.achievements = None
        self.default_action = "Noop"
        self.failed_candidates: list[str] = []

    def _update_progress(self, info: dict):
        self.score_tracker = 0 + sum(
            [1.0 for k, v in info["achievements"].items() if v > 0]
        )
        self.achievements = info["achievements"]
        return self.score_tracker

    def _step(self, action: str) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        obs, reward, done, info = self.base_env.step(
            MAP_DISPLAY_ACTION_TO_ENGINE_ACTION[action]
        )
        self.step_count += 1
        truncated = self.step_count >= self.config.max_episode_steps
        if truncated:
            done = True

        return obs, reward, bool(done), truncated, info

    def _build_observation(self, obs: np.ndarray, info: dict) -> Observation:
        language_observation = self.renderer(info)
        return Observation(
            text=Text(
                short_term_context=language_observation.short_term_context,
                long_term_context=language_observation.long_term_context,
            ),
            image=(
                Image.fromarray(self.base_env.render()).convert("RGB")
                if self.config.render_image
                else None
            ),
            obs=obs,
        )

    def reset(self, seed: Optional[int] = None) -> OnResetExperience[WorldState]:
        """Reset the environment and return both language and symbolic observations."""
        # Crafter's `reset` method does not accept a seed, so we set it after init.
        self.base_env.reset()
        self.base_env._seed = seed
        self.step_count = 0
        self.score_tracker = 0
        self.achievements = None

        assert self.base_env._step is not None
        assert self.base_env._view is not None

        # Create language observation
        # One problem is that Crafter is an old-style env that does not return an
        # info dict upon calling `reset()`
        # We _could_ use the exported state here, but for consistentcy with Balrog, we will take
        # a no-op action and use the info dict from the step call.
        obs, _, _, _, info = self._step(self.default_action)

        language_observation = self.renderer(info)
        language_obs = Observation(
            text=Text(
                short_term_context=language_observation.short_term_context,
                long_term_context=language_observation.long_term_context,
            ),
            image=(
                Image.fromarray(self.base_env.render()).convert("RGB")
                if self.config.render_image
                else None
            ),
            obs=obs,
        )

        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )

        return OnResetExperience(
            obs=language_obs,
            info=world_state,
        )

    def step(self, action: str) -> Experience[WorldState]:
        obs, reward, done, truncated, info = self._step(action)
        language_obs = self._build_observation(obs, info)
        self.score_tracker = self._update_progress(info)
        assert self.base_env._step is not None
        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )
        return Experience(
            obs=language_obs,
            reward=reward,
            done=done,
            action=action,
            truncated=truncated,
            info=world_state,
        )

    def get_stats(self) -> dict:
        return {
            "score": self.score_tracker,
            "progression": float(self.score_tracker) / 22.0,
            "achievements": self.achievements,
        }

    def check_action_validity(self, candidate_action: str) -> str:
        if candidate_action in MAP_DISPLAY_ACTION_TO_ENGINE_ACTION:
            return candidate_action
        else:
            self.failed_candidates.append(candidate_action)
            return self.default_action

    def get_instruction_prompt(self, instructions: str | None = None) -> str:
        return self.config.instruction_prompt


implements(EnvironmentProtocol[WorldState])(LanguageSymbolicWrapper)
