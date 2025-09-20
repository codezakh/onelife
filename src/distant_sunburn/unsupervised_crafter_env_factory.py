from dataclasses import dataclass

import crafter.constants
import numpy as np
from crafter.env import Env as BaseCrafterEnv
from crafter.state_export import WorldState, export_world_state
from PIL import Image
from pydantic import ConfigDict

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
from typing import Optional, Any
from typing import Protocol
from crafter.state_export import Position

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


@dataclass
class TextRendererOutput:
    long_term_context: str
    short_term_context: str


class TextRendererProtocol(Protocol):
    def __call__(self, world_state: WorldState) -> TextRendererOutput: ...


class UnsupervisedTextRenderer:
    """
    Text renderer for unsupervised Crafter environment that reads from WorldState.

    Shows:
    - Player status (sleeping/dead)
    - Local view: closest material/entity of each type within view range
    - Distant view: closest material/entity of each type outside view range
    - Inventory
    - What the player is targeting
    """

    # Interesting materials to show to the agent
    INTERESTING_MATERIALS = [
        "tree",
        "stone",
        "coal",
        "iron",
        "diamond",
        "water",
        "grass",
        "table",
        "furnace",
        "plant",
    ]

    # Entity types to show to the agent
    ENTITY_TYPES = ["cow", "zombie", "skeleton", "arrow", "fence"]

    # Vital stats to show in inventory
    VITALS = ["health", "food", "drink", "energy"]

    def __init__(self):
        pass

    def _manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _get_direction_description(self, from_pos: Position, to_pos: Position) -> str:
        """Get relative direction description from one position to another."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y

        directions = []
        if dy < 0:
            directions.append("north")
        elif dy > 0:
            directions.append("south")
        if dx < 0:
            directions.append("west")
        elif dx > 0:
            directions.append("east")

        return "-".join(directions) if directions else "here"

    def _find_closest_material(
        self, world_state: WorldState, material_name: str, within_range: bool = True
    ) -> tuple[Position, int] | None:
        """Find the closest material of the given type.

        Args:
            world_state: The world state to search
            material_name: Name of material to find
            within_range: If True, search within view range; if False, search outside view range

        Returns:
            Tuple of (position, distance) or None if not found
        """
        player_pos = world_state.player.position
        view_range = max(world_state.view)

        closest_pos = None
        closest_distance = float("inf")

        for x in range(world_state.size[0]):
            for y in range(world_state.size[1]):
                try:
                    material = world_state.materials[x][y]
                    if material == material_name:
                        pos = Position(x=x, y=y)
                        distance = self._manhattan_distance(player_pos, pos)

                        # Check if this position is within/outside the desired range
                        is_within = distance <= view_range
                        if is_within == within_range and distance < closest_distance:
                            closest_pos = pos
                            closest_distance = distance
                except IndexError:
                    continue

        return (closest_pos, int(closest_distance)) if closest_pos is not None else None

    def _find_closest_entity(
        self, world_state: WorldState, entity_type: str, within_range: bool = True
    ) -> tuple[Position, int] | None:
        """Find the closest entity of the given type.

        Args:
            world_state: The world state to search
            entity_type: Name of entity type to find
            within_range: If True, search within view range; if False, search outside view range

        Returns:
            Tuple of (position, distance) or None if not found
        """
        player_pos = world_state.player.position
        view_range = max(world_state.view)

        closest_pos = None
        closest_distance = float("inf")

        for obj in world_state.objects:
            if obj.name == entity_type:
                distance = self._manhattan_distance(player_pos, obj.position)

                # Check if this position is within/outside the desired range
                is_within = distance <= view_range
                if is_within == within_range and distance < closest_distance:
                    closest_pos = obj.position
                    closest_distance = distance

        return (closest_pos, int(closest_distance)) if closest_pos is not None else None

    def _describe_player_status(self, world_state: WorldState) -> str:
        """Describe the player's current status."""
        if world_state.player.sleeping:
            return "You are sleeping and cannot take actions until energy is full.\n\n"
        elif world_state.player.health <= 0:
            return "You are dead.\n\n"
        else:
            return ""

    def _describe_target(self, world_state: WorldState) -> str:
        """Describe what the player is currently targeting."""
        target_pos = world_state.player.position + world_state.player.facing
        material, obj = world_state.get_tile(target_pos)

        if obj is not None:
            return f"You are targeting a {obj.name} at ({target_pos.x}, {target_pos.y}).\n\n"
        elif material is not None:
            return (
                f"You are targeting {material} at ({target_pos.x}, {target_pos.y}).\n\n"
            )
        else:
            return f"You are targeting empty space at ({target_pos.x}, {target_pos.y}).\n\n"

    def _describe_local_view(self, world_state: WorldState) -> str:
        """Describe the local view - closest materials and entities within view range."""
        result = "Local view (within {} steps):\n".format(max(world_state.view))

        # Find closest materials within range
        material_descriptions = []
        for material in self.INTERESTING_MATERIALS:
            closest = self._find_closest_material(
                world_state, material, within_range=True
            )
            if closest is not None:
                pos, distance = closest
                direction = self._get_direction_description(
                    world_state.player.position, pos
                )
                material_descriptions.append(
                    f"- {material}: {distance} steps {direction} at ({pos.x}, {pos.y})"
                )
            else:
                material_descriptions.append(f"- {material}: not present")

        # Find closest entities within range
        entity_descriptions = []
        for entity_type in self.ENTITY_TYPES:
            closest = self._find_closest_entity(
                world_state, entity_type, within_range=True
            )
            if closest is not None:
                pos, distance = closest
                direction = self._get_direction_description(
                    world_state.player.position, pos
                )
                entity_descriptions.append(
                    f"- {entity_type}: {distance} steps {direction} at ({pos.x}, {pos.y})"
                )
            else:
                entity_descriptions.append(f"- {entity_type}: not present")

        if material_descriptions:
            result += "Materials:\n" + "\n".join(material_descriptions) + "\n\n"
        if entity_descriptions:
            result += "Entities:\n" + "\n".join(entity_descriptions) + "\n\n"

        return result

    def _describe_distant_view(self, world_state: WorldState) -> str:
        """Describe the distant view - closest materials and entities outside view range."""
        result = "Distant view (beyond {} steps):\n".format(max(world_state.view))

        # Find closest materials outside range
        material_descriptions = []
        for material in self.INTERESTING_MATERIALS:
            closest = self._find_closest_material(
                world_state, material, within_range=False
            )
            if closest is not None:
                pos, distance = closest
                direction = self._get_direction_description(
                    world_state.player.position, pos
                )
                material_descriptions.append(
                    f"- {material}: {distance} steps {direction} at ({pos.x}, {pos.y})"
                )
            else:
                material_descriptions.append(f"- {material}: not present in world")

        # Find closest entities outside range
        entity_descriptions = []
        for entity_type in self.ENTITY_TYPES:
            closest = self._find_closest_entity(
                world_state, entity_type, within_range=False
            )
            if closest is not None:
                pos, distance = closest
                direction = self._get_direction_description(
                    world_state.player.position, pos
                )
                entity_descriptions.append(
                    f"- {entity_type}: {distance} steps {direction} at ({pos.x}, {pos.y})"
                )
            else:
                entity_descriptions.append(f"- {entity_type}: not present in world")

        if material_descriptions:
            result += "Materials:\n" + "\n".join(material_descriptions) + "\n\n"
        if entity_descriptions:
            result += "Entities:\n" + "\n".join(entity_descriptions) + "\n\n"

        return result

    def _describe_inventory(self, world_state: WorldState) -> str:
        """Describe the player's inventory."""
        result = "Your status:\n"

        # Show vitals
        for vital in self.VITALS:
            value = getattr(world_state.player.inventory, vital, 0)
            result += f"- {vital}: {value}/9\n"

        result += "\nYour inventory:\n"

        # Show other items
        inventory_items = []
        for item_name in [
            "sapling",
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "wood_sword",
            "stone_sword",
            "iron_sword",
        ]:
            value = getattr(world_state.player.inventory, item_name, 0)
            if value > 0:
                inventory_items.append(f"- {item_name}: {value}")

        if inventory_items:
            result += "\n".join(inventory_items)
        else:
            result += "You have no items in your inventory."

        return result

    def __call__(self, world_state: WorldState) -> TextRendererOutput:
        """Render the world state as text for the language model."""
        result = ""

        # Player status
        result += self._describe_player_status(world_state)

        # What player is targeting
        result += self._describe_target(world_state)

        # Local view
        result += self._describe_local_view(world_state)

        # Distant view
        result += self._describe_distant_view(world_state)

        return TextRendererOutput(
            long_term_context=result.strip(),
            short_term_context=self._describe_inventory(world_state),
        )


class UnsupervisedCrafterEnvironmentConfig(EnvironmentConfig):
    area: tuple[int, int]
    view: tuple[int, int]
    size: tuple[int, int]
    text_renderer: Any
    instruction_prompt: str
    reward: bool
    seed: Optional[int] = None
    name: str = "crafter"
    task: str = "open_ended"
    max_episode_steps: int = Field(default=2000)
    render_image: bool = Field(default=False)

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)


def build_base_environment(
    config: UnsupervisedCrafterEnvironmentConfig,
) -> BaseCrafterEnv:
    return BaseCrafterEnv(
        area=config.area,
        view=config.view,
        size=config.size,
        reward=config.reward,
        length=config.max_episode_steps,
        seed=config.seed,
    )


class LanguageSymbolicWrapper:
    def __init__(self, config: UnsupervisedCrafterEnvironmentConfig):
        self.config = config
        self.base_env = build_base_environment(config)
        self.renderer = config.text_renderer
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

    def _build_observation(
        self, obs: np.ndarray, info: dict, world_state: WorldState
    ) -> Observation:
        language_observation = self.renderer(world_state)
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

        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )

        language_observation = self.renderer(world_state)
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

        return OnResetExperience(
            obs=language_obs,
            info=world_state,
        )

    def step(self, action: str) -> Experience[WorldState]:
        obs, reward, done, truncated, info = self._step(action)
        assert self.base_env._step is not None
        world_state = export_world_state(
            self.base_env._world,
            tuple(self.base_env._view),
            self.base_env._step,
        )
        language_obs = self._build_observation(obs, info, world_state)
        self.score_tracker = self._update_progress(info)
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
