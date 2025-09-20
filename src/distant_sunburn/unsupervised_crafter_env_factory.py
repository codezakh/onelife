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


MAP_DISPLAY_ACTION_TO_DESCRIPTION = {
    "Noop": "Used to do nothing.",
    "Move West": "Used to move west.",
    "Move East": "Used to move east.",
    "Move North": "Used to move north.",
    "Move South": "Used to move south.",
    "Do": "Used to interact with the world. Depending on the target, can be used for resource collection or combat.",
    "Sleep": "Puts the player to sleep.",
    "Place Stone": "Used to place a stone in the world.",
    "Place Table": "Used to place a table in the world.",
    "Place Furnace": "Used to place a furnace in the world.",
    "Place Plant": "Used to place a plant in the world.",
    "Make Wood Pickaxe": "Used to create a wood pickaxe.",
    "Make Stone Pickaxe": "Used to create a stone pickaxe.",
    "Make Iron Pickaxe": "Used to create an iron pickaxe.",
    "Make Wood Sword": "Used to create a wood sword.",
    "Make Stone Sword": "Used to create a stone sword.",
    "Make Iron Sword": "Used to create an iron sword.",
}


def get_instruction_prompt():
    action_strings = ",\n".join(
        f"{action}: {MAP_DISPLAY_ACTION_TO_DESCRIPTION[action]}"
        for action in MAP_DISPLAY_ACTION_TO_ENGINE_ACTION
    )
    instruction_prompt = f"""
You are an explorer in an unknown digital world. Your mission is to experience as many of the world's hidden mechanics as possible. Your recorded experiences will be analyzed later to create a complete map of the world's physical laws.

The world's laws follow a simple pattern: `IF (a specific situation occurs) AND (you take an ACTION), THEN (a certain outcome happens).`

To succeed, you must trigger as many different `IF-THEN` scenarios as you can.

**What to Expect in the World:**
This world is complex and may be dangerous.
- **Hostile Entities:** You may encounter creatures that are hostile and will attack you.
- **Resource Collection:** The world contains raw materials that can be gathered, though there may be preconditions for collection.
- **Item Production:** You have the ability to craft useful items from raw materials, though there may be preconditions for production.
- **Combat:** You can engage in combat with the entities you encounter.

Your primary goal is to discover the rules governing these activities. 
You will need to explore the game world by moving around and interacting with the entitiesa and materials in the world.
It is important that you survive long enough to experience as many of the world's mechanics as possible.
If an action has no effect, repeating it over and over again is not a good strategy.
If an action has no effect, you may not have fulfilled the preconditions for the action to have an effect.
Try to get actions to succeed by thinking about necessary preconditions and trying to fulfill them.
Try out as many of the actions as you can, do not get stuck repeating a single action.
Aim to experience a broad range of mechanics, rather than getting stuck on a single category.
Before starting, set some objectives for yourself to guide your exploration.

The following are the only valid actions you can take:

{action_strings}.

You will now receive observations from the world. Begin your exploration.
""".strip()

    return instruction_prompt


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

    def _facing_to_direction(self, facing: Position) -> str:
        """Convert facing position tuple to directional name."""
        if facing.x == 0 and facing.y == 1:
            return "south"
        elif facing.x == 0 and facing.y == -1:
            return "north"
        elif facing.x == 1 and facing.y == 0:
            return "east"
        elif facing.x == -1 and facing.y == 0:
            return "west"
        else:
            return f"({facing.x}, {facing.y})"  # Fallback for unexpected values

    def _get_direction_description(self, from_pos: Position, to_pos: Position) -> str:
        """Get relative direction description from one position to another."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y

        if dx == 0 and dy == 0:
            return "here"

        parts = []
        if dy != 0:
            if dy > 0:
                parts.append(f"{dy} steps south")
            else:
                parts.append(f"{-dy} steps north")

        if dx != 0:
            if dx > 0:
                parts.append(f"{dx} steps east")
            else:
                parts.append(f"{-dx} steps west")

        return ", ".join(parts)

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

        # Use all objects in the world, not just update range
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

    def _render_ascii_map(self, world_state: WorldState) -> str:
        """Render an ASCII map of the local area around the player."""
        player_pos = world_state.player.position
        view_range = max(world_state.view)
        map_size = 2 * view_range + 1  # Make map size match view range
        half_size = view_range

        # Create the map grid
        map_lines = []
        for dy in range(-half_size, half_size + 1):
            line = ""
            for dx in range(-half_size, half_size + 1):
                pos = Position(x=player_pos.x + dx, y=player_pos.y + dy)
                material, entity = world_state.get_tile(pos)

                # Entities take precedence over materials, but only if within view range
                char = " "  # Default to empty space
                if entity is not None:
                    distance = self._manhattan_distance(player_pos, pos)
                    if distance <= view_range:  # Only show entities within view range
                        if entity.name == "player":
                            char = "@"
                        elif entity.name == "cow":
                            char = "c"
                        elif entity.name == "zombie":
                            char = "Z"
                        elif entity.name == "skeleton":
                            char = "S"
                        elif entity.name == "arrow":
                            char = "^"
                        elif entity.name == "plant":
                            char = "p"
                        elif entity.name == "fence":
                            char = "|"
                        else:
                            char = "?"  # Unknown entity

                # If no entity or entity is outside view range, show material
                if char == " " and material is not None:
                    if material == "water":
                        char = "~"
                    elif material == "grass":
                        char = "."
                    elif material == "stone":
                        char = "#"
                    elif material == "path":
                        char = "="
                    elif material == "sand":
                        char = ":"
                    elif material == "tree":
                        char = "T"
                    elif material == "lava":
                        char = "L"
                    elif material == "coal":
                        char = "C"
                    elif material == "iron":
                        char = "I"
                    elif material == "diamond":
                        char = "D"
                    elif material == "table":
                        char = "+"
                    elif material == "furnace":
                        char = "F"
                    else:
                        char = "?"  # Unknown material

                line += char
            map_lines.append(line)

        # Create the legend
        legend = (
            "Legend:\n"
            "Materials: ~water .grass #stone =path :sand Ttree Llava Ccoal Iiron Ddiamond +table Ffurnace\n"
            "Entities:  @player ccow Zzombie Sskeleton ^arrow pplant |fence"
        )

        return (
            f"Local map ({map_size}x{map_size}):\n"
            + "\n".join(map_lines)
            + "\n\n"
            + legend
        )

    def _describe_local_view(self, world_state: WorldState) -> str:
        """Describe the local view - closest materials and entities within view range."""
        result = "Local view (within {} steps):\n".format(max(world_state.view))

        # Add ASCII map
        result += self._render_ascii_map(world_state) + "\n"

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
        """Describe the player's comprehensive status and inventory."""
        result = "Your status:\n"

        # Show vitals
        for vital in self.VITALS:
            value = getattr(world_state.player.inventory, vital, 0)
            result += f"- {vital}: {value}/9\n"

        # Show position and facing
        pos = world_state.player.position
        facing = world_state.player.facing
        result += f"- position: ({pos.x}, {pos.y})\n"
        result += f"- facing: {self._facing_to_direction(facing)}\n"

        # Show sleeping status
        result += f"- sleeping: {world_state.player.sleeping}\n"

        # Show internal state variables
        result += f"- thirst: {world_state.player.thirst:.2f}\n"
        result += f"- hunger: {world_state.player.hunger:.2f}\n"
        result += f"- fatigue: {world_state.player.fatigue:.2f}\n"
        result += f"- recovery: {world_state.player.recover:.2f}\n"

        result += "\nYour inventory:\n"

        # Show all items with their counts
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
            inventory_items.append(f"- {item_name}: {value}")

        result += "\n".join(inventory_items)

        return result

    def _describe_inventory_compact(self, world_state: WorldState) -> str:
        """Describe the player's status and inventory (compact version for long-term context)."""
        result = "Your status:\n"

        # Show vitals
        for vital in self.VITALS:
            value = getattr(world_state.player.inventory, vital, 0)
            result += f"- {vital}: {value}/9\n"

        # Show position and facing
        pos = world_state.player.position
        facing = world_state.player.facing
        result += f"- position: ({pos.x}, {pos.y})\n"
        result += f"- facing: {self._facing_to_direction(facing)}\n"

        # Show sleeping status
        result += f"- sleeping: {world_state.player.sleeping}\n"

        # Show internal state variables
        result += f"- thirst: {world_state.player.thirst:.2f}\n"
        result += f"- hunger: {world_state.player.hunger:.2f}\n"
        result += f"- fatigue: {world_state.player.fatigue:.2f}\n"
        result += f"- recovery: {world_state.player.recover:.2f}\n"

        result += "\nYour inventory:\n"

        # Show only items with counts > 0
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

        # Player status and inventory (comprehensive player state)
        result += self._describe_inventory(world_state) + "\n\n"

        # Player status
        result += self._describe_player_status(world_state)

        # What player is targeting
        result += self._describe_target(world_state)

        # Local view
        result += self._describe_local_view(world_state)

        # Distant view
        result += self._describe_distant_view(world_state)

        return TextRendererOutput(
            long_term_context=self._describe_inventory_compact(world_state),
            short_term_context=result.strip(),
        )


class UnsupervisedCrafterEnvironmentConfig(EnvironmentConfig):
    area: tuple[int, int]
    view: tuple[int, int]
    size: tuple[int, int]
    reward: bool
    seed: Optional[int] = None
    name: str = "crafter"
    task: str = "open_ended"
    max_episode_steps: int = Field(default=2000)
    render_image: bool = Field(default=False)
    instruction_prompt: str = Field(default=get_instruction_prompt())


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
        self.renderer = UnsupervisedTextRenderer()
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
