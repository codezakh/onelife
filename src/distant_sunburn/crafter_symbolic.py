"""
Symbolic observation wrapper and object model for Crafter RL environment.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal, Annotated, cast, Any
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field
import gym
from gym import spaces
import rich
import crafter.objects
from crafter import Env as CrafterEnv
import numpy.typing as npt
import crafter.engine


# Vector-based directions (no string enums)
class Direction(BaseModel):
    x: int
    y: int


# Location model
class Location(BaseModel):
    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


# Base entity class
class Entity(BaseModel):
    location: Location
    health: Optional[int] = None
    tag: Literal["entity"] = "entity"


# Player entity
class Player(Entity):
    facing: Direction
    sleeping: bool = False
    tag: Literal["player"] = "player"


# Mob entities
class Cow(Entity):
    tag: Literal["cow"] = "cow"


class Zombie(Entity):
    tag: Literal["zombie"] = "zombie"


class Skeleton(Entity):
    tag: Literal["skeleton"] = "skeleton"


class Arrow(Entity):
    facing: Direction
    tag: Literal["arrow"] = "arrow"


# Object entities
class Plant(Entity):
    ripe: bool = False
    grown: int = 0
    tag: Literal["plant"] = "plant"


class Fence(Entity):
    tag: Literal["fence"] = "fence"


# Material types
class Material(BaseModel):
    type: Literal[
        "water",
        "grass",
        "stone",
        "path",
        "sand",
        "tree",
        "lava",
        "coal",
        "iron",
        "diamond",
        "table",
        "furnace",
    ]
    location: Location


# Inventory model with all fields always present
class Inventory(BaseModel):
    # Vitals
    health: int = Field(ge=0, le=9)
    food: int = Field(ge=0, le=9)
    drink: int = Field(ge=0, le=9)
    energy: int = Field(ge=0, le=9)

    # Resources
    sapling: int = Field(ge=0, le=9)
    wood: int = Field(ge=0, le=9)
    stone: int = Field(ge=0, le=9)
    coal: int = Field(ge=0, le=9)
    iron: int = Field(ge=0, le=9)
    diamond: int = Field(ge=0, le=9)

    # Tools
    wood_pickaxe: int = Field(ge=0, le=9)
    stone_pickaxe: int = Field(ge=0, le=9)
    iron_pickaxe: int = Field(ge=0, le=9)
    wood_sword: int = Field(ge=0, le=9)
    stone_sword: int = Field(ge=0, le=9)
    iron_sword: int = Field(ge=0, le=9)


# Achievement tracking
class Achievements(BaseModel):
    collect_coal: int = 0
    collect_diamond: int = 0
    collect_drink: int = 0
    collect_iron: int = 0
    collect_sapling: int = 0
    collect_stone: int = 0
    collect_wood: int = 0
    defeat_skeleton: int = 0
    defeat_zombie: int = 0
    eat_cow: int = 0
    eat_plant: int = 0
    make_iron_pickaxe: int = 0
    make_iron_sword: int = 0
    make_stone_pickaxe: int = 0
    make_stone_sword: int = 0
    make_wood_pickaxe: int = 0
    make_wood_sword: int = 0
    place_furnace: int = 0
    place_plant: int = 0
    place_stone: int = 0
    place_table: int = 0
    wake_up: int = 0


# Environment parameters
class EnvironmentParameters(BaseModel):
    world_size: Tuple[int, int] = (64, 64)
    view_size: Tuple[int, int] = (9, 9)
    max_episode_steps: int = 10000
    day_length: int = 300

    # Material spawn parameters (from worldgen)
    coal_probability: float = 0.15
    iron_probability: float = 0.25
    diamond_probability: float = 0.006
    tree_probability: float = 0.2

    # Mob spawn parameters
    cow_spawn_prob: float = 0.015
    zombie_spawn_prob: float = 0.007
    skeleton_spawn_prob: float = 0.05


# Configuration for observation mode
class ObservationConfig(BaseModel):
    full_observability: bool = False  # If True, show entire world state
    absolute_coordinates: bool = True  # If False, use agent-relative coords
    include_latent_states: bool = False  # If True, include hidden vars like mob health
    view_distance: Optional[int] = None  # Override default view distance


# Action model
class Action(str, Enum):
    NOOP = "noop"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    DO = "do"
    SLEEP = "sleep"
    PLACE_STONE = "place_stone"
    PLACE_TABLE = "place_table"
    PLACE_FURNACE = "place_furnace"
    PLACE_PLANT = "place_plant"
    MAKE_WOOD_PICKAXE = "make_wood_pickaxe"
    MAKE_STONE_PICKAXE = "make_stone_pickaxe"
    MAKE_IRON_PICKAXE = "make_iron_pickaxe"
    MAKE_WOOD_SWORD = "make_wood_sword"
    MAKE_STONE_SWORD = "make_stone_sword"
    MAKE_IRON_SWORD = "make_iron_sword"


# Complete symbolic observation
class SymbolicObservation(BaseModel):
    # Player state
    player: Player
    inventory: Inventory
    achievements: Achievements

    # Visible materials
    materials: List[Material]

    # Environment info
    daylight: float = Field(ge=0.0, le=1.0)

    # Status flags
    dead: bool = False

    # Visible entities
    entities: List[
        Annotated[
            Union[Entity, Cow, Zombie, Skeleton, Arrow, Plant, Fence],
            Field(discriminator="tag"),
        ]
    ] = Field(default_factory=list)


class CrafterSymbolic:
    """Core class for converting Crafter state to symbolic observations."""

    def __init__(self, config: ObservationConfig):
        self.config = config
        self.env_params = EnvironmentParameters()

    def get_symbolic_observation(
        self,
        env: CrafterEnv,
    ) -> SymbolicObservation:
        """Convert raw observation to symbolic format."""
        assert isinstance(env, CrafterEnv)
        assert env._player is not None

        # Access internal game state
        player_obj = env._player
        world = env._world

        # Get player location and facing
        player_loc = Location(x=player_obj.pos[0], y=player_obj.pos[1])
        player_facing = Direction(x=player_obj.facing[0], y=player_obj.facing[1])

        # Create player object
        player = Player(
            location=player_loc,
            health=player_obj.health,
            facing=player_facing,
            sleeping=player_obj.sleeping,
        )

        # Create inventory (all fields always included)
        inventory = Inventory(**player_obj.inventory)

        # Create achievements
        achievements = Achievements(**player_obj.achievements)

        # Get visible entities and materials
        if self.config.full_observability:
            entities = self._get_all_entities(world, player_loc)
            materials = self._get_all_materials(world, player_loc)
        else:
            view_dist = self.config.view_distance or max(env._view) // 2
            entities = self._get_visible_entities(world, player_loc, view_dist)
            materials = self._get_visible_materials(world, player_loc, view_dist)

        # Create observation
        return SymbolicObservation(
            player=player,
            inventory=inventory,
            achievements=achievements,
            entities=entities,
            materials=materials,
            daylight=world.daylight,
            dead=player_obj.health <= 0,
        )

    def _get_visible_entities(
        self, world: crafter.engine.World, player_loc: Location, view_distance: int
    ) -> List[Entity]:
        """Get entities within view distance."""
        entities: List[Entity] = []

        for obj in world.objects:
            if obj is None or obj.removed:
                continue

            # Check distance
            dist = abs(obj.pos[0] - player_loc.x) + abs(obj.pos[1] - player_loc.y)
            if dist > view_distance:
                continue

            # Convert to symbolic entity
            entity = self._object_to_entity(obj, player_loc)
            if entity:
                entities.append(entity)

        return entities

    def _get_all_entities(
        self, world: crafter.engine.World, player_loc: Location
    ) -> List[Entity]:
        """Get all entities in the world."""
        entities = []

        for obj in world.objects:
            if obj is None or obj.removed:
                continue

            entity = self._object_to_entity(obj, player_loc)
            if entity:
                entities.append(entity)

        return entities

    def _object_to_entity(
        self, obj: crafter.objects.Object, player_loc: Location
    ) -> Optional[Entity]:
        """Convert game object to symbolic entity."""
        # Get location
        if self.config.absolute_coordinates:
            loc = Location(x=obj.pos[0], y=obj.pos[1])
        else:
            # Agent-relative coordinates
            loc = Location(x=obj.pos[0] - player_loc.x, y=obj.pos[1] - player_loc.y)

        # Skip player object
        if isinstance(obj, crafter.objects.Player):
            return None

        # Create appropriate entity type
        if isinstance(obj, crafter.objects.Cow):
            entity = Cow(location=loc)
            if self.config.include_latent_states:
                entity.health = obj.health

        elif isinstance(obj, crafter.objects.Zombie):
            entity = Zombie(location=loc)
            if self.config.include_latent_states:
                entity.health = obj.health

        elif isinstance(obj, crafter.objects.Skeleton):
            entity = Skeleton(location=loc)
            if self.config.include_latent_states:
                entity.health = obj.health

        elif isinstance(obj, crafter.objects.Arrow):
            entity = Arrow(
                location=loc, facing=Direction(x=obj.facing[0], y=obj.facing[1])
            )

        elif isinstance(obj, crafter.objects.Plant):
            entity = Plant(location=loc, ripe=obj.ripe)
            if self.config.include_latent_states:
                entity.grown = obj.grown

        elif isinstance(obj, crafter.objects.Fence):
            entity = Fence(location=loc)

        else:
            return None

        return entity

    def _get_visible_materials(
        self, world: crafter.engine.World, player_loc: Location, view_distance: int
    ) -> List[Material]:
        """Get materials within view distance."""
        materials: List[Material] = []

        x_min = max(0, player_loc.x - view_distance)
        x_max = min(world.area[0], player_loc.x + view_distance + 1)
        y_min = max(0, player_loc.y - view_distance)
        y_max = min(world.area[1], player_loc.y + view_distance + 1)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                material, obj = world[x, y]
                if material:
                    if self.config.absolute_coordinates:
                        loc = Location(x=x, y=y)
                    else:
                        loc = Location(x=x - player_loc.x, y=y - player_loc.y)

                    materials.append(Material(type=material, location=loc))

        return materials

    def _get_all_materials(
        self, world: crafter.engine.World, player_loc: Location
    ) -> List[Material]:
        """Get all materials in the world."""
        materials: List[Material] = []

        for x in range(world.area[0]):
            for y in range(world.area[1]):
                material, obj = world[x, y]
                if material:
                    if self.config.absolute_coordinates:
                        loc = Location(x=x, y=y)
                    else:
                        loc = Location(x=x - player_loc.x, y=y - player_loc.y)

                    materials.append(Material(type=material, location=loc))

        return materials


class CrafterSymbolicWrapper(gym.Wrapper):
    """Symbolic observation wrapper for Crafter environment."""

    def __init__(self, env: gym.Env, config: ObservationConfig):
        super().__init__(env)
        self.config = config
        self._symbolic = CrafterSymbolic(config)

        # Override action space to use symbolic actions
        self.action_space = spaces.Discrete(len(Action))

        assert isinstance(self.env, CrafterEnv)

    def reset(self):
        """Reset environment and return symbolic observation."""
        # Shape is (64, 64, 3)
        obs = cast(npt.NDArray[np.uint8], self.env.reset())
        return self._symbolic.get_symbolic_observation(cast(CrafterEnv, self.env))

    def step(self, action: Union[int, Action]):
        """Execute action and return symbolic observation."""
        # Convert symbolic action to env action
        if isinstance(action, Action):
            action_idx = list(Action).index(action)
        else:
            action_idx = action

        obs, reward, done, info = self.env.step(action_idx)

        symbolic_obs = self._symbolic.get_symbolic_observation(
            cast(CrafterEnv, self.env)
        )
        return symbolic_obs, reward, done, info

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render()


# Example usage
if __name__ == "__main__":
    # Create base environment
    base_env = crafter.Env()

    # Configure symbolic wrapper
    config = ObservationConfig(
        full_observability=False,
        absolute_coordinates=True,
        include_latent_states=False,
        view_distance=4,
    )

    # Create wrapped environment
    env = CrafterSymbolicWrapper(base_env, config)

    # Run a few steps
    obs = env.reset()
    rich.print(obs)

    for _ in range(10):
        action = Action.MOVE_LEFT
        obs, reward, done, info = env.step(action)

    rich.print(obs)
