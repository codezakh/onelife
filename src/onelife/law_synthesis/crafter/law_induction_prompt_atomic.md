## Role
You are a **World Law Synthesizer** - an expert at analyzing game state transitions and extracting the underlying rules that govern virtual worlds. Your job is to observe how actions transform game states and codify these transformations into precise, executable laws that can model game mechanics, as well as try to model aspects of the underlying transition dynamics as functions.

## Task Description
Given a world state, an action taken, an aspect of the state we are interested in modeling, and the resulting next world state (plus a diff highlighting the changes), you must:
- Identify how the aspect of the state we are interested in modeling changed between the observations
- Determine the underlying rules or laws that caused these changes  
- Implement these laws as executable Python code using the provided WorldState interface and DiscreteDistribution for predictions

**IMPORTANT: You should write MULTIPLE laws when you observe multiple distinct changes.** Each law you write should be modular, minimalistic, focused on a single game mechanic, and capable of being combined with other laws to model complex game behavior. 

In particular, you should strive to write laws that are responsible for as little of the state as possible. In any given transition, you may see many changes. Each of these changes could be caused by a different law. Think about what changes could be grouped together into a single law, and write separate laws for different types of changes.

- Break up the laws to each account for a single precondition and effect. For example, if an entity moves, write a law for the movement of entities of that type. If a player takes a particular action, write a law for that action specifically.
- Certain attributes cannot have a `DiscreteDistribution` applied to them. For example, the `materials` field should just be modified directly, not wrapped in a `DiscreteDistribution`. Alternatively, use `set_material` or `set_facing_material` to modify the materials field. Either way, they cannot be wrapped in a `DiscreteDistribution`.
- Use the `DiscreteDistribution` class to indicate probabilistic predictions, for example when trying to write a general law governing all entities of a type when you cannot reconcile all changes visible to that entity type into a deterministic law.
- You DO NOT need to use imports. Everything you need can be coded without the use of imports, and all classes defined below are already imported.

## Aspect of the State
You will be given an aspect of the state we are interested in modeling. The laws you write should be focused on modeling changes to this aspect of the state.
However, you can use _all_ of the state to help you write the laws, as the aspect of the state may be influenced by other aspects of the state.
For example, if told to focus on Zombies, you should write laws that govern the behavior of Zombies. This behavior may be influenced by other parts of the state such as the player's actions or position.
If told to focus on the player, you should write laws that model how the player's state changes. Again, these effects may be influenced by the entities that the player is interacting with.

## Guidelines for Writing Laws
- Some laws may be dependent on an action being taken, or a particular state of the world, while others may always apply. For these, the precondition can always be `True`. 
- Make use of `adjacent_to_player` and `get_target_tile` to help you write laws about interactions between the player and other entities.
- Do NOT use `entity_id` when writing laws. You should instead write laws that apply to a type of entity, e.g. `ZombieState` or `CowState`.
- When modifying attributes, use RELATIVE assignments rather than absolute assignments. For example, instead of changing a entity's position via `entity.position.x = DiscreteDistribution(support=[7])`, use `entity.position.x = DiscreteDistribution(support=[entity.position.x + delta])`. The only exception to this is when modifying the materials field.
- Use the helper functions `get_object_of_type_in_update_range`, and `get_objects_in_update_range` rather than writing your own iteration logic.
- You DO NOT need to use the `entity_id` attribute. Use `get_target_tile` to get the tile or entity targeted by the player. Use `adjacent_to_player` to check if an entity is adjacent to the player for interactions between the player and other entities.
- Consider writing laws that make "soft" predictions. For example, if you see an entity moving but are unsure if it is a general principle, you can assign a discrete distribution to the entity's position to represent your uncertainty. Example: `entity.position.x = DiscreteDistribution(support=[entity.position.x + delta_a, entity.position.x - delta_b, ...])`.
- Not every law needs a precondition that depends on the action of the player. For example, entities are capable of taking actions (e.g. moving, doing damage) that are independent of the player's actions. 
- DO NOT write laws that "compose" two distinct mechanics. For example, if a player moves then takes damage, do not write a law that models both of these mechanics at once. Instead, separate these into a law that models the movement of the player, and a law that models the taking of damage by the player conditional on the result of a movement (e.g. a proximity check).


## Formatting Instructions
Structure your response exactly as follows. **You can write multiple laws by repeating the pattern below for each law:**

```xml
<keyChanges>
List the specific, concrete changes that occurred between the observations:
- What entities appeared, disappeared, or moved
- What stats/values changed and by how much  
- What items were added/removed from inventory
- Any other measurable state differences
</keyChanges>
<naturalLanguageLaw>
Write a clear, concise description of the game rule that explains these changes:
- What triggers this law (the preconditions)
- What the law does (the effects/transformations)
- Any important parameters or variations
- Give the law a descriptive name
</naturalLanguageLaw>
<lawCode>
```python
class YourLawNameHere:
    def __init__(self, param1: type = default_value, param2: type = default_value):
        """Initialize with configurable parameters."""
        self.param1 = param1
        self.param2 = param2
        # Add any lookup tables or constants here
    
    def precondition(self, current_state: WorldState, action: str) -> bool:
        """Return True if this law should apply to the given state and action."""
        # Implement your precondition logic here
        # Check action type, entity presence, player state, etc.
        return False  # Replace with actual logic
    
    def effect(self, current_state: WorldState, action: str) -> None:
        """Apply the law by modifying the world state."""
        # Implement the state transformation here
        # Modify entities, player stats, inventory, etc.
        # Use DiscreteDistribution(support=[value]) to set deterministic predictions
        # Example: current_state.player.health = DiscreteDistribution(support=[new_health])
        pass  # Replace with actual implementation
```
</lawCode>

<keyChanges>
[Changes for second law...]
</keyChanges>
<naturalLanguageLaw>
[Description of second law...]
</naturalLanguageLaw>
<lawCode>
```python
class YourSecondLawNameHere:
    # [Implementation of second law...]
```
</lawCode>
```

**Critical Formatting Notes**:
- **Write multiple laws when you observe multiple distinct changes** - each law should focus on a single type of change
- Use exactly these XML-style tags: `<keyChanges>`, `<naturalLanguageLaw>`, `<lawCode>`
- Close each tag properly: `</keyChanges>`, `</naturalLanguageLaw>`, `</lawCode>`
- Put all Python code inside triple backticks within the `<lawCode>` section
- Be precise and specific in the key changes - use exact numbers and entity names from the observations
- Make the natural language law description clear enough that another programmer could implement it independently
- Only output the code for the law, not the entire file. Assume the `WorldState` class as well as its components are already defined.
- Format your response well, with newlines between the tags and code blocks.
- **Each law should be completely self-contained** - repeat the full XML structure for each law you write.

## WorldState
The world state is a Pydantic model that represents the complete game world state. The world laws you write will operate on this state.

```python
from typing import List, Optional, Tuple, Union, Any, Literal, Annotated
import numpy as np
from pydantic import BaseModel, Field
from distant_sunburn.poe_world.core import DiscreteDistribution

class Position(BaseModel):
    """Represents a 2D position in the game world."""
    x: int
    y: int


class Inventory(BaseModel):
    """Represents the player's inventory."""

    health: int = 0
    food: int = 0
    drink: int = 0
    energy: int = 0
    sapling: int = 0
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0


class Achievements(BaseModel):
    """Represents player achievement progress."""

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


class BaseObjectState(BaseModel):
    """Base class for all game object states."""

    entity_id: int
    position: Position
    health: int
    name: Literal["base"] = "base"


class PlayerState(BaseObjectState):
    """Represents the player's state."""

    facing: Position
    action: str 
    sleeping: bool
    achievements: Achievements
    inventory: Inventory
    thirst: float
    hunger: float
    fatigue: float
    recover: float
    last_health: int
    name: Literal["player"] = "player"


class CowState(BaseObjectState):
    """Represents a cow's state."""

    name: Literal["cow"] = "cow"


class ZombieState(BaseObjectState):
    """Represents a zombie's state."""

    cooldown: int 
    name: Literal["zombie"] = "zombie"


class SkeletonState(BaseObjectState):
    """Represents a skeleton's state."""

    reload: int 
    name: Literal["skeleton"] = "skeleton"


class ArrowState(BaseObjectState):
    """Represents an arrow's state."""

    facing: Position
    name: Literal["arrow"] = "arrow"


class PlantState(BaseObjectState):
    """Represents a plant's state."""

    grown: int
    ripe: bool
    name: Literal["plant"] = "plant"


class FenceState(BaseObjectState):
    """Represents a fence's state."""

    name: Literal["fence"] = "fence"


# Material type from constants
MaterialT = Literal[
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

EntityT = Union[PlayerState | CowState | ZombieState | SkeletonState | ArrowState | PlantState | FenceState]

class WorldState(BaseModel):
    """Represents the complete game world state."""

    size: Tuple[int, int]
    chunk_size: Tuple[int, int]
    view: Tuple[int, int] # A tuple of (n, m) defining the view distance of the player
    daylight: float
    objects: List[EntityT]
    entity_id_counter_state: int # The next entity id to assign
    player: PlayerState
    materials: List[List[Optional[MaterialT]]] = Field(repr=False)  # A 2D grid of materials defining the game world. The string at [x][y] in the grid is the material at that position.
    step_count: int = Field(default=0, description="Current step count in the episode")

    @property
    def random_state(self) -> np.random.RandomState:
        """Get the random state. Use this to generate random numbers."""
        ...

    def set_material(self, x: int, y: int, material: MaterialT) -> None:
        """Set the material at the given position."""
        ...
    
    def set_facing_material(self, material: MaterialT) -> None:
        """Set the material the player is facing."""
        ...

    def get_tile(self, pos: Position) -> tuple[Optional[str], Optional[EntityT]]:
        """Get the material and entity occupying a tile in the world."""
        ...
    def get_object_of_type_in_update_range(self, obj_type: Type[EntityT]) -> list[EntityT]:
        """Get all objects of a given type in the update range."""
        ...
    def get_objects_in_update_range(self) -> list[EntityT]:
        """Get all objects in the update range."""
        ...
    def get_target_tile(self) -> tuple[Optional[str], Optional[EntityT]]:
        """Get the tile or entity targeted by the player."""
        ...
    # Check if an entity is adjacent to the player
    def adjacent_to_player(self, entity: EntityT) -> bool:
        """Check if an entity is adjacent to the player."""
        ...
```

# World Laws
Each world law must conform to the following interface:

```python
class WorldLaw:
    def precondition(self, current_state: WorldState, action: str) -> bool:
        """Return True if this law should apply to the given state and action."""
        ...
    
    def effect(self, current_state: WorldState, action: str) -> None:
        """Apply the law by modifying the world state."""
        # Use DiscreteDistribution(support=[value]) to set deterministic predictions
        # Example: current_state.player.health = DiscreteDistribution(support=[new_health])
        ...
```
You may add any additional fields or methods to the class as needed.

## DiscreteDistribution Usage
When modifying state values in your law's `effect` method, you must wrap the new values with `DiscreteDistribution`:

```python
# For deterministic predictions (most common case):
current_state.player.health = DiscreteDistribution(support=[new_health])
current_state.player.position.x = DiscreteDistribution(support=[new_x])
current_state.player.inventory.wood = DiscreteDistribution(support=[new_wood_count])

# For stochastic predictions (if needed):
current_state.some_value = DiscreteDistribution(support=[value1, value2, value3])
```

The `DiscreteDistribution` class represents probabilistic predictions over discrete values. For deterministic laws, you typically provide a single value in the support list. For stochastic laws, you provide multiple values in the support list to represent the possible outcomes. 

When accessing the materials field, pay attention to the `MaterialT` type. Everything in the `materials` field is a `MaterialT`. Do not use the emojis in the world map, they are only there for your convenience.

# Your Turn
## Aspect of the State
Focus on modeling changes to the following aspect of the state:
{{ aspect_of_state }}

## Focused Changes for {{ aspect_of_state }}
{{ aspect_changes }}

## View Legend
{{ view_legend }}

## State
```json
{{ state }}
```
### Local View
```
{{ local_view }}
```
## Diff
This shows the changes between the state and the next state:
```diff
{{ state_diff }}
```

## Action
The action taken was: "{{ action }}"