# PoE World Law Structure Limitations: A Scientific Analysis

**Date:** 2025-01-27  
**Purpose:** Scientific analysis of fundamental limitations in PoE World's law structure and loss function design

## Abstract

This report provides a systematic analysis of the Product of Experts (PoE) World system's law structure and loss function limitations. Through examination of the original implementation, we identify fundamental architectural constraints that restrict the system's applicability to Atari-style environments and prevent generalization to more complex symbolic game environments such as Crafter or 12-distant-sunburn. The analysis reveals that the loss function design, attribute system, and object representation are deeply coupled to simple 2D physics-based game mechanics and cannot be trivially adapted to complex symbolic state spaces without significant architectural redesign.

## 1. Introduction

The PoE World system demonstrates effective online learning of world models through expert synthesis for Atari-style environments. However, the law structure and loss function architecture exhibit strong coupling to specific environmental characteristics that limit its generalizability. This report systematically examines these limitations through analysis of the original implementation, focusing on the loss function design, attribute system, and object representation constraints.

## 2. Fundamental Loss Function Limitations

### 2.1 Discrete Value Space Assumption

The PoE World loss function fundamentally assumes that all state attributes can be represented as discrete integer values within fixed ranges.

**Evidence from Implementation:**
```python
# From external/poe-world/classes/helper.py lines 1129-1130
all_possible_velocities = np.arange(-Constants.MAX_ABS_VELOCITY,
                                    Constants.MAX_ABS_VELOCITY + 1)
```

**Game-Specific Velocity Ranges:**
```python
# From external/poe-world/classes/helper.py lines 39-63
MONTEZUMA_REVENGE_MAX_ABS_VELOCITY = 15
PONG_MAX_ABS_VELOCITY = 30
PITFALL_MAX_ABS_VELOCITY = 15
BREAKOUT_MAX_ABS_VELOCITY = 15
```

**Loss Function Implementation:**
```python
# From external/poe-world/classes/helper.py lines 1070-1085
if isinstance(obj_dist.velocity_x, RandomValues):
    logprobs += obj_dist.velocity_x.evaluate_logprobs(obj.velocity_x)
else:
    logprobs += 0 if obj_dist.velocity_x == obj.velocity_x else LOG_IMPOSSIBLE_VALUE
```

**Limitation:** This discrete value assumption cannot handle:
- Continuous float values (e.g., `thirst: float` in Crafter)
- Unbounded numerical ranges
- Precision-sensitive measurements
- Complex data types (vectors, matrices, nested structures)

### 2.2 Hardcoded Attribute Set

The loss function operates on a fixed set of attributes that are hardcoded for each game environment.

**Core Attribute Set:**
```python
# From external/poe-world/classes/helper.py lines 1127-1150
def fill_unset_values_with_uniform(obj_list_dist, size_change_flag=False):
    # Hardcoded attribute set
    if not isinstance(obj_dist.velocity_x, RandomValues):
        obj_dist.velocity_x = RandomValues(all_possible_velocities)
    if not isinstance(obj_dist.velocity_y, RandomValues):
        obj_dist.velocity_y = RandomValues(all_possible_velocities)
    if not isinstance(obj_dist.deleted, RandomValues):
        obj_dist.deleted = RandomValues(np.arange(2))  # Binary: 0 or 1
```

**Evidence of Attribute Hardcoding:**
```python
# From external/poe-world/classes/helper.py lines 338-405
class Obj:
    def __init__(self, ...):
        self.velocity_x, self.velocity_y = 0, 0
        self.deleted = 0
        self.history = {
            'velocity_x': [],
            'velocity_y': [],
            'deleted': [1],
            'touch_below': [],
            'w_change': [],
            'h_change': []
        }
```

**Limitation:** The system cannot handle:
- Dynamic attribute discovery
- Nested object structures
- Complex data types
- Game-specific attributes not in the predefined set

### 2.3 Binary Existence Model

The system assumes a simple binary existence model through the `deleted` attribute.

**Evidence:**
```python
# From external/poe-world/classes/helper.py lines 1090-1095
if isinstance(obj_dist.deleted, RandomValues):
    logprobs += 100 * obj_dist.deleted.evaluate_logprobs(obj.deleted)
else:
    logprobs += 0 if obj_dist.deleted == obj.deleted else LOG_IMPOSSIBLE_VALUE
```

**Limitation:** This binary model cannot represent:
- Complex state transitions (e.g., sleeping, crafting, combat states)
- Multi-stage object lifecycles
- Conditional existence based on game state
- Complex creation/destruction mechanics

## 3. Object Representation Constraints

### 3.1 Flat Attribute Structure

The system assumes a flat attribute structure for all objects, preventing representation of hierarchical state.

**Evidence from Object Structure:**
```python
# From external/poe-world/classes/helper.py lines 338-405
class Obj:
    def __init__(self, ...):
        self.velocity_x = 0
        self.velocity_y = 0
        self.deleted = 0
        self.w_change = 0
        self.h_change = 0
        # No support for nested structures
```

**Limitation:** Cannot represent complex state structures such as:
```python
# Crafter's hierarchical state (impossible in PoE World)
player.inventory.wood = 5
player.inventory.stone = 3
player.achievements.collect_wood = 1
player.achievements.defeat_skeleton = 0
```

### 3.2 Game-Specific Object Type Dictionaries

The system requires hardcoded object type dictionaries for each game environment.

**Evidence:**
```python
# From external/poe-world/classes/game_utils/montezuma.py
montezuma_revenge_wh_dict = {
    'player': (8, 20), 'skull': (7, 13), 'spider': (8, 11),
    'key': (7, 15), 'amulet': (6, 15), 'torch': (6, 13),
    'platform': (8, 4), 'ladder': (8, 4), 'wall': (8, 4),
    'disappearing_platform': (8, 4)
}

# From external/poe-world/classes/game_utils/pong.py
pong_wh_dict = {
    'player': (4, 15), 'ball': (2, 4), 'enemy': (4, 15),
    'wall': (148, 5), 'zone': (5, 169)
}
```

**Limitation:** The system cannot:
- Discover new object types automatically
- Handle dynamic object creation/destruction
- Adapt to environments with unknown object types
- Scale to environments with hundreds of object types

### 3.3 Fixed Physics Model

The system assumes a specific 2D physics model with velocity-based movement.

**Evidence from Loss Function:**
```python
# From external/poe-world/classes/helper.py lines 1070-1085
if isinstance(obj_dist.velocity_x, RandomValues):
    logprobs += obj_dist.velocity_x.evaluate_logprobs(obj.velocity_x)
if isinstance(obj_dist.velocity_y, RandomValues):
    logprobs += obj_dist.velocity_y.evaluate_logprobs(obj.velocity_y)
```

**Limitation:** Cannot handle:
- 3D environments
- Complex physics (collision detection, bouncing, chaos)
- Non-velocity-based movement (teleportation, discrete steps)
- Continuous physics with floating-point precision

## 4. Loss Function Evaluation Constraints

### 4.1 Exact Value Matching Requirement

The loss function requires exact value matching for non-RandomValues attributes.

**Evidence:**
```python
# From external/poe-world/classes/helper.py lines 1075-1080
else:
    logprobs += 0 if obj_dist.velocity_x == obj.velocity_x else LOG_IMPOSSIBLE_VALUE
```

**Limitation:** This exact matching requirement:
- Cannot handle floating-point precision issues
- Fails with continuous values that may have small variations
- Requires perfect discretization of continuous values
- Cannot handle approximate or fuzzy matching

### 4.2 Fixed Evaluation Modes

The loss function has only two evaluation modes: position-based and velocity-based.

**Evidence:**
```python
# From external/poe-world/classes/helper.py lines 1050-1122
def evaluate_logprobs_of_obj_list(obj_list_dist, obj_list, by_pos=False, size_change_flag=False):
    if by_pos:
        # Position-based evaluation
        if isinstance(obj_dist.x, RandomValues):
            logprobs += obj_dist.x.evaluate_logprobs(obj.x)
    else:
        # Velocity-based evaluation
        if isinstance(obj_dist.velocity_x, RandomValues):
            logprobs += obj_dist.velocity_x.evaluate_logprobs(obj.velocity_x)
```

**Limitation:** Cannot evaluate:
- Complex state relationships
- Multi-attribute dependencies
- Conditional state changes
- Event-driven state transitions

### 4.3 Object Matching Assumptions

The loss function assumes objects can be matched by ID across frames.

**Evidence:**
```python
# From external/poe-world/classes/helper.py lines 1058-1060
good_pairs, leftover_list1, leftover_list2 = match_two_obj_lists(
    obj_list_dist, obj_list)
```

**Limitation:** This matching assumption breaks down when:
- Objects are created/destroyed frequently
- Object IDs are not persistent
- Complex object relationships exist
- Objects have dynamic properties that affect matching

## 5. Expert Function Constraints

### 5.1 Fixed Function Signature

Expert functions must follow a specific signature pattern that assumes object iteration.

**Evidence from Prompt Templates:**
```python
# From external/poe-world/prompts/synthesizer.py lines 1525-1530
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    obj_list = obj_list.deepcopy()
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Physics-based modifications
        pass
    return obj_list
```

**Limitation:** This signature cannot handle:
- Functions that operate on entire state
- Functions with complex parameters
- Functions that modify multiple object types simultaneously
- Functions that depend on global state

### 5.2 Physics-Based Modification Assumption

Expert functions are designed to modify object physics properties.

**Evidence from Expert Examples:**
```python
# From external/poe-world/mr_world_model_seed0.txt lines 1965-1980
def alter_player_objects(obj_list: ObjList, action: str, touch_side=3, touch_percent=0.1) -> ObjList:
    if action == 'LEFT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        for player_obj in player_objs:
            player_obj.velocity_x = RandomValues([player_obj.velocity_x + 2, player_obj.velocity_x - 2])
    return obj_list
```

**Limitation:** Cannot model:
- Complex game logic
- Resource management
- Achievement systems
- Event-driven mechanics

## 6. Scalability and Extensibility Issues

### 6.1 Exponential Complexity with Attributes

Adding more attributes to the system creates exponential complexity in the loss function.

**Evidence:** The loss function evaluates each attribute independently:
```python
# From external/poe-world/classes/helper.py lines 1070-1100
# Each attribute adds another evaluation term
if isinstance(obj_dist.velocity_x, RandomValues):
    logprobs += obj_dist.velocity_x.evaluate_logprobs(obj.velocity_x)
if isinstance(obj_dist.velocity_y, RandomValues):
    logprobs += obj_dist.velocity_y.evaluate_logprobs(obj.velocity_y)
if isinstance(obj_dist.deleted, RandomValues):
    logprobs += 100 * obj_dist.deleted.evaluate_logprobs(obj.deleted)
```

**Limitation:** This approach does not scale to:
- Environments with hundreds of attributes
- Complex attribute relationships
- Conditional attribute evaluation
- Hierarchical attribute structures

### 6.2 Game-Specific Configuration Requirements

Each new game environment requires extensive configuration.

**Evidence:**
```python
# From external/poe-world/classes/helper.py lines 25-70
def set_global_constants(env_name):
    if env_name == 'MontezumaRevenge':
        Constants.set_constants(montezuma_revenge_wh_dict, ...)
    elif env_name == 'Pong':
        Constants.set_constants(pong_wh_dict, ...)
    else:
        raise NotImplementedError
```

**Limitation:** This approach:
- Requires manual configuration for each new environment
- Cannot automatically adapt to new environments
- Creates maintenance burden for environment-specific code
- Prevents generalization to unknown environments

## 7. Evidence from Complex Environment Analysis

### 7.1 Crafter Environment Incompatibility

Analysis of the Crafter environment reveals fundamental incompatibilities:

**Crafter State Structure:**
```python
# From external/crafter_refactored/crafter/state_export.py
class PlayerState(BaseObjectState):
    facing: Position
    action: str
    sleeping: bool
    achievements: Achievements  # Nested structure
    inventory: Inventory       # Nested structure
    thirst: float             # Continuous value
    hunger: float             # Continuous value
    fatigue: float            # Continuous value
    recover: float            # Continuous value
```

**Incompatibilities:**
1. **Nested Structures**: `achievements` and `inventory` are nested objects, not flat attributes
2. **Continuous Values**: `thirst`, `hunger`, `fatigue`, `recover` are floats, not discrete integers
3. **Complex Types**: `Position` objects cannot be represented as simple velocity attributes
4. **Boolean States**: `sleeping` state cannot be represented by the binary `deleted` attribute

### 7.2 Distant Sunburn Environment Incompatibility

Analysis of the Distant Sunburn environment reveals additional incompatibilities:

**Distant Sunburn State Structure:**
```python
# From external/12-distant-sunburn-gameenv/src/distant_sunburn_gameenv/engine/state.py
class Velocity(BaseModel):
    dx: float = 0.0
    dy: float = 0.0

class DamageEvent(BaseEvent):
    event_type: Literal[EventType.DAMAGE] = EventType.DAMAGE
    target_entity_id: int
    damage_amount: int
    position: Position
    source_type: str
    source_entity_id: Optional[int]
```

**Incompatibilities:**
1. **Complex Physics**: `Velocity` objects with continuous `dx`, `dy` values
2. **Event System**: `DamageEvent` objects represent complex temporal relationships
3. **Optional Attributes**: `source_entity_id` can be `None`, not supported by the discrete value system
4. **Event Logging**: The system has no concept of event history or temporal relationships

## 8. Conclusion

The PoE World law structure and loss function design exhibit fundamental limitations that prevent generalization to complex symbolic environments. These limitations are not superficial configuration issues but deep architectural constraints that would require complete system redesign to overcome.

**Key Limitations:**
1. **Discrete Value Assumption**: Cannot handle continuous or complex data types
2. **Hardcoded Attribute Sets**: Cannot adapt to dynamic or hierarchical state structures
3. **Binary Existence Model**: Cannot represent complex state transitions
4. **Fixed Physics Model**: Cannot handle complex or non-physics-based mechanics
5. **Game-Specific Configuration**: Cannot automatically adapt to new environments
6. **Exponential Complexity**: Does not scale to environments with many attributes

**Evidence-Based Conclusion:**
The PoE World system is fundamentally designed for simple 2D Atari-style environments with discrete state spaces and physics-based mechanics. Adapting it to complex symbolic environments like Crafter or Distant Sunburn would require complete architectural redesign rather than incremental extensions. The current system's limitations are architectural and algorithmic, not configuration issues that can be trivially resolved.

**Implications for Future Work:**
Any attempt to generalize PoE World to complex environments must address these fundamental architectural limitations. This may require:
1. Redesigning the loss function to handle continuous and complex data types
2. Creating flexible object representation systems
3. Developing adaptive attribute discovery mechanisms
4. Building support for hierarchical state structures
5. Implementing event-driven state transition models

The evidence presented in this report demonstrates that PoE World's current architecture is not suitable for complex symbolic environments and that significant architectural innovation is required for such applications.
