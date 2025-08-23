# PoE World Synthesizer Limitations: A Scientific Analysis

**Date:** 2025-01-27  
**Purpose:** Scientific analysis of architectural limitations in PoE World's synthesizer system

## Abstract

This report provides a systematic analysis of the Product of Experts (PoE) World synthesizer system's architectural limitations. Through examination of the original implementation, we identify fundamental constraints that restrict the system's applicability to Atari-style environments and prevent generalization to more complex symbolic game environments such as Crafter or 12-distant-sunburn. The analysis reveals that the synthesizer pattern is deeply coupled to physics-based, object-centric game mechanics and cannot be trivially adapted to symbolic state spaces without significant architectural redesign.

## 1. Introduction

The PoE World system demonstrates effective online learning of world models through expert synthesis for Atari-style environments. However, the synthesizer architecture exhibits strong coupling to specific environmental characteristics that limit its generalizability. This report systematically examines these limitations through analysis of the original implementation in `external/poe-world/learners/synthesizer.py` and `external/poe-world/prompts/synthesizer.py`.

## 2. Fundamental Architectural Constraints

### 2.1 Object-Centric State Representation

The PoE World synthesizer system is fundamentally designed around **physical object iteration** rather than symbolic state manipulation. This manifests in several architectural decisions:

**Core Assumption:** Game state consists of collections of physical objects that can be iterated over and modified independently.

**Evidence from Implementation:**
```python
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Physics-based modifications
        pass
    return obj_list
```

**Limitation:** This pattern cannot represent symbolic state structures where:
- State is a single hierarchical object (e.g., `player.inventory.wood`)
- No physical objects exist to iterate over
- State changes involve complex relationships between multiple attributes

### 2.2 Physics-Based Mechanics Assumption

The synthesizer system assumes **continuous physics-based mechanics** typical of Atari games:

**Core Assumptions:**
- Objects have continuous spatial properties (position, velocity, size)
- Interactions occur through spatial relationships (collision, proximity)
- State changes follow physical laws (momentum, gravity, bouncing)

**Evidence from Prompt Templates:**
- `explain_event_prompt` focuses on velocity and position changes
- `explain_event_snapping_prompt` handles spatial alignment
- `interpret_obj_interact_prompt` assumes spatial interactions

**Limitation:** Cannot model discrete symbolic mechanics such as:
- Resource management (inventory, health, achievements)
- Complex game rules (crafting recipes, achievement unlocks)
- Event-driven systems (damage events, healing events)

### 2.3 Simple Interaction Model

The system assumes **binary spatial interactions** between objects:

**Core Assumption:** Objects interact through simple spatial relationships (touching, proximity).

**Evidence:**
```python
def touches(obj: Obj) -> bool:
    Returns whether this Obj is touching the input obj (True/False)
```

**Limitation:** Cannot model complex symbolic interactions such as:
- Multi-step crafting processes
- Achievement progress tracking
- Complex combat systems with cooldowns and status effects

## 3. Synthesizer Type Analysis

The PoE World system implements 17 specialized synthesizers, each designed for specific Atari-style mechanics:

### 3.1 Action-Based Synthesizers

**ActionSynthesizer** (lines 161-205)
- **Purpose:** Immediate effects of player actions on objects
- **Atari Specificity:** Assumes actions directly modify object properties
- **Limitation:** Cannot handle complex action chains or delayed effects

**MultiTimestepActionSynthesizer** (lines 206-333)
- **Purpose:** Delayed effects of actions over multiple timesteps
- **Atari Specificity:** Assumes POMDP-style delayed physics effects
- **Limitation:** Cannot model complex game logic with conditional triggers

### 3.2 Physics-Based Synthesizers

**MultiTimestepMomentumSynthesizer** (lines 334-529)
- **Purpose:** Momentum changes over multiple timesteps
- **Atari Specificity:** Assumes continuous velocity and momentum physics
- **Limitation:** Cannot model discrete state transitions or symbolic mechanics

**VelocitySynthesizer** (lines 1089-1136)
- **Purpose:** Velocity-related changes
- **Atari Specificity:** Assumes continuous velocity vectors
- **Limitation:** Cannot model discrete movement or teleportation mechanics

**MultiTimestepVelocitySynthesizer** (lines 1137-1330)
- **Purpose:** Velocity evolution over time
- **Atari Specificity:** Assumes continuous velocity trajectories
- **Limitation:** Cannot model pathfinding or complex movement patterns

### 3.3 Object Lifecycle Synthesizers

**PassiveCreationSynthesizer** (lines 1053-1088)
- **Purpose:** Object creation not caused by actions
- **Atari Specificity:** Assumes simple spawning mechanics
- **Limitation:** Cannot model complex entity lifecycle systems

**MultiTimestepSizeChangeSynthesizer** (lines 530-678)
- **Purpose:** Size changes over multiple timesteps
- **Atari Specificity:** Assumes continuous size scaling
- **Limitation:** Cannot model discrete growth stages or state transitions

### 3.4 Status-Based Synthesizers

**MultiTimestepStatusChangeSynthesizer** (lines 679-788)
- **Purpose:** Status changes over multiple timesteps
- **Atari Specificity:** Assumes simple visibility/appearance changes
- **Limitation:** Cannot model complex status systems (health, hunger, fatigue)

**MultiTimestepStatusChangeVelocityModeSynthesizer** (lines 789-897)
- **Purpose:** Status changes with velocity considerations
- **Atari Specificity:** Assumes status affects movement physics
- **Limitation:** Cannot model status effects on game logic

**MultiTimestepStatusChangeSizeModeSynthesizer** (lines 898-1006)
- **Purpose:** Status changes with size considerations
- **Atari Specificity:** Assumes status affects physical size
- **Limitation:** Cannot model status effects on capabilities

### 3.5 Movement Synthesizers

**PassiveMovementSynthesizer** (lines 1007-1052)
- **Purpose:** Movement not caused by actions
- **Atari Specificity:** Assumes autonomous physics-based movement
- **Limitation:** Cannot model AI-driven movement or pathfinding

**NoInteractPassiveMovementSynthesizer** (lines 1402-1447)
- **Purpose:** Movement without interactions
- **Atari Specificity:** Assumes isolated physics movement
- **Limitation:** Cannot model movement influenced by game state

**VelocityTrackingSynthesizer** (lines 1331-1401)
- **Purpose:** Tracking velocity patterns
- **Atari Specificity:** Assumes continuous velocity tracking
- **Limitation:** Cannot model discrete movement patterns

### 3.6 Interaction Synthesizers

**PlayerInteractionSynthesizer** (lines 1448-1480)
- **Purpose:** Player-specific interactions
- **Atari Specificity:** Assumes simple player-object collisions
- **Limitation:** Cannot model complex player-game interactions

**SnappingSynthesizer** (lines 1481-1504)
- **Purpose:** Object snapping/alignment
- **Atari Specificity:** Assumes spatial alignment mechanics
- **Limitation:** Cannot model logical alignment or positioning

**ConstraintsSynthesizer** (lines 1505-1528)
- **Purpose:** Learning object constraints
- **Atari Specificity:** Assumes physical constraints (walls, boundaries)
- **Limitation:** Cannot model logical constraints or game rules

### 3.7 System Synthesizers

**RestartSynthesizer** (lines 1529-1717)
- **Purpose:** Game restart events
- **Atari Specificity:** Assumes simple game state reset
- **Limitation:** Cannot model complex game state transitions

## 4. Prompt Template Limitations: Evidence from Code

### 4.1 Fixed Object Structure Assumption

**Evidence from `explain_event_prompt` (lines 8-100):**
```python
class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)
```

**Specific Limitations:**
- **No nested attributes:** Cannot represent `player.inventory.wood` or `player.achievements.collect_wood`
- **No complex data types:** Only supports integers and RandomValues, not lists, dictionaries, or enums
- **No relationship structures:** Cannot model crafting recipes, achievement dependencies, or complex game rules

### 4.2 Physics-Based Instruction Set

**Evidence from `explain_event_prompt` (lines 50-60):**
```python
Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of action '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are conflicting changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
6. Please use if-condition to indicate that the effects only happen because of action '{action}'
```

**Specific Limitations:**
- **Single attribute changes only:** Rule 1 explicitly restricts functions to modifying "one attribute" - cannot handle multi-attribute relationships like crafting (requires wood AND stone)
- **Spatial interaction only:** Rule 3 mandates using `Obj.touches` - cannot model logical relationships like "has_item" or "can_craft"
- **Velocity assumption:** Rule 5 assumes "velocities of input objects are integers" - cannot model discrete state transitions

### 4.3 Spatial Interaction Assumption

**Evidence from `interpret_obj_interact_prompt` (lines 867-950):**
```python
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to +0
2. The player objects that touch an unknown object set their y-axis velocity to -4
3. The player objects that touch an ladder object set their x-axis velocity to +0
4. The player objects that touch an ladder object set their y-axis velocity to -4
```

**Specific Limitations:**
- **Binary spatial relationships only:** Examples show only "touching" or "not touching" - cannot model complex logical relationships
- **Velocity-based reasoning:** All examples involve velocity changes - cannot model discrete state changes like inventory updates
- **No conditional logic:** Cannot express "if player has wood_pickaxe AND is near stone, then collect stone"

### 4.4 Position-Centric Object Model

**Evidence from `explain_event_snapping_prompt` (lines 670-750):**
```python
class Obj:
    Attributes:
        center_x (int | RandomValues): x-axis center position of the object
        center_y (int | RandomValues): y-axis center position of the object
        left_side (int | RandomValues): left x-axis position of the object
        right_side (int | RandomValues): right x-axis position of the object
        top_side (int | RandomValues): top y-axis position of the object
        bottom_side (int | RandomValues): bottom y-axis position of the object
```

**Specific Limitations:**
- **Spatial positioning only:** All attributes are spatial coordinates - cannot represent abstract properties like "health", "hunger", or "achievement_progress"
- **No symbolic properties:** Cannot model discrete game state like inventory items, tool durability, or achievement flags

### 4.5 Momentum-Based Physics Assumption

**Evidence from `interpret_obj_momentum_pomdp_x_prompt` (lines 950-1000):**
```python
Here's an example with car objects:
Example input list of objects:
car object (id = 0) with x-axis velocity = -3,
car object (id = 0) is at x=32,

Example output list of object changes:
- The car object (id = 0) sets x-axis velocity to [-4, -2, +0]

Example reasons:
1. The car objects with negative x-axis velocity and x-axis position less than or equal to 32 set their x-axis velocity to [-4, -2, +0]
```

**Specific Limitations:**
- **Continuous physics only:** Examples assume continuous velocity and momentum - cannot model discrete movement or teleportation
- **Position-velocity relationships:** Assumes physics-based relationships between position and velocity - cannot model game logic like "if player has key, unlock door"

### 4.6 Object Creation/Deletion Model

**Evidence from `interpret_velocity_creation_prompt` (lines 1200-1250):**
```python
Example output list of object changes:
- The player object (id = 0) is deleted
- A new player object is created at (x=76,y=73)

Example reasons:
1. If player objects touch an unknown object, a new player object is created at (x=76, y=73)
2. If player objects touch a ladder object, a new player object is created at (x=76, y=73)
3. The player objects that touch an unknown object are deleted
4. The player objects that touch a ladder object are deleted
```

**Specific Limitations:**
- **Spatial creation only:** Object creation requires spatial coordinates - cannot model abstract object creation like "add item to inventory"
- **Simple deletion model:** Only supports complete object deletion - cannot model partial state changes like "reduce health by 1"
- **No complex lifecycle:** Cannot model complex entity lifecycles like "plant grows over time" or "enemy spawns when conditions met"

### 4.7 Fixed Function Signature Assumption

**Evidence from all prompt templates:**
```python
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}')
    for {obj_type}_obj in {obj_type}_objs:
        # Physics-based modifications
        pass
    return obj_list
```

**Specific Limitations:**
- **Object iteration requirement:** Function signature assumes iterating over objects of a specific type - cannot model state-wide changes that don't involve object iteration
- **Return value constraint:** Must return `ObjList` - cannot model functions that modify state in-place without returning a new object list
- **Action parameter only:** Only receives `action` parameter - cannot model complex state-dependent behaviors that require access to full game state
- **No context parameter:** Cannot receive additional context like "player position", "inventory contents", or "achievement progress"

## 5. State Space Mismatch Analysis

### 5.1 Crafter State Structure

**Crafter State Characteristics:**
```python
class PlayerState:
    inventory: Inventory  # Nested structure
    achievements: Achievements  # Progress tracking
    health: int  # Discrete value
    hunger: float  # Continuous but bounded
    fatigue: float  # Continuous but bounded
```

**Mismatch with PoE World:**
- No physical objects to iterate over
- Complex nested state structure
- Discrete and continuous mixed attributes
- Relationship-based state changes

### 5.2 12-Distant-Sunburn State Structure

**12-Distant-Sunburn State Characteristics:**
```python
class State:
    player: Player  # Complex player state
    entities: list[Entity]  # Heterogeneous entity list
    event_log: EventLog  # Event-driven system
    turn_count: int  # Time-based mechanics
```

**Mismatch with PoE World:**
- Event-driven architecture
- Complex entity interactions
- Time-based mechanics
- Heterogeneous entity types

## 6. Expert Function Protocol Limitations

### 6.1 Mutation-Based State Changes

**Current Protocol:**
```python
def __call__(self, current_state: MetadataT, action: str, **context: Any) -> None:
    # Mutate state in-place
```

**Limitation:** Cannot handle:
- Immutable state structures
- Complex validation logic
- Side effects or external state changes

### 6.2 Simple Context Model

**Current Context:**
- Limited to simple key-value pairs
- No structured context objects
- No relationship information

**Limitation:** Cannot express:
- Complex game state relationships
- Historical context (previous states)
- Environmental context (world state)

## 7. Conclusion

The PoE World synthesizer system exhibits fundamental architectural constraints that prevent its direct application to symbolic game environments. The system is deeply coupled to:

1. **Physics-based mechanics** rather than symbolic game logic
2. **Object-centric state representation** rather than hierarchical symbolic state
3. **Spatial interaction models** rather than logical relationship models
4. **Continuous attribute changes** rather than discrete state transitions

These limitations are not superficial implementation details but represent core architectural assumptions that would require fundamental redesign to overcome. The synthesizer pattern demonstrates effective learning for its intended domain but cannot be trivially extended to more complex symbolic environments without significant architectural changes.

## References

1. PoE World Implementation: `external/poe-world/learners/synthesizer.py`
2. PoE World Prompts: `external/poe-world/prompts/synthesizer.py`
3. Crafter State Export: `external/crafter_refactored/crafter/state_export.py`
4. 12-Distant-Sunburn Laws: `external/12-distant-sunburn-gameenv/src/distant_sunburn_gameenv/engine/laws.py`
5. 12-Distant-Sunburn State: `external/12-distant-sunburn-gameenv/src/distant_sunburn_gameenv/engine/state.py`
