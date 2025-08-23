# PRD: Simple 1D Test Environment for PoE-World

**Version:** 1.0
**Date:** 2025-08-20
**Author:** System

## 1. Overview

This document specifies the requirements for a simple, configurable, and fully reproducible 1D grid world environment. The primary purpose of this environment is to serve as an initial testbed for the PoE-World project. Its simplicity is intentional, allowing for the manual creation of programmatic "expert" laws and the verification of the project's core expert synthesis and weight-fitting pipeline.

The environment's logic will be purely functional and symbolic, with no graphical or user-facing components. It will feature an object-centric state representation and mechanics implemented as a sequence of modular "world laws," mirroring the architecture of more complex target environments.

## 2. Goals and Objectives

*   **Provide a Functional Core:** The environment must be implemented as two pure functions: `initial_state()` and `transition_function(state, action)`.
*   **Implement Object-Centric State:** The world state must be represented by structured data objects (e.g., Python `dataclasses`), not raw arrays, to facilitate object-centric learning.
*   **Model Separable Dynamics:** The environment must contain distinct and independent mechanics (switched movement, slipperiness, stochastic object state changes) that can be modeled by separate, hand-written experts.
*   **Ensure Full Reproducibility:** Given the same initial seed, applying the same sequence of actions must always produce the exact same sequence of states.
*   **Be Easily Configurable:** Key parameters like world size, slipperiness, and stochastic event probabilities must be configurable during initialization.

## 3. System Architecture & Design Principles

### 3.1. Functional Core

The environment's dynamics are defined by a pure transition function. This function takes the current state and an action and returns a new state, without any side effects or modification of the original state. This design ensures testability and predictability.

`new_state = transition_function(current_state, action)`

### 3.2. Law-Based Mechanics

The `transition_function` does not contain monolithic game logic. Instead, it acts as an orchestrator that applies a sequence of modular "World Law" objects to the state. Each law is responsible for a single aspect of the world's dynamics (e.g., player movement, light behavior). This modularity is critical for testing the PoE-World framework's ability to learn independent causal mechanisms.

```python
# Pseudocode for the transition function
def transition_function(state: GameState, action: Action) -> GameState:
    new_state = copy.deepcopy(state)
    
    for law in WORLD_LAWS:
        law.apply(new_state, action)
        
    return new_state
```

### 3.3. Object-Centric State

The state of the world is a container for distinct, typed objects. This contrasts with representing the state as a simple grid or array of numbers. For example, the state object will have attributes like `state.player` and `state.lights`, where each is an object with its own attributes.

### 3.4. State-Managed Randomness

To ensure reproducibility, all stochastic events are driven by a single `random.Random` instance that is part of the game state itself. This object is passed through each transition, ensuring that a given seed and action sequence will always produce the identical outcome.

## 4. Detailed Specifications

### 4.1. State Representation

The environment's state will be defined using Python `dataclasses`.

```python
import random
from dataclasses import dataclass
from typing import List

@dataclass
class Player:
    position: int

@dataclass
class Light:
    position: int
    is_on: bool

@dataclass(frozen=True) # Configuration is immutable
class WorldConfig:
    width: int
    switch_point: int # The first coordinate in the "switched" half

@dataclass
class GameState:
    config: WorldConfig
    player: Player
    lights: List[Light]
    rng: random.Random
```

### 4.2. Actions

Actions are represented by a simple enumeration.

```python
from enum import Enum, auto

class Action(Enum):
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    STAY = auto()
```

### 4.3. World Generation (`initial_state`)

This function creates the deterministic starting state for the environment.

*   **Signature:** `initial_state(width: int = 12, num_lights: int = 2, seed: int = 42) -> GameState`
*   **Logic:**
    1.  Creates a `WorldConfig` instance. `width` is set by the argument, and `switch_point` is calculated as `width // 2`.
    2.  Instantiates the `Player` at a fixed starting position, e.g., `position = width // 4`.
    3.  Instantiates `num_lights` `Light` objects at fixed, distinct positions (e.g., one in each half of the world). Their initial `is_on` state can be fixed (e.g., `False`).
    4.  Initializes a `random.Random` instance with the provided `seed`.
    5.  Returns a fully populated `GameState` object.

### 4.4. World Laws

The environment will have two distinct laws, each with its own configuration.

#### 4.4.1. `MovementLaw`

This law governs the player's position.

*   **Initialization:** `MovementLaw(slip_probability: float = 0.1)`
*   **Logic (`apply` method):**
    1.  If the action is `STAY`, do nothing.
    2.  Determine the base direction: -1 for `MOVE_LEFT`, +1 for `MOVE_RIGHT`.
    3.  **Switched Zone Check:** Check if `state.player.position >= state.config.switch_point`. If true, multiply the direction by -1.
    4.  **Slipperiness Check:** Draw a random float from `state.rng.random()`. If the float is less than `slip_probability`, multiply the direction by -1 again.
    5.  Calculate the `new_position` by adding the final direction to the current position.
    6.  **Boundary Check:** Clamp the `new_position` to be within `[0, state.config.width - 1]`.
    7.  Update `state.player.position` to the clamped `new_position`.

#### 4.4.2. `LightLaw`

This law governs the state of the `Light` objects, representing a simple stochastic process independent of player actions.

*   **Initialization:** `LightLaw(toggle_probability: float = 0.2)`
*   **Logic (`apply` method):**
    1.  Iterate through each `light` in `state.lights`.
    2.  For each light, draw a random float from `state.rng.random()`.
    3.  If the float is less than `toggle_probability`, flip the boolean value of `light.is_on`.

### 4.5. Transition Function (`transition_function`)

This function orchestrates the application of the laws.

*   **Signature:** `transition_function(state: GameState, action: Action, laws: List[Law]) -> GameState`
*   **Logic:**
    1.  Create a deep copy of the input `state`.
    2.  Iterate through the provided `laws` list.
    3.  For each `law`, call its `apply` method with the new state and the action.
    4.  Return the modified new state.

## 5. Testing Strategy

The environment must be accompanied by a comprehensive test suite to validate its mechanics.

*   **Unit Tests for Laws:**
    *   **`MovementLaw`:**
        *   Test standard movement in the non-switched zone with zero slipperiness.
        *   Test that movement is inverted in the switched zone with zero slipperiness.
        *   Test boundary conditions: ensure the player cannot move past position 0 or `width - 1`.
        *   Test deterministic slipperiness (`slip_probability=1.0`): movement in the normal zone should be inverted.
        *   Test that the `STAY` action results in no change of position.
    *   **`LightLaw`:**
        *   Test with `toggle_probability=0.0`: lights should never change state.
        *   Test with `toggle_probability=1.0`: lights should always flip their state.

*   **Integration Tests for `transition_function`:**
    *   **Reproducibility:** Create an initial state with a specific seed. Apply a fixed sequence of 20 random actions. Store the final state. Reset the environment with the same seed and apply the same action sequence. Assert that the new final state is identical to the stored one.
    *   **State Independence:** Verify that the `transition_function` does not modify the original state object passed into it.

## 6. Out of Scope

*   A graphical (GUI) or terminal (TUI) interface for visualization or interaction.
*   An agent, policy, or planner for making decisions in the environment.
*   Any data collection logic, such as an experience buffer. The component is only responsible for the `initial_state` and `transition_function`.