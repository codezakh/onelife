I have reviewed the provided documents and the plan to write hand-written experts. The core components from the project's PRD, such as the `ExpertFunction` protocol and the `RandomValues` class, are well-defined and can be directly applied to the simple 1D environment. There are no blocking issues or implementation gaps that prevent the creation of a derivative PRD for these experts.

Here is the PRD for the hand-written correct and incorrect experts.

***

# PRD: Hand-Written Experts for the 1D Test Environment

**Version:** 1.0
**Date:** 2025-08-20
**Author:** System

## 1. Overview

This document specifies the requirements for implementing two sets of hand-written, programmatic experts for the "Simple 1D Test Environment." The purpose of these experts is to provide a ground-truth test case for the PoE-World inference pipeline.

The two sets are:
1.  **Correct Experts:** A set of functions that perfectly model the known, true mechanics of the 1D environment.
2.  **Incorrect Experts:** A set of functions that model plausible but incorrect mechanics.

When these experts are fed into the weight-fitting component of the PoE-World system, the pipeline is considered successful if it assigns high weights to the correct experts and low or zero weights to the incorrect ones. These experts are for testing purposes only and are not part of the environment's internal logic.

## 2. Goals and Objectives

*   Define a set of correct expert functions that precisely replicate the behavior of the `MovementLaw` and `LightLaw` from the environment.
*   Define a set of deliberately incorrect expert functions that model flawed physics.
*   Ensure all experts conform to the `ExpertFunction` protocol defined in the main project specification.
*   Ensure all experts use the `RandomValues` class to represent their predictions probabilistically.
*   Establish a clear naming convention to distinguish correct experts from incorrect ones for testability.

## 3. Required Context from Environment PRD

An engineer implementing these experts must be familiar with the following components from the "Simple 1D Test Environment" PRD.

### 3.1. State and Action Definitions

The experts will operate on the following data structures.

```python
# State Representation
@dataclass
class Player:
    position: int

@dataclass
class Light:
    position: int
    is_on: bool

@dataclass(frozen=True)
class WorldConfig:
    width: int
    switch_point: int

@dataclass
class GameState:
    config: WorldConfig
    player: Player
    lights: List[Light]
    rng: random.Random

# Action Enumeration
class Action(Enum):
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    STAY = auto()
```

### 3.2. True Environment Mechanics (The "Laws")

The correct experts must perfectly model these mechanics.

*   **`MovementLaw(slip_probability)`:**
    *   Player position is bounded within `[0, width - 1]`.
    *   If the player is at or beyond `switch_point`, the effect of `MOVE_LEFT` and `MOVE_RIGHT` is inverted.
    *   There is a `slip_probability` chance that the intended direction of movement will be inverted. This check happens *after* the switched zone check.
*   **`LightLaw(toggle_probability)`:**
    *   This law is independent of the player's action.
    *   Each light in the state has a `toggle_probability` chance of having its `is_on` attribute flipped on each turn.

## 4. Core Components for Expert Implementation

All experts must adhere to the following protocol and use the specified data structure for their predictions.

### 4.1. The `ExpertFunction` Protocol

This protocol defines the required interface for any expert. The function should not return a value but instead **mutate the `current_state` object in-place** by assigning `RandomValues` objects to the attributes it predicts.

```python
from typing import Protocol
# Assume GameState and Action are imported

class ExpertFunction(Protocol):
    """
    Protocol defining the interface that all expert functions must implement.
    """
    def __call__(self, current_state: GameState, action: Action) -> None:
        """
        Executes the expert's logic on the current state.

        This function modifies current_state in-place by assigning
        RandomValues objects to attributes that the expert has an opinion about.
        """
        ...
```

### 4.2. The `RandomValues` Class

Experts express their predictions not as primitive values, but as `RandomValues` objects. For these hand-written experts, we will create "sharp" distributions, meaning the `values` array will contain only a single possible outcome. The broader system is responsible for adding noise to these predictions later.

```python
import numpy as np

@attrs.define
class RandomValues:
    """
    Represents a discrete probability distribution over a set of integer or boolean values.
    """
    values: np.ndarray
    logscores: np.ndarray = attrs.field()

    @logscores.default
    def _default_logscores(self) -> np.ndarray:
        """Defaults to uniform logscores if not provided."""
        return np.zeros_like(self.values, dtype=float)

# Example Usage:
# To predict the player's next position is 5:
# current_state.player.position = RandomValues(values=np.array([5]))
#
# To predict a light will be on (True):
# light.is_on = RandomValues(values=np.array([True]))
```

## 5. Specifications for Correct Experts

These experts should be named with a `correct_` prefix. They must use the `rng` object from the `current_state` for any stochastic predictions to ensure their predictions can be validated against the environment's true evolution.

### 5.1. `correct_movement_expert`

*   **Target Object Type:** `player`
*   **Description:** Perfectly models the `MovementLaw`.
*   **Logic:**
    1.  Check the `action`. If `STAY`, do not modify the player's state.
    2.  Determine the initial `direction` (-1 for `MOVE_LEFT`, +1 for `MOVE_RIGHT`).
    3.  Check if `current_state.player.position` is in the switched zone (`>= switch_point`). If so, invert `direction`.
    4.  Using `current_state.rng`, check for a slip event against a hardcoded `slip_probability` (e.g., 0.1, matching the law's configuration). If a slip occurs, invert `direction`.
    5.  Calculate the `new_position` and clamp it to the world boundaries `[0, width - 1]`.
    6.  Assign the final prediction: `current_state.player.position = RandomValues(values=np.array([new_position]))`.

### 5.2. `correct_light_expert`

*   **Target Object Type:** `light`
*   **Description:** Perfectly models the `LightLaw`.
*   **Logic:**
    1.  Iterate through each `light` in `current_state.lights`.
    2.  Using `current_state.rng`, check for a toggle event against a hardcoded `toggle_probability` (e.g., 0.2, matching the law's configuration).
    3.  If the event occurs, the new state is `not light.is_on`.
    4.  If the event does not occur, the new state is `light.is_on`.
    5.  Assign the final prediction for that light: `light.is_on = RandomValues(values=np.array([new_state_boolean]))`.

## 6. Specifications for Incorrect Experts

These experts should be named with an `incorrect_` prefix. They introduce specific, deliberate flaws in their model of the world.

### 6.1. `incorrect_movement_expert_ignores_switch`

*   **Target Object Type:** `player`
*   **Description:** Models player movement but completely ignores the switched-zone mechanic. It should still correctly model boundaries and slipperiness.
*   **Logic:**
    1.  Follows the same logic as `correct_movement_expert` but **skips step 3** (the switched zone check).
    2.  It correctly calculates direction, checks for slipperiness, clamps to boundaries, and assigns the `RandomValues` object.

### 6.2. `incorrect_movement_expert_ignores_slip`

*   **Target Object Type:** `player`
*   **Description:** Models player movement but assumes the world is never slippery. It should still correctly model the switched zone and boundaries.
*   **Logic:**
    1.  Follows the same logic as `correct_movement_expert` but **skips step 4** (the slipperiness check).
    2.  It correctly calculates direction, checks for the switched zone, clamps to boundaries, and assigns the `RandomValues` object.

### 6.3. `incorrect_light_expert_is_deterministic`

*   **Target Object Type:** `light`
*   **Description:** Incorrectly models the light behavior as deterministic, assuming lights always toggle.
*   **Logic:**
    1.  Iterate through each `light` in `current_state.lights`.
    2.  Always predict the new state will be `not light.is_on`.
    3.  Assign the prediction: `light.is_on = RandomValues(values=np.array([not light.is_on]))`.

### 6.4. `incorrect_light_expert_action_dependent`

*   **Target Object Type:** `light`
*   **Description:** Incorrectly models that lights only toggle if the player moves right.
*   **Logic:**
    1.  Check if `action == Action.MOVE_RIGHT`.
    2.  If true, iterate through lights and predict they will all toggle (`not light.is_on`).
    3.  If false, iterate through lights and predict they will not change (`light.is_on`).
    4.  Assign the `RandomValues` object for each light accordingly.