"""
Hand-written experts for the 1D test environment.

This module contains both correct and incorrect expert functions that model
the mechanics of the 1D test environment. These experts are used to test
the PoE-World weight-fitting pipeline.

Correct experts perfectly model the true environment mechanics, while incorrect
experts introduce deliberate flaws to test the system's ability to distinguish
between good and bad models.
"""

import numpy as np
from typing import Any

from ..core import RandomValues, ExpertFunction
from ...simple_1d_env.environment import GameState, Action
from ...typing_utils import implements


def correct_movement_expert(
    current_state: GameState, action: Action, **context: Any
) -> None:
    """
    Correct expert that perfectly models the MovementLaw.

    This expert models:
    - Player position is bounded within [0, width - 1]
    - If the player is at or beyond switch_point, the effect of MOVE_LEFT and MOVE_RIGHT is inverted
    - There is a slip_probability chance that the intended direction will be inverted
    - The slip check happens after the switched zone check

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # If STAY action, do not modify the player's state
    if action == Action.STAY:
        return

    # Determine the initial direction
    if action == Action.MOVE_LEFT:
        direction = -1
    elif action == Action.MOVE_RIGHT:
        direction = 1
    else:
        # Unknown action, don't modify state
        return

    # Check if player is in the switched zone
    if current_state.player.position >= current_state.config.switch_point:
        direction *= -1

    # Check for slip event using the state's RNG
    slip_probability = 0.1  # Hardcoded to match the law's configuration
    if current_state.rng.random() < slip_probability:
        direction *= -1

    # Calculate new position and clamp to boundaries
    new_position = current_state.player.position + direction
    new_position = max(0, min(new_position, current_state.config.width - 1))

    # Assign the prediction using RandomValues
    current_state.player.position = RandomValues(values=np.array([new_position]))  # type: ignore


def correct_light_expert(
    current_state: GameState, action: Action, **context: Any
) -> None:
    """
    Correct expert that perfectly models the LightLaw.

    This expert models:
    - Each light has a toggle_probability chance of having its is_on attribute flipped
    - The light behavior is independent of the player's action

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken (unused in this expert)
        **context: Additional context (unused)
    """
    toggle_probability = 0.2  # Hardcoded to match the law's configuration

    # Iterate through each light
    for light in current_state.lights:
        # Check for toggle event using the state's RNG
        if current_state.rng.random() < toggle_probability:
            new_state = not light.is_on
        else:
            new_state = light.is_on

        # Assign the prediction using RandomValues
        light.is_on = RandomValues(values=np.array([new_state]))  # type: ignore


def incorrect_movement_expert_ignores_switch(
    current_state: GameState, action: Action, **context: Any
) -> None:
    """
    Incorrect expert that ignores the switched zone mechanic.

    This expert models player movement but completely ignores the switched-zone mechanic.
    It still correctly models boundaries and slipperiness.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # If STAY action, do not modify the player's state
    if action == Action.STAY:
        return

    # Determine the initial direction
    if action == Action.MOVE_LEFT:
        direction = -1
    elif action == Action.MOVE_RIGHT:
        direction = 1
    else:
        # Unknown action, don't modify state
        return

    # NOTE: This expert SKIPS the switched zone check (step 3 from correct expert)

    # Check for slip event using the state's RNG
    slip_probability = 0.1  # Hardcoded to match the law's configuration
    if current_state.rng.random() < slip_probability:
        direction *= -1

    # Calculate new position and clamp to boundaries
    new_position = current_state.player.position + direction
    new_position = max(0, min(new_position, current_state.config.width - 1))

    # Assign the prediction using RandomValues
    current_state.player.position = RandomValues(values=np.array([new_position]))  # type: ignore


def incorrect_movement_expert_ignores_slip(
    current_state: GameState, action: Action, **context: Any
) -> None:
    """
    Incorrect expert that ignores the slipperiness mechanic.

    This expert models player movement but assumes the world is never slippery.
    It still correctly models the switched zone and boundaries.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # If STAY action, do not modify the player's state
    if action == Action.STAY:
        return

    # Determine the initial direction
    if action == Action.MOVE_LEFT:
        direction = -1
    elif action == Action.MOVE_RIGHT:
        direction = 1
    else:
        # Unknown action, don't modify state
        return

    # Check if player is in the switched zone
    if current_state.player.position >= current_state.config.switch_point:
        direction *= -1

    # NOTE: This expert SKIPS the slipperiness check (step 4 from correct expert)

    # Calculate new position and clamp to boundaries
    new_position = current_state.player.position + direction
    new_position = max(0, min(new_position, current_state.config.width - 1))

    # Assign the prediction using RandomValues
    current_state.player.position = RandomValues(values=np.array([new_position]))  # type: ignore


def incorrect_light_expert_is_deterministic(
    current_state: GameState, action: Action, **context: Any
) -> None:
    """
    Incorrect expert that models light behavior as deterministic.

    This expert incorrectly models that lights always toggle their state,
    ignoring the stochastic nature of the true light behavior.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken (unused in this expert)
        **context: Additional context (unused)
    """
    # Iterate through each light
    for light in current_state.lights:
        # Always predict the light will toggle (incorrect deterministic behavior)
        new_state = not light.is_on

        # Assign the prediction using RandomValues
        light.is_on = RandomValues(values=np.array([new_state]))  # type: ignore


# Collection of all experts for easy access
CORRECT_EXPERTS = [
    correct_movement_expert,
    correct_light_expert,
]

INCORRECT_EXPERTS = [
    incorrect_movement_expert_ignores_switch,
    incorrect_movement_expert_ignores_slip,
    incorrect_light_expert_is_deterministic,
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
