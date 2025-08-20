"""
Simple 1D Test Environment for PoE-World

This module implements a configurable, reproducible 1D grid world environment
for testing the PoE-World expert synthesis and weight-fitting pipeline.
"""

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Union
from ..core import RandomValues

from loguru import logger


class Action(Enum):
    """Available actions in the 1D environment."""

    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    STAY = auto()


@dataclass
class Player:
    """Represents the player in the 1D world."""

    position: int


@dataclass
class Light:
    """Represents a light object in the 1D world."""

    position: int
    is_on: bool


@dataclass(frozen=True)
class WorldConfig:
    """Immutable configuration for the world."""

    width: int
    switch_point: int  # The first coordinate in the "switched" half


@dataclass
class GameState:
    """Complete state of the 1D game world."""

    config: WorldConfig
    player: Player
    lights: list[Light]
    rng: random.Random


class Law(Protocol):
    """Protocol for world laws that can be applied to game states."""

    def apply(self, state: GameState, action: Action) -> None:
        """Apply this law to the given state and action."""
        ...


class MovementLaw:
    """Law governing player movement with switched zones and slipperiness."""

    def __init__(self, slip_probability: float = 0.1):
        """Initialize the movement law.

        Args:
            slip_probability: Probability of movement direction being inverted
        """
        self.slip_probability = slip_probability
        logger.debug(
            f"Initialized MovementLaw with slip_probability={slip_probability}"
        )

    def apply(self, state: GameState, action: Action) -> None:
        """Apply movement law to the current state.

        Args:
            state: Current game state
            action: Action to apply
        """
        if action == Action.STAY:
            return

        # Determine base direction
        if action == Action.MOVE_LEFT:
            direction = -1
        elif action == Action.MOVE_RIGHT:
            direction = 1
        else:
            logger.warning(f"Unknown action: {action}")
            return

        # Check if in switched zone
        if state.player.position >= state.config.switch_point:
            direction *= -1
            logger.debug(f"Player in switched zone, direction inverted to {direction}")

        # Check for slipperiness
        if state.rng.random() < self.slip_probability:
            direction *= -1
            logger.debug(f"Slippery! Direction inverted to {direction}")

        # Calculate new position
        new_position = state.player.position + direction

        # Apply boundary constraints
        new_position = max(0, min(new_position, state.config.width - 1))

        # Update player position
        old_position = state.player.position
        state.player.position = new_position

        logger.debug(f"Player moved from {old_position} to {new_position}")


class LightLaw:
    """Law governing light state changes."""

    def __init__(self, toggle_probability: float = 0.2):
        """Initialize the light law.

        Args:
            toggle_probability: Probability of a light toggling its state
        """
        self.toggle_probability = toggle_probability
        logger.debug(
            f"Initialized LightLaw with toggle_probability={toggle_probability}"
        )

    def apply(self, state: GameState, action: Action) -> None:
        """Apply light law to the current state.

        Args:
            state: Current game state
            action: Action to apply (not used in this law)
        """
        for light in state.lights:
            if state.rng.random() < self.toggle_probability:
                light.is_on = not light.is_on
                logger.debug(
                    f"Light at position {light.position} toggled to {light.is_on}"
                )


def initial_state(width: int = 12, num_lights: int = 2, seed: int = 42) -> GameState:
    """Create the initial state for the 1D environment.

    Args:
        width: Width of the 1D world
        num_lights: Number of lights to place in the world
        seed: Random seed for reproducibility

    Returns:
        Initial game state
    """
    # Create configuration
    switch_point = width // 2
    config = WorldConfig(width=width, switch_point=switch_point)

    # Create player at fixed starting position
    player = Player(position=width // 4)

    # Create lights at fixed positions
    lights = []
    for i in range(num_lights):
        # Place lights in different halves of the world
        if i == 0:
            light_pos = width // 4  # First quarter
        else:
            light_pos = 3 * width // 4  # Third quarter
        lights.append(Light(position=light_pos, is_on=False))

    # Initialize random number generator
    rng = random.Random(seed)

    state = GameState(config=config, player=player, lights=lights, rng=rng)

    logger.info(
        f"Created initial state: width={width}, player_pos={player.position}, "
        f"num_lights={num_lights}, seed={seed}"
    )

    return state


def transition_function(state: GameState, action: Action, laws: list[Law]) -> GameState:
    """Apply the transition function to move from current state to next state.

    Args:
        state: Current game state
        action: Action to apply
        laws: List of laws to apply in order

    Returns:
        New game state after applying the action and laws
    """
    # Create deep copy to avoid modifying original state
    new_state = copy.deepcopy(state)

    # Apply each law in sequence
    for law in laws:
        law.apply(new_state, action)

    logger.debug(f"Applied action {action} with {len(laws)} laws")

    return new_state


DEFAULT_MOVEMENT_LAW = MovementLaw()
DEFAULT_LIGHT_LAW = LightLaw()
DEFAULT_LAWS = [DEFAULT_MOVEMENT_LAW, DEFAULT_LIGHT_LAW]
