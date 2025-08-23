"""
Component implementations for the hybrid evaluation framework.

This module provides concrete implementations of the injected component
protocols for different environments and use cases.
"""

import copy
import random
from typing import Any, TypeVar

import jsonpatch

from ..core import (
    SymbolicTransition,
    SymbolicEnvironment,
)
from ...poe_world.benchmark_1d.environment import (
    GameState,
    Action,
    WorldConfig,
    initial_state,
)

SymbolicStateT = TypeVar("SymbolicStateT")


class RandomPolicy1DTrajectoryCollector:
    """Random policy trajectory collector for 1D environment."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.STAY]

    def collect_transitions(
        self, environment: SymbolicEnvironment[GameState], num_transitions: int
    ) -> list[SymbolicTransition[GameState]]:
        """Collect transitions using random policy."""
        transitions = []
        state = initial_state(seed=self.rng.randint(0, 2**31 - 1))

        for _ in range(num_transitions):
            action = self.rng.choice(self.actions)
            next_state = environment.transition(state, action)
            transitions.append(SymbolicTransition(state, action, next_state))
            state = next_state

        return transitions


class JSONPatchEditDistance:
    """Edit distance calculator using JSON patch for serializable states."""

    def compute_distance(self, state1: GameState, state2: GameState) -> int:
        """Compute distance using JSON patch operations."""
        json1 = self._to_json(state1)
        json2 = self._to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))

    def _to_json(self, state: GameState) -> dict:
        """Convert state to JSON-serializable format."""
        return {
            "player_position": state.player.position,
            "lights": [(light.position, light.is_on) for light in state.lights],
            # Exclude non-serializable fields like RNG
        }


class StructuralEditDistance:
    """Structural edit distance for complex states."""

    def compute_distance(self, state1: Any, state2: Any) -> float:
        """Compute distance based on structural differences."""
        # Focus on semantically meaningful differences
        distance = 0

        # Player differences
        if state1.player.position != state2.player.position:
            distance += abs(state1.player.position - state2.player.position)

        # Light state differences
        for light1, light2 in zip(state1.lights, state2.lights):
            if light1.position != light2.position:
                distance += 1
            if light1.is_on != light2.is_on:
                distance += 1

        return distance


class TemporalDistractorGenerator:
    """Generate distractors from temporally distant transitions."""

    def __init__(self, gap: int = 50):
        self.gap = gap

    def generate_distractors(
        self,
        transition: SymbolicTransition[SymbolicStateT],
        all_transitions: list[SymbolicTransition[SymbolicStateT]],
        num_distractors: int,
    ) -> list[SymbolicStateT]:
        """Generate distractors from temporally distant states."""
        current_idx = all_transitions.index(transition)
        eligible_indices = [
            i for i in range(len(all_transitions)) if abs(i - current_idx) > self.gap
        ]

        if len(eligible_indices) < num_distractors:
            return [all_transitions[i].next_metadata for i in eligible_indices]

        selected_indices = random.sample(eligible_indices, num_distractors)
        return [all_transitions[i].next_metadata for i in selected_indices]


class Semantic1DDistractorGenerator:
    """Generate semantically plausible distractors for 1D environment."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.mutators = [
            self._mutate_player_position,
            self._mutate_light_states,
        ]

    def generate_distractors(
        self,
        transition: SymbolicTransition[GameState],
        all_transitions: list[SymbolicTransition[GameState]],
        num_distractors: int,
    ) -> list[GameState]:
        """Generate distractors using semantic mutations."""
        distractors = []
        for _ in range(num_distractors):
            mutator = random.choice(self.mutators)
            distractor = mutator(transition.next_metadata)
            distractors.append(distractor)
        return distractors

    def _mutate_player_position(self, state: GameState) -> GameState:
        """Mutate player position to create plausible distractors."""
        new_state = copy.deepcopy(state)
        new_state.player.position = random.choice(
            [
                state.player.position + 2,  # Jump too far
                -1,  # Out of bounds
                self.config.width,  # Out of bounds
            ]
        )
        return new_state

    def _mutate_light_states(self, state: GameState) -> GameState:
        """Mutate light states to create plausible distractors."""
        new_state = copy.deepcopy(state)
        for light in new_state.lights:
            if random.random() < 0.5:
                light.is_on = not light.is_on
        return new_state
