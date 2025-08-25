"""
Component implementations for the hybrid evaluation framework.

This module provides concrete implementations of the injected component
protocols for different environments and use cases.
"""

import copy
import random
from typing import TypeVar
from distant_sunburn.typing_utils import implements

import jsonpatch

from ..core import (
    SymbolicTransition,
    SymbolicTransitionFunction,
    DistractorGenerator,
    EditDistanceCalculator,
)
from ...simple_1d_env.environment import (
    GameState,
    Action,
    WorldConfig,
)

SymbolicStateT = TypeVar("SymbolicStateT")


class RandomPolicy1DTrajectoryCollector:
    """Random policy trajectory collector for 1D environment."""

    def __init__(self, rng: random.Random, initial_state: GameState):
        self.rng = rng
        self.initial_state = initial_state
        self.actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.STAY]

    def collect_transitions(
        self,
        transition_function: SymbolicTransitionFunction[GameState, Action],
        num_transitions: int,
    ) -> list[SymbolicTransition[GameState, Action]]:
        """Collect transitions using random policy."""
        transitions = []
        state = self.initial_state

        for _ in range(num_transitions):
            action = self.rng.choice(self.actions)
            next_state = transition_function(state, action)
            transitions.append(SymbolicTransition(state, action, next_state))
            state = next_state

        return transitions


class JSONPatchEditDistance:
    @staticmethod
    def _gamestate_to_json(state: GameState) -> dict:
        """Convert a GameState object to JSON.

        Normally, this would be handled by a serialization library such as
        Pydantic or cattrs, but the game state is simple enough that we can do it manually here.
        """
        # Note: we convert to int and bool here to avoid issues with JSON serialization
        # of NumPy dtypes like int64.
        return {
            "player": {"position": int(state.player.position)},
            "lights": [
                {"position": int(light.position), "is_on": bool(light.is_on)}
                for light in state.lights
            ],
            # Exclude the RNG state, which is not easy to serialize.
        }

    def __call__(self, state1: GameState, state2: GameState) -> int:
        """Compute the edit distance between two GameState objects using JSON patch."""
        json1 = self._gamestate_to_json(state1)
        json2 = self._gamestate_to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))


implements(EditDistanceCalculator[GameState])(JSONPatchEditDistance)


class Semantic1DDistractorGenerator:
    """Generate semantically plausible distractors for 1D environment."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.mutators = [
            self._mutate_player_position,
            self._mutate_light_states,
        ]

    def __call__(
        self,
        transition: SymbolicTransition[GameState, Action],
        all_transitions: list[SymbolicTransition[GameState, Action]],
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


implements(DistractorGenerator[GameState, Action])(Semantic1DDistractorGenerator)
