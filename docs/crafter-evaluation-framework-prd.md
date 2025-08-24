# PRD: Hybrid Evaluation Framework for Crafter (Revised Architecture)

## 1. Overview

This document outlines the engineering requirements for extending our hybrid evaluation framework to the Crafter environment. This revised version incorporates a cleaner, more functional architecture based on critical feedback, ensuring better code isolation and clarity for the implementer.

The primary goal is to create a robust, configurable, and maintainable testing suite for Crafter world models. This involves two key workstreams:
1.  **Trajectory Collection**: Implementing flexible strategies for collecting interesting and diverse state transitions from the environment.
2.  **Distractor Generation**: Creating a structured and categorized set of state mutations to test a world model's fine-grained understanding of game mechanics.

**Reference Document:** [Hybrid Evaluation Framework for Symbolic WMs](docs/1-hybrid-evaluation-framework-for-symbolic-wms.md)

## 2. Core Architecture

We will continue to use the hexagonal architecture defined in `src/distant_sunburn/evaluator/core.py`. The main task is to implement concrete versions of the protocols for the Crafter environment (`crafter.state_export.WorldState`).

## 3. Workstream 1: Trajectory Collection

To ensure robustness and testability, we will adopt a more functional approach. Instead of passing a mutable `Env` object between components, scenarios will be responsible for describing a desired `WorldState`, which the collection strategy will then execute.

### 3.1. State Manipulation Helpers

To facilitate the creation of specific world states for scenarios, we use a functional approach. The core pattern is to generate a blank `WorldState`, reconstruct the `World` object from it, programmatically modify the world to achieve the desired starting conditions, and then export a new `WorldState`.

Several helper functions support this process.

```python
# In: src/distant_sunburn/evaluator/crafter/utils.py

from crafter.engine import World
import crafter.objects as crafter_objects

def find_player(world: World) -> crafter_objects.Player:
    """Finds the player in the world."""
    for obj in world.objects:
        if isinstance(obj, crafter_objects.Player):
            return obj
    raise ValueError("No player found in world")
```

### 3.2. Architecture: Strategy-Based Collection

The `CrafterTrajectoryCollector` will instantiate and run a given collection strategy.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py

from ..core import SymbolicTransition, TrajectoryCollector
from crafter.state_export import WorldState
from typing import Protocol

class CollectionStrategy(Protocol):
    """A protocol for a single method of collecting transitions in Crafter."""
    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        ...

class CrafterTrajectoryCollector(TrajectoryCollector[WorldState]):
    def __init__(self, strategy: CollectionStrategy):
        self.strategy = strategy

    def collect_transitions(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        return self.strategy.collect(num_transitions)
```

### 3.3. Strategy 1: Random Movement Policy

This strategy creates its own internal `Env` to generate states.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py
from crafter.env import Env
from crafter.functional_env import EnvConfig
from .utils import get_world_state 
import random

class RandomMovementStrategy(CollectionStrategy):
    def __init__(self, env_config: EnvConfig, seed: int):
        self.env_config = env_config
        self.rng = random.Random(seed)
        self.movement_actions = ["move_left", "move_right", "move_up", "move_down"]

    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        env = Env(self.env_config)
        env.reset()
        
        transitions = []
        state = get_world_state(env)

        for _ in range(num_transitions):
            action = self.rng.choice(self.movement_actions)
            prev_state = state
            
            env.step(action)
            state = get_world_state(env)
            
            transitions.append(SymbolicTransition(prev_state, action, state))
        
        return transitions
```

### 3.4. Strategy 2: Scenario-Based Collection

This strategy uses `Scenario` objects that are stateless factories for `WorldState` objects.

**`Scenario` Protocol:**
```python
# In: src/distant_sunburn/evaluator/crafter/scenarios.py
from crafter.state_export import WorldState
from crafter.constants import ActionT
from typing import Protocol

class Scenario(Protocol):
    @property
    def name(self) -> str: ...

    def get_initial_state(self) -> WorldState:
        """Creates and returns the specific starting WorldState for this scenario."""
        ...

    def get_actions(self) -> list[ActionT]: ...
```

**`ScenarioBasedStrategy`:** This strategy uses the pure `crafter.functional_env.transition` function to step through the scenario. It correctly converts action strings to indices before processing.

```python
# In: src/distant_sunburn/evaluator/crafter/components.py
from crafter.functional_env import transition
from crafter import constants
from .scenarios import Scenario

class ScenarioBasedStrategy(CollectionStrategy):
    def __init__(self, scenarios: list[Scenario]):
        self.scenarios = scenarios

    def collect(self, num_transitions: int) -> list[SymbolicTransition[WorldState]]:
        transitions = []
        for scenario in self.scenarios:
            initial_state = scenario.get_initial_state()
            actions = scenario.get_actions()

            state = initial_state
            for action in actions:
                prev_state = state
                action_index = constants.actions.index(action)
                state, _ = transition(prev_state, action_index)
                transitions.append(SymbolicTransition(prev_state, action, state))
        
        return transitions
```

## 4. Workstream 2: Distractor Generation (Unchanged)
The architecture for distractor generation remains the same as it is already functional and decoupled.

## 5. Factory and Final Assembly

The factory will be updated to reflect the new stateless strategy architecture.

```python
# In: src/distant_sunburn/evaluator/crafter/factory.py

from .components import (
    CrafterTrajectoryCollector, 
    ScenarioBasedStrategy,
    CrafterDistractorGenerator,
    JSONPatchEditDistance
)
from .scenarios import CraftWoodenPickaxeScenario, CowMovementScenario
from ..core import EvaluationContext, EvaluationConfig
from crafter.functional_env import EnvConfig
from crafter.state_export import WorldState


class CrafterEvaluationFactory:
    def __init__(self, env_config: EnvConfig, policy_seed: int = 42):
        self.env_config = env_config
        self.policy_seed = policy_seed

    def create_context(
        self, config: EvaluationConfig, num_transitions: int
    ) -> EvaluationContext[WorldState]:

        scenarios = [CraftWoodenPickaxeScenario(), CowMovementScenario()]
        strategy = ScenarioBasedStrategy(scenarios)
        
        collector = CrafterTrajectoryCollector(strategy)
        test_transitions = collector.collect_transitions(num_transitions)

        distractor_generator = CrafterDistractorGenerator(seed=self.policy_seed)

        return EvaluationContext(
            config=config,
            test_transitions=test_transitions,
            distractor_generator=distractor_generator,
            edit_distance_calculator=JSONPatchEditDistance(),
        )
```

## 6. Implementation Plan (Revised)

1.  **Setup:**
    *   Create `src/distant_sunburn/evaluator/crafter/utils.py` and implement helper functions like `find_player`.
    *   Create `src/distant_sunburn/evaluator/crafter/scenarios.py` and `.../mutators.py`.
    *   Ensure test helpers from `crafter.testing_helpers` are accessible.
2.  **Implement Trajectory Collection:**
    *   Implement `CollectionStrategy` protocol and `CrafterTrajectoryCollector` in `components.py`.
    *   Implement the `RandomMovementStrategy`.
    *   Define the `Scenario` protocol in `scenarios.py`.
    *   Implement scenarios (e.g., `CraftWoodenPickaxeScenario`) using the functional, stateless approach.
    *   Implement the `ScenarioBasedStrategy`, ensuring action strings are converted to indices.
3.  **Implement Distractor Generation:** (No change in plan)
4.  **Update Factory:** Modify `CrafterEvaluationFactory` to match the new design.
5.  **Testing:** Write unit tests for scenarios verifying their outcomes and for strategies verifying correct transition collection.

## 7. Appendix: Detailed Scenario Implementation Guide (Revised)

This guide demonstrates creating a scenario with the cleaner, functional architecture.

### Step 1 & 2: Define Goal and Pre-conditions (Unchanged)
- **Goal:** Test crafting a wooden pickaxe.
- **Pre-conditions:** Player needs 2 wood, must be next to a table.

### Step 3: Code the Scenario

The scenario encapsulates its own setup logic, creating the desired starting state from scratch without relying on a mutable `Env` instance. This improves testability and isolation.

```python
# In: src/distant_sunburn/evaluator/crafter/scenarios.py

from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
    initial_state,
)
from crafter.state_export import WorldState
from crafter.constants import ActionT
from .utils import find_player
from crafter.testing_helpers import (
    player_utils,
    world_utils,
)

class CraftWoodenPickaxeScenario(Scenario):
    @property
    def name(self) -> str:
        return "craft_wooden_pickaxe"

    def get_initial_state(self) -> WorldState:
        """
        Creates a world state from scratch with the desired 
        starting conditions.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))
        player_utils.set_player_facing(player, (0, 1))
        world_utils.set_tile_material(world, (5, 6), "table")
        player_utils.set_player_inventory_item(player, "wood", 2)
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)

        return export_world_state(world, view=view, step_count=0)

    def get_actions(self) -> list[ActionT]:
        return ["make_wood_pickaxe"]
```

### Step 4: Integrate the Scenario (Unchanged)

The integration in the factory remains the same conceptually: instantiate the scenario and add it to the strategy's list.

## 8. Changelog
- **2025-08-24:** Updated PRD to reflect the implemented functional approach for scenario creation. `Scenario.get_initial_state` is now stateless, and the implementation guide uses `initial_state` and `reconstruct_world_from_state` instead of a temporary `Env` object. Trajectory collection strategies updated to convert action strings to indices before calling `transition`.
