# PRD: Crafter Hybrid Evaluation Framework Implementation

## Context

Implementation of the hybrid evaluation framework for Crafter, a complex symbolic environment with rich state spaces, stochastic mechanics, and diverse gameplay systems. This builds upon the existing hybrid evaluation framework in `src/distant_sunburn/evaluator/` to provide comprehensive evaluation of world models in a significantly more complex environment than our 1D test case.

## Overview

Crafter presents unique challenges compared to the 1D environment:
- **Complex State Space**: Rich object hierarchies, inventory systems, player stats, achievements, and environmental state
- **Large Action Space**: 17 distinct actions covering movement, crafting, placement, and special actions
- **Stochastic Elements**: Random mob spawning, environmental changes, and probabilistic events
- **Temporal Dependencies**: Day/night cycles, object lifecycles, and state progression over time

Our approach will use **scenario-based trajectory collection** rather than purely random policies, enabling targeted testing of specific game mechanics while maintaining the framework's generality.

## Architecture

### Core Components (Reused from Existing Framework)
- `HybridEvaluator`: Core evaluation orchestrator (no changes needed)
- `EvaluationConfig`: Configuration management (minor extensions)
- `EvaluationResults`: Results aggregation (minor extensions)

### New Crafter-Specific Components
```
src/distant_sunburn/evaluator/
├── adapters.py              # Add CrafterAdapter
├── components.py            # Add Crafter-specific implementations
└── crafter_scenarios.py     # New: Scenario definitions and policies
```

## Implementation Strategy

### 1. Crafter Environment Adapter

```python
class CrafterAdapter:
    """Complete adapter for Crafter environment evaluation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
    
    def create_environment(self) -> SymbolicEnvironment[WorldState]:
        return CrafterEnvironmentWrapper(self.seed)
    
    def create_trajectory_collector(self) -> TrajectoryCollector[WorldState]:
        return CrafterScenarioTrajectoryCollector(self.rng)
    
    def create_edit_distance_calculator(self) -> EditDistanceCalculator[WorldState]:
        return CrafterJSONPatchEditDistance()
    
    def create_distractor_generator(self) -> DistractorGenerator[WorldState]:
        return CrafterSemanticDistractorGenerator(self.rng)
```

### 2. Environment Wrapper

```python
class CrafterEnvironmentWrapper:
    """Minimal Crafter environment wrapper using functional interface."""
    
    def __init__(self, seed: int):
        self.base_seed = seed  # Stored but not needed for transition
    
    def transition(self, state: WorldState, action: int) -> WorldState:
        """Apply action using functional transition function."""
        # The functional transition is already deterministic because
        # the WorldState includes serialized_random_state which is
        # restored during world reconstruction
        next_state, _ = transition(state, action)
        return next_state
```

### 3. Scenario-Based Trajectory Collection

#### Core Philosophy
Instead of purely random policies, we define **focused scenarios** that exercise specific game mechanics. Each scenario creates controlled initial conditions and executes a targeted policy.

```python
@dataclass
class CrafterScenario:
    """Defines an initial state and policy for targeted mechanic testing."""
    
    name: str
    create_initial_state: Callable[[], WorldState]
    policy: Callable[[WorldState], int]  # State -> Action
    max_steps: int
    description: str

class CrafterScenarioTrajectoryCollector:
    """Collect trajectories from predefined scenarios."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.scenarios = [
            self._movement_scenario(),
            self._crafting_scenario(),
            self._zombie_interaction_scenario(),
            self._resource_gathering_scenario(),
            self._placement_scenario(),
        ]
    
    def collect_transitions(
        self, environment: SymbolicEnvironment[WorldState], num_transitions: int
    ) -> list[SymbolicTransition[WorldState]]:
        """Collect transitions by running multiple scenarios."""
        transitions = []
        transitions_per_scenario = num_transitions // len(self.scenarios)
        
        for scenario in self.scenarios:
            scenario_transitions = self._run_scenario(
                environment, scenario, transitions_per_scenario
            )
            transitions.extend(scenario_transitions)
        
        return transitions[:num_transitions]  # Ensure exact count
```

#### Example Scenarios

**Movement Scenario**: Tests basic navigation and physics
```python
def _movement_scenario(self) -> CrafterScenario:
    def create_state() -> WorldState:
        # Create controlled environment: open area with known obstacles
        state = self._create_base_state()
        
        # Set specific player position
        state.player.position = Position(x=10, y=10)
        
        # Add known obstacles for collision testing
        state.materials[12][10] = "stone"  # Wall to the right
        state.materials[10][8] = "water"   # Water above
        
        return state
    
    def movement_policy(state: WorldState) -> int:
        # Systematic movement: right -> blocked, up -> water, left -> success, down -> success
        move_sequence = [
            constants.actions.index("move_right"),  # Should fail (stone)
            constants.actions.index("move_up"),     # Should fail (water)
            constants.actions.index("move_left"),   # Should succeed
            constants.actions.index("move_down"),   # Should succeed
        ]
        # Cycle through movements
        step_in_scenario = getattr(state, '_scenario_step', 0)
        return move_sequence[step_in_scenario % len(move_sequence)]
    
    return CrafterScenario(
        name="movement",
        create_initial_state=create_state,
        policy=movement_policy,
        max_steps=20,
        description="Tests movement mechanics and collision detection"
    )
```

**Crafting Scenario**: Tests crafting mechanics
```python
def _crafting_scenario(self) -> CrafterScenario:
    def create_state() -> WorldState:
        state = self._create_base_state()
        
        # Setup crafting environment
        state.materials[9][10] = "table"   # Table nearby
        state.materials[11][10] = "furnace"  # Furnace nearby
        
        # Give player crafting materials
        state.player.inventory.wood = 3
        state.player.inventory.stone = 2
        state.player.inventory.coal = 1
        state.player.inventory.iron = 1
        
        return state
    
    def crafting_policy(state: WorldState) -> int:
        # Systematic crafting progression
        if state.player.inventory.wood_pickaxe == 0:
            return constants.actions.index("make_wood_pickaxe")
        elif state.player.inventory.stone_pickaxe == 0:
            return constants.actions.index("make_stone_pickaxe")
        elif state.player.inventory.iron_pickaxe == 0:
            return constants.actions.index("make_iron_pickaxe")
        else:
            return constants.actions.index("noop")
    
    return CrafterScenario(
        name="crafting",
        create_initial_state=create_state,
        policy=crafting_policy,
        max_steps=10,
        description="Tests crafting mechanics and resource consumption"
    )

def _zombie_interaction_scenario(self) -> CrafterScenario:
    def create_state() -> WorldState:
        state = self._create_base_state()
        
        # Place player in center of open area
        state.player.position = Position(x=10, y=10)
        
        # Add zombie nearby for interaction
        # Note: ZombieState should be imported from crafter.state_export
        zombie_state = ZombieState(
            entity_id=100,  # Use high ID to avoid conflicts
            position=Position(x=12, y=10),  # 2 tiles to the right
            health=6,  # Default zombie health
            cooldown=0,  # Ready to act
            removed=False,
        )
        state.objects.append(zombie_state)
        
        # Give player a weapon for combat testing
        state.player.inventory.wood_sword = 1
        
        # Ensure surrounding area is walkable grass
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= 10+dx < state.size[0] and 0 <= 10+dy < state.size[1]:
                    state.materials[10+dx][10+dy] = "grass"
        
        return state
    
    def zombie_interaction_policy(state: WorldState) -> int:
        # Simple policy: face zombie and attack, or move toward it
        player_pos = state.player.position
        
        # Find the zombie
        zombie_pos = None
        for obj in state.objects:
            if hasattr(obj, 'name') and obj.name == "zombie":
                zombie_pos = obj.position
                break
        
        if zombie_pos is None:
            return constants.actions.index("noop")
        
        # Calculate relative position
        dx = zombie_pos.x - player_pos.x
        dy = zombie_pos.y - player_pos.y
        
        # If zombie is adjacent, attack
        if abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0):
            # Face the zombie first, then attack
            if dx > 0:
                state.player.facing = Position(x=1, y=0)
            elif dx < 0:
                state.player.facing = Position(x=-1, y=0)
            elif dy > 0:
                state.player.facing = Position(x=0, y=1)
            elif dy < 0:
                state.player.facing = Position(x=0, y=-1)
            return constants.actions.index("do")  # Attack action
        
        # Otherwise, move toward zombie
        if abs(dx) > abs(dy):
            return constants.actions.index("move_right" if dx > 0 else "move_left")
        else:
            return constants.actions.index("move_down" if dy > 0 else "move_up")
    
    return CrafterScenario(
        name="zombie_interaction",
        create_initial_state=create_state,
        policy=zombie_interaction_policy,
        max_steps=15,
        description="Tests mob interaction, combat mechanics, and mob AI behavior"
    )
```

### 4. JSON Patch Edit Distance for Crafter

```python
class CrafterJSONPatchEditDistance:
    """JSON patch edit distance specifically tuned for Crafter states."""
    
    def compute_distance(self, state1: WorldState, state2: WorldState) -> int:
        """Compute edit distance excluding non-deterministic fields."""
        json1 = self._to_comparable_json(state1)
        json2 = self._to_comparable_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))
    
    def _to_comparable_json(self, state: WorldState) -> dict:
        """Convert state to JSON, excluding non-deterministic fields."""
        # Use model_dump but exclude randomness-related fields
        state_dict = state.model_dump(exclude={
            "serialized_random_state",  # Random state is non-deterministic
            "event_bus",                 # Event bus can vary
        })
        
        # Sort objects by entity_id for consistent comparison
        if "objects" in state_dict:
            state_dict["objects"] = sorted(
                state_dict["objects"], 
                key=lambda obj: obj.get("entity_id", 0)
            )
        
        # Sort chunks for consistency
        if "chunks" in state_dict:
            state_dict["chunks"] = sorted(
                state_dict["chunks"],
                key=lambda chunk: chunk["chunk_key"]
            )
            for chunk in state_dict["chunks"]:
                chunk["objects"] = sorted(chunk["objects"])
        
        return state_dict
```

### 5. Semantic Distractor Generation

```python
class CrafterSemanticDistractorGenerator:
    """Generate semantically plausible but incorrect distractors for Crafter."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.mutators = [
            self._mutate_player_position,
            self._mutate_player_inventory,
            self._mutate_player_stats,
            self._mutate_object_positions,
            self._mutate_object_states,
            self._mutate_world_materials,
            self._violate_game_rules,
        ]
    
    def generate_distractors(
        self,
        transition: SymbolicTransition[WorldState],
        all_transitions: list[SymbolicTransition[WorldState]],
        num_distractors: int,
    ) -> list[WorldState]:
        """Generate distractors using semantic mutations."""
        distractors = []
        
        # Mix of temporal distractors (sanity check) and semantic mutations
        num_temporal = min(2, num_distractors // 2)
        num_semantic = num_distractors - num_temporal
        
        # Temporal distractors (basic sanity check)
        temporal_distractors = self._generate_temporal_distractors(
            transition, all_transitions, num_temporal
        )
        distractors.extend(temporal_distractors)
        
        # Semantic mutations (fine-grained testing)
        for _ in range(num_semantic):
            mutator = self.rng.choice(self.mutators)
            distractor = mutator(transition.next_metadata)
            distractors.append(distractor)
        
        return distractors
    
    def _mutate_player_inventory(self, state: WorldState) -> WorldState:
        """Create invalid inventory states."""
        new_state = copy.deepcopy(state)
        
        mutations = [
            # Add items that couldn't be obtained
            lambda: setattr(new_state.player.inventory, "diamond", 5),
            # Remove items that should be present
            lambda: setattr(new_state.player.inventory, "wood", 0),
            # Exceed maximum quantities
            lambda: setattr(new_state.player.inventory, "stone", 15),
            # Negative quantities
            lambda: setattr(new_state.player.inventory, "health", -1),
        ]
        
        self.rng.choice(mutations)()
        return new_state
    
    def _mutate_player_position(self, state: WorldState) -> WorldState:
        """Create invalid player positions."""
        new_state = copy.deepcopy(state)
        
        mutations = [
            # Out of bounds positions
            Position(x=-1, y=10),
            Position(x=state.size[0], y=10),
            # Inside solid objects
            Position(x=5, y=5),  # Assuming there's a wall here
            # Impossible teleportation
            Position(x=50, y=50),
        ]
        
        new_state.player.position = self.rng.choice(mutations)
        return new_state
    
    def _violate_game_rules(self, state: WorldState) -> WorldState:
        """Create states that violate fundamental game rules."""
        new_state = copy.deepcopy(state)
        
        violations = [
            # Player in impossible state
            lambda: setattr(new_state.player, "sleeping", True) and 
                    setattr(new_state.player.position, "x", new_state.player.position.x + 1),
            # Objects in impossible positions
            lambda: self._place_object_in_solid_tile(new_state),
            # Impossible material combinations
            lambda: self._create_impossible_material_state(new_state),
        ]
        
        self.rng.choice(violations)()
        return new_state
```

### 6. Baseline World Models for Sanity Testing

```python
class TrueCrafterWorldModel:
    """Perfect world model using actual transition function."""
    
    def __init__(self, environment: SymbolicEnvironment[WorldState]):
        self.environment = environment
    
    def sample_next_state(self, current_state: WorldState, action: int) -> WorldState:
        return self.environment.transition(current_state, action)
    
    def evaluate_log_probability(
        self, next_state: WorldState, current_state: WorldState, action: int
    ) -> float:
        true_next = self.environment.transition(current_state, action)
        # Use JSON comparison for state equality
        return 0.0 if self._states_equal(next_state, true_next) else -math.inf
    
    def _states_equal(self, state1: WorldState, state2: WorldState) -> bool:
        # Compare using same logic as edit distance calculator
        calc = CrafterJSONPatchEditDistance()
        return calc.compute_distance(state1, state2) == 0

class NullCrafterWorldModel:
    """Baseline model that predicts no state changes."""
    
    def sample_next_state(self, current_state: WorldState, action: int) -> WorldState:
        # Return copy with only step count incremented
        new_state = copy.deepcopy(current_state)
        new_state.step_count += 1
        return new_state
    
    def evaluate_log_probability(
        self, next_state: WorldState, current_state: WorldState, action: int
    ) -> float:
        # High probability for states that only differ by step count
        expected_null = self.sample_next_state(current_state, action)
        calc = CrafterJSONPatchEditDistance()
        distance = calc.compute_distance(next_state, expected_null)
        
        if distance == 0:
            return 0.0  # Perfect match
        elif distance <= 3:
            return -2.0  # Small changes somewhat likely
        else:
            return -8.0  # Large changes very unlikely

class RandomCrafterWorldModel:
    """Model that generates random but structurally valid states."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
    
    def sample_next_state(self, current_state: WorldState, action: int) -> WorldState:
        new_state = copy.deepcopy(current_state)
        new_state.step_count += 1
        
        # Apply random but valid mutations
        if self.rng.random() < 0.3:  # 30% chance of position change
            new_state.player.position = Position(
                x=max(0, min(current_state.size[0]-1, 
                           current_state.player.position.x + self.rng.randint(-1, 1))),
                y=max(0, min(current_state.size[1]-1, 
                           current_state.player.position.y + self.rng.randint(-1, 1)))
            )
        
        if self.rng.random() < 0.2:  # 20% chance of inventory change
            item = self.rng.choice(list(constants.items.keys()))
            current_amount = getattr(new_state.player.inventory, item)
            max_amount = constants.items[item].max
            new_amount = max(0, min(max_amount, current_amount + self.rng.randint(-1, 1)))
            setattr(new_state.player.inventory, item, new_amount)
        
        return new_state
    
    def evaluate_log_probability(
        self, next_state: WorldState, current_state: WorldState, action: int
    ) -> float:
        # Assign uniform low probability to all states
        return -5.0
```

## Usage Example

```python
def create_crafter_evaluator(seed: int = 42) -> HybridEvaluator[WorldState]:
    """Create a complete Crafter evaluator."""
    adapter = CrafterAdapter(seed=seed)
    
    return HybridEvaluator(
        config=EvaluationConfig(
            num_transitions=100,
            num_distractors=5,
            random_seed=seed
        ),
        trajectory_collector=adapter.create_trajectory_collector(),
        edit_distance_calc=adapter.create_edit_distance_calculator(),
        distractor_generator=adapter.create_distractor_generator(),
    )

def run_crafter_sanity_test():
    """Run sanity test comparing true vs baseline models."""
    adapter = CrafterAdapter(seed=42)
    environment = adapter.create_environment()
    evaluator = create_crafter_evaluator(seed=42)
    
    # Create models
    true_model = TrueCrafterWorldModel(environment)
    null_model = NullCrafterWorldModel()
    random_model = RandomCrafterWorldModel(random.Random(42))
    
    # Evaluate all models
    true_results = evaluator.evaluate(true_model, environment)
    null_results = evaluator.evaluate(null_model, environment)
    random_results = evaluator.evaluate(random_model, environment)
    
    # Sanity checks
    assert true_results.mean_generative_error < null_results.mean_generative_error
    assert true_results.mean_generative_error < random_results.mean_generative_error
    assert true_results.discriminative_accuracy > 0.95
    assert null_results.discriminative_accuracy < 0.3
    assert random_results.discriminative_accuracy < 0.2
    
    print(f"True model: gen_error={true_results.mean_generative_error:.2f}, "
          f"disc_acc={true_results.discriminative_accuracy:.2f}")
    print(f"Null model: gen_error={null_results.mean_generative_error:.2f}, "
          f"disc_acc={null_results.discriminative_accuracy:.2f}")
    print(f"Random model: gen_error={random_results.mean_generative_error:.2f}, "
          f"disc_acc={random_results.discriminative_accuracy:.2f}")
```

## Implementation Considerations

### 1. State Determinism and RNG Handling
- **Critical**: Exclude `serialized_random_state` and `event_bus` from JSON comparisons
- **Deterministic Evaluation**: The functional transition function is inherently deterministic because `WorldState.serialized_random_state` contains the RNG state, which is restored during world reconstruction
- **State Copying**: Use `copy.deepcopy()` for all state mutations to prevent side effects
- **No Additional Seeding**: Unlike the 1D environment, Crafter's functional interface handles randomness through state serialization, eliminating the need for manual RNG management

### 2. Scenario Design Principles
- **Focused Testing**: Each scenario targets specific mechanics (movement, crafting, combat)
- **Controlled Conditions**: Known initial states enable predictable behavior testing
- **Coverage**: Scenarios collectively exercise diverse game systems
- **Extensibility**: Easy to add new scenarios for additional mechanics

### 3. Edit Distance Tuning
- **JSON Patch Approach**: Leverage existing Pydantic serialization with exclusions
- **Consistency**: Sort collections (objects, chunks) by ID for stable comparisons
- **Relevance**: Focus on gameplay-relevant state differences, ignore implementation details

### 4. Distractor Quality
- **Temporal Sanity**: Include distractors from distant time steps as basic sanity check
- **Semantic Plausibility**: Generate states that violate game rules in subtle, realistic ways
- **Difficulty Gradient**: Mix obvious violations (teleportation) with subtle ones (inventory inconsistencies)

## Success Criteria

### Sanity Test Requirements
1. **True Model Dominance**: True transition model achieves >95% discriminative accuracy
2. **Baseline Separation**: True model outperforms null/random models by large margins
3. **Consistency**: Results stable across multiple random seeds
4. **Coverage**: Scenarios exercise all major game mechanics

### Performance Metrics
- **Evaluation Speed**: <30 seconds for 100 transitions on standard hardware
- **Memory Usage**: <2GB peak memory during evaluation
- **Determinism**: Identical results for identical seeds

## Future Extensions

### Additional Scenarios
- **Mob Interaction**: Combat with zombies and skeletons
- **Day/Night Cycles**: Time-dependent behavior testing
- **Resource Chains**: Complex crafting dependency testing
- **Survival Mechanics**: Hunger, thirst, and fatigue testing

### Advanced Distractors
- **Causal Violations**: States that violate action-consequence relationships
- **Temporal Inconsistencies**: States that couldn't result from given action sequences
- **Physics Violations**: States that break spatial or material constraints

### Evaluation Extensions
- **Mechanic-Specific Metrics**: Separate metrics for different game systems
- **Temporal Analysis**: Evaluation of state progression over longer horizons
- **Stochastic Robustness**: Testing with varying levels of environmental randomness

---

This PRD provides a comprehensive framework for evaluating world models in Crafter while maintaining the flexibility and rigor of the original hybrid evaluation approach. The scenario-based trajectory collection addresses the complexity challenge while ensuring thorough coverage of the game's diverse mechanics.
