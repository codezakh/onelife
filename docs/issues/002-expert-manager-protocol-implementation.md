# Issue #002: Implement ExpertManagerProtocol by wrapping existing weight_fitter and world_model components

## Overview

We need to implement the `ExpertManagerProtocol` defined in `src/distant_sunburn/poe_world/object_model_learner.py` by creating a wrapper class that coordinates our existing `MaxLikelihoodWeightFitter` and `PoEWorldModel` components. This will enable our `ObjectModelOrchestrator` to work with concrete implementations rather than just protocols.

## Background

### Architecture Context

Our PoE-World implementation follows the external `poe-world` architecture where:
- `ObjectModelOrchestrator` (our generic class) corresponds to `ObjModelLearner` in external poe-world
- `ExpertManagerProtocol` corresponds to `MoEObjModel` in external poe-world
- Each object type has separate managers for creation and non-creation experts

### Current Components

We have two well-tested components that can be wrapped:

1. **`MaxLikelihoodWeightFitter`** (`src/distant_sunburn/poe_world/weight_fitter.py`):
   - Handles weight learning via PyTorch optimization
   - Takes expert functions and transitions, returns weighted experts
   - Uses L-BFGS optimizer with L1 regularization

2. **`PoEWorldModel`** (`src/distant_sunburn/poe_world/world_model.py`):
   - Manages a collection of weighted experts
   - Provides `evaluate_log_probability()` for surprise detection
   - Handles expert prediction combination via Product of Experts
   - Has `with_new_experts()` for immutable updates

### Protocol Requirements

The `ExpertManagerProtocol` requires these methods:

```python
def add_experts(self, experts: List[WeightedExpert]) -> None
def fit_weights(self, transitions: List[SymbolicTransition], fast_mode: bool = False) -> None
def prune_experts(self) -> None
def evaluate_log_probability(self, state, action, next_state) -> float
def get_experts(self) -> List[WeightedExpert]
def save(self, checkpoint_path: str) -> None
def load(self, checkpoint_path: str) -> bool
```

## Implementation Plan

### 1. Create ExpertManager Wrapper Class

**File**: `src/distant_sunburn/poe_world/expert_manager.py`

Create a class that wraps and coordinates the existing components:

```python
class ExpertManager(Generic[SymbolicStateT, ActionT]):
    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        weight_fitter: MaxLikelihoodWeightFitter[SymbolicStateT],
    ):
        self.observable_extractor = observable_extractor
        self.weight_fitter = weight_fitter
        self.world_model = PoEWorldModel(observable_extractor, [])
```

### 2. Implement Required Methods

#### Easy Mappings (Direct delegation):
- `add_experts()` → `PoEWorldModel.with_new_experts()`
- `evaluate_log_probability()` → `PoEWorldModel.evaluate_log_probability()`
- `get_experts()` → `PoEWorldModel.experts` property

#### Coordination Required:
- `fit_weights()` → Coordinate between `MaxLikelihoodWeightFitter.fit()` and `PoEWorldModel` updates
- `prune_experts()` → **NEW**: Remove experts with weights below threshold
- `save()`/`load()` → **NEW**: Checkpoint expert state and weights

### 3. Handle Missing Functionality

#### Pruning Logic
Implement `prune_experts()` to remove experts with low/zero weights:

```python
def prune_experts(self, weight_threshold: float = 0.01) -> None:
    """Remove experts with weights below threshold."""
    remaining_experts = [
        expert for expert in self.world_model.experts 
        if expert.weight >= weight_threshold
    ]
    self.world_model = PoEWorldModel(
        self.observable_extractor, remaining_experts
    )
```

#### Checkpointing
Implement save/load for expert state:

```python
def save(self, checkpoint_path: str) -> None:
    """Save expert state to checkpoint."""
    checkpoint_data = {
        'experts': [
            {
                'expert_function': expert.expert_function,
                'weight': expert.weight
            }
            for expert in self.world_model.experts
        ]
    }
    # Save using pickle or similar

def load(self, checkpoint_path: str) -> bool:
    """Load expert state from checkpoint."""
    # Load and reconstruct world_model
```

#### Fast Mode Support
Modify `fit_weights()` to support incremental updates by fitting only new experts:

```python
def fit_weights(self, transitions: List[SymbolicTransition], fast_mode: bool = False) -> None:
    if fast_mode:
        # Fast mode: Only fit weights for newly added experts
        # This approach avoids complex masking by passing only new experts to the fitter
        
        # Identify new experts (those added since last fit)
        new_experts = [expert for expert in self.world_model.experts if not expert.is_fitted]
        
        if new_experts:
            # Fit only new experts using existing weight fitter
            new_expert_functions = [expert.expert_function for expert in new_experts]
            new_weighted_experts = self.weight_fitter.fit(new_expert_functions, transitions)
            
            # Update weights for new experts while preserving existing weights
            self._update_weights_for_new_experts(new_weighted_experts)
            
            # Mark new experts as fitted
            for expert in new_experts:
                expert.is_fitted = True
    else:
        # Full mode: Fit all experts (current behavior)
        all_expert_functions = [expert.expert_function for expert in self.world_model.experts]
        all_weighted_experts = self.weight_fitter.fit(all_expert_functions, transitions)
        self.world_model = PoEWorldModel(self.observable_extractor, all_weighted_experts)
        
        # Mark all experts as fitted
        for expert in self.world_model.experts:
            expert.is_fitted = True
```

**Implementation Details:**
- Track `is_fitted` flag on each expert to identify new vs. existing experts
- Use `_update_weights_for_new_experts()` helper to merge new weights with existing ones
- Preserve existing expert weights during fast mode updates
- This approach mirrors the external poe-world fast fitting strategy but with cleaner architecture

## Key Files to Examine

### Core Implementation Files:
- `src/distant_sunburn/poe_world/object_model_learner.py` - Protocol definition and orchestrator
- `src/distant_sunburn/poe_world/weight_fitter.py` - Weight learning component
- `src/distant_sunburn/poe_world/world_model.py` - Expert management component
- `src/distant_sunburn/poe_world/core.py` - Core types and protocols

### Reference Implementation:
- `external/poe-world/learners/models.py` - External MoEObjModel implementation
- `external/poe-world/learners/obj_model_learner.py` - External ObjModelLearner

### Type Definitions:
- `src/distant_sunburn/poe_world/core.py` - `WeightedExpert`, `SymbolicTransition`, etc.

## Testing Strategy

### Basic Test Structure

Create `tests/poe_world/test_expert_manager.py`:

```python
def test_expert_manager_basic_operations():
    """Test basic expert manager operations."""
    # Setup
    observable_extractor = MockObservableExtractor()
    weight_fitter = MaxLikelihoodWeightFitter(observable_extractor)
    manager = ExpertManager(observable_extractor, weight_fitter)
    
    # Test add_experts
    experts = [WeightedExpert(expert_function, weight=1.0)]
    manager.add_experts(experts)
    assert len(manager.get_experts()) == 1
    
    # Test evaluate_log_probability
    log_prob = manager.evaluate_log_probability(state, action, next_state)
    assert isinstance(log_prob, float)
    
    # Test fit_weights
    transitions = [SymbolicTransition(state, action, next_state)]
    manager.fit_weights(transitions)
    
    # Test pruning
    manager.prune_experts(weight_threshold=0.5)
    
    # Test save/load
    manager.save("test_checkpoint.pkl")
    new_manager = ExpertManager(observable_extractor, weight_fitter)
    assert new_manager.load("test_checkpoint.pkl")
```

### Integration Test

Test with `ObjectModelOrchestrator`:

```python
def test_expert_manager_with_orchestrator():
    """Test expert manager integration with orchestrator."""
    # Create managers
    non_creation_manager = ExpertManager(observable_extractor, weight_fitter)
    creation_manager = ExpertManager(observable_extractor, weight_fitter)
    
    # Create orchestrator
    orchestrator = ObjectModelOrchestrator(
        object_type="player",
        non_creation_expert_manager=non_creation_manager,
        creation_expert_manager=creation_manager,
        non_creation_synthesizer=mock_synthesizer,
        creation_synthesizer=mock_synthesizer,
        config=LearningConfig()
    )
    
    # Test learning loop
    orchestrator.add_datapoint(state, action, next_state)
    result = orchestrator.infer_moe()
    assert isinstance(result, ObjectTypeModel)
```

## Design Decisions

### 1. Immutable Updates
Follow `PoEWorldModel.with_new_experts()` pattern for immutable expert updates to avoid state mutation issues.

### 2. Weight Threshold for Pruning
Use configurable threshold (default 0.01) to remove ineffective experts while preserving potentially useful ones.

### 3. Checkpoint Format
Use pickle for simplicity initially, but consider JSON for better versioning and debugging.

### 4. Fast Mode Implementation
For fast mode, use the simplified approach of fitting only new experts:

**Core Strategy:**
- Track which experts have been fitted using an `is_fitted` flag
- During fast mode, identify and fit only newly added experts
- Preserve existing expert weights without modification
- Use the existing `MaxLikelihoodWeightFitter` without any modifications

**Implementation Steps:**
1. Add `is_fitted: bool` field to `WeightedExpert` or track separately in `ExpertManager`
2. Implement `_update_weights_for_new_experts()` helper method
3. Modify `fit_weights()` to support both full and fast modes
4. Ensure proper state tracking across save/load operations

**Benefits:**
- No changes needed to the weight fitter component
- Simpler and more maintainable than masking approaches
- Preserves the single responsibility principle
- Easier to test and debug

## Acceptance Criteria

- [ ] `ExpertManager` class implements `ExpertManagerProtocol`
- [ ] All required methods are implemented and tested
- [ ] Wrapper correctly coordinates `MaxLikelihoodWeightFitter` and `PoEWorldModel`
- [ ] Pruning removes experts below weight threshold
- [ ] Checkpointing saves/loads expert state correctly
- [ ] Fast mode provides faster weight fitting
- [ ] Integration test with `ObjectModelOrchestrator` passes
- [ ] Type annotations are correct and complete
- [ ] Documentation is clear and comprehensive

## Notes for Implementation

### Key Insights from Previous Discussion:
- The `ExpertManagerProtocol` corresponds to `MoEObjModel` in external poe-world
- Each object type needs separate managers for creation vs non-creation experts
- The orchestrator uses both managers to evaluate surprise: `max(non_creation_log_prob, creation_log_prob)`
- Weight fitting should preserve expert functions while updating weights
- The `PoEWorldModel` already handles the Product of Experts combination logic

### Potential Challenges:
- Ensuring type safety across the wrapper interface
- Handling edge cases in pruning (e.g., all experts below threshold)
- Managing checkpoint compatibility across versions
- Tracking expert fitting state across save/load operations
- Ensuring fast mode doesn't lead to suboptimal weight combinations

### Performance Considerations:
- Expert evaluation can be expensive - consider caching
- Weight fitting scales with number of experts and transitions
- Checkpointing should be efficient for large expert collections
