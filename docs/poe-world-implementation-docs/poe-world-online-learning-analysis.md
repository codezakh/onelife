# PoE World Online Learning Analysis

**Date:** 2025-01-27  
**Author:** AI Assistant  
**Purpose:** Analysis of PoE World's online learning implementation and claims

## Executive Summary

This document analyzes PoE World's online learning capabilities as described in the arXiv paper versus its actual implementation. The investigation was prompted by questions about whether the method's online learning claims matched its implementation approach.

**Key Finding:** PoE World implements a hybrid approach combining offline batch learning from prerecorded demonstrations with limited online refinement during agent execution. While online updates do occur, they are more constrained than the paper's description might suggest.

## Instigating Question

The analysis was requested to address this specific question:

> "The arXiv paper mentions that the PoE-World method is online. I'm not certain that my reading of the method is correct, so please check the paper to confirm. The part that is confusing to me is that it seems like the method *as implemented in the codebase* is not online in a meaningful sense — it appears to use a list of prerecorded actions. Can you confirm that my understanding here is correct?"

## Methodology

The analysis involved:

1. **Reading the PRD** - Understanding the intended reimplementation goals
2. **Examining the arXiv paper** - Identifying claims about online learning
3. **Analyzing the codebase** - Tracing execution flow and data handling
4. **Cross-referencing claims vs. implementation** - Identifying discrepancies

## Claims in the arXiv Paper

### Overview Figure Caption
> "These programs are refined online in later environment interactions."

### Learning Algorithm Section
The paper describes the learning process as:
```
We begin with a demonstration trajectory, learn a world model, and then begin to act in the world according to that model.
As the agent acts, it collects more trajectory data, which it uses to update or "debug" its model.
```

And states:
> "We repeat this loop every time there are new observations."

### Implications
These statements suggest a continuous online learning process where:
- The model starts with initial demonstrations
- The agent acts in the environment
- New data is collected during action
- The model updates incrementally with new observations
- This cycle repeats continuously

## Actual Implementation Analysis

### 1. Prerecorded Action Sequences

**Location:** `actions_lists/` directory

**Evidence:**
- `actions_list_pong.py` contains hardcoded action sequences like:
  ```python
  pong_actions_basic1 = ['NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', ...]
  ```
- `actions_list_montezuma.py` (97KB, 1201 lines) contains massive predefined action lists
- `actions_list_pitfall.py` (30KB, 416 lines) contains more hardcoded sequences

**Impact:** These are not dynamically generated actions from an online agent, but static, manually crafted sequences.

### 2. Batch Data Loading

**Location:** `run.py` main execution

**Evidence:**
```python
# Load observations -- use the same observation for both non-prime and prime versions
observations, actions, game_states = load_atari_observations(
    config.task.replace('Alt', '') + config.obs_suffix)
```

**Impact:** All data is loaded at once from pickle files, not collected incrementally during execution.

### 3. Sequential Batch Processing

**Location:** `learners/obj_model_learner.py` - `infer_moe()` method

**Evidence:**
```python
while self.processed_obs_count < len(self.transitions):
    indices = np.arange(
        self.processed_obs_count,
        min(self.processed_obs_count + self.batch_size, len(self.transitions)))
```

**Impact:** Data is processed in fixed-size batches (default 10) through the entire dataset, not as new observations arrive.

### 4. Data Collection Process

**Location:** `make_observations.py`

**Evidence:**
```python
def make_observations(config: DictConfig, actions: List[str], name: str) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay.
    """
    for idx, action in enumerate(actions):
        if action == 'RESTART':
            obs, game_state = env.reset()
        else:
            obs, game_state = env.step(action)
```

**Impact:** This is a one-time, offline process that executes predefined action sequences and saves results to pickle files.

### 5. Online Updates During Agent Execution

**Location:** `agents/agent.py`

**Fast Updates:**
```python
# fast update of world model
if self.config.agent.fast_world_update and ct % (self.config.agent.max_iter // 10) == 0:
    self.world_learner.update_world_model(
        cur_c, fast=True, player_only=self.config.agent.update_player_only)
```

**Permanent Updates:**
```python
if self.config.agent.permanent_world_update and (not success):
    # Permanently update (with slow learning) world model
    self.world_learner.update_world_model(
        all_c, fast=False, player_only=self.config.agent.update_player_only)
```

**Characteristics of Online Updates:**
- **Frequency:** Fast updates occur every 10% of max iterations
- **Scope:** Updates focus on player models (configurable via `update_player_only`)
- **Trigger:** Permanent updates occur when the agent fails to achieve goals
- **Data Source:** Updates use accumulated experience from the current episode
- **Method:** Uses the same weight fitting and pruning process as offline learning

## Detailed Analysis of Online Learning Implementation

### The `update_world_model` Method

**Location:** `learners/world_model_learner.py` - `PoEWorldLearner.update_world_model()`

**Method Signature:**
```python
def update_world_model(self, c, fast=False, player_only=False) -> WorldModel:
    """
    Update existing world model with new observations.
    
    Args:
        c: New observations to incorporate
        fast: Whether to use fast inference mode
        player_only: Whether to update only player models
        
    Returns:
        Updated composed world model
    """
```

**Core Implementation:**
```python
obj_type_models: List[ObjTypeModel] = []
constraints = None
for obj_type in self.all_obj_types:
    if obj_type != 'player' and player_only:
        obj_type_models.append(self.obj_model_learners[obj_type].return_obj_type_model())
        continue
    log.info(f'Updating ObjModel for obj_type "{obj_type}" (fast={fast})...')
    obj_model_learner = self.obj_model_learners[obj_type]
    for x in c:
        obj_model_learner.add_datapoint(x)
    if fast:
        obj_type_model = obj_model_learner.fast_infer_moe()
    else:
        obj_type_model = obj_model_learner.slow_infer_moe()
    obj_type_models.append(obj_type_model)
    if obj_type == 'player':
        constraints = obj_model_learner.return_constraints()

self.world_model = WorldModel(obj_type_models, constraints)
return self.world_model
```

**Key Components:**
1. **Data Ingestion:** New observations are added to each object model learner via `add_datapoint()`
2. **Model Selection:** Chooses between `fast_infer_moe()` and `slow_infer_moe()` based on the `fast` parameter
3. **Scope Control:** The `player_only` parameter determines whether to update all object types or just the player
4. **Model Reconstruction:** Creates a new `WorldModel` by combining updated object type models

### Fast vs Slow Inference Methods

**Location:** `learners/obj_model_learner.py`

#### Fast Inference (`fast_infer_moe()`)

**Purpose:** Quick, lightweight updates during agent execution

**Key Characteristics:**
- **Synthesizer Scope:** Only runs "fully observable MDP synthesizers" (no POMDP synthesizers)
- **Data Scope:** Processes only new observations since last checkpoint
- **Weight Fitting:** Uses `fit_only_new_weights()` which preserves existing expert weights
- **Pruning:** Performs basic pruning but skips constraint synthesis

**Implementation Details:**
```python
def fast_infer_moe(self) -> MoEObjModel:
    # Only process new observations
    indices = np.arange(self.processed_obs_count, len(self.transitions))
    
    # Run only fully observable MDP synthesizers
    to_be_run = [
        self._a_infer_moe_at_transition(self.transitions[idx:idx + 1],
                                        with_constraint=False)
        for idx in to_be_run_indices
    ]
    
    # Use fast weight fitting
    self._update_moe(self.transitions, fast_fitting=True)
    
    return self.return_obj_type_model()
```

#### Slow Inference (`slow_infer_moe()`)

**Purpose:** Comprehensive model updates when agent fails

**Key Characteristics:**
- **Synthesizer Scope:** Runs both fully observable MDP and POMDP synthesizers
- **Data Scope:** Processes all observations with full historical context
- **Weight Fitting:** Uses `fit_weights()` which refits all expert weights
- **Pruning:** Performs comprehensive pruning including constraint synthesis
- **Persistence:** Saves checkpoints to disk

**Implementation Details:**
```python
def slow_infer_moe(self) -> MoEObjModel:
    # Process all observations with full history
    to_be_run = [
        self._a_infer_moe_at_transition(self.transitions[:idx + 1],
                                        with_constraint=False)
        for idx in to_be_run_indices
    ]
    to_be_run = to_be_run + [
        self._a_infer_moe_at_transition(self.transitions[:idx + 1],
                                        with_constraint=False,
                                        with_full_history=True)
        for idx in to_be_run_indices
    ]
    
    # Use comprehensive weight fitting
    self._update_moe(self.transitions, fast_fitting=False)
    
    # Save checkpoint
    self.save()
    
    return self.return_obj_type_model()
```

### Weight Fitting Differences

**Location:** `learners/models.py` - `MoEObjModel` class

#### Fast Weight Fitting (`fit_only_new_weights()`)

**Purpose:** Incremental updates that preserve existing expert knowledge

**Key Characteristics:**
- **Weight Preservation:** Freezes weights of previously fitted experts
- **New Expert Initialization:** Sets weights of new experts to 0.01
- **Selective Optimization:** Only optimizes weights for newly added experts
- **Efficiency:** Avoids recomputing distributions for existing experts

**Implementation:**
```python
def fit_only_new_weights(self, c: List[Any], include_l1_loss: bool = True) -> None:
    # Find the last fitted expert
    freeze_before = -1
    for idx in range(len(self.fitteds) - 1, -1, -1):
        if self.fitteds[idx]:
            freeze_before = idx
            break
        self.params[idx] = 0.01  # Initialize new expert weights
    
    # Fit only new weights while preserving old ones
    new_params = list(
        self._fit_weights_helper(c, freeze_before=freeze_before, include_l1_loss=include_l1_loss))
```

#### Slow Weight Fitting (`fit_weights()`)

**Purpose:** Comprehensive refitting of all expert weights

**Key Characteristics:**
- **Full Refitting:** Optimizes weights for all experts simultaneously
- **Pruning:** Performs bad rule pruning before fitting
- **Comprehensive:** Recomputes all expert distributions
- **Robustness:** Ensures optimal weight configuration for all data

**Implementation:**
```python
def fit_weights(self, c: List[StateTransitionTriplet], include_l1_loss: bool = True) -> None:
    # Prune bad programs first
    for x in c:
        self._prune_bad_rules(x.input_state, x.event)
    
    # Precompute all distributions
    for idx, x in enumerate(c):
        # ... distribution computation ...
    
    # Fit all weights together
    new_params = list(self._fit_weights_helper(c, include_l1_loss=include_l1_loss))
```

### Agent Design and World Model Usage

**Location:** `agents/agent.py` and `agents/mcts.py`

#### Agent Architecture

The agent uses a hierarchical planning approach:

1. **High-Level Planning:** Abstract state space planning using symbolic representations
2. **Low-Level Execution:** MCTS-based action selection using the world model
3. **Online Learning:** Incremental model updates during execution

#### World Model Integration

**Planning Process:**
```python
def run_low_level(self, cur_obj_list, cur_game_state, n_budget_iterations, target_abstract_state, target_id=None):
    # Save snapshot for potential rollback
    if self.config.agent.fast_world_update:
        self.world_learner.save_snapshot()
    
    # Use world model for MCTS planning
    world_model = self.world_learner.world_model
    new_plan = self.mcts.search(cur_obj_list, target_abstract_state, world_model, iterations=n_budget_iterations)
    
    # Execute plan and collect experience
    for action in plan:
        cur_obj_list, cur_game_state = self.atari_env.step(action)
        # ... collect experience ...
    
    # Fast updates during execution
    if self.config.agent.fast_world_update and ct % (self.config.agent.max_iter // 10) == 0:
        self.world_learner.update_world_model(cur_c, fast=True, player_only=self.config.agent.update_player_only)
    
    # Rollback fast updates
    if self.config.agent.fast_world_update:
        self.world_learner.load_snapshot()
    
    # Permanent updates on failure
    if self.config.agent.permanent_world_update and (not success):
        self.world_learner.update_world_model(all_c, fast=False, player_only=self.config.agent.update_player_only)
```

#### MCTS World Model Usage

**Location:** `agents/mcts.py` - `GameState` class

The MCTS planner uses the world model in several ways:

1. **State Prediction:** `sample_next_scene()` predicts future states
2. **Action Simulation:** `perform_action_seq()` simulates action sequences
3. **Stability Checking:** `is_stable_state()` determines if states are stable
4. **Abstract State Generation:** `get_abstract_state()` creates symbolic representations

**Key Methods:**
```python
def old_perform_action_seq(self, action_seq):
    """Simulate action sequence using world model"""
    cur_obj_list = self.cur_obj_list.deepcopy()
    memory = cur_obj_list.memory
    
    for action in action_seq:
        old_obj_list = cur_obj_list
        # Use world model to predict next state
        cur_obj_list = self.world_model.sample_next_scene(cur_obj_list, action, memory=memory, det=self.config.det_world_model)
        memory.add_obj_list_and_action(old_obj_list, action)
        
        # Check for death conditions
        if self._check_death(cur_obj_list, old_obj_list):
            died = True
            break
    
    return GameState(self.world_model, ObjListWithMemory(cur_obj_list, memory), ...)
```

#### Policy Definition

The agent's policy is defined through the combination of:

1. **Abstract Planning:** Symbolic state transitions in the abstract graph
2. **MCTS Action Selection:** Monte Carlo Tree Search for low-level action sequences
3. **World Model Guidance:** The learned Product of Experts model guides both planning and execution

**Policy Components:**
- **High-Level:** Abstract state space navigation using learned skills
- **Low-Level:** MCTS-based action selection with world model simulation
- **Learning:** Incremental model updates based on execution experience

## Implementation Characteristics

### 1. Data Sources
- **Initial Learning:** Prerecorded action sequences from demonstrations
- **Online Updates:** Real-time experience collected during agent execution

### 2. Learning Process
- **Initial Phase:** Batch processing of demonstration data
- **Online Phase:** Incremental updates during agent interaction

### 3. Online Learning Scope
- **Frequency:** Periodic updates (every 10% of iterations) rather than continuous
- **Scope:** Primarily player-focused updates rather than comprehensive model updates
- **Trigger:** Failure-driven permanent updates rather than continuous refinement

### 4. Learning Continuity
- **Initial:** One-time batch learning from demonstrations
- **Ongoing:** Conditional online refinement during agent execution

## Conclusion

The analysis reveals that PoE World implements a hybrid learning approach that combines offline batch learning with online refinement capabilities.

### PoE World's Learning Approach
- **Initial Learning:** Offline batch processing of prerecorded demonstrations
- **Online Refinement:** Periodic updates during agent execution, with both fast and permanent update mechanisms
- **Scope:** Primarily focused on player model updates rather than comprehensive world model updates

### Comparison with Paper Claims
The paper's description of "refined online in later environment interactions" is technically accurate but may suggest a more comprehensive online learning system than what is implemented. The actual online updates are:
- Periodic rather than continuous
- Limited in scope (primarily player-focused)
- Triggered by specific conditions (failure events)

### Implications for Reimplementation
This analysis supports the PRD's approach of building a more comprehensive online learning system that includes:
- Continuous experience buffer management
- Incremental model updates for all object types
- Proper checkpointing and resumption
- True online learning pipeline

The original PoE World demonstrates the Product of Experts approach effectively but could benefit from more robust online learning capabilities as outlined in the reimplementation PRD.

## Files Analyzed

### Core Execution
- `run.py` - Main entry point and data loading
- `make_observations.py` - Data collection process
- `data/atari.py` - Data loading utilities

### Learning Pipeline
- `learners/world_model_learner.py` - Top-level learning orchestration
- `learners/obj_model_learner.py` - Object-specific learning logic
- `learners/models.py` - Model implementations

### Agent Execution
- `agents/agent.py` - Agent planning and execution with limited online updates

### Data Sources
- `actions_lists/` - Prerecorded action sequences
- `saved_data/` - Pickled observation data

### Documentation
- `poe-world-arxiv.md` - Original paper claims
- `prd.md` - Reimplementation requirements
- `poe-world-codebase-outline.md` - Codebase structure analysis
