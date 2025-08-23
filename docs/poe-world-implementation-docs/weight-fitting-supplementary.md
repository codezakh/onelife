# PoE World Weight Fitting: Technical Implementation Details

## Unified End-to-End Process

The PoE World weight fitting process transforms a collection of synthesized expert programs into a probabilistic world model. Here's the complete story:

### The Complete Pipeline

1. **Expert Functions**: Python functions that receive the full game state and can modify any objects of their target type (e.g., all balls). Each expert encodes a hypothesis about game physics or dynamics.

2. **Expert Execution**: For each training transition, every expert runs on the input state, potentially modifying multiple objects simultaneously to model interactions and coordinated behaviors.

3. **Object Selection**: After expert execution, only objects of the target type are extracted for loss computation (ball experts → ball objects only).

4. **Attribute-Level Predictions**: Each expert's modifications are converted to `RandomValues` distributions for each object attribute (velocity_x, velocity_y, deleted, etc.).

5. **PoE Combination**: For each object's each attribute, expert predictions are combined using learned weights via matrix multiplication of log-probabilities.

6. **Loss Computation**: The system evaluates how well the combined prediction matches the observed attribute value, accumulating loss over all attributes, all objects, all training examples.

7. **Weight Optimization**: L-BFGS optimizer updates expert weights to maximize the total log-likelihood of observed data.

### Key Insight: Granular Loss Structure

The loss is computed at the most granular level possible:

```python
total_loss = 0
for training_example in dataset:
    for object_of_target_type in training_example:
        for attribute in ['velocity_x', 'velocity_y', 'deleted']:
            # Combine expert predictions using current weights
            combined_prediction = combine_expert_predictions(attribute, weights)
            # Evaluate observed value under combined prediction
            observed_value = getattr(object, attribute)
            log_prob = combined_prediction.evaluate_logprobs(observed_value)
            total_loss += log_prob
```

**The loss must be defined for every single object's every single attribute** - there's no higher-level aggregation. This granular approach allows the system to learn precise expert weights that capture fine-grained dynamics.

## Overview

This document provides detailed technical explanation of how this weight fitting process works, including the mathematical operations, tensor shapes, and implementation details.

## Key Concepts

### 1. Expert Programs as Probabilistic Predictors

Each expert program in PoE World is a deterministic Python function that takes a current state and action as input and produces a **probabilistic prediction** of the next state. This is achieved through the `RandomValues` data structure:

```python
class RandomValues:
    def __init__(self, values: Sequence[float], logscores: Optional[np.ndarray] = None):
        self.values = np.array(values)  # Possible discrete values
        self.logscores = logscores or np.zeros_like(self.values)  # Log-probabilities
```

**Critical Insight**: Expert programs don't return single values. Instead, they assign `RandomValues` objects to state attributes, representing probability distributions over possible outcomes.

### 2. Product of Experts (PoE) Combination

The core innovation of PoE World is how multiple expert predictions are combined. For each attribute of the next state:

1. **Collect Expert Predictions**: Gather `RandomValues` objects from all experts for that attribute
2. **Weighted Combination**: The final log-probability is the weighted sum of expert log-probabilities:
   ```
   log_prob_final = Σ(θ_i * log_prob_expert_i)
   ```
   where `θ_i` are the learned weights.

## Data Flow Architecture

### 1. Object-Type Grouping

**Critical Design Decision**: Experts are **not fitted globally**. Instead, the system groups experts by the object type they predict:

```python
# From world_model_learner.py
for obj_type in self.all_obj_types:
    obj_model_learner = self.obj_model_learners[obj_type]
    obj_type_model = obj_model_learner.infer_moe()
```

This means:
- Player experts are fitted together
- Ball experts are fitted together  
- Brick experts are fitted together
- etc.

**Why This Matters**: This prevents experts for different object types from interfering with each other during optimization, leading to more stable and interpretable weights.

### 2. Creation vs. Non-Creation Separation

For each object type, there are **two separate MoE models**:

```python
# From obj_model_learner.py
self.moe_non_creation = MoEObjModel('non_creation', ...)
self.moe_creation = MoEObjModel('creation', ...)
```

- **Non-creation experts**: Predict how existing objects change (movement, velocity changes, etc.)
- **Creation experts**: Predict when new objects appear

This separation is crucial because creation and non-creation behaviors have fundamentally different dynamics and should be modeled separately.

## The Weight Fitting Process

### 1. Parameter Dimensionality and Structure

**CRITICAL CLARIFICATION**: The parameters being optimized are **per-expert weights**, not per-object weights. Here's the key insight:

- Each expert is a Python function that can make predictions about **any number of objects** of its target type
- The weight array `self.params` has **exactly one weight per expert**: `len(self.params) == len(self.rules)`  
- When a new expert is added, the parameter array grows by exactly 1: `self.params = self.params + ([0.5] * n_new_rules)`

**Example**: If you have 3 ball experts, you have exactly 3 weights `[θ₁, θ₂, θ₃]`, regardless of whether there are 4 balls or 7 balls in any given state.

### 2. How Experts Handle Variable Object Counts

The key to understanding dynamic scaling is the **object selector mechanism**:

```python
# From models.py _get_obj_list_dists_helper()
obj_list_next_dist = callable(pre_stepped_obj_list_prev.deepcopy(), event)
objs_dist = self.objects_selector(obj_list_next_dist, pre_stepped_obj_list_prev)
```

**What happens**:
1. **Expert execution**: Each expert function receives the **entire state** and can modify any objects of its type
2. **Object selection**: The `objects_selector` extracts only the objects of the target type (e.g., all balls)
3. **Attribute-wise prediction**: Each expert assigns `RandomValues` to attributes of each selected object

**Ball example**: 
- Expert 1 might assign `RandomValues([0, -1, 1])` to `velocity_x` for each ball
- Expert 2 might assign `RandomValues([0])` to `velocity_x` for each ball  
- Expert 3 might not modify `velocity_x` (gets uniform distribution)

### 3. The Loss Function: Attribute-by-Attribute Combination

The loss function operates **per-attribute, per-object**. Here's the detailed process:

```python
# Pseudocode showing the actual combination logic from combine_random_values()
def combine_expert_predictions(expert_predictions, weights):
    # expert_predictions[i] = RandomValues from expert i for this attribute
    # weights = [θ₁, θ₂, θ₃, ...] - one per expert
    
    logscores_matrix = np.stack([pred.logscores for pred in expert_predictions])  # Shape: (n_experts, n_values)
    combined_logscores = logscores_matrix.T @ weights  # Matrix multiplication!
    
    return RandomValues(expert_predictions[0].values, logscores=combined_logscores)
```

**Key insight**: The PoE combination happens via **matrix multiplication** where:
- Each row represents possible values for an attribute  
- Each column represents an expert's opinion
- Weights are multiplied with expert logscores and summed

### 4. Precomputation for Efficiency

Before optimization begins, the system precomputes expert predictions for all training examples:

```python
# From models.py fit_weights()
for idx, x in enumerate(c):
    if len(self.precompute_dist) <= idx:
        self.precompute_dist.append(
            self._get_obj_list_dists(x.input_state, x.event, memory))
```

**What gets precomputed**: For each expert and each training transition, the system computes the `RandomValues` objects that the expert would assign to each attribute.

**Why precompute**: This avoids re-running expert programs during optimization, which would be computationally expensive.

### 5. Loss Computation: The Granular Structure

The loss computation operates at the finest possible granularity - per attribute, per object, per training example. Here's the detailed breakdown:

#### Expert Execution and Object Selection

```python
# For each expert and each training transition
obj_list_next_dist = expert_function(input_state.deepcopy(), action)  # Expert sees full state
relevant_objects = self.objects_selector(obj_list_next_dist, input_state)  # Extract target type objects
```

**Critical insight**: Expert functions receive the **entire game state** and can modify multiple objects simultaneously. This allows modeling of interactions (e.g., ball collisions, player-platform physics). However, only objects of the target type contribute to the loss.

#### Attribute-Level Loss Accumulation

```python
def evaluate_logprobs(self, input_state, event, output_state, params):
    total_log_prob = 0
    
    # Get objects of target type from output state
    target_objects = self.objects_selector(output_state, input_state)
    
    for obj in target_objects:  # Each ball, each player, etc.
        for attr_name in ['velocity_x', 'velocity_y', 'deleted']:  # Each attribute
            # 1. Get expert predictions for this object's attribute (from precomputed cache)
            expert_predictions = [expert_pred[obj_idx][attr_name] for expert_pred in precomputed_predictions]
            
            # 2. Combine using expert weights
            combined_distribution = combine_random_values(expert_predictions, params)
            
            # 3. Evaluate observed value
            observed_value = getattr(obj, attr_name)
            log_prob = combined_distribution.evaluate_logprobs(observed_value)
            
            # 4. Accumulate
            total_log_prob += log_prob
    
    return total_log_prob
```

#### Complete Objective Function

```python
def _objective(self, params, training_data):
    total_loss = 0
    for transition in training_data:
        transition_log_prob = self.evaluate_logprobs(
            transition.input_state, transition.action, transition.output_state, params)
        total_loss += transition_log_prob
    
    return -total_loss  # Negative log-likelihood for minimization
```

**Loss Structure Summary**:
- **Training examples**: N transitions
- **Objects per transition**: Variable (e.g., 4 balls in one transition, 7 in another)  
- **Attributes per object**: 3 (velocity_x, velocity_y, deleted) or more
- **Total loss terms**: N × (sum of objects across transitions) × 3 attributes

The loss must be computed for every single attribute of every single object. There's no aggregation at the object level - each attribute contributes independently to learning the expert weights.

### 6. Concrete Example: Ball Velocity Prediction

Let's walk through a specific example to make this concrete:

**Scenario**: 2 balls, 3 experts, predicting `velocity_x` for ball #1

**Expert Predictions** (from precomputation):
- Expert 1: `RandomValues(values=[0, 1, -1], logscores=[0.1, 0.8, 0.1])` 
- Expert 2: `RandomValues(values=[0, 1, -1], logscores=[0.9, 0.05, 0.05])`
- Expert 3: `RandomValues(values=[0, 1, -1], logscores=[0.33, 0.33, 0.34])` (uniform)

**Current Weights**: `θ = [0.7, 0.2, 0.1]`

**PoE Combination**:
```python
# Matrix multiplication: logscores_matrix.T @ weights
logscores_matrix = [[0.1, 0.8, 0.1],    # Expert 1 logscores
                    [0.9, 0.05, 0.05],   # Expert 2 logscores  
                    [0.33, 0.33, 0.34]]  # Expert 3 logscores

combined_logscores = [0.1*0.7 + 0.9*0.2 + 0.33*0.1,    # For value 0
                      0.8*0.7 + 0.05*0.2 + 0.33*0.1,   # For value 1  
                      0.1*0.7 + 0.05*0.2 + 0.34*0.1]   # For value -1
                    = [0.32, 0.60, 0.08]
```

**Loss Computation**:
- If observed `velocity_x` = 1, then `log_prob = 0.60`
- If observed `velocity_x` = 0, then `log_prob = 0.32`  
- If observed `velocity_x` = -1, then `log_prob = 0.08`

**Key Insight**: Expert 1 strongly prefers `velocity_x = 1` and has high weight (0.7), so the combined prediction also strongly prefers `velocity_x = 1`. The system learns weights that make experts with better predictions more influential.

## Understanding LogScores and Tensor Shapes

### Where Do LogScores Come From?

This is a crucial point that was missing from the original explanation. **Expert functions don't directly specify logscores** - they only specify **which values are possible**. The logscores are added through a multi-step process:

#### Step 1: Expert Functions Create "Sharp" Distributions

When an expert function runs, it assigns `RandomValues` with **only the values it believes are possible**:

```python
# From programs.py - Expert function example
if action == 'LEFT':
    player_obj.velocity_x = RandomValues([-3])  # Only value -3 is possible
    # No logscores specified - defaults to [0.0]
```

**Initial state**: `RandomValues(values=[-3], logscores=[0.0])`

#### Step 2: "Noise Addition" Creates Full Probability Distributions

The system then calls `add_noise_to_random_values()` to expand this to cover all possible values:

```python
# From add_noise_to_random_values()
all_possible_values = np.arange(-20, 21)  # All possible velocities: [-20, -19, ..., 20]
new_logscores = np.full(41, -10.0)        # Shape: (41,) filled with -10.0 (very low probability)

# Set the expert's preferred value to have much higher probability
new_logscores[17] = 0.0  # Index 17 corresponds to value -3

# Result: RandomValues(values=[-20, -19, ..., 20], logscores=[-10, -10, ..., 0, ..., -10])
```

**After noise addition**: 
- `values`: `[-20, -19, -18, ..., -3, -2, -1, 0, 1, ..., 20]` (shape: 41)
- `logscores`: `[-10, -10, -10, ..., 0, -10, -10, -10, -10, ..., -10]` (shape: 41)

#### Step 3: Uniform Distributions for Unmodified Attributes

If an expert doesn't modify an attribute, `fill_unset_values_with_uniform()` creates a uniform distribution:

```python
# For attributes the expert didn't touch
obj.velocity_y = RandomValues(all_possible_velocities)  # values=[-20, ..., 20], logscores=[0, 0, ..., 0]
```

### Complete Shape Analysis

Let's trace through a concrete example with **2 balls, 3 experts, predicting velocity_x**:

#### Expert Predictions (After Noise Addition):
```python
# All experts predict over same value space: [-20, -19, ..., 20] (41 values)

# Expert 1: Strongly prefers velocity_x = -3
expert1_ball1_vx = RandomValues(values=[-20, ..., 20], 
                               logscores=[-10, ..., 0, ..., -10])  # Shape: (41,)

# Expert 2: Strongly prefers velocity_x = 0  
expert2_ball1_vx = RandomValues(values=[-20, ..., 20],
                               logscores=[-10, ..., 0, ..., -10])  # Shape: (41,)

# Expert 3: Uniform (didn't modify this attribute)
expert3_ball1_vx = RandomValues(values=[-20, ..., 20],
                               logscores=[0, 0, ..., 0])          # Shape: (41,)
```

#### Matrix Combination:

```python
# From combine_random_values() - the key operation
logscores_lst = [expert1_ball1_vx.logscores,    # Shape: (41,)
                expert2_ball1_vx.logscores,     # Shape: (41,) 
                expert3_ball1_vx.logscores]     # Shape: (41,)

logscores_matrix = np.stack(logscores_lst)      # Shape: (3, 41) - 3 experts, 41 values
weights = np.array([0.7, 0.2, 0.1])           # Shape: (3,)   - 3 expert weights

combined_logscores = logscores_matrix.T @ weights  # Shape: (41,) - final distribution
```

**Matrix multiplication breakdown**:
```python
# logscores_matrix.T has shape (41, 3)
# weights has shape (3,)
# Result has shape (41,)

# For value -3 (index 17):
combined_logscores[17] = 0.0*0.7 + (-10)*0.2 + 0.0*0.1 = -2.0

# For value 0 (index 20):  
combined_logscores[20] = (-10)*0.7 + 0.0*0.2 + 0.0*0.1 = -7.0

# For other values (e.g., index 0):
combined_logscores[0] = (-10)*0.7 + (-10)*0.2 + 0.0*0.1 = -9.0
```

### The Key Insight: Sharp vs. Smooth Distributions

**Expert functions create sharp distributions** (one value has logscore 0, others have logscore -10), but **the PoE combination creates smooth distributions** through weighted averaging. The final combined distribution has different logscores for different values, creating a proper probability distribution where the expert weights determine the relative influence of each expert's opinion.

### How Loss Function Uses These LogScores

The final step connects these combined logscores to the loss function:

```python
# From RandomValues.evaluate_logprobs()
def evaluate_logprobs(self, value: float, temp: float = 1) -> float:
    # Convert logscores to proper log-probabilities
    self.logprobs = self.logscores / temp - logsumexp(self.logscores / temp, -1)
    
    # Find the log-probability of the observed value
    if value in self.values:
        return self.logprobs[np.where(self.values == value)[0][0]]
    return LOG_IMPOSSIBLE_VALUE  # -inf for impossible values
```

**Concrete Example**: If the observed `velocity_x = -3` and our combined logscores are `[-9.0, -9.0, ..., -2.0, ..., -7.0]`:

1. **Normalize to log-probabilities**: `logprobs = logscores - logsumexp(logscores)`
2. **Look up observed value**: `log_prob = logprobs[17]` (index for value -3)
3. **Add to loss**: `total_loss += log_prob`

**The gradient flows backwards**: When the optimizer updates expert weights to maximize this log-probability, it's learning to give higher weights to experts whose predictions are more consistent with the observed data.

### Summary of Shapes Throughout the Pipeline

For **3 experts, 2 balls, velocity_x attribute**:

1. **Expert functions output**: 3 × `RandomValues(values=[specific_value], logscores=[0.0])` 
2. **After noise addition**: 3 × `RandomValues(values=[41 values], logscores=[41 values])`
3. **Matrix combination**: `(41, 3) @ (3,) = (41,)` → Final distribution
4. **Loss computation**: Single scalar per attribute per object
5. **Total loss**: Sum over all attributes, all objects, all training examples
6. **Optimization**: Update the 3 expert weights to minimize total loss

This explains why the system can handle variable object counts - the same expert weights are applied to each object independently, and the loss simply accumulates more terms when there are more objects.

### 7. Dynamic Scaling: Handling Variable Object Counts

**Your specific question**: "How does it work when we have 4 balls in one transition and then 7 balls in the other transition?"

**Answer**: The weight fitting handles this seamlessly because:

1. **Expert weights are object-count agnostic**: The same 3 weights `[θ₁, θ₂, θ₃]` are used regardless of object count

2. **Loss accumulates across all objects**: 
   - Transition with 4 balls: Loss = sum over 4 balls × (3 attributes each) = 12 terms
   - Transition with 7 balls: Loss = sum over 7 balls × (3 attributes each) = 21 terms
   - Total loss = 33 terms, all using the same expert weights

3. **Each expert sees all objects**: When an expert function runs, it receives the full state and can assign `RandomValues` to all balls simultaneously

**Code evidence**:
```python
# From combine_obj_list_dists() - this handles any number of objects
for j in range(len_obj_list):  # Iterate over however many objects exist
    velocity_x = combine_random_values(random_values_x_lst, weights, use_torch)
    velocity_y = combine_random_values(random_values_y_lst, weights, use_torch)
    # Same weights used for every object
```

**Why this works**: Expert weights represent **how much to trust each expert's opinion**, not **weights per object**. An expert that's good at predicting ball physics should be trusted equally for all balls.

### 8. Optimization Algorithm

The system uses **L-BFGS** (Limited-memory BFGS) optimization:

```python
# From models.py _fit_weights_helper()
if self.config.moe.optim == 'lbfgs':
    optimizer = optim.LBFGS([weights], lr=self.config.moe.lr, 
                           line_search_fn='strong_wolfe')
```

**Why L-BFGS**: 
- It's a quasi-Newton method that approximates the Hessian matrix
- Works well for smooth, convex optimization problems
- More efficient than gradient descent for this type of problem
- The paper found it worked better than Adam or SGD

### 5. Regularization

The objective includes L1 regularization to encourage sparsity:

```python
loss = self._objective(weights, c, ...) + l1_weight * weights.abs().sum()
```

This helps prevent overfitting and encourages the model to use fewer, more important experts.

## Fast vs. Slow Weight Fitting

The system implements two weight fitting strategies:

### Fast Weight Fitting (`fit_only_new_weights()`)

```python
# From models.py fit_only_new_weights()
freeze_before = -1
for idx in range(len(self.fitteds) - 1, -1, -1):
    if self.fitteds[idx]:
        freeze_before = idx
        break
    self.params[idx] = 0.01  # Initialize new weights

new_params = list(self._fit_weights_helper(c, freeze_before=freeze_before))
```

**Purpose**: Only fit weights for newly added experts while preserving existing weights.

**Benefits**: 
- Much faster than refitting all weights
- Preserves learned knowledge from previous iterations
- Useful for online learning scenarios

### Slow Weight Fitting (`fit_weights()`)

```python
# From models.py fit_weights()
new_params = list(self._fit_weights_helper(c, include_l1_loss=include_l1_loss))
```

**Purpose**: Refit all expert weights from scratch.

**Benefits**:
- More thorough optimization
- Can correct for suboptimal weights from previous iterations
- Better final model quality

## Batch Processing and Sampling

For computational efficiency, the system uses batch sampling during optimization:

```python
# From models.py _fit_weights_helper()
indices = np.random.choice(len(c), batch_size, replace=False) if batch_size < len(c) else np.arange(len(c))
loss = self._objective(weights, c, use_torch=True, indices=indices, use_precompute=True)
```

**Benefits**:
- Reduces memory usage for large datasets
- Speeds up optimization
- Provides some regularization through stochasticity

## Weight Initialization and Constraints

### Initialization

```python
# From models.py _fit_weights_helper()
if self.config.moe.continue_params:
    weights = nn.Parameter(torch.tensor(self.params, dtype=torch.float32, device=device))
else:
    weights = nn.Parameter(torch.tensor(np.ones_like(self.params) * 0.5, dtype=torch.float32, device=device))
```

**Default**: All weights start at 0.5 (uniform contribution from all experts)

**Continuation**: Can continue from previous weights if `continue_params` is enabled

### Constraints

```python
# From models.py _objective()
if use_torch:
    params = torch.clamp(params, min=0, max=10)
else:
    params = np.clip(params, 0, 10)
```

**Bounds**: Weights are constrained to [0, 10] to prevent numerical instability

**Non-negativity**: Ensures experts can only contribute positively to predictions

## Pruning After Weight Fitting

After weight fitting, experts with low weights are pruned:

```python
# From models.py prune_programs()
def prune_programs(self):
    threshold = 0.01  # Configurable threshold
    keep_indices = [i for i, weight in enumerate(self.params) if weight > threshold]
    # Remove experts and weights below threshold
```

**Purpose**: Remove useless experts to keep the model compact and interpretable

**Threshold**: Typically 0.01, meaning experts contributing less than 1% are removed

## Implementation Details

### PyTorch Integration

The system uses PyTorch for automatic differentiation:

```python
# From models.py _fit_weights_helper()
weights = nn.Parameter(torch.tensor(self.params, dtype=torch.float32, device=device))
```

**Benefits**:
- Automatic gradient computation
- GPU acceleration support
- Integration with PyTorch optimizers

### Memory Management

The precomputation cache can grow large:

```python
# From models.py _get_obj_list_dists()
if self.cache_enabled:
    k = f'{obj_list_prev}{event}'
    # ... cache key construction
    if k in self.cache:
        return self.cache[k]
```

**Cache Strategy**: 
- Caches expert predictions to avoid recomputation
- Can be disabled for memory-constrained environments
- Keys include state, action, and recent history

## Summary

The weight fitting process in PoE World is a sophisticated implementation of maximum likelihood estimation for Product of Experts models. Key design decisions include:

1. **Object-type grouping**: Experts are fitted separately by object type
2. **Creation/non-creation separation**: Different models for different types of state changes
3. **Precomputation**: Expert predictions are cached for efficiency
4. **L-BFGS optimization**: Quasi-Newton method for smooth optimization
5. **L1 regularization**: Encourages sparsity and prevents overfitting
6. **Fast/slow fitting modes**: Trade-offs between speed and thoroughness
7. **Pruning**: Removes low-weight experts to maintain model quality

This architecture enables the system to learn interpretable, probabilistic world models from programmatic experts while maintaining computational efficiency and numerical stability.

## Key Clarifications Added

**Parameter Structure**: The parameters are **per-expert weights** (not per-object). For N experts, there are exactly N parameters regardless of how many objects exist in any given state.

**Loss Function**: The loss operates through **weighted linear combination** of expert log-probabilities, computed via matrix multiplication for each attribute of each object.

**Dynamic Scaling**: Variable object counts are handled naturally because:
- Expert weights are object-count agnostic
- Each expert function can handle any number of objects
- Loss accumulates across all objects using the same expert weights
- More objects simply means more terms in the loss sum

**Data Flow**: Input data (state transitions) → Expert functions → RandomValues predictions → Weighted combination → Log-probability evaluation → Loss computation → Weight optimization

This clarifies how the "traditional deep learning" paradigm applies: parameters (expert weights) interact with input data (state transitions) through a differentiable computation graph where expert predictions are combined and evaluated against ground truth observations.
