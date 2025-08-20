# PoE World Weight Fitting: Technical Implementation Details

## Overview

This document provides a detailed technical explanation of how expert weights are fitted in the PoE World system. The weight fitting process is a critical component that transforms a collection of synthesized expert programs into a probabilistic world model through maximum likelihood estimation.

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

### 1. Precomputation for Efficiency

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

### 2. The Objective Function

The core of the weight fitting is the negative log-likelihood objective:

```python
# From models.py _objective()
def _objective(self, params, c, use_torch=True, indices=None, use_precompute=True):
    res = 0
    for idx in indices:
        x = c[idx]
        value = self.evaluate_logprobs(
            x.input_state, x.event, x.output_state, 
            params=params, use_torch=use_torch, precompute_index=idx)
        res += value
    return -res / len(c) * 1000  # Negative log-likelihood
```

**Key Components**:
- `evaluate_logprobs()`: Computes the log-probability of the observed next state under the current PoE model
- The negative sign converts maximization to minimization
- The scaling factor (1000) helps with numerical stability

### 3. Log-Probability Computation

The `evaluate_logprobs()` method implements the PoE combination:

```python
# Pseudocode from the implementation
def evaluate_logprobs(self, input_state, event, output_state, params):
    # 1. Get expert predictions (from precomputed cache)
    expert_predictions = self.precompute_dist[precompute_index]
    
    # 2. For each attribute of the output state
    total_log_prob = 0
    for attr_name in output_state.attributes:
        # 3. Collect expert predictions for this attribute
        attr_predictions = [expert_pred[attr_name] for expert_pred in expert_predictions]
        
        # 4. Combine using weighted sum of log-probabilities
        combined_logscores = sum(
            param * pred.logscores for param, pred in zip(params, attr_predictions)
        )
        
        # 5. Evaluate probability of observed value
        observed_value = getattr(output_state, attr_name)
        log_prob = self._evaluate_log_probability(combined_logscores, observed_value)
        total_log_prob += log_prob
    
    return total_log_prob
```

### 4. Optimization Algorithm

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
