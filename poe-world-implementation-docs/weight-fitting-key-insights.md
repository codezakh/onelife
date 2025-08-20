# Key Insights for PRD: Weight Fitting Process

## Critical Design Decisions to Include in PRD

### 1. Object-Type Grouping (CRITICAL)
**Current PRD Issue**: The PRD doesn't mention that experts are grouped by object type during weight fitting.

**Implementation Detail**: 
- Each object type (player, ball, brick, etc.) has its own separate MoE model
- Weight fitting is performed independently for each object type
- This prevents experts for different object types from interfering with each other

**Why This Matters**: This is a crucial optimization that makes the system much more stable and interpretable. Without this grouping, the optimization would be much harder and less reliable.

### 2. Creation vs. Non-Creation Separation
**Current PRD Issue**: The PRD doesn't distinguish between creation and non-creation experts.

**Implementation Detail**:
- For each object type, there are TWO separate MoE models:
  - `moe_non_creation`: Predicts how existing objects change
  - `moe_creation`: Predicts when new objects appear
- These are fitted separately because they have fundamentally different dynamics

### 3. Precomputation Strategy
**Current PRD Issue**: The PRD mentions batch sampling but doesn't explain the precomputation strategy.

**Implementation Detail**:
- Before optimization, expert predictions are precomputed for all training examples
- This avoids re-running expert programs during optimization (which would be very expensive)
- The precomputed cache stores `RandomValues` objects for each expert-transition pair

### 4. L-BFGS Optimization
**Current PRD Issue**: The PRD mentions "scipy.optimize" but doesn't specify the algorithm.

**Implementation Detail**:
- Uses L-BFGS (Limited-memory BFGS) optimizer from PyTorch
- This is a quasi-Newton method that works much better than gradient descent for this problem
- The paper found it outperformed Adam and SGD

### 5. Fast vs. Slow Weight Fitting
**Current PRD Issue**: The PRD doesn't mention the two different fitting modes.

**Implementation Detail**:
- **Fast fitting**: Only fits weights for newly added experts, preserves existing weights
- **Slow fitting**: Refits all expert weights from scratch
- This is crucial for online learning efficiency

### 6. Weight Constraints and Initialization
**Current PRD Issue**: The PRD doesn't specify weight bounds or initialization.

**Implementation Detail**:
- Weights are constrained to [0, 10] to prevent numerical instability
- Default initialization is 0.5 (uniform contribution from all experts)
- Can continue from previous weights if desired

## Recommended PRD Updates

### Section 5.4: MaxLikelihoodWeightFitter
Add these details:

```python
class MaxLikelihoodWeightFitter(implements WeightFitterProtocol):
    """
    Fits weights to experts using maximum likelihood estimation.
    
    CRITICAL: Experts are grouped by object type. Each object type has separate
    creation and non-creation models that are fitted independently.
    """
    
    def fit(self, experts: list[ExpertSourceCode], transitions: list[SymbolicTransition[MetadataT]]) -> list[WeightedExpert]:
        """
        Fit weights using L-BFGS optimization with L1 regularization.
        
        Process:
        1. Group experts by object type and creation/non-creation
        2. Precompute expert predictions for all transitions
        3. Optimize weights using L-BFGS with bounds [0, 10]
        4. Apply L1 regularization to encourage sparsity
        5. Return weighted experts
        """
```

### Section 5.2: PoEWorldModel
Add this detail about the probabilistic prediction mechanism:

```python
# In the "Probabilistic Prediction Mechanism" section:
# 2. Generate Expert Outputs: For a given current_state and action, the PoEWorldModel 
#    iterates through its list of weighted experts. For each expert:
#    a. A deep copy of the current_state is created.
#    b. The expert's source_code is compiled into an ExpertFunction via the ExpertCompilerProtocol
#    c. The compiled expert function is executed via the ExpertExecutorProtocol, which mutates 
#       the attributes of the copied state in-place.
#    d. CRITICAL: The expert assigns RandomValues objects to attributes, not primitive values.
#       For example: state.player.inventory.wood = RandomValues(values=np.array([1, 2, 3]), logscores=np.array([0.1, 0.5, 0.2]))
#    e. The mutated state copy, containing probabilistic attribute values, is the expert's output distribution.
```

### Section 6: Online Learning Pipeline
Add this detail about the update cycle:

```python
# In step 5 "Fit Weights":
# 5. Fit Weights: Pass the combined expert list and a representative batch of symbolic 
#    transitions from the buffer to the fitter to get a new list of WeightedExpert s.
#    NOTE: The fitter will automatically group experts by object type and fit creation 
#    vs. non-creation models separately. This is handled internally by the fitter.
```

## Data Structures to Add

### RandomValues
The PRD should include this data structure definition:

```python
@attrs.define
class RandomValues:
    """
    Represents a discrete probability distribution over a set of integer values.
    This is the core mechanism for interpreting deterministic expert outputs
    as probabilistic predictions.
    """
    values: np.ndarray  # Possible discrete values
    logscores: np.ndarray = attrs.field()  # Log-probabilities

    @logscores.default
    def _default_logscores(self) -> np.ndarray:
        """Defaults to uniform logscores if not provided."""
        return np.zeros_like(self.values, dtype=float)

    def sample(self) -> int:
        """Samples a value from the distribution."""
        probabilities = np.exp(self.logscores - logsumexp(self.logscores))
        return np.random.choice(self.values, p=probabilities)

    def evaluate_log_probability(self, value: int) -> float:
        """Calculates the log-probability of a given value."""
        log_probs = self.logscores - logsumexp(self.logscores)
        try:
            return log_probs[np.where(self.values == value)[0][0]]
        except IndexError:
            return -np.inf  # Value not in distribution
```

## Summary

The key insight is that PoE World's weight fitting is much more sophisticated than a simple "fit weights to experts" approach. The object-type grouping, creation/non-creation separation, and precomputation strategy are all critical design decisions that make the system work effectively. These details should be incorporated into the PRD to ensure the reimplementation captures the essential aspects of the original system.
