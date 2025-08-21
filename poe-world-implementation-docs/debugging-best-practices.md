# PoE-World Debugging Best Practices

## Overview

This document captures critical debugging methodologies learned while implementing the PoE-World weight fitting system. These practices help isolate issues in complex machine learning pipelines where multiple components interact.

## Core Principle: Start Simple, Test Incrementally

The most effective debugging strategy for PoE-World is to **start with the simplest possible case that should work** and incrementally add complexity. This follows the principle of "minimal failing examples."

## Critical Bug Pattern: Gradient Flow Issues

### The Problem We Solved

**Symptom**: All experts receive identical weights despite having dramatically different prediction quality.

**Root Cause**: Breaking PyTorch's computation graph by calling `.detach().numpy()` in the middle of loss computation.

**Lesson**: In PyTorch-based optimization, **preserve gradients throughout the entire forward pass**. Any conversion to numpy arrays or calls to `.detach()` will sever gradient flow.

### Debugging Methodology That Worked

#### Step 1: Create a Single-Transition Test Case

When weight fitting fails, create the **simplest possible test case**:

```python
def create_clear_transition():
    """
    Create ONE transition where the difference between good/bad experts is obvious.
    
    Example: Player at switch zone boundary, action that should be inverted.
    - Good expert: Knows about switch zone, predicts correctly
    - Bad expert: Ignores switch zone, predicts incorrectly
    """
```

**Why this works**:
- Eliminates statistical noise from large datasets
- Makes expert differences crystal clear
- Allows manual verification of expected behavior

#### Step 2: Verify Loss Computation Before Optimization

**Before running the optimizer**, manually test the loss function:

```python
# Test different weight combinations manually
test_weights = [
    ([1.0, 0.0], "Only correct expert"),
    ([0.0, 1.0], "Only incorrect expert"), 
    ([0.5, 0.5], "Equal weights")
]

for weights, description in test_weights:
    loss = compute_loss(experts, transitions, weights)
    print(f"{description}: Loss = {loss}")
```

**Expected results**:
- "Only correct expert" should have **lowest loss**
- "Only incorrect expert" should have **highest loss**
- Equal weights should be **in between**

**If this doesn't work**: The problem is in loss computation, not optimization.

#### Step 3: Test Optimization Only After Loss Verification

Once loss computation is verified, test the optimizer:

```python
# Start with simple quadratic test
def simple_test():
    # Target: minimize (x[0] - 0.8)^2 + (x[1] - 0.2)^2
    # Expected result: weights converge to [0.8, 0.2]
    
# Then test actual loss function
def actual_test():
    # Use the verified loss function
    # Expected: weights should move toward correct expert
```

## Key Debugging Tools

### 1. Single Transition Tests

Create unit tests that use **exactly one transition** with obvious expert differences:

```python
def test_single_transition_weight_fitting():
    # Scenario: Clear-cut case where one expert is obviously better
    transition = create_clear_transition()
    experts = [correct_expert, incorrect_expert]
    
    # Verify loss computation
    loss_correct = compute_loss([correct_expert], [transition])
    loss_incorrect = compute_loss([incorrect_expert], [transition])
    assert loss_correct < loss_incorrect
    
    # Verify optimization
    fitted_experts = fitter.fit(experts, [transition])
    assert fitted_experts[0].weight > fitted_experts[1].weight
```

### 2. Gradient Flow Verification

When using PyTorch, verify gradients are flowing:

```python
def test_gradient_flow():
    weights = torch.tensor([0.5, 0.5], requires_grad=True)
    loss = compute_loss_function(weights, data)
    loss.backward()
    
    # Check gradients exist and are different
    assert weights.grad is not None
    assert not torch.allclose(weights.grad[0], weights.grad[1])
```

### 3. Loss Function Validation

Test loss function behavior with known scenarios:

```python
def test_loss_function_behavior():
    # Perfect prediction should have low loss
    perfect_loss = loss_function(perfect_prediction, observed_data)
    
    # Random prediction should have higher loss  
    random_loss = loss_function(random_prediction, observed_data)
    
    assert perfect_loss < random_loss
```

## PyTorch Best Practices for PoE-World

### ✅ Do: Preserve Gradient Flow

```python
# CORRECT: Keep tensors in PyTorch throughout computation
def combine_expert_predictions_torch(predictions, weights):
    logscores_matrix = torch.stack([
        torch.tensor(pred.logscores, dtype=torch.float32) 
        for pred in predictions
    ])
    combined_logscores = logscores_matrix.T @ weights
    return values_tensor, combined_logscores  # Keep as tensors!

def evaluate_log_probability_torch(values, logscores, observed):
    log_probs = logscores - torch.logsumexp(logscores, dim=0)
    mask = (values == observed)
    return log_probs[mask][0]  # Return tensor, not float
```

### ❌ Don't: Break Gradient Flow

```python
# WRONG: Converting to numpy breaks gradients
def combine_expert_predictions_broken(predictions, weights):
    logscores_matrix = torch.stack([...])
    combined_logscores = logscores_matrix.T @ weights
    
    return RandomValues(
        values=predictions[0].values,
        logscores=combined_logscores.detach().numpy()  # ❌ BREAKS GRADIENTS!
    )
```

### Key Insight: Two Versions Pattern

When integrating PyTorch optimization with existing code:

1. **Keep original function** for non-optimization uses
2. **Create `_torch` version** that preserves gradients for optimization
3. **Use appropriate version** based on context

```python
# For general use (returns numpy)
def combine_expert_predictions(predictions, weights):
    # ... implementation that returns RandomValues with numpy arrays

# For optimization (preserves gradients)  
def combine_expert_predictions_torch(predictions, weights):
    # ... implementation that returns PyTorch tensors
```

## Testing Strategy

### Unit Tests for Core Components

1. **Single transition loss computation**
2. **Single transition optimization** 
3. **Gradient flow verification**
4. **Expert prediction quality**

### Integration Tests

1. **Multiple transitions with known expert quality**
2. **Full weight fitting pipeline**
3. **Checkpointing and resumption**

### Property-Based Tests

1. **Loss should decrease with better predictions**
2. **Weights should sum to reasonable values**
3. **Optimization should converge**

## Common Pitfalls

### 1. Statistical Masking

**Problem**: Using large, complex datasets that mask fundamental issues.

**Solution**: Start with single, clear examples before scaling up.

### 2. Premature Optimization

**Problem**: Trying to optimize before verifying loss computation works.

**Solution**: Test loss function in isolation first.

### 3. Hidden State Issues

**Problem**: Cached or mutable state affecting reproducibility.

**Solution**: Use fresh instances and fixed random seeds in tests.

### 4. Silent Gradient Failures

**Problem**: PyTorch optimizers continuing to run even when gradients are broken.

**Solution**: Explicitly check gradient values in tests.

## Debugging Checklist

When weight fitting isn't working:

- [ ] Create single transition test case with obvious expert differences
- [ ] Manually verify loss computation gives expected ordering
- [ ] Check that PyTorch gradients are non-zero and different
- [ ] Test optimizer on simple quadratic function first
- [ ] Verify expert predictions are actually different
- [ ] Check for gradient flow breaks (`.detach()`, `.numpy()`)
- [ ] Test with fixed random seeds for reproducibility
- [ ] Add logging to track loss/weight changes during optimization

## Summary

The key insight is **incremental verification**: test each component in isolation with the simplest possible inputs before combining them. This methodology proved critical for identifying the gradient flow issue that was preventing weight learning.

**Remember**: If the system can't learn on a single, obvious example, it won't learn on complex datasets. Start simple, verify thoroughly, then scale up.
