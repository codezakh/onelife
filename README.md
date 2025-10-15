# 12-distant-sunburn

A research codebase for learning probabilistic, programmatic world models from environment interactions. This project implements methods for synthesizing world models that capture the transition dynamics of symbolic environments, with a focus on complex, partially observable, stochastic domains like Crafter.

## Quick Start

### Installation

Clone the repository and initialize submodules:

```bash
git clone --depth 1 <repository-url>
cd 12-distant-sunburn
git submodule update --init
```

Install dependencies using `uv`:

```bash
uv sync
```

### Running Tests

Run the test suite from the `onelife` directory:

```bash
cd onelife
uv run --env-file .env pytest tests/
```

To run a specific test:

```bash
uv run --env-file .env pytest tests/integration/crafter/test_poe_world_fitting_and_eval.py -v
```

### Crafter Environment Integration Tests
The integration tests provide the clearest view of how the main components work together. These tests demonstrate the complete pipeline: generating training data, fitting a world model, and evaluating its performance.

Two main integration tests demonstrate the full system on the Crafter environment:

**PoE-World** (`tests/integration/crafter/test_poe_world_fitting_and_eval.py`):
**OneLife** (`tests/integration/crafter/test_our_method_fitting_and_eval.py`):

Both tests follow this pattern:

1. **Data Generation**: Collect transitions `(s, a, s')` from a random policy
2. **Model Fitting**: Learn weights for handwritten experts/laws using maximum likelihood
3. **Evaluation**: Test the model on held-out scenarios using the hybrid evaluation framework
4. **Analysis**: Compare discriminative accuracy, edit distance, and normalized recall metrics

### Simple 1D Environment 
For debugging and understanding the core algorithms, the Simple 1D environment provides a minimal testbed (`tests/integration/simple_1d_env/test_poe_world_fitting_and_eval.py`):