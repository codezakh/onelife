# [One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration](onelife-worldmodel.github.io)

## Abstract
Symbolic world modeling is the task of inferring and representing the transitional dynamics of an environment as an executable program. Previous research on symbolic world modeling has focused on largely deterministic environments with abundant interaction data, simple mechanics, and human-provided guidance. We address the more realistic and challenging problem of learning a symbolic world model in a complex, stochastic environment with severe constraints: a limited interaction budget where the agent has only "one life" to explore a hostile environment and no external guidance in the form of human-provided, environment-specific rewards or goals. We introduce OneLife, a framework that models world dynamics through conditionally activated programmatic laws within a probabilistic programming framework. Each law operates through a precondition-effect structure, allowing it to remain silent on irrelevant aspects of the world state and predict only the attributes it directly governs. This creates a dynamic computation graph that routes both inference and optimization only through relevant laws for each transition, avoiding the scaling challenges that arise when all laws must contribute to predictions about a complex, hierarchical state space, and enabling accurate learning of stochastic dynamics even when most rules are inactive at any given moment. To evaluate our approach under these demanding constraints, we introduce a new evaluation protocol that measures (a) state ranking, the ability to distinguish plausible future states from implausible ones, and (b) state fidelity, the ability to generate future states that closely resemble reality. We develop and evaluate our framework on Crafter-OO, our reimplementation of the popular Crafter environment that exposes a structured, object-oriented symbolic state and a pure transition function that operates on that state alone. OneLife can successfully learn key environment dynamics from minimal, unguided interaction, outperforming a strong baseline on 16 out of 23 scenarios tested. We also demonstrate the world model's utility for planning, where rollouts simulated within the world model successfully identify superior strategies in goal-oriented tasks. Our work establishes a foundation for autonomously constructing programmatic world models of unknown, complex environments.


## Quick Start
### Installation

Clone the repository and initialize submodules:

```bash
git clone --depth 1 git@github.com:codezakh/onelife.git 
cd onelife 
git submodule update --init
```

Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

### Running Tests

Run the test suite from the `onelife` directory:

```bash
cd onelife
uv run --env-file .env pytest tests/
```

You can use this as the `.env` file:
```
GEMINI_API_KEY=
OPENAI_API_KEY=
```

The keys can be left blank, though some tests will fail.

To run a specific test:

```bash
uv run --env-file .env pytest tests/integration/crafter/test_poe_world_fitting_and_eval.py -v
```

### Crafter Environment Integration Tests
The integration tests provide the clearest view of how the main components work together. These tests demonstrate the complete pipeline: generating training data, fitting a world model, and evaluating its performance.

Two main integration tests demonstrate the full system on the Crafter environment:

- **PoE-World** (`tests/integration/crafter/test_poe_world_fitting_and_eval.py`):
- **OneLife** (`tests/integration/crafter/test_our_method_fitting_and_eval.py`):

Both tests follow this pattern:

1. **Data Generation**: Collect transitions `(s, a, s')` from a random policy
2. **Model Fitting**: Learn weights for handwritten experts/laws using maximum likelihood
3. **Evaluation**: Test the model on held-out scenarios using the hybrid evaluation framework
4. **Analysis**: Compare discriminative accuracy, edit distance, and normalized recall metrics

### Simple 1D Environment 
For debugging and understanding the core algorithms, the Simple 1D environment provides a minimal testbed (`tests/integration/simple_1d_env/test_poe_world_fitting_and_eval.py`):

# Crafter-OO
The Crafter-OO code is located at [codezakh/crafter_oo](https://github.com/codezakh/crafter_oo).
You don't have to install it separately, as the installation instructions already handle installing it as a submodule.

# Credits
The methods, code, and components in this repository are heavily inspired by the following projects:
- [PoE-World](https://github.com/topwasu/poe-world)
- [Crafter](https://github.com/danijar/crafter)
- [Balrog](https://github.com/balrog-ai/BALROG)

# Citation
```
@inproceedings{khan2025onelife,
  title={One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration},
  author={Khan, Zaid and Prasad, Archiki and Stengel-Eskin, Elias and Cho, Jaemin and Bansal, Mohit},
  journal={arXiv preprint arXiv:2510.12088},
  year={2025}
}
```
