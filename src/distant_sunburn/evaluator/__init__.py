"""
Hybrid Evaluation Framework for Symbolic World Models

This module implements the hybrid evaluation framework that combines generative
and discriminative tests to measure both utility (for planning) and scientific
accuracy (probability distribution understanding) of world models.
"""

from .core import (
    EvaluatableWorldModel,
    SymbolicTransitionFunction,
    TrajectoryCollector,
    EditDistanceCalculator,
    DistractorGenerator,
    SymbolicTransition,
    EvaluationConfig,
    EvaluationResults,
    Evaluator,
)
from .simple_1d_env.adapters import (
    Environment1DAdapter,
)
from .simple_1d_env.components import (
    RandomPolicy1DTrajectoryCollector,
    JSONPatchEditDistance,
    Semantic1DDistractorGenerator,
)
from .baselines import (
    TrueTransitionWorldModel,
    NullWorldModel,
    RandomWorldModel,
)

__all__ = [
    # Core interfaces and classes
    "EvaluatableWorldModel",
    "SymbolicTransitionFunction",
    "TrajectoryCollector",
    "EditDistanceCalculator",
    "DistractorGenerator",
    "SymbolicTransition",
    "EvaluationConfig",
    "EvaluationResults",
    "Evaluator",
    # Environment adapters
    "Environment1DAdapter",
    # Component implementations
    "RandomPolicy1DTrajectoryCollector",
    "JSONPatchEditDistance",
    "Semantic1DDistractorGenerator",
    # Baseline world models
    "TrueTransitionWorldModel",
    "NullWorldModel",
    "RandomWorldModel",
]
