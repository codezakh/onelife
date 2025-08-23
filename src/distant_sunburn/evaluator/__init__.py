"""
Hybrid Evaluation Framework for Symbolic World Models

This module implements the hybrid evaluation framework that combines generative
and discriminative tests to measure both utility (for planning) and scientific
accuracy (probability distribution understanding) of world models.
"""

from .core import (
    EvaluatableWorldModel,
    SymbolicEnvironment,
    TrajectoryCollector,
    EditDistanceCalculator,
    DistractorGenerator,
    SymbolicTransition,
    EvaluationConfig,
    EvaluationResults,
    HybridEvaluator,
)
from .simple_1d_env.adapters import (
    Environment1DAdapter,
    Environment1DWrapper,
)
from .simple_1d_env.components import (
    RandomPolicy1DTrajectoryCollector,
    JSONPatchEditDistance,
    StructuralEditDistance,
    TemporalDistractorGenerator,
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
    "SymbolicEnvironment",
    "TrajectoryCollector",
    "EditDistanceCalculator",
    "DistractorGenerator",
    "SymbolicTransition",
    "EvaluationConfig",
    "EvaluationResults",
    "HybridEvaluator",
    # Environment adapters
    "Environment1DAdapter",
    "Environment1DWrapper",
    # Component implementations
    "RandomPolicy1DTrajectoryCollector",
    "JSONPatchEditDistance",
    "StructuralEditDistance",
    "TemporalDistractorGenerator",
    "Semantic1DDistractorGenerator",
    # Baseline world models
    "TrueTransitionWorldModel",
    "NullWorldModel",
    "RandomWorldModel",
]
