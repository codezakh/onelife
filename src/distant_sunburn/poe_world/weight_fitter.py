"""
Maximum likelihood weight fitter for PoE-World experts.

This module implements the core weight fitting logic using PyTorch optimization
to learn expert weights that maximize the log-likelihood of observed transitions.

The current implementation works specifically for the simple 1D environment.
"""

import copy
from typing import Generic, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from ..typing_utils import implements
from .core import (
    ExpertFunction,
    ObservableExtractorProtocol,
    DiscreteDistribution,
    SymbolicTransition,
    WeightedExpert,
    WeightFitterProtocol,
    ObservableId,
)


def combine_expert_predictions_for_attr(
    expert_predictions: list[DiscreteDistribution], weights: torch.Tensor
) -> DiscreteDistribution:
    """
    Combine expert predictions for a single attribute using learned weights.

    Implements Product of Experts (PoE) combination: weighted sum of expert log-probabilities.
    Used for inference/evaluation. For optimization, use the _torch version.

    WARNING: Breaks gradient flow with .detach().numpy().

    Args:
        expert_predictions: List of RandomValues from each expert for this attribute.
            Each RandomValues contains predictions for the same set of possible values.
        weights: Tensor of expert weights [n_experts] with dtype=torch.float32.
            weights[i] determines how much expert i's prediction contributes.

    Returns:
        Combined RandomValues distribution representing the ensemble prediction.
    """
    if not expert_predictions:
        raise ValueError("No expert predictions provided")

    # Stack logscores from all experts into matrix [n_experts, n_values]
    # Each row is one expert's predictions for all possible values
    logscores_matrix = torch.stack(
        [
            torch.tensor(pred.logscores, dtype=torch.float32)
            for pred in expert_predictions
        ]
    )

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    # This computes: combined_logscores[value] = sum(weight[i] * expert_logscore[i][value])
    combined_logscores = logscores_matrix.T @ weights

    # Return combined distribution using the same values as the first expert
    # WARNING: .detach().numpy() breaks gradient flow - use _torch version for optimization
    return DiscreteDistribution(
        support=expert_predictions[0].support,
        logscores=combined_logscores.detach().numpy(),
    )


def combine_expert_predictions_for_attr_torch(
    expert_predictions: list[DiscreteDistribution], weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-native version of expert prediction combination that preserves gradients.

    Performs same PoE combination but keeps operations in tensor space for gradient flow.
    Essential for weight fitting optimization.

    Args:
        expert_predictions: List of RandomValues from each expert for this attribute.
            Each RandomValues contains predictions for the same set of possible values.
        weights: Tensor of expert weights [n_experts] with dtype=torch.float32.
            weights[i] determines how much expert i's prediction contributes.

    Returns:
        Tuple of:
        - support_tensor: Tensor of possible values [n_values] with dtype=torch.int32.
          These are the possible values for this attribute (e.g., possible positions).
        - weighted_logscores: Tensor of weighted log-scores [n_values] with dtype=torch.float32.
          weighted_logscores[i] is the log-score for support_tensor[i] after weighting.
          These are NOT normalized log-probabilities.
    """
    if not expert_predictions:
        raise ValueError("No expert predictions provided")

    # Stack logscores from all experts into matrix [n_experts, n_values]
    # Each row is one expert's predictions for all possible values
    logscores_matrix = torch.stack(
        [
            torch.tensor(pred.logscores, dtype=torch.float32)
            for pred in expert_predictions
        ]
    )

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    # This computes: combined_logscores[value] = sum(weight[i] * expert_logscore[i][value])
    combined_logscores = logscores_matrix.T @ weights

    # Also return values as tensor for PyTorch operations
    values_tensor = torch.tensor(expert_predictions[0].support, dtype=torch.int32)

    return values_tensor, combined_logscores


def eval_expert_predictions_logprob_for_attr_torch(
    support_tensor: torch.Tensor,
    raw_logscores: torch.Tensor,
    observed_outcome: int,
) -> torch.Tensor:
    """
    Evaluate log-probability of observed outcome under combined expert predictions.

    Normalizes raw log-scores to log-probabilities using log-sum-exp, then extracts
    probability for the observed outcome. Core computation for maximum likelihood fitting.

    Args:
        support_tensor: Tensor of possible values [n_values] with dtype=torch.int32.
            The support of the distribution - all possible values this attribute can take.
            support_tensor[i] corresponds to raw_logscores[i].
        raw_logscores: Tensor of raw log-scores [n_values] with dtype=torch.float32.
            Raw log-scores from weighted expert combination (NOT normalized probabilities).
        observed_outcome: The actual observed value to evaluate (e.g., ground truth).
            Must be one of the values in support_tensor for finite log-probability.

    Returns:
        Log probability tensor [1] with dtype=torch.float32.
        The log-probability of the observed_outcome under the normalized distribution.
        Returns -inf if observed_outcome is not in support_tensor.
    """
    # Normalize to log probabilities using log-sum-exp trick for numerical stability
    # log_probs[i] = raw_logscores[i] - log(sum(exp(raw_logscores)))
    log_probs = raw_logscores - torch.logsumexp(raw_logscores, dim=0)

    # Find the index of the observed value
    mask = support_tensor == observed_outcome

    if mask.any():
        # Return the log probability for the observed value
        return log_probs[mask][0]  # Take first match
    else:
        # Value not possible under this distribution
        return torch.tensor(-float("inf"), dtype=torch.float32)


def compute_log_prob_for_attr_from_expert_predictions_torch(
    expert_predictions: list[DiscreteDistribution],
    weights: torch.Tensor,
    observed_outcome: int,
) -> torch.Tensor:
    """
    Compute the log probability of an observed outcome under expert predictions.

    Each of the expert predictions is expected to be a discrete distribution over the
    same set of possible values. The weights are used to combine the expert predictions
    into a single distribution. The log probability of the observed outcome under this
    combined distribution is returned.

    Args:
        expert_predictions: List of expert predictions for this attribute.
            Each DiscreteDistribution should have the same support (same possible values).
            Length: n_experts
        weights: Tensor of expert weights [n_experts] with dtype=torch.float32.
            weights[i] determines how much expert i's prediction contributes.
        observed_outcome: The actual observed value to evaluate (e.g., ground truth).
            Must be one of the values in support_tensor for finite log-probability.
            Scalar integer value.

    Returns:
        Log probability tensor [1] with dtype=torch.float32.
        The log-probability of the observed_outcome under the normalized distribution.
        Returns -inf if observed_outcome is not in support_tensor.
    """
    # Combine expert predictions into a single distribution
    values_tensor, combined_logscores = combine_expert_predictions_for_attr_torch(
        expert_predictions, weights
    )

    # Evaluate log-probability of observed outcome under combined distribution
    return eval_expert_predictions_logprob_for_attr_torch(
        values_tensor, combined_logscores, observed_outcome
    )


SymbolicStateT = TypeVar("SymbolicStateT")


class MaxLikelihoodWeightFitter(Generic[SymbolicStateT]):
    """
    Learns weights for a set of expert functions that model an environment.

    Args:
        observable_extractor: Extracts observable attributes from the state. These are
        the attributes that expert functions may predict and will be used to compute the
        loss and find the weights that maximize the likelihood of the set of loss functions.
        learning_rate: Learning rate for the optimizer.
        max_iterations: Maximum number of iterations to run the optimizer.
        batch_size: Batch size for the optimizer.
        l1_weight: Weight for the L1 regularization term.
        weight_bounds: Bounds for the weights.
    """

    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        batch_size: int = 1000,
        l1_weight: float = 0.001,
        weight_bounds: Tuple[float, float] = (0.0, 10.0),
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.weight_bounds = weight_bounds

        self.observable_extractor = observable_extractor

    def fit(
        self,
        experts: list[ExpertFunction[SymbolicStateT]],
        transitions: list[SymbolicTransition[SymbolicStateT]],
    ) -> list[WeightedExpert]:
        """
        Fit expert weights using maximum likelihood estimation.

        Args:
            experts: List of expert functions to fit weights for
            transitions: Training data as symbolic transitions

        Returns:
            List of weighted experts with learned weights
        """
        if not experts or not transitions:
            return []

        logger.info(
            f"Fitting weights for {len(experts)} experts on {len(transitions)} transitions"
        )

        # Precompute expert predictions for all transitions
        expert_predictions = self._precompute_expert_predictions(experts, transitions)

        # Sample batch if dataset is large
        if len(transitions) > self.batch_size:
            indices = np.random.choice(len(transitions), self.batch_size, replace=False)
            sampled_transitions = [transitions[i] for i in indices]
            sampled_predictions = [expert_predictions[i] for i in indices]
        else:
            sampled_transitions = transitions
            sampled_predictions = expert_predictions

        # Initialize weights
        weights = nn.Parameter(torch.ones(len(experts), dtype=torch.float32) * 0.5)

        # Set up L-BFGS optimizer
        optimizer = optim.LBFGS(
            [weights], lr=self.learning_rate, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()

            # Clamp weights to bounds
            with torch.no_grad():
                weights.clamp_(self.weight_bounds[0], self.weight_bounds[1])

            # Compute negative log-likelihood loss
            loss = self._compute_loss(weights, sampled_transitions, sampled_predictions)

            # Add L1 regularization
            l1_penalty = self.l1_weight * torch.abs(weights).sum()
            total_loss = loss + l1_penalty

            total_loss.backward()
            return total_loss

        # Run optimization
        for iteration in range(self.max_iterations):
            loss = optimizer.step(closure)
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}, Loss: {loss.item():.6f}")

        # Create weighted experts with final weights
        final_weights = weights.detach().numpy()
        weighted_experts = []

        for i, (expert, weight) in enumerate(zip(experts, final_weights)):
            weighted_experts.append(
                WeightedExpert(expert_function=expert, weight=float(weight))
            )
            logger.debug(f"Expert {i}: weight = {weight:.4f}")

        return weighted_experts

    def _precompute_expert_predictions(
        self,
        experts: list[ExpertFunction[SymbolicStateT]],
        transitions: list[SymbolicTransition[SymbolicStateT]],
    ) -> list[list[dict[ObservableId, DiscreteDistribution]]]:
        """
        Precompute expert predictions for all transitions to avoid repeated execution.

        Returns:
            List of expert predictions [n_transitions][n_experts][attribute_name]
        """
        preds_for_all_transitions: list[
            list[dict[ObservableId, DiscreteDistribution]]
        ] = []

        for transition in transitions:
            preds_for_transition: list[dict[ObservableId, DiscreteDistribution]] = []

            # Each expert make a prediction for all observable attributes
            for expert in experts:
                # Deep copy state and run expert
                state_copy = copy.deepcopy(transition.prev_metadata)
                expert(state_copy, transition.action)

                # Extract predictions for each attribute
                preds_from_expert = (
                    self.observable_extractor.extract_attribute_predictions(state_copy)
                )
                preds_for_transition.append(preds_from_expert)

            preds_for_all_transitions.append(preds_for_transition)

        return preds_for_all_transitions

    def _compute_loss(
        self,
        weights: torch.Tensor,
        transitions: list[SymbolicTransition[SymbolicStateT]],
        expert_preds_per_transition: list[
            list[dict[ObservableId, DiscreteDistribution]]
        ],
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss for the given weights.

        CRITICAL: This function preserves PyTorch gradient flow by using the _torch
        versions of combination and evaluation functions. Breaking the gradient flow
        here (e.g., by calling .detach() or .numpy()) will prevent weight learning.

        The loss is computed per-attribute, per-object, per-transition as described
        in the supplementary material.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        for i, transition in enumerate(transitions):
            transition_predictions = expert_preds_per_transition[i]

            # Get observed values from next state
            observed_values = self.observable_extractor.get_observed_outcomes(
                transition.next_metadata
            )

            # Compute loss for each attribute
            for attr_name, observed_value in observed_values.items():
                # Get expert predictions for this attribute
                # Design choice: Wrap in try-except to handle cases where experts don't predict
                # certain attributes (e.g., when entities spawn/despawn). We assume missing
                # predictions are due to entity lifecycle changes and skip the loss computation.
                try:
                    attr_predictions = [
                        pred[attr_name] for pred in transition_predictions
                    ]

                    log_prob = compute_log_prob_for_attr_from_expert_predictions_torch(
                        attr_predictions, weights, observed_value
                    )

                    # Accumulate negative log-likelihood
                    total_loss -= log_prob
                except KeyError:
                    # Skip loss computation for attributes that experts didn't predict
                    # This typically happens when entities spawn/despawn between states
                    logger.debug(
                        f"Skipping loss computation for {attr_name} - missing expert predictions"
                    )
                    continue

        return total_loss


implements(WeightFitterProtocol)(MaxLikelihoodWeightFitter)
