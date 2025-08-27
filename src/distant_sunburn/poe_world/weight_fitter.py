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
    RandomValues,
    SymbolicTransition,
    WeightedExpert,
    WeightFitterProtocol,
    ObservableId,
)


def expand_to_full_domain(
    rv: RandomValues, all_possible_values: np.ndarray, noise_logscore: float = -10.0
) -> RandomValues:
    """
    Expand a RandomValues distribution to cover all possible values in the domain.

    Expert functions often only predict a subset of possible values for an attribute
    (e.g., only predicting position changes when the expert thinks movement occurs).
    This function expands such partial distributions to cover the full domain by
    assigning a low probability (noise_logscore) to values the expert didn't predict.

    This is necessary for proper combination of expert predictions, as all experts
    must have distributions over the same set of possible values to be combined
    via weighted averaging.

    Args:
        rv: The partial RandomValues distribution from an expert
        all_possible_values: Array of all possible values for this attribute
        noise_logscore: Log-probability assigned to values not predicted by the expert

    Returns:
        RandomValues distribution covering the full domain
    """
    new_logscores = np.full_like(all_possible_values, noise_logscore, dtype=np.float32)
    for i, val in enumerate(rv.values):
        if val in all_possible_values:
            idx = np.where(all_possible_values == val)[0][0]
            new_logscores[idx] = rv.logscores[i]
    return RandomValues(values=all_possible_values, logscores=new_logscores)


def combine_expert_predictions_for_attr(
    expert_predictions: list[RandomValues], weights: torch.Tensor
) -> RandomValues:
    """
    Combine multiple expert predictions for a single attribute using learned weights.

    This implements the core Product of Experts (PoE) combination rule: the combined
    log-probability for each value is the weighted sum of individual expert log-probabilities.
    The weights determine how much each expert's opinion contributes to the final prediction.

    This function is used for inference and evaluation where gradient preservation
    is not needed. For weight fitting optimization, use the _torch version instead.

    WARNING: This function breaks PyTorch gradient flow by calling .detach().numpy().
    For weight fitting optimization, use combine_expert_predictions_torch() instead
    to preserve gradients. This function is kept for non-optimization use cases.

    Args:
        expert_predictions: List of RandomValues from each expert for this attribute
        weights: Tensor of expert weights [n_experts] - determines relative importance

    Returns:
        Combined RandomValues distribution representing the ensemble prediction
    """
    if not expert_predictions:
        raise ValueError("No expert predictions provided")

    # Stack logscores from all experts into matrix [n_experts, n_values]
    logscores_matrix = torch.stack(
        [
            torch.tensor(pred.logscores, dtype=torch.float32)
            for pred in expert_predictions
        ]
    )

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    combined_logscores = logscores_matrix.T @ weights

    # Return combined distribution using the same values as the first expert
    # WARNING: .detach().numpy() breaks gradient flow - use _torch version for optimization
    return RandomValues(
        values=expert_predictions[0].values,
        logscores=combined_logscores.detach().numpy(),
    )


def combine_expert_predictions_for_attr_torch(
    expert_predictions: list[RandomValues], weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch-native version of expert prediction combination that preserves gradients.

    This function performs the same PoE combination as combine_expert_predictions_for_attr
    but keeps all operations in PyTorch tensor space to maintain gradient flow for
    backpropagation. This is essential for the weight fitting optimization process,
    where gradients must flow from the loss function back to the expert weights.

    The combination rule is: combined_logscore[value] = sum(weight[i] * expert_logscore[i][value])
    for each expert i and each possible value.

    CRITICAL: This function preserves PyTorch gradient flow by keeping all operations
    in tensor form. Use this version for weight fitting optimization. Never call
    .detach() or .numpy() on the returned tensors during optimization.

    Args:
        expert_predictions: List of RandomValues from each expert for this attribute
        weights: Tensor of expert weights [n_experts] - determines relative importance

    Returns:
        Tuple of (values_tensor, combined_logscores_tensor) preserving gradients
    """
    if not expert_predictions:
        raise ValueError("No expert predictions provided")

    # Stack logscores from all experts into matrix [n_experts, n_values]
    logscores_matrix = torch.stack(
        [
            torch.tensor(pred.logscores, dtype=torch.float32)
            for pred in expert_predictions
        ]
    )

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    combined_logscores = logscores_matrix.T @ weights

    # Also return values as tensor for PyTorch operations
    values_tensor = torch.tensor(expert_predictions[0].values, dtype=torch.int32)

    return values_tensor, combined_logscores


def eval_expert_predictions_logprob_for_attr_torch(
    values_tensor: torch.Tensor, combined_logscores: torch.Tensor, observed_value: int
) -> torch.Tensor:
    """
    Evaluate the log-probability of an observed value under combined expert predictions.

    This function computes the log-probability of a specific observed value (e.g.,
    the actual next position of an object) under the combined distribution from all
    weighted experts. This is the core computation needed for maximum likelihood
    weight fitting - we want to maximize the probability of observed transitions.

    The function first normalizes the combined log-scores to proper log-probabilities
    using the log-sum-exp trick, then extracts the probability for the observed value.
    If the observed value is not in the support of the distribution, it returns -inf.

    CRITICAL: This function preserves PyTorch gradient flow by operating entirely
    in tensor space. The returned tensor maintains gradients with respect to the
    input logscores, enabling backpropagation through expert weight optimization.

    Args:
        values_tensor: Tensor of possible values [n_values] - the support of the distribution
        combined_logscores: Combined logscores tensor [n_values] from weighted experts
        observed_value: The actual observed value to evaluate (e.g., ground truth)

    Returns:
        Log probability tensor of the observed value (preserves gradients)
    """
    # Normalize to log probabilities
    log_probs = combined_logscores - torch.logsumexp(combined_logscores, dim=0)

    # Find the index of the observed value
    mask = values_tensor == observed_value

    if mask.any():
        # Return the log probability for the observed value
        return log_probs[mask][0]  # Take first match
    else:
        # Value not possible under this distribution
        return torch.tensor(-float("inf"), dtype=torch.float32)


SymbolicStateT = TypeVar("SymbolicStateT")


class MaxLikelihoodWeightFitter(Generic[SymbolicStateT]):
    """
    Maximum likelihood weight fitter using PyTorch L-BFGS optimization.

    This implementation follows the design outlined in the PRD:
    - Uses L-BFGS optimization for smooth convergence
    - Constrains weights to [0, 10] range for numerical stability
    - Supports batch sampling for computational efficiency
    - Includes L1 regularization to encourage sparsity
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
    ) -> list[list[dict[ObservableId, RandomValues]]]:
        """
        Precompute expert predictions for all transitions to avoid repeated execution.

        Returns:
            List of expert predictions [n_transitions][n_experts][attribute_name]
        """
        preds_for_all_transitions: list[list[dict[ObservableId, RandomValues]]] = []

        for transition in transitions:
            preds_for_transition: list[dict[ObservableId, RandomValues]] = []

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
        expert_preds_per_transition: list[list[dict[ObservableId, RandomValues]]],
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
            observed_values = self.observable_extractor.get_observed_values(
                transition.next_metadata
            )

            # Compute loss for each attribute
            for attr_name, observed_value in observed_values.items():
                # Get expert predictions for this attribute
                attr_predictions = [pred[attr_name] for pred in transition_predictions]

                # Use PyTorch-native combination to preserve gradients
                values_tensor, combined_logscores = (
                    combine_expert_predictions_for_attr_torch(attr_predictions, weights)
                )

                # Evaluate log-probability using PyTorch operations
                log_prob = eval_expert_predictions_logprob_for_attr_torch(
                    values_tensor, combined_logscores, observed_value
                )

                # Accumulate negative log-likelihood
                total_loss -= log_prob

        return total_loss


implements(WeightFitterProtocol)(MaxLikelihoodWeightFitter)
