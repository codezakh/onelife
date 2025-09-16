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
from ..poe_world.core import (
    ObservableExtractorProtocol,
    DiscreteDistribution,
    ObservableId,
)
from ..our_method.core import (
    LawProtocol,
    WeightedLaw,
    SymbolicTransition,
    LawOptimizerProtocol,
)
from tqdm.auto import tqdm
from typing import Sequence
from dataclasses import dataclass


# Global floor for logscores to prevent -inf/NaN from propagating through PoE
LOGSCORE_FLOOR = -50.0  # ~1.9e-22 probability


@dataclass
class LawApplicationOutcome:
    precondition_met: bool
    observed_effects: dict[ObservableId, DiscreteDistribution] | None = None


def combine_expert_predictions_for_attr(
    expert_predictions: Sequence[DiscreteDistribution], weights: torch.Tensor
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
    # Sanitize and clamp to avoid -inf/NaN dominating the combination
    logscores_matrix = torch.where(
        torch.isfinite(logscores_matrix),
        logscores_matrix,
        torch.full_like(logscores_matrix, LOGSCORE_FLOOR),
    )
    logscores_matrix = torch.clamp(logscores_matrix, min=LOGSCORE_FLOOR)

    # Matrix multiplication: [n_values, n_experts] @ [n_experts] = [n_values]
    # This computes: combined_logscores[value] = sum(weight[i] * expert_logscore[i][value])
    try:
        combined_logscores = logscores_matrix.T @ weights
    except RuntimeError:
        logger.opt(exception=True).error(
            "Could not aggregate logscores",
            extra={
                "logscores_matrix": logscores_matrix.shape,
                "weights": weights.shape,
                "expert_predictions": [
                    pred.logscores.shape for pred in expert_predictions
                ],
            },
        )
        raise

    # Return combined distribution using the same values as the first expert
    # WARNING: .detach().numpy() breaks gradient flow - use _torch version for optimization
    return DiscreteDistribution(
        support=expert_predictions[0].support,
        logscores=combined_logscores.detach().numpy(),
    )


def combine_expert_predictions_for_attr_torch(
    expert_predictions: Sequence[DiscreteDistribution], weights: torch.Tensor
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
    # Sanitize and clamp to avoid -inf/NaN dominating the combination
    logscores_matrix = torch.where(
        torch.isfinite(logscores_matrix),
        logscores_matrix,
        torch.full_like(logscores_matrix, LOGSCORE_FLOOR),
    )
    logscores_matrix = torch.clamp(logscores_matrix, min=LOGSCORE_FLOOR)

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
    expert_predictions: Sequence[DiscreteDistribution],
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
        laws: Sequence[LawProtocol[SymbolicStateT]],
        transitions: Sequence[SymbolicTransition[SymbolicStateT]],
    ) -> list[WeightedLaw[SymbolicStateT]]:
        """
        Fit law weights using maximum likelihood estimation.

        Args:
            laws: List of laws to fit weights for
            transitions: Training data as symbolic transitions

        Returns:
            List of weighted laws with learned weights. The returned list maintains
            the same order as the input laws list - laws[i] corresponds to
            returned_weighted_laws[i].

        Note:
            All returned WeightedLaw instances have is_fitted=True to indicate
            they have been fitted with learned weights.
        """
        if not laws or not transitions:
            return []

        logger.info(
            f"Fitting weights for {len(laws)} experts on {len(transitions)} transitions"
        )

        # Precompute expert predictions for all transitions
        law_predictions = self._precompute_law_predictions(laws, transitions)

        # Sample batch if dataset is large
        if len(transitions) > self.batch_size:
            indices = np.random.choice(len(transitions), self.batch_size, replace=False)
            sampled_transitions = [transitions[i] for i in indices]
            sampled_predictions = [law_predictions[i] for i in indices]
        else:
            sampled_transitions = transitions
            sampled_predictions = law_predictions

        # Initialize weights
        weights = nn.Parameter(torch.ones(len(laws), dtype=torch.float32) * 0.5)

        # Set up L-BFGS optimizer
        optimizer = optim.LBFGS(
            [weights],
            lr=self.learning_rate,
            line_search_fn="strong_wolfe",
            max_iter=5,
        )

        def closure():
            optimizer.zero_grad()

            # Clamp weights to bounds
            with torch.no_grad():
                weights.clamp_(self.weight_bounds[0], self.weight_bounds[1])

            # Baseline scalar implementation
            loss = self._compute_loss(weights, sampled_transitions, sampled_predictions)

            logger.info(f"Loss: {loss.item():.6f}")

            # Add L1 regularization
            l1_penalty = self.l1_weight * torch.abs(weights).sum()
            total_loss = loss + l1_penalty

            total_loss.backward()
            return total_loss

        # Run optimization
        for iteration in tqdm(
            range(self.max_iterations),
            desc="Fitting weights",
            total=self.max_iterations,
        ):
            loss = optimizer.step(closure)
            logger.info(f"Iteration {iteration}, Loss: {loss.item():.6f}")

        # Ensure bounds are enforced on final weights before returning
        with torch.no_grad():
            weights.clamp_(self.weight_bounds[0], self.weight_bounds[1])

        # Create weighted experts with final weights
        final_weights = weights.detach().numpy()
        weighted_laws = []

        for i, (law, weight) in enumerate(zip(laws, final_weights)):
            weighted_laws.append(
                WeightedLaw(
                    law=law,
                    weight=float(weight),
                    is_fitted=True,
                )
            )
            logger.debug(f"Law {i}: weight = {weight:.4f}")

        return weighted_laws

    def _precompute_law_predictions(
        self,
        laws: Sequence[LawProtocol[SymbolicStateT]],
        transitions: Sequence[SymbolicTransition[SymbolicStateT]],
    ) -> list[list[LawApplicationOutcome]]:
        """
        Precompute expert predictions for all transitions to avoid repeated execution.

        Returns:
            List of expert predictions [n_transitions][n_experts][attribute_name]
        """
        preds_for_all_transitions: list[list[LawApplicationOutcome]] = []

        for transition in tqdm(transitions, desc="Precomputing expert predictions"):
            preds_for_transition: list[LawApplicationOutcome] = []

            for law in laws:
                state_copy = copy.deepcopy(transition.prev_state)
                if not law.precondition(state_copy, transition.action):
                    preds_for_transition.append(
                        LawApplicationOutcome(precondition_met=False)
                    )
                    continue

                law.effect(state_copy, transition.action)

                # Extract predictions for each attribute
                preds_from_law = (
                    self.observable_extractor.extract_attribute_predictions(state_copy)
                )
                preds_for_transition.append(
                    LawApplicationOutcome(
                        precondition_met=True, observed_effects=preds_from_law
                    )
                )

            preds_for_all_transitions.append(preds_for_transition)

        return preds_for_all_transitions

    def _compute_loss(
        self,
        weights: torch.Tensor,
        transitions: Sequence[SymbolicTransition[SymbolicStateT]],
        expert_preds_per_transition: Sequence[Sequence[LawApplicationOutcome]],
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
                transition.next_state
            )

            # Compute loss for each attribute
            for attr_name, observed_value in observed_values.items():
                attr_predictions: list[DiscreteDistribution] = []

                active_laws: list[int] = []

                for law_idx, law_prediction in enumerate(transition_predictions):
                    if not law_prediction.precondition_met:
                        # This law did not make a prediction for this transitions
                        continue
                    try:
                        assert law_prediction.observed_effects is not None
                        attr_predictions.append(
                            law_prediction.observed_effects[attr_name]
                        )
                    except KeyError:
                        # This law did not make a prediction for this attribute
                        continue
                    else:
                        active_laws.append(law_idx)

                if not active_laws:
                    # This transition did not have any laws that made a prediction for this attribute
                    continue

                # Gather the weights for the active laws
                active_weights = weights[active_laws]

                log_prob = compute_log_prob_for_attr_from_expert_predictions_torch(
                    attr_predictions, active_weights, observed_value
                )

                # Accumulate negative log-likelihood
                total_loss -= log_prob

        return total_loss


implements(LawOptimizerProtocol)(MaxLikelihoodWeightFitter)
