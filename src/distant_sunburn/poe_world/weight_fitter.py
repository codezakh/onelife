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
from tqdm.auto import tqdm


# Global floor for logscores to prevent -inf/NaN from propagating through PoE
LOGSCORE_FLOOR = -50.0  # ~1.9e-22 probability


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
        use_parallel_loss: bool = False,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.weight_bounds = weight_bounds
        self.use_parallel_loss = use_parallel_loss

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
            List of weighted experts with learned weights. The returned list maintains
            the same order as the input experts list - experts[i] corresponds to
            returned_weighted_experts[i].

        Note:
            All returned WeightedExpert instances have is_fitted=True to indicate
            they have been fitted with learned weights.
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
            [weights],
            lr=self.learning_rate,
            line_search_fn="strong_wolfe",
            max_iter=5,
        )

        # Optionally precompute loss buckets once per optimization run to remove
        # Python overhead from the closure. Buckets are independent of weights.
        precomputed_buckets = None
        if self.use_parallel_loss:
            precomputed_buckets = self.build_loss_buckets(
                sampled_transitions,
                sampled_predictions,
                device=weights.device,
            )

        def closure():
            optimizer.zero_grad()

            # Clamp weights to bounds
            with torch.no_grad():
                weights.clamp_(self.weight_bounds[0], self.weight_bounds[1])

            # Compute negative log-likelihood loss
            if self.use_parallel_loss:
                # Vectorized, bucketed loss using precomputed tensors
                assert precomputed_buckets is not None
                loss = self.compute_buckets_loss(weights, precomputed_buckets)
            else:
                # Baseline scalar implementation
                loss = self._compute_loss(
                    weights, sampled_transitions, sampled_predictions
                )

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
        weighted_experts = []

        for i, (expert, weight) in enumerate(zip(experts, final_weights)):
            weighted_experts.append(
                WeightedExpert(
                    expert_function=expert,
                    weight=float(weight),
                    is_fitted=True,
                )
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

        for transition in tqdm(transitions, desc="Precomputing expert predictions"):
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

                    # Debug instrumentation: disabled by default
                    if False:
                        self._debug_attr_loss_diagnostics(
                            attr_name=attr_name,
                            observed_value=observed_value,
                            attr_predictions=attr_predictions,
                            weights=weights,
                        )

                    log_prob = compute_log_prob_for_attr_from_expert_predictions_torch(
                        attr_predictions, weights, observed_value
                    )

                    # Accumulate negative log-likelihood
                    total_loss -= log_prob
                except KeyError:
                    # Skip loss computation for attributes that experts didn't predict
                    # This typically happens when entities spawn/despawn between states
                    continue

        return total_loss

    def _debug_attr_loss_diagnostics(
        self,
        attr_name: ObservableId,
        observed_value: int,
        attr_predictions: list[DiscreteDistribution],
        weights: torch.Tensor,
    ) -> None:
        """
        Emit targeted diagnostics for a single attribute's loss computation.

        Checks:
        - Support equality across experts
        - Observed value membership in support
        - Non-finite logscores in any expert prediction
        - Finiteness of combined raw logscore at the observed index
        """
        # 1) Check supports are identical across experts
        same_support = all(
            np.array_equal(attr_predictions[0].support, p.support)
            for p in attr_predictions[1:]
        )
        if not same_support:
            with logger.contextualize(attribute=str(attr_name)):
                logger.warning("Support mismatch across experts")

        # 2) Check observed value is in support
        try:
            observed_int = int(observed_value)
        except Exception:
            observed_int = observed_value
        support_set = set(attr_predictions[0].support.tolist())
        if observed_int not in support_set:
            with logger.contextualize(attribute=str(attr_name), observed=observed_int):
                logger.warning("Observed outcome not in support")

        # 3) Check for non-finite logscores
        bad_experts = [
            idx
            for idx, p in enumerate(attr_predictions)
            if not np.isfinite(p.logscores).all()
        ]
        if bad_experts:
            with logger.contextualize(attribute=str(attr_name), experts=bad_experts):
                logger.warning("Non-finite logscores present")

        # 4) Check combined raw logscore at observed index
        values_tensor, combined_logscores = combine_expert_predictions_for_attr_torch(
            attr_predictions, weights
        )
        mask = values_tensor == observed_int
        if mask.any():
            combined_obs = combined_logscores[mask][0]
            if not torch.isfinite(combined_obs):
                with logger.contextualize(attribute=str(attr_name)):
                    logger.warning("Combined raw logscore at observed index is -inf")

    # ============================= Parallelized Path =============================
    # The following methods provide a vectorized, extractor-agnostic implementation
    # of the loss computation that supports dynamic attributes and supports without
    # global padding. They can be used directly or enabled via the constructor flag
    # `use_parallel_loss`.

    def build_loss_buckets(
        self,
        transitions: list[SymbolicTransition[SymbolicStateT]],
        expert_preds_per_transition: list[
            list[dict[ObservableId, DiscreteDistribution]]
        ],
        device: torch.device | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Build per-support buckets of loss inputs for vectorized computation.

        This routine flattens the (transition, attribute) pairs into "items" and groups
        items by their unified support so that we can batch them without padding. The
        grouping is extractor-agnostic and uses only `DiscreteDistribution` and the
        protocol methods of the `ObservableExtractor` passed at construction.

        Dataflow (per item):
        1) Take observed outcome for attribute from `next_metadata`.
        2) Collect each expert's predicted `DiscreteDistribution` for that attribute for
           the specific transition. If any expert is missing the attribute, skip item.
        3) Compute the union of supports across experts (1D int array of length V).
        4) Expand every expert distribution to this union using noise of LOGSCORE_FLOOR.
        5) Record the observed index `obs_idx` in the union support if present; otherwise
           mark as not present.

        Bucketing: Items with the same union support array (same content and length)
        are grouped into a bucket. For each bucket we produce tensors:
        - logscores: float32 tensor of shape [B, E, V]
            B: number of items in the bucket
            E: number of experts
            V: number of values in this bucket's support
            Entry [b, e, v] is the logscore expert e assigns to support[v] for item b.
            Values are sanitized and clamped at LOGSCORE_FLOOR.
        - obs_idx: int64 tensor of shape [B]
            Index into the last dimension (V) of `logscores`/combined logits for the
            observed outcome for item b. When observed outcome is not present in the
            union support, this index is set to 0 (unused placeholder) and a mask is used.
        - obs_present: bool tensor of shape [B]
            True if the observed outcome is in the union support, False otherwise.
        - support: int32 tensor of shape [V]
            The actual support values for this bucket. Not used in the hot path; kept
            for diagnostics and symmetry with the scalar path.

        Args:
            transitions: Training transitions in the sampled batch.
            expert_preds_per_transition: Precomputed expert predictions matching
                `transitions`: list over transitions → list over experts → dict
                mapping ObservableId → DiscreteDistribution.
            device: Optional torch device to place resulting tensors on. If None,
                uses CPU. For GPU execution, pass `weights.device`.

        Returns:
            A list of bucket dictionaries with keys: `logscores`, `obs_idx`,
            `obs_present`, `support`.
        """
        if device is None:
            device = torch.device("cpu")

        # Internal accumulators keyed by support signature
        # Signature uses bytes of the support array to avoid padding across families
        buckets_acc: dict[tuple[int, bytes], dict[str, list]] = {}
        sig_supports: dict[tuple[int, bytes], np.ndarray] = {}

        for i, transition in enumerate(transitions):
            transition_predictions = expert_preds_per_transition[i]

            # Observed values from next state
            observed_values = self.observable_extractor.get_observed_outcomes(
                transition.next_metadata
            )

            for attr_name, observed_value in observed_values.items():
                # Gather each expert's prediction for this attribute. If any is missing,
                # skip this item (matches baseline behavior).
                try:
                    per_expert_preds = [
                        pred[attr_name] for pred in transition_predictions
                    ]
                except KeyError:
                    continue

                # 1) Compute union of supports across all experts
                supports = [p.support for p in per_expert_preds]
                union_support = np.unique(np.concatenate(supports)).astype(np.int32)

                # 2) Expand each expert's distribution to the union support
                expanded_logscores = []  # will become shape [E, V]
                for p in per_expert_preds:
                    expanded = p.expand_support(
                        union_support, noise_logscore=LOGSCORE_FLOOR
                    )
                    expanded_logscores.append(expanded.logscores.astype(np.float32))

                logscores_e_v = np.stack(expanded_logscores, axis=0)  # [E, V]

                # 3) Determine observed index and presence
                try:
                    observed_int = int(observed_value)
                except Exception:
                    observed_int = observed_value
                where = np.where(union_support == observed_int)[0]
                if len(where) > 0:
                    obs_idx = int(where[0])
                    obs_present = True
                else:
                    obs_idx = 0  # placeholder, will be ignored via mask
                    obs_present = False

                # 4) Bucket by support signature
                sig = (len(union_support), union_support.tobytes())
                if sig not in buckets_acc:
                    buckets_acc[sig] = {
                        "logscores": [],
                        "obs_idx": [],
                        "obs_present": [],
                    }
                    sig_supports[sig] = union_support

                b = buckets_acc[sig]
                b["logscores"].append(logscores_e_v)
                b["obs_idx"].append(obs_idx)
                b["obs_present"].append(obs_present)

        # Materialize torch tensors per bucket
        buckets: list[dict[str, torch.Tensor]] = []
        for sig, data in buckets_acc.items():
            logscores_np = np.stack(data["logscores"], axis=0)  # [B, E, V]
            # Sanitize and clamp
            logscores = torch.tensor(logscores_np, dtype=torch.float32, device=device)
            logscores = torch.where(
                torch.isfinite(logscores),
                logscores,
                torch.full_like(logscores, LOGSCORE_FLOOR),
            )
            logscores = torch.clamp(logscores, min=LOGSCORE_FLOOR)

            obs_idx = torch.tensor(data["obs_idx"], dtype=torch.long, device=device)
            obs_present = torch.tensor(
                data["obs_present"], dtype=torch.bool, device=device
            )
            support = torch.tensor(sig_supports[sig], dtype=torch.int32, device=device)

            buckets.append(
                {
                    "logscores": logscores,  # [B, E, V]
                    "obs_idx": obs_idx,  # [B]
                    "obs_present": obs_present,  # [B]
                    "support": support,  # [V]
                }
            )

        return buckets

    def compute_buckets_loss(
        self,
        weights: torch.Tensor,
        buckets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute total negative log-likelihood using pre-built buckets.

        Shapes:
        - weights: [E]
        - For each bucket b:
            - logscores_b: [B, E, V]
            - obs_idx_b:   [B]
            - obs_present_b: [B] (bool)
            - support_b:   [V] (not used in math)

        Computation per bucket:
            combined = einsum('bev,e->bv', logscores_b, weights)  # [B, V]
            logZ     = logsumexp(combined, dim=1)                 # [B]
            obs_logit = combined.gather(1, safe_idx[:, None]).squeeze(1)  # [B]
            obs_logprob = where(obs_present, obs_logit - logZ, -inf)      # [B]
            nll_b = -sum(obs_logprob)

        Returns:
            Scalar torch.float32 tensor representing the total negative log-likelihood.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=weights.device)

        for bucket in buckets:
            logscores = bucket["logscores"]  # [B, E, V]
            obs_idx = bucket["obs_idx"]  # [B]
            obs_present = bucket["obs_present"]  # [B]

            # Ensure tensors are on the same device as weights
            if logscores.device != weights.device:
                logscores = logscores.to(weights.device)
                obs_idx = obs_idx.to(weights.device)
                obs_present = obs_present.to(weights.device)

            # [B, V]
            combined = torch.einsum("bev,e->bv", logscores, weights)
            logZ = torch.logsumexp(combined, dim=1)

            # Gather observed logits. For items with missing observed values in support,
            # we will ignore the gathered value using the mask and set logprob to -inf.
            safe_idx = torch.clamp(obs_idx, min=0)
            obs_logits = combined.gather(1, safe_idx[:, None]).squeeze(1)

            obs_logprob = obs_logits - logZ
            neg_inf = torch.tensor(
                -float("inf"), dtype=obs_logprob.dtype, device=obs_logprob.device
            )
            obs_logprob = torch.where(obs_present, obs_logprob, neg_inf)

            total_loss = total_loss - torch.sum(obs_logprob)

        return total_loss


implements(WeightFitterProtocol)(MaxLikelihoodWeightFitter)
