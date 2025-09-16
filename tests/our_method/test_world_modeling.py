from dataclasses import dataclass, field

import numpy as np
import torch

from distant_sunburn.our_method.core import LawFunctionWrapper, WeightedLaw
from distant_sunburn.our_method.optimization import combine_expert_predictions_for_attr
from distant_sunburn.our_method.world_modeling import LawMixture
from distant_sunburn.poe_world.core import DiscreteDistribution, ObservableId
from typing import TypeAlias, Mapping


ExpertIndex: TypeAlias = int


@dataclass
class State:
    attr: int


class AlwaysOnHasEffectLaw:
    def precondition(self, current_state: State, action: str) -> bool:
        return True

    def effect(self, current_state: State, action: str) -> None:
        current_state.attr = DiscreteDistribution(support=[current_state.attr + 1])  # type: ignore


class AlwaysOnNoEffectLaw:
    def precondition(self, current_state: State, action: str) -> bool:
        return True

    def effect(self, current_state: State, action: str) -> None:
        pass


@dataclass
class ObservableExtractor:
    attr_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 10))

    def extract_attribute_predictions(
        self, state: State
    ) -> dict[ObservableId, DiscreteDistribution]:
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        if isinstance(state.attr, DiscreteDistribution):
            predictions[ObservableId("attr")] = state.attr.expand_support(
                self.attr_domain
            )
        return predictions

    def get_observed_outcomes(self, state: State) -> dict[ObservableId, int]:
        return {
            ObservableId("attr"): state.attr,
        }

    def apply_expert_predictions(
        self,
        new_state: State,
        expert_predictions: Mapping[
            ObservableId, Mapping[ExpertIndex, DiscreteDistribution]
        ],
        weights: torch.Tensor,
    ) -> State:
        if "attr" in expert_predictions:
            law_id_to_preds = expert_predictions[ObservableId("attr")]
            active_indices = list(law_id_to_preds.keys())
            weights = weights[active_indices]
            raw_preds = list(law_id_to_preds.values())
            combined_dist = combine_expert_predictions_for_attr(raw_preds, weights)
            new_state.attr = combined_dist.sample()

        return new_state


def test_sample_next_state_when_always_on_law_no_effect():
    """
    When there is a law that is always on but has no effect,
    we should still be able to sample a next state.
    This used to be broken and the test exists to ensure it doesn't break again.
    """

    mixture = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=[
            WeightedLaw(
                law=LawFunctionWrapper.from_non_runtime_created(AlwaysOnHasEffectLaw()),
                weight=1.0,
                is_fitted=True,
            ),
            WeightedLaw(
                law=LawFunctionWrapper.from_non_runtime_created(AlwaysOnNoEffectLaw()),
                weight=1.0,
                is_fitted=True,
            ),
        ],
    )

    state = State(attr=0)
    mixture.sample_next_state(state, "action")


def test_evaluate_log_probability_when_always_on_law_no_effect():
    """
    When there is a law that is always on but has no effect,
    we should still be able to evaluate the log probability of a transition.
    This used to be broken and the test exists to ensure it doesn't break again.
    """

    mixture = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=[
            WeightedLaw(
                law=LawFunctionWrapper.from_non_runtime_created(AlwaysOnHasEffectLaw()),
                weight=1.0,
                is_fitted=True,
            ),
            WeightedLaw(
                law=LawFunctionWrapper.from_non_runtime_created(AlwaysOnNoEffectLaw()),
                weight=1.0,
                is_fitted=True,
            ),
        ],
    )

    state = State(attr=0)
    log_prob = mixture.evaluate_log_probability(state, "action", State(attr=1))
    # Log probability should be close to 0
    assert np.isclose(log_prob, 0.0, rtol=0, atol=1e-3)
