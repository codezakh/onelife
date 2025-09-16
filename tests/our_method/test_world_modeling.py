from distant_sunburn.our_method.world_modeling import LawMixture
from dataclasses import dataclass
from distant_sunburn.our_method.core import LawFunctionWrapper, WeightedLaw
from distant_sunburn.poe_world.core import ObservableId, DiscreteDistribution
import numpy as np
from dataclasses import field
import torch
from distant_sunburn.our_method.optimization import combine_expert_predictions_for_attr


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
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> State:
        if "attr" in expert_predictions:
            a_preds = expert_predictions[ObservableId("attr")]
            combined_dist = combine_expert_predictions_for_attr(a_preds, weights)
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
