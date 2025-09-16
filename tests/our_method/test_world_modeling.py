from distant_sunburn.our_method.world_modeling import LawMixture
from dataclasses import dataclass
from distant_sunburn.our_method.core import LawFunctionWrapper, WeightedLaw
from distant_sunburn.poe_world.core import ObservableId, DiscreteDistribution
import numpy as np
from dataclasses import field
import torch
from distant_sunburn.our_method.optimization import combine_expert_predictions_for_attr


@dataclass
class TestState:
    a: int


class TestLawA:
    def precondition(self, current_state: TestState, action: str) -> bool:
        return True

    def effect(self, current_state: TestState, action: str) -> None:
        current_state.a = DiscreteDistribution(support=[current_state.a + 1])  # type: ignore


class NoOpLaw:
    def precondition(self, current_state: TestState, action: str) -> bool:
        return True

    def effect(self, current_state: TestState, action: str) -> None:
        pass


@dataclass
class TestObservableExtractor:
    a_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 10))

    def extract_attribute_predictions(
        self, state: TestState
    ) -> dict[ObservableId, DiscreteDistribution]:
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        if isinstance(state.a, DiscreteDistribution):
            predictions[ObservableId("a")] = state.a.expand_support(self.a_domain)
        return predictions

    def get_observed_outcomes(self, state: TestState) -> dict[ObservableId, int]:
        return {
            ObservableId("a"): state.a,
        }

    def apply_expert_predictions(
        self,
        new_state: TestState,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> TestState:
        if "a" in expert_predictions:
            a_preds = expert_predictions[ObservableId("a")]
            combined_dist = combine_expert_predictions_for_attr(a_preds, weights)
            new_state.a = combined_dist.sample()

        return new_state


def test_sample_next_state():

    law_a = LawFunctionWrapper.from_non_runtime_created(TestLawA())
    no_op_law = LawFunctionWrapper.from_non_runtime_created(NoOpLaw())

    mixture = LawMixture(
        observable_extractor=TestObservableExtractor(),
        weighted_laws=[
            WeightedLaw(law=law_a, weight=1.0, is_fitted=True),
            WeightedLaw(law=no_op_law, weight=1.0, is_fitted=True),
        ],
    )

    state = TestState(a=0)
    mixture.sample_next_state(state, "action")
