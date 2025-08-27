from ...typing_utils import implements
from crafter.state_export import WorldState
from ..core import ObservableExtractorProtocol, ObservableId, DiscreteDistribution
import torch


class ObservableExtractor:
    def __init__(self):
        # Define possible values for attributes of interest in Crafter
        # We keep this fairly generic to avoid hardcoding domain knowledge
        # that the world modeler should have to discover itself.
        # To simplify things, we will defined a domain for each _type_ of attribute
        # and then use the type to determine the domain.
        # So all attributes of type int will have the same domain, etc.
        pass

    def extract_attribute_predictions(
        self, state: WorldState
    ) -> dict[ObservableId, DiscreteDistribution]:
        """
        Extract probabilistic predictions from a state after expert execution.
        """
        pass

    def get_observed_outcomes(self, state: WorldState) -> dict[ObservableId, int]:
        """
        Extract ground truth observed values from a state.
        """
        pass

    @staticmethod
    def apply_expert_predictions(
        new_state: WorldState,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> WorldState:
        pass


implements(ObservableExtractorProtocol[WorldState])(ObservableExtractor)
