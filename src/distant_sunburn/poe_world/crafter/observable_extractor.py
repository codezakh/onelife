"""
PoE-World was originally designed for 2D physics-based environments that are
object-centric, like Atari Pong or 2D platformers like Montezuma's Revenge.

The "state" in these environments is a list of objects with commmon attributes
like position, velocity, etc.

In Crafter, the state is much more complex, and hierarchical. Since our aim is
to evaluate the ability of a more generic approach to learn about the world, we
want to avoid hardcoding domain knowledge about the world, such as how to extract
interesting observables from the state and so on.

So, we will stick to the original design of PoE-World, and define an extractor which
operates on objects in the world state that are similar to the objects in the original
environments.

This corresponds to the player's position, health, and the position and health of nearby
game entities. Like in PoE-World for physics-based environments, we will ignore static
objects like tiles of the game world or crafting stations.
"""

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
