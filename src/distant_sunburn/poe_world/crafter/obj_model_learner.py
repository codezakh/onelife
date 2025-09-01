"""
Object Model Learner for Crafter environment.

This module implements the ObjModelLearner class that manages the synthesis process
for a specific object type, following the PoE-World design document.
"""

from typing import List, Optional
from loguru import logger
import copy

from crafter.state_export import WorldState
from ..core import SymbolicTransition, WorldModelProtocol, WeightedExpert
from ..world_model import PoEWorldModel
from .observable_extractor import ObservableExtractor
from .synthesizer import CrafterExpertSynthesizer, SynthesizedExpert
from ...litellm_utils import GeminiLiteLlmParams


class ObjModelLearner:
    def __init__(
        self,
        object_type: str,
        llm_params: Optional[GeminiLiteLlmParams] = None,
        surprise_threshold: float = -2.0,
    ):
        """
        Initialize the object model learner.

        Args:
            object_type: Type of object this learner is responsible for
            llm_params: LLM parameters for synthesis
            surprise_threshold: Log probability threshold for "surprising" transitions
        """
        self.object_type = object_type
        self.surprise_threshold = surprise_threshold

        # Initialize components
        self.observable_extractor = ObservableExtractor()
        self.world_model = PoEWorldModel(
            observable_extractor=self.observable_extractor,
            weighted_experts=[],
        )
        self.synthesizer = CrafterExpertSynthesizer(llm_params=llm_params)

        logger.info(f"Initialized ObjModelLearner for {object_type}")

    async def process_transitions(
        self,
        transitions: List[SymbolicTransition[WorldState]],
    ) -> List[SynthesizedExpert]:
        """
        Process a sequence of transitions and synthesize experts for surprising ones.

        Args:
            transitions: Sequence of state transitions to process

        Returns:
            List of newly synthesized experts
        """
        if not transitions:
            return []

        # Filter for surprising transitions using the world model
        surprising_transitions = self._filter_surprising_transitions(transitions)

        if not surprising_transitions:
            logger.info(f"No surprising transitions found for {self.object_type}")
            return []

        # Synthesize experts for surprising transitions
        new_experts: List[SynthesizedExpert] = []
        for transition in surprising_transitions:
            try:
                experts = await self._synthesize_for_transition(transition)
                new_experts.extend(experts)
            except Exception as e:
                logger.error(f"Failed to synthesize experts for transition: {e}")
                continue

        # Add new experts to the world model
        if new_experts:
            self._add_experts_to_model(new_experts)
            logger.info(f"Added {len(new_experts)} new experts for {self.object_type}")

        return new_experts

    def _filter_surprising_transitions(
        self,
        transitions: List[SymbolicTransition[WorldState]],
    ) -> List[SymbolicTransition[WorldState]]:
        """
        Filter transitions that are "surprising" (low probability under current model).

        A transition is considered surprising if the log probability of the observed
        outcome, according to the current mixture of experts, is below the threshold.
        """
        surprising: List[SymbolicTransition[WorldState]] = []

        for transition in transitions:
            try:
                # Compute log probability of the observed transition
                log_prob = self.world_model.evaluate_log_probability(
                    state=transition.prev_metadata,
                    action=transition.action,
                    next_state=transition.next_metadata,
                )

                # If log probability is below threshold, transition is surprising
                if log_prob < self.surprise_threshold:
                    surprising.append(transition)
                    logger.debug(
                        f"Surprising transition for {self.object_type}: "
                        f"log_prob={log_prob:.3f} < {self.surprise_threshold}"
                    )

            except Exception as e:
                logger.warning(f"Failed to evaluate transition probability: {e}")
                # If we can't evaluate, treat as surprising to be safe
                surprising.append(transition)

        return surprising

    async def _synthesize_for_transition(
        self,
        transition: SymbolicTransition[WorldState],
    ) -> List[SynthesizedExpert]:
        """
        Synthesize experts for a specific surprising transition.

        Args:
            transition: The surprising transition to explain

        Returns:
            List of synthesized experts
        """
        # Create custom view of the state (only objects of target type)
        custom_input_state = self._create_custom_view(transition.prev_metadata)
        custom_output_state = self._create_custom_view(transition.next_metadata)

        # Create custom transition with filtered states
        custom_transition = SymbolicTransition(
            prev_metadata=custom_input_state,
            action=transition.action,
            next_metadata=custom_output_state,
        )

        # Synthesize experts using the synthesizer
        experts = await self.synthesizer.synthesize_experts(
            transitions=[custom_transition],
            object_type=self.object_type,
        )

        return experts

    def _create_custom_view(self, state: WorldState) -> WorldState:
        """
        Create a custom view of the state that only includes objects of the target type.

        This follows the PoE-World design principle that each object type only sees
        its own objects, preventing cross-object-type rules.
        """

        # Create a deep copy to avoid modifying the original state
        custom_state = copy.deepcopy(state)

        # Filter objects to only include the target type
        custom_state.objects = [
            obj for obj in state.objects if obj.name == self.object_type
        ]

        return custom_state

    def _add_experts_to_model(self, experts: List[SynthesizedExpert]) -> None:
        """
        Add synthesized experts to the world model.

        This converts SynthesizedExpert objects to WeightedExpert objects
        and adds them to the world model. Initial weights are set to 1.0.
        """
        # Convert SynthesizedExpert to WeightedExpert
        weighted_experts: list[WeightedExpert] = []
        for expert in experts:
            # TODO: Compile the expert code into an actual function
            # For now, we'll create a placeholder
            weighted_expert = WeightedExpert(
                expert_function=expert.code,  # This should be a callable function
                weight=1.0,  # Initial weight
            )
            weighted_experts.append(weighted_expert)

        # Add to world model
        self.world_model = self.world_model.with_new_experts(weighted_experts)

    @property
    def model(self) -> WorldModelProtocol[WorldState]:
        """Get the current world model for this object type."""
        return self.world_model
