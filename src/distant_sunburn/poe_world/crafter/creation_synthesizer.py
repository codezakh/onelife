"""
PoE-World creation synthesizer for the Crafter environment.

This module implements the expert synthesis algorithm that generates Python code
to explain observed object lifecycle events (creation, deletion, replacement)
in the Crafter environment.
"""

import ast
from typing import List, Optional

from crafter.state_export import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    Position,
    SkeletonState,
    WorldState,
    ZombieState,
)
from loguru import logger

from ...litellm_utils import GeminiLiteLlmParams, LiteLlmMessage, LiteLlmRequest
from ...local_code_execution import ExecWithLimitedNamespace
from ...typing_utils import implements
from ..core import (
    DiscreteDistribution,
    ExpertFunction,
    ExpertSynthesizerProtocol,
    SymbolicTransition,
    WeightedExpert,
)


class CrafterCreationSynthesizer:
    """
    Creation-focused expert synthesizer for the Crafter environment.

    This synthesizer uses LLM calls to generate Python expert functions that
    explain observed object lifecycle events (creation, deletion, replacement).
    It follows the PoE-World approach of surprise-driven synthesis, only generating
    experts for transitions that the current model cannot explain well.
    """

    def __init__(self, llm_params: Optional[GeminiLiteLlmParams] = None):
        """
        Initialize the synthesizer.

        Args:
            llm_params: LLM parameters for synthesis. If None, uses default Gemini params.
        """
        self.llm_params = llm_params or GeminiLiteLlmParams()

    async def synthesize_experts(
        self,
        transitions: List[SymbolicTransition[WorldState]],
        object_type: str,
    ) -> List[WeightedExpert]:
        """
        Synthesize expert programs from state transitions.

        This method expects transitions that have already been filtered for "surprising"
        ones by the calling ObjModelLearner. The synthesizer focuses purely on
        generating experts from the provided transitions.

        Args:
            transitions: Sequence of state transitions to analyze (already filtered for surprising ones)
            object_type: Type of object to synthesize experts for

        Returns:
            List of WeightedExpert objects containing compiled expert functions
        """
        if not transitions:
            return []

        # Generate experts for all provided transitions (assumed to be surprising)
        experts = []
        for transition in transitions:
            try:
                expert = await self._synthesize_expert_for_transition(
                    transition, object_type
                )
                if expert:
                    experts.append(expert)
            except Exception as e:
                logger.error(f"Failed to synthesize expert for transition: {e}")
                continue

        return experts

    async def _synthesize_expert_for_transition(
        self,
        transition: SymbolicTransition[WorldState],
        object_type: str,
    ) -> Optional[WeightedExpert]:
        """
        Synthesize a single expert for a specific transition.

        Args:
            transition: The state transition to explain
            object_type: Type of object to focus on

        Returns:
            WeightedExpert or None if synthesis failed
        """
        # Create prompt for the LLM
        prompt = self._create_synthesis_prompt(transition, object_type)

        # Call LLM
        request = LiteLlmRequest(
            messages=[
                LiteLlmMessage(role="system", content=self._get_system_prompt()),
                LiteLlmMessage(role="user", content=prompt),
            ],
            params=self.llm_params,
        )

        try:
            response = request()
            code = response.choices[0].message.content

            if not code:
                logger.warning("Empty response from LLM")
                return None

            # Extract and validate the generated code
            expert_code = self._extract_expert_function(code)
            if not expert_code:
                logger.warning(
                    "Failed to extract valid expert function from LLM response"
                )
                return None

            # Validate the code
            if not self._validate_expert_code(expert_code):
                logger.warning(
                    f"Generated expert code failed validation:\n{expert_code}"
                )
                return None

            # Compile the expert function
            expert_function = self._compile_expert_function(expert_code, object_type)
            if not expert_function:
                logger.warning("Failed to compile expert function")
                return None

            return WeightedExpert(
                expert_function=expert_function,
                weight=1.0,
                is_fitted=False,
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for creation expert synthesis."""
        return """You are an expert at analyzing game state transitions and generating Python functions that explain object lifecycle events.

Your task is to generate a Python function that modifies a WorldState object to explain observed object creation, deletion, or replacement events. The function should:

1. Take a WorldState object and an action as input
2. Modify the state in-place by:
   - Creating new objects and appending them to current_state.objects
   - Deleting existing objects by removing them from current_state.objects
   - Replacing objects by deleting old ones and creating new ones
3. Use DiscreteDistribution(support=[value]) for probabilistic predictions
4. Return None (modify state in-place)

The function should be named `alter_{object_type}_objects` where {object_type} is the type of object being modified.

IMPORTANT: Use the correct object creation and deletion patterns:

**Creating new objects:**
- Instantiate the appropriate state class (e.g., CowState, ZombieState)
- Set entity_id to a unique value (use current_state.entity_id_counter_state + offset)
- Set position, health, and other required attributes
- Append to current_state.objects

**Deleting objects:**
- Find objects by name and remove them from current_state.objects using list operations
- Use list comprehension or filter to create a new objects list without the deleted objects

**Example creation:**
```python
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "spawn_cow":
        # Create a new cow at a specific position
        new_cow = CowState(
            entity_id=current_state.entity_id_counter_state + 1,
            position=Position(x=50, y=30),
            health=10,
            name="cow"
        )
        current_state.objects.append(new_cow)
```

**Deletion Pattern:**
```python
# To delete objects, use list comprehension with appropriate conditions:
# current_state.objects = [obj for obj in current_state.objects if <keep_condition>]
# 
# Examples:
# - Keep all objects except zombies: if obj.name != "zombie"
# - Keep all objects except zombies at position (10, 20): if not (obj.name == "zombie" and obj.position.x == 10 and obj.position.y == 20)
# - Keep all objects except zombies with health <= 0: if not (obj.name == "zombie" and obj.health <= 0)
```

**Example replacement:**
```python
def alter_plant_objects(current_state: WorldState, action: str) -> None:
    if action == "grow_plant":
        # Remove old plants and create new ones
        current_state.objects = [obj for obj in current_state.objects if obj.name != "plant"]
        
        # Create new plant
        new_plant = PlantState(
            entity_id=current_state.entity_id_counter_state + 1,
            position=Position(x=25, y=35),
            health=5,
            name="plant",
            grown=10,
            ripe=True
        )
        current_state.objects.append(new_plant)
```

Generate only the function code, no explanations or markdown formatting."""

    def _create_synthesis_prompt(
        self,
        transition: SymbolicTransition[WorldState],
        object_type: str,
    ) -> str:
        """Create a prompt for synthesizing a creation expert for a specific transition."""

        # Extract key lifecycle changes
        changes = self._extract_lifecycle_changes(transition)

        prompt = f"""Analyze this state transition and generate a Python function that explains the object lifecycle changes:

**Action:** {transition.action}

**Key Lifecycle Changes:**
{changes}

**Object Type:** {object_type}

Please generate a Python function named `alter_{object_type}_objects` that explains these lifecycle changes. The function should:
- Take `current_state: WorldState` and `action: str` as parameters
- Modify the current_state in-place by creating, deleting, or replacing objects
- Return None
- Handle object lifecycle events:
  * Object creation: Instantiate state classes and append to current_state.objects
  * Object deletion: Remove objects from current_state.objects using list operations
  * Object replacement: Delete old objects and create new ones

**WorldState Structure:**
- `current_state.objects`: List of game objects (PlayerState, CowState, ZombieState, etc.)
- `current_state.entity_id_counter_state`: Integer counter for generating unique entity IDs
- `current_state.size`: Tuple of (width, height) for world dimensions
- `current_state.player`: PlayerState object representing the player

**Available Object Types and Their Required Attributes:**

**CowState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "cow")

**ZombieState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "zombie")
- cooldown: int

**SkeletonState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "skeleton")
- reload: int

**ArrowState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "arrow")
- facing: Position(x: int, y: int)

**PlantState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "plant")
- grown: int
- ripe: bool

**FenceState:**
- entity_id: int
- position: Position(x: int, y: int)
- health: int
- name: str (must be "fence")

**Position Class:**
- x: int
- y: int

**Important Notes:**
- Always use `current_state.entity_id_counter_state + offset` for new entity IDs
- Use list comprehension to remove objects: `current_state.objects = [obj for obj in current_state.objects if <keep_condition>]`
- Append new objects: `current_state.objects.append(new_object)`
- Set all required attributes when creating objects
- For deletion, think carefully about which specific objects should be removed based on the transition

Generate ONLY the function code. DO NOT re-write out the class definitions or state definition.
Generate ONLY the function code.
If you DO NOT generate only the function code, INCREDIBLY BAD THINGS WILL HAPPEN.
!!! IMPORTANT !!!
GENERATE ONLY THE FUNCTION CODE.
!!! IMPORTANT !!!
DO NOT USE ANY IMPORT STATEMENTS.
DO NOT DEFINE ANY CLASSES.
!!! IMPORTANT !!!
"""

        return prompt

    def _extract_lifecycle_changes(
        self, transition: SymbolicTransition[WorldState]
    ) -> str:
        """Extract a human-readable description of object lifecycle changes."""
        prev_state = transition.prev_metadata
        next_state = transition.next_metadata
        changes = []

        # Count entities by type in both states
        prev_counts = {}
        next_counts = {}

        for obj in prev_state.objects:
            prev_counts[obj.name] = prev_counts.get(obj.name, 0) + 1

        for obj in next_state.objects:
            next_counts[obj.name] = next_counts.get(obj.name, 0) + 1

        # Detect entity creation/deletion by type
        for entity_type in set(prev_counts.keys()) | set(next_counts.keys()):
            prev_count = prev_counts.get(entity_type, 0)
            next_count = next_counts.get(entity_type, 0)

            if next_count > prev_count:
                # Entity creation
                changes.append(
                    f"- {entity_type} count increased from {prev_count} to {next_count} (created {next_count - prev_count})"
                )
            elif next_count < prev_count:
                # Entity deletion
                changes.append(
                    f"- {entity_type} count decreased from {prev_count} to {next_count} (deleted {prev_count - next_count})"
                )

        # Detect specific entity lifecycle events
        prev_entities = {e.entity_id: e for e in prev_state.objects}
        next_entities = {e.entity_id: e for e in next_state.objects}

        # Check for entity removal
        for entity_id in prev_entities:
            if entity_id not in next_entities:
                entity = prev_entities[entity_id]
                changes.append(f"- {entity.name} (ID: {entity_id}) was removed")

        # Check for new entities
        for entity_id in next_entities:
            if entity_id not in prev_entities:
                entity = next_entities[entity_id]
                changes.append(
                    f"- New {entity.name} (ID: {entity_id}) appeared at ({entity.position.x}, {entity.position.y})"
                )

        return (
            "\n".join(changes)
            if changes
            else "- No significant lifecycle changes detected"
        )

    def _extract_expert_function(self, llm_response: str) -> Optional[str]:
        """Extract the expert function code from the LLM response using AST parsing."""
        try:
            # Parse the entire response to get the AST
            tree = ast.parse(llm_response)

            # Find all function definitions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for the function that matches our expected naming pattern
                    if node.name.startswith("alter_"):
                        # Extract the function source code directly from the AST
                        function_code = ast.unparse(node)
                        functions.append(function_code)

            # Return the first matching function, or None if none found
            if functions:
                logger.info(f"Found {len(functions)} matching functions")
                return functions[0]
            else:
                logger.warning(
                    "No functions starting with 'alter_' found in LLM response"
                )
                return None

        except SyntaxError as e:
            logger.error(f"Failed to parse LLM response as Python code: {e}")
            logger.debug(
                f"LLM response that caused syntax error: {llm_response[:200]}..."
            )
            return None

        except Exception as e:
            logger.error(f"Unexpected error during function extraction: {e}")
            logger.debug(f"LLM response: {llm_response[:200]}...")
            return None

    def _validate_expert_code(self, code: str) -> bool:
        """Validate that the generated expert code is syntactically correct."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _compile_expert_function(
        self, code: str, object_type: str
    ) -> Optional[ExpertFunction[WorldState]]:
        """Compile the generated code into a callable expert function."""
        try:
            function_name = f"alter_{object_type}_objects"

            # Create executor with access to required classes

            executor = ExecWithLimitedNamespace(
                inherited_scope={
                    "WorldState": WorldState,
                    "CowState": CowState,
                    "ZombieState": ZombieState,
                    "SkeletonState": SkeletonState,
                    "ArrowState": ArrowState,
                    "PlantState": PlantState,
                    "FenceState": FenceState,
                    "Position": Position,
                    "DiscreteDistribution": DiscreteDistribution,
                },
                allowed_names={
                    "WorldState",
                    "CowState",
                    "ZombieState",
                    "SkeletonState",
                    "ArrowState",
                    "PlantState",
                    "FenceState",
                    "Position",
                    "DiscreteDistribution",
                },
            )

            # Compile the code
            executor(code)

            # Extract the compiled function from the namespace
            expert_function = executor.namespace[function_name]

            # Set the expert source code
            expert_function.__source_code__ = code

            return expert_function

        except Exception as e:
            logger.error(f"Failed to compile expert function: {e}")
            return None


implements(ExpertSynthesizerProtocol[WorldState])(CrafterCreationSynthesizer)
