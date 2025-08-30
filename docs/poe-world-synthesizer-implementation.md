# PoE-World Synthesizer Implementation

**Date:** 2025-01-27  
**Purpose:** Technical specification of PoE-World's synthesizer implementation for reimplementation guidance

## Overview

This document describes the implementation of PoE-World's expert synthesis algorithm, which transforms environment transitions into programmatic expert functions. The synthesis process involves multiple specialized synthesizers that generate Python code to explain observed state changes.

## Core Interfaces

### 1. ExpertSynthesizer Protocol

```python
from typing import Protocol, List, Awaitable
from abc import ABC, abstractmethod

class ExpertSynthesizer(Protocol):
    """Protocol for expert synthesis from state transitions."""
    
    async def synthesize_experts(
        self, 
        transitions: List[StateTransition], 
        object_type: str
    ) -> List[Expert]:
        """
        Synthesize expert programs from state transitions.
        
        Args:
            transitions: Sequence of state transitions to analyze
            object_type: Type of object to synthesize experts for
            
        Returns:
            List of synthesized expert programs
        """
        ...

class Synthesizer(ABC):
    """Base class for PoE-World synthesizers."""
    
    def __init__(self, config: LearningConfig, obj_type: str, llm: LLMClient):
        self.config = config
        self.obj_type = obj_type
        self.llm = llm
        self.objects_selector = ObjTypeObjSelector(obj_type)
        self.interactions_selector = ObjTypeInteractionSelector(obj_type)
        self.cache_x = {}
    
    @abstractmethod
    async def a_synthesize(
        self, 
        transitions: List[StateTransitionTriplet], 
        **kwargs
    ) -> List[str]:
        """Synthesize Python code strings from transitions."""
        pass
```

**Location in PoE-World:** `external/poe-world/learners/synthesizer.py:44-100`

### 2. StateTransition Interface

```python
class StateTransition:
    """Represents a single state transition with input, action, and output."""
    
    def __init__(
        self,
        input_state: ObjectState,
        action: Action,
        output_state: ObjectState,
        input_game_state: Optional[GameState] = None,
        output_game_state: Optional[GameState] = None
    ):
        self.input_state = input_state
        self.action = action
        self.output_state = output_state
        self.input_game_state = input_game_state
        self.output_game_state = output_game_state
```

**Location in PoE-World:** `external/poe-world/classes/helper.py:75-140`

### 3. ObjectState Interface

```python
class ObjectState:
    """Represents the state of all objects in the environment."""
    
    def get_objects_by_type(self, obj_type: str) -> List[GameObject]:
        """Get all objects of a specific type."""
        pass
    
    def get_object_interactions(self) -> List[ObjectInteraction]:
        """Get all object interactions (touching objects)."""
        pass
    
    def deepcopy(self) -> ObjectState:
        """Create a deep copy of the state."""
        pass

class GameObject:
    """Represents a single game object with attributes."""
    
    def __init__(self, id: int, obj_type: str, **attributes):
        self.id = id
        self.obj_type = obj_type
        self.attributes = attributes  # position, velocity, deleted, etc.
    
    def touches(self, other: GameObject) -> bool:
        """Check if this object touches another object."""
        pass
```

**Location in PoE-World:** `external/poe-world/classes/helper.py:300-700`

### 4. LLMClient Interface

```python
class LLMClient(Protocol):
    """Protocol for LLM interaction during synthesis."""
    
    async def aprompt(
        self, 
        prompts: List[str], 
        temperature: float = 0, 
        seed: int = None
    ) -> List[str]:
        """Send prompts to LLM and return responses."""
        ...
    
    def setup_cache(self, mode: str, database_path: str) -> None:
        """Setup caching for LLM responses."""
        ...
```

**Location in PoE-World:** `external/poe-world/learners/obj_model_learner.py:40-50`

## Synthesis Pipeline Composition

### 1. Transition Processing Pipeline

The synthesis pipeline processes transitions through several stages:

```python
class SynthesisPipeline:
    """Orchestrates the synthesis of experts from transitions."""
    
    def __init__(self, config: LearningConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.synthesizers = self._create_synthesizers()
    
    async def synthesize_experts(
        self, 
        transitions: List[StateTransition], 
        object_type: str
    ) -> List[Expert]:
        """Main synthesis pipeline."""
        
        # Step 1: Filter transitions for object type
        object_transitions = self._filter_by_object_type(transitions, object_type)
        
        # Step 2: Extract state changes
        effects = self._extract_state_changes(object_transitions)
        
        # Step 3: Generate natural language descriptions
        observations = await self._generate_observations(object_transitions, effects)
        
        # Step 4: Synthesize code via LLM
        code_strings = await self._synthesize_code(observations, object_type)
        
        # Step 5: Parse and validate code
        experts = self._parse_experts(code_strings, object_type)
        
        return experts
```

**Location in PoE-World:** `external/poe-world/learners/synthesizer.py:100-160`

### 2. State Change Extraction

```python
def _extract_state_changes(self, transitions: List[StateTransition]) -> List[str]:
    """Extract natural language descriptions of state changes."""
    effects = []
    
    for transition in transitions:
        input_objects = self.objects_selector(transition.input_state)
        output_objects = self.objects_selector(transition.output_state)
        
        # Track object creation, deletion, and attribute changes
        input_ids = [obj.id for obj in input_objects]
        
        for obj in output_objects:
            if obj.deleted == 1:
                effects.append(f'The {obj.str_w_id()} is deleted')
            elif obj.id not in input_ids:
                effects.append(f'A new {obj.obj_type} object is created at (x={obj.x},y={obj.y})')
            else:
                effects.append(f'The {obj.str_w_id()} sets x-axis velocity to {"%+d" % obj.velocity_x}')
                effects.append(f'The {obj.str_w_id()} sets y-axis velocity to {"%+d" % obj.velocity_y}')
    
    return effects
```

**Location in PoE-World:** `external/poe-world/learners/synthesizer.py:110-140`

### 3. LLM Code Generation

```python
async def _synthesize_code(
    self, 
    observations: List[str], 
    object_type: str
) -> List[str]:
    """Generate Python code from observations via LLM."""
    
    prompt = explain_event_prompt.format(
        obj_type=object_type,
        obs_lst_txt=list_to_bullets(observations),
        action=action,
        n=4
    )
    
    responses = await self.llm.aprompt([prompt], temperature=0, seed=self.config.seed)
    
    # Extract Python code blocks from responses
    code_strings = []
    for response in responses:
        codes = process_llm_response_to_codes(response)
        code_strings.extend(codes)
    
    return code_strings
```

**Location in PoE-World:** `external/poe-world/learners/synthesizer.py:150-160`

## Synthesizer Types

PoE-World implements 17 specialized synthesizer types, each handling different aspects of environment dynamics:

### 1. ActionSynthesizer

**Purpose:** Synthesizes experts for immediate effects of actions when objects interact.

**Key Logic:**
- Focuses on transitions where objects are touching
- Uses recent transitions within `synth_window` (typically 1-3 timesteps)
- Generates experts that explain immediate action effects

**Location:** `external/poe-world/learners/synthesizer.py:161-205`

**Example Expert Generated:**
```python
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    player_objs = obj_list.get_objs_by_obj_type('player')
    for player_obj in player_objs:
        if action == 'UP':
            ladder_objs = obj_list.get_objs_by_obj_type('ladder')
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj):
                    player_obj.velocity_y = RandomValues([-1, -2])
                    break
    return obj_list
```

### 2. MultiTimestepActionSynthesizer

**Purpose:** Handles delayed effects of actions over multiple timesteps (POMDP).

**Key Logic:**
- Searches through history up to 15 timesteps for action causes
- Handles cases where action effects are delayed
- Returns tuples of (code, context_length) for POMDP experts

**Location:** `external/poe-world/learners/synthesizer.py:206-333`

### 3. PassiveMovementSynthesizer

**Purpose:** Synthesizes experts for movement not caused by player actions.

**Key Logic:**
- Focuses on autonomous entity movement
- Handles gravity, momentum, and passive physics
- Generates experts for natural object behavior

**Location:** `external/poe-world/learners/synthesizer.py:1007-1052`

### 4. VelocitySynthesizer

**Purpose:** Handles velocity-related changes and momentum.

**Key Logic:**
- Tracks velocity patterns and changes
- Handles acceleration, deceleration, and direction changes
- Generates experts for movement physics

**Location:** `external/poe-world/learners/synthesizer.py:1089-1136`

### 5. ConstraintsSynthesizer

**Purpose:** Learns physical constraints and impossible states.

**Key Logic:**
- Identifies physically impossible configurations
- Generates constraint functions that rule out invalid states
- Used for Montezuma's Revenge platform alignment

**Location:** `external/poe-world/learners/synthesizer.py:1505-1528`

### 6. RestartSynthesizer

**Purpose:** Handles game restart events and state resets.

**Key Logic:**
- Manages game state transitions to RESTART
- Handles object respawning and position resets
- Generates experts for game lifecycle events

**Location:** `external/poe-world/learners/synthesizer.py:1529-1717`

### 7. Specialized Synthesizers

Additional synthesizers handle specific aspects:

- **MultiTimestepMomentumSynthesizer**: Momentum changes over time
- **MultiTimestepSizeChangeSynthesizer**: Object size changes
- **MultiTimestepStatusChangeSynthesizer**: State transitions
- **PassiveCreationSynthesizer**: Object creation events
- **SnappingSynthesizer**: Object alignment and positioning
- **PlayerInteractionSynthesizer**: Player-specific interactions

**Location:** `external/poe-world/learners/synthesizer.py:334-1504`

## Synthesis Orchestration

### Synthesizer Selection Logic

```python
def _select_synthesizers(
    self, 
    transition: StateTransition, 
    use_full_history: bool = False
) -> List[Synthesizer]:
    """Select appropriate synthesizers based on transition characteristics."""
    
    if transition.output_game_state == GameState.RESTART:
        return self.restart_synthesizers
    elif use_full_history:
        return self.pomdp_synthesizers
    else:
        synthesizers = self.normal_synthesizers
        if self.constraint_synthesizers:
            synthesizers.extend(self.constraint_synthesizers)
        return synthesizers
```

**Location in PoE-World:** `external/poe-world/learners/obj_model_learner.py:340-380`

### Parallel Synthesis Execution

```python
async def _execute_synthesizers(
    self, 
    transitions: List[StateTransition], 
    synthesizers: List[Synthesizer]
) -> List[str]:
    """Execute multiple synthesizers in parallel."""
    
    # Create async tasks for each synthesizer
    synthesis_tasks = [
        synthesizer.a_synthesize(transitions)
        for synthesizer in synthesizers
    ]
    
    # Execute all synthesizers in parallel
    results = await asyncio.gather(*synthesis_tasks)
    
    # Flatten results
    all_experts = []
    for result in results:
        if isinstance(result, list):
            all_experts.extend(result)
        elif isinstance(result, tuple):
            # Handle POMDP results with context length
            experts, context_lengths = zip(*result)
            all_experts.extend(experts)
    
    return all_experts
```

**Location in PoE-World:** `external/poe-world/learners/obj_model_learner.py:380-400`

## Complete ExpertSynthesizer Implementation

```python
from typing import List, Awaitable, Protocol
from abc import ABC, abstractmethod
import asyncio

class ExpertSynthesizer(Protocol):
    """Protocol for expert synthesis from state transitions."""
    
    async def synthesize_experts(
        self, 
        transitions: List[StateTransition], 
        object_type: str
    ) -> List[Expert]:
        """Synthesize expert programs from state transitions."""
        ...

class PoEWorldExpertSynthesizer:
    """Complete implementation of PoE-World's expert synthesis algorithm."""
    
    def __init__(self, config: LearningConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.synthesizers = self._create_synthesizers()
    
    def _create_synthesizers(self) -> Dict[str, List[Synthesizer]]:
        """Create synthesizers for different object types and modes."""
        return {
            'normal': [
                ActionSynthesizer,
                PassiveMovementSynthesizer,
                VelocitySynthesizer,
                SnappingSynthesizer
            ],
            'restart': [
                RestartSynthesizer,
                PassiveCreationSynthesizer
            ],
            'constraint': [
                ConstraintsSynthesizer
            ],
            'pomdp': [
                MultiTimestepActionSynthesizer,
                MultiTimestepVelocitySynthesizer,
                MultiTimestepStatusChangeSynthesizer
            ]
        }
    
    async def synthesize_experts(
        self, 
        transitions: List[StateTransition], 
        object_type: str
    ) -> List[Expert]:
        """Main synthesis pipeline implementation."""
        
        # Step 1: Filter transitions for object type
        object_transitions = self._filter_by_object_type(transitions, object_type)
        
        if not object_transitions:
            return []
        
        # Step 2: Select appropriate synthesizers
        synthesizers = self._select_synthesizers(object_transitions[-1])
        
        # Step 3: Execute synthesizers in parallel
        expert_codes = await self._execute_synthesizers(object_transitions, synthesizers)
        
        # Step 4: Parse code into Expert objects
        experts = []
        for code in expert_codes:
            try:
                expert = self._parse_expert_code(code, object_type)
                experts.append(expert)
            except Exception as e:
                # Log parsing errors but continue
                continue
        
        return experts
    
    def _filter_by_object_type(
        self, 
        transitions: List[StateTransition], 
        object_type: str
    ) -> List[StateTransition]:
        """Filter transitions to include only those with the target object type."""
        filtered = []
        for transition in transitions:
            input_objects = transition.input_state.get_objects_by_type(object_type)
            output_objects = transition.output_state.get_objects_by_type(object_type)
            if input_objects or output_objects:
                filtered.append(transition)
        return filtered
    
    def _select_synthesizers(self, transition: StateTransition) -> List[Synthesizer]:
        """Select appropriate synthesizers based on transition characteristics."""
        if transition.output_game_state == GameState.RESTART:
            return [s(self.config, self.obj_type, self.llm) 
                   for s in self.synthesizers['restart']]
        else:
            return [s(self.config, self.obj_type, self.llm) 
                   for s in self.synthesizers['normal']]
    
    async def _execute_synthesizers(
        self, 
        transitions: List[StateTransition], 
        synthesizers: List[Synthesizer]
    ) -> List[str]:
        """Execute multiple synthesizers in parallel."""
        synthesis_tasks = [
            synthesizer.a_synthesize(transitions)
            for synthesizer in synthesizers
        ]
        
        results = await asyncio.gather(*synthesis_tasks, return_exceptions=True)
        
        all_codes = []
        for result in results:
            if isinstance(result, Exception):
                continue  # Skip failed synthesizers
            if isinstance(result, list):
                all_codes.extend(result)
            elif isinstance(result, tuple):
                # Handle POMDP results
                codes, context_lengths = zip(*result)
                all_codes.extend(codes)
        
        return all_codes
    
    def _parse_expert_code(self, code: str, object_type: str) -> Expert:
        """Parse Python code string into Expert object."""
        # Extract function name and validate format
        if not code.startswith(f'def alter_{object_type}_objects'):
            raise ValueError(f"Invalid expert code format for {object_type}")
        
        return Expert(
            code=code,
            context_length=-1,  # Default to MDP mode
            object_type=object_type
        )

class ActionSynthesizer(Synthesizer):
    """Synthesizer for immediate action effects."""
    
    async def a_synthesize(
        self, 
        transitions: List[StateTransitionTriplet], 
        **kwargs
    ) -> List[str]:
        """Synthesize experts for action-related events."""
        
        action = transitions[-1].event
        recent_transitions = transitions[-self.config.synthesizer.synth_window:]
        
        # Extract state changes
        effects = self._extract_state_changes(recent_transitions)
        
        if not effects:
            return []
        
        # Generate observations
        observations = await self._generate_observations(recent_transitions, effects)
        
        # Synthesize code
        prompt = explain_event_prompt.format(
            obj_type=self.obj_type,
            obs_lst_txt=list_to_bullets(observations),
            action=action,
            n=4
        )
        
        responses = await self.llm.aprompt([prompt], temperature=0, seed=self.config.seed)
        
        # Parse responses
        codes = []
        for response in responses:
            codes.extend(process_llm_response_to_codes(response))
        
        return codes
    
    def _extract_state_changes(self, transitions: List[StateTransitionTriplet]) -> List[str]:
        """Extract natural language descriptions of state changes."""
        effects = []
        
        for transition in transitions:
            input_objects = self.objects_selector(transition.input_state)
            output_objects = self.objects_selector(transition.output_state)
            
            input_ids = [obj.id for obj in input_objects]
            
            for obj in output_objects:
                if obj.deleted == 1:
                    effects.append(f'The {obj.str_w_id()} is deleted')
                elif obj.id not in input_ids:
                    effects.append(f'A new {obj.obj_type} object is created at (x={obj.x},y={obj.y})')
                else:
                    effects.append(f'The {obj.str_w_id()} sets x-axis velocity to {"%+d" % obj.velocity_x}')
                    effects.append(f'The {obj.str_w_id()} sets y-axis velocity to {"%+d" % obj.velocity_y}')
        
        return effects
    
    async def _generate_observations(
        self, 
        transitions: List[StateTransitionTriplet], 
        effects: List[str]
    ) -> List[str]:
        """Generate natural language observations from effects."""
        prompt = interpret_obj_interact_prompt.format(obj_type=self.obj_type)
        
        input_text = self._prepare_input_text(transitions[-1])
        effects_text = list_to_bullets(effects)
        
        full_prompt = prompt.format(input=input_text, effects=effects_text)
        
        responses = await self.llm.aprompt([full_prompt], temperature=0, seed=self.config.seed)
        
        observations = []
        for response in responses:
            observations.extend(parse_listed_output(response))
        
        return observations
    
    def _prepare_input_text(self, transition: StateTransitionTriplet) -> str:
        """Prepare input text describing objects and interactions."""
        input_objects = self.objects_selector(transition.input_state)
        input_interactions = self.interactions_selector(
            transition.input_state.get_obj_interactions()
        )
        
        objects_text = '\n'.join([obj.str_w_id() for obj in input_objects])
        interactions_text = '\n'.join([str(interaction) for interaction in input_interactions])
        
        return f"{objects_text}\n{interactions_text}"
```