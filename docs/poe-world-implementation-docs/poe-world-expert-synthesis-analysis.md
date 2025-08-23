# PoE World Expert Synthesis Analysis

**Date:** 2025-01-27  
**Purpose:** Comprehensive analysis of PoE World's expert synthesis approach for reimplementation guidance

## Overview

This document provides a detailed analysis of how PoE World synthesizes programmatic experts from environment transitions. The analysis covers batching strategies, information provided to synthesizers, synthesizer types, and their applicability to Crafter.

## Key Findings Summary

1. **Batching Strategy**: PoE World uses a sophisticated batching approach that processes transitions in batches of 10, but only synthesizes "surprising" transitions (those with low model probability)
2. **Multiple Synthesizer Types**: The system uses 15+ specialized synthesizers rather than a single general-purpose one
3. **Rich Context**: Synthesizers receive detailed state transitions, object interactions, and natural language descriptions
4. **Crafter Applicability**: 8 synthesizer types would be needed for Crafter's state space

## Detailed Analysis

### 1. Batching Strategy for Transitions

**Location in Codebase:**
- `external/poe-world/learners/obj_model_learner.py:67` - `self.batch_size = 10`
- `external/poe-world/learners/obj_model_learner.py:301-325` - `_grab_surprising_indices()` method
- `external/poe-world/learners/obj_model_learner.py:148-180` - Main batching loop in `infer_moe()`

**Key Implementation Details:**
- **Batch Size**: Fixed at 10 transitions per batch
- **Surprising Selection**: Only transitions with low log-probability under current model are synthesized
- **Synthesis Window**: Each synthesizer looks at recent transitions within `config.synthesizer.synth_window`
- **Efficiency**: This approach focuses computational resources on learning new patterns

**Relevant Paper Sections:**
- arxiv paper doesn't explicitly detail batching, but mentions "online learning" and "continuous refinement"

### 2. Information Provided to Synthesizers

**Location in Codebase:**
- `external/poe-world/learners/synthesizer.py:44-100` - Base `Synthesizer` class
- `external/poe-world/prompts/synthesizer.py:1-100` - Prompt templates
- `external/poe-world/learners/synthesizer.py:161-205` - `ActionSynthesizer` implementation

**Information Structure:**
1. **State Transitions**: `StateTransitionTriplet` objects with input state, action, output state
2. **Object Interactions**: Which objects are touching/interacting
3. **Natural Language Descriptions**: Converted state changes (e.g., "player touches ladder")
4. **Object-Specific Context**: Filtered to relevant object types
5. **Class Definitions**: Detailed Python class documentation in prompts

**Prompt Structure:**
- `external/poe-world/prompts/synthesizer.py:8-50` - `explain_event_prompt` template
- Includes class definitions for `RandomValues`, `Obj`, `ObjList`
- Instructions for generating Python functions that mutate state

### 3. Multiple Synthesizer Types

**Location in Codebase:**
- `external/poe-world/learners/synthesizer.py:161-1717` - All synthesizer implementations
- `external/poe-world/learners/obj_model_learner.py:25-60` - Synthesizer initialization

**Complete List of Synthesizers:**

1. **ActionSynthesizer** (lines 161-205)
   - Handles action-related events when objects are interacting
   - Focuses on immediate effects of actions

2. **MultiTimestepActionSynthesizer** (lines 206-333)
   - Action synthesis with longer history (POMDP)
   - Handles delayed effects of actions

3. **MultiTimestepMomentumSynthesizer** (lines 334-529)
   - Momentum changes over multiple timesteps
   - Tracks velocity patterns

4. **MultiTimestepSizeChangeSynthesizer** (lines 530-678)
   - Size changes over multiple timesteps
   - Handles object growth/shrinking

5. **MultiTimestepStatusChangeSynthesizer** (lines 679-788)
   - Status changes over multiple timesteps
   - Handles state transitions

6. **MultiTimestepStatusChangeVelocityModeSynthesizer** (lines 789-897)
   - Status changes with velocity considerations
   - Combines status and movement

7. **MultiTimestepStatusChangeSizeModeSynthesizer** (lines 898-1006)
   - Status changes with size considerations
   - Combines status and size

8. **PassiveMovementSynthesizer** (lines 1007-1052)
   - Movement not caused by actions
   - Handles autonomous entity movement

9. **PassiveCreationSynthesizer** (lines 1053-1088)
   - Object creation not caused by actions
   - Handles spawning/generation

10. **VelocitySynthesizer** (lines 1089-1136)
    - Velocity-related changes
    - Handles speed/direction changes

11. **MultiTimestepVelocitySynthesizer** (lines 1137-1330)
    - Velocity changes over multiple timesteps
    - Tracks velocity evolution

12. **VelocityTrackingSynthesizer** (lines 1331-1401)
    - Tracking velocity patterns
    - Handles complex movement patterns

13. **NoInteractPassiveMovementSynthesizer** (lines 1402-1447)
    - Movement without interactions
    - Handles isolated entity movement

14. **PlayerInteractionSynthesizer** (lines 1448-1480)
    - Player-specific interactions
    - Handles player-object interactions

15. **SnappingSynthesizer** (lines 1481-1504)
    - Object snapping/alignment
    - Handles positioning constraints

16. **ConstraintsSynthesizer** (lines 1505-1528)
    - Learning object constraints
    - Handles physical limitations

17. **RestartSynthesizer** (lines 1529-1717)
    - Game restart events
    - Handles game state resets

**Relevant Paper Sections:**
- arxiv paper mentions "ActionSynthesizer" specifically in section about synthesizer modules
- Describes the general approach of using multiple specialized synthesizers

### 4. Synthesizer Selection and Usage

**Location in Codebase:**
- `external/poe-world/learners/obj_model_learner.py:25-60` - Synthesizer initialization
- `external/poe-world/learners/obj_model_learner.py:340-400` - `_a_infer_moe_at_transition()` method

**Selection Logic:**
- Different synthesizers are used based on:
  - Game state (RESTART vs normal)
  - Whether to use full history (POMDP mode)
  - Whether to include constraints
- Synthesizers are hardcoded per object type in the learner initialization

### 5. Crafter-Specific Synthesizer Requirements

**Based on Crafter State Structure** (`external/crafter_refactored/crafter/state_export.py`):

1. **ActionSynthesizer**
   - Player actions: move, collect, craft, attack, sleep
   - Immediate effects of player actions

2. **InventorySynthesizer**
   - Inventory changes: collecting resources, crafting items
   - Resource management dynamics

3. **HealthSynthesizer**
   - Health, hunger, thirst, fatigue changes
   - Player status management

4. **AchievementSynthesizer**
   - Achievement progress tracking
   - Goal completion dynamics

5. **EntityLifecycleSynthesizer**
   - Entity creation/deletion: plants growing, enemies spawning
   - Lifecycle management

6. **MovementSynthesizer**
   - Entity movement patterns
   - Pathfinding and navigation

7. **InteractionSynthesizer**
   - Entity interactions: combat, harvesting
   - Interaction mechanics

8. **EnvironmentSynthesizer**
   - Environmental changes: daylight, materials
   - World state evolution

### 6. Implementation Insights

1. **Separation of Concerns**: Each synthesizer handles a specific aspect of world dynamics
2. **Asynchronous Processing**: Uses `asyncio.gather()` for parallel LLM calls
3. **Caching**: Implements caching for expensive operations
4. **Modular Design**: Synthesizers are pluggable components

**Potential Improvements for Crafter:**
1. **Single General Synthesizer**: Start with one synthesizer as proposed in PRD
2. **Symbolic State**: Crafter's symbolic state simplifies object tracking
3. **Stable IDs**: Crafter objects have stable IDs, unlike Atari games
4. **Simplified Interactions**: Crafter has simpler interaction patterns

### 7. Navigation Quick Reference

**Key Files for Expert Synthesis:**
- `external/poe-world/learners/synthesizer.py` - All synthesizer implementations
- `external/poe-world/learners/obj_model_learner.py` - Main learning orchestration
- `external/poe-world/prompts/synthesizer.py` - Prompt templates
- `external/poe-world/learners/models.py` - MoE model implementation

**Key Methods:**
- `ObjModelLearner.infer_moe()` - Main learning loop
- `ObjModelLearner._grab_surprising_indices()` - Transition selection
- `ObjModelLearner._a_infer_moe_at_transition()` - Synthesis orchestration
- `Synthesizer.a_synthesize()` - Individual synthesis

**Paper References:**
- arxiv paper section on synthesizer modules (mentions ActionSynthesizer)
- Overview of the PoE approach and expert combination

## Conclusion

The original PoE World implementation demonstrates that **specialized synthesizers with sophisticated batching strategies** are effective for learning world models. However, for Crafter, a **single general-purpose synthesizer** with comprehensive prompts may be sufficient initially, given the simpler state space and stable object IDs.

The key insight is that **focusing on surprising transitions** is more important than having multiple synthesizer types. This batching strategy should be prioritized in the reimplementation.
