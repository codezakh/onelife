# PoE World Domain Analysis: Core Algorithm Structure

**Date:** 2025-01-27  
**Purpose:** Domain-driven design analysis of PoE World's core algorithm structure

## Preamble

This document provides a distilled interpretation of PoE World's algorithm structure, expressed through domain-driven design principles. This is not a literal rendering of the PoE World codebase, but rather our analysis of the core domain objects and their interactions that implement the Product of Experts approach to world model learning.

The analysis focuses on the essential components and their interfaces, abstracting away implementation details to reveal the fundamental structure of the algorithm.

## Core Domain Objects

### 1. Experience Management

**ExperienceBuffer**
- `add_transition(transition: StateTransition) -> None`
- `get_recent_transitions(count: int) -> List[StateTransition]`
- `get_all_transitions() -> List[StateTransition]`
- `clear() -> None`

**StateTransition**
- `input_state: ObjectState`
- `action: Action`
- `output_state: ObjectState`
- `input_game_state: GameState`
- `output_game_state: GameState`

### 2. World Model Representation

**WorldModel**
- `predict_next_state(current_state: ObjectState, action: Action, history: List[StateTransition]) -> ObjectState`
- `evaluate_log_probability(state: ObjectState, action: Action, next_state: ObjectState) -> float`
- `sample_next_state(current_state: ObjectState, action: Action) -> ObjectState`
- `get_experts() -> List[WeightedExpert]`
- `with_new_experts(experts: List[WeightedExpert]) -> WorldModel`

**WeightedExpert**
- `expert: Expert`
- `weight: float`

**Expert**
- `code: str`  # Python function as string
- `context_length: int`  # -1 for MDP, >0 for POMDP
- `execute(state: ObjectState, action: Action, history: List[StateTransition]) -> ObjectState`

### 3. Expert Synthesis

**ExpertSynthesizer**
- `synthesize_experts(transitions: List[StateTransition], object_type: str) -> List[Expert]`
- `synthesis_mode: SynthesisMode`  # MDP or POMDP

**SurprisingTransitionSelector**
- `select_surprising_transitions(transitions: List[StateTransition], world_model: WorldModel, batch_size: int) -> List[int]`
- `surprise_threshold: float`

### 4. Weight Optimization

**WeightFitter**
- `fit_weights(experts: List[Expert], transitions: List[StateTransition], fast_mode: bool) -> List[WeightedExpert]`
- `fit_only_new_weights(experts: List[Expert], transitions: List[StateTransition]) -> List[WeightedExpert]`

**ExpertPruner**
- `prune_experts(weighted_experts: List[WeightedExpert], threshold: float) -> List[WeightedExpert]`

### 5. Object State Management

**ObjectState**
- `objects: List[GameObject]`
- `get_objects_by_type(obj_type: str) -> List[GameObject]`
- `get_object_by_id(obj_id: int) -> GameObject`
- `deepcopy() -> ObjectState`

**GameObject**
- `id: int`
- `obj_type: str`
- `attributes: Dict[str, Any]`  # position, velocity, etc.
- `touches(other: GameObject) -> bool`

### 6. Planning and Execution

**Goal**
- `target_object_type: str`
- `target_object_id: Optional[int]`
- `is_achieved(state: ObjectState) -> bool`

**Planner**
- `plan_action(current_state: ObjectState, goal: Goal, world_model: WorldModel) -> Action`
- `planning_budget: int`

**Agent**
- `execute_action(action: Action) -> StateTransition`
- `get_current_state() -> ObjectState`
- `reset() -> ObjectState`

### 7. Online Learning Orchestration

**OnlineLearner**
- `experience_buffer: ExperienceBuffer`
- `world_model: WorldModel`
- `synthesizer: ExpertSynthesizer`
- `weight_fitter: WeightFitter`
- `pruner: ExpertPruner`
- `surprising_selector: SurprisingTransitionSelector`
- `update_cycle(new_transitions: List[StateTransition]) -> None`
- `save_checkpoint() -> None`
- `load_checkpoint() -> None`

**LearningConfig**
- `batch_size: int`
- `surprise_threshold: float`
- `pruning_threshold: float`
- `fast_update_frequency: int`
- `max_experts_per_object_type: int`

## Core Algorithm Implementation

```python
class PoEWorldLearningLoop:
    """
    Core learning loop implementing PoE World's online learning algorithm.
    This is the main orchestrator that coordinates all domain objects.
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.experience_buffer = ExperienceBuffer()
        self.world_model = WorldModel()
        self.synthesizer = ExpertSynthesizer()
        self.weight_fitter = WeightFitter()
        self.pruner = ExpertPruner()
        self.surprising_selector = SurprisingTransitionSelector(
            threshold=config.surprise_threshold
        )
        
    def initialize_from_demonstrations(self, demonstrations: List[StateTransition]) -> None:
        """
        Initialize world model from pre-recorded demonstration data.
        This is the offline learning phase.
        """
        # Add all demonstration data to experience buffer
        for transition in demonstrations:
            self.experience_buffer.add_transition(transition)
        
        # Perform initial expert synthesis and weight fitting
        self._perform_learning_cycle()
        
    def online_learning_cycle(self, new_transitions: List[StateTransition]) -> None:
        """
        Main online learning cycle that updates the world model with new experience.
        This is called after each batch of new transitions from agent interaction.
        """
        # Step 1: Add new data to experience buffer
        for transition in new_transitions:
            self.experience_buffer.add_transition(transition)
        
        # Step 2: Identify surprising transitions that need new experts
        all_transitions = self.experience_buffer.get_all_transitions()
        surprising_indices = self.surprising_selector.select_surprising_transitions(
            all_transitions, self.world_model, self.config.batch_size
        )
        
        if not surprising_indices:
            return  # No learning needed
        
        # Step 3: Extract surprising transitions for synthesis
        surprising_transitions = [all_transitions[i] for i in surprising_indices]
        
        # Step 4: Synthesize new experts for each object type
        new_experts = []
        for object_type in self._get_object_types(surprising_transitions):
            object_transitions = self._filter_by_object_type(surprising_transitions, object_type)
            experts = self.synthesizer.synthesize_experts(object_transitions, object_type)
            new_experts.extend(experts)
        
        if not new_experts:
            return  # No new experts synthesized
        
        # Step 5: Combine with existing experts
        existing_experts = self.world_model.get_experts()
        all_experts = existing_experts + new_experts
        
        # Step 6: Fit weights (fast mode for online updates)
        weighted_experts = self.weight_fitter.fit_only_new_weights(
            all_experts, all_transitions
        )
        
        # Step 7: Prune low-weight experts
        pruned_experts = self.pruner.prune_experts(
            weighted_experts, self.config.pruning_threshold
        )
        
        # Step 8: Update world model
        self.world_model = self.world_model.with_new_experts(pruned_experts)
        
    def _perform_learning_cycle(self) -> None:
        """
        Internal method for the complete learning cycle (synthesis + fitting + pruning).
        Used for both initial learning and comprehensive updates.
        """
        all_transitions = self.experience_buffer.get_all_transitions()
        
        # Synthesize experts for all object types
        new_experts = []
        for object_type in self._get_object_types(all_transitions):
            object_transitions = self._filter_by_object_type(all_transitions, object_type)
            experts = self.synthesizer.synthesize_experts(object_transitions, object_type)
            new_experts.extend(experts)
        
        # Fit all weights together (comprehensive mode)
        weighted_experts = self.weight_fitter.fit_weights(
            new_experts, all_transitions, fast_mode=False
        )
        
        # Prune experts
        pruned_experts = self.pruner.prune_experts(
            weighted_experts, self.config.pruning_threshold
        )
        
        # Update world model
        self.world_model = self.world_model.with_new_experts(pruned_experts)
```

## Agent Integration

```python
class PoEWorldAgent:
    """
    Agent that uses the learned world model for planning and execution.
    Integrates with the online learning loop.
    """
    
    def __init__(self, learning_loop: PoEWorldLearningLoop, planner: Planner, goal: Goal):
        self.learning_loop = learning_loop
        self.planner = planner
        self.goal = goal
        self.agent = Agent()
        self.transition_buffer = []
        
    def execute_episode(self) -> bool:
        """
        Execute a single episode, collecting experience and triggering online learning.
        Returns True if goal was achieved.
        """
        current_state = self.agent.reset()
        
        while not self.goal.is_achieved(current_state):
            # Use world model for planning
            action = self.planner.plan_action(
                current_state, self.goal, self.learning_loop.world_model
            )
            
            # Execute action and collect transition
            transition = self.agent.execute_action(action)
            self.transition_buffer.append(transition)
            
            # Fast online updates during execution
            if len(self.transition_buffer) % self.learning_loop.config.fast_update_frequency == 0:
                self.learning_loop.online_learning_cycle(self.transition_buffer)
                self.transition_buffer = []  # Clear buffer after update
            
            current_state = transition.output_state
            
            # Check for episode termination
            if self._is_episode_terminated(transition):
                break
        
        # Comprehensive update at end of episode
        if self.transition_buffer:
            self.learning_loop.online_learning_cycle(self.transition_buffer)
            self.transition_buffer = []
        
        return self.goal.is_achieved(current_state)
```

## Key Learning Points

### 1. Surprising Transition Selection
The algorithm only synthesizes experts for transitions that the current world model cannot explain well (log-probability below threshold). This focuses computational resources on learning new patterns.

### 2. Dual Update Modes
- **Fast Updates**: During agent execution, only fit weights for newly added experts
- **Comprehensive Updates**: At episode end or on failure, refit all expert weights together

### 3. Object-Type Modularity
Experts are synthesized separately for each object type, enabling modular learning of independent causal mechanisms.

### 4. Experience Buffer Management
The experience buffer accumulates all transitions and provides the data source for both expert synthesis and weight fitting.

## Goal Definition and Sources

### Goal Specification in PoE World

PoE World uses **hardcoded, game-specific goals** that are defined at the configuration level and passed to the agent. The goals are not learned or discovered by the system, but rather manually specified based on the target environment.

### Goal Definition Location

**Primary Location:** `external/poe-world/classes/envs/env.py`

```python
def get_goal_obj_type_by_game(config):
    if config.task.startswith('MontezumaRevenge'):
        return 'key'  # Hardcoded goal: collect the key
    elif config.task.startswith('Pong'):
        return 'ball'  # Hardcoded goal: hit the ball
    else:
        raise NotImplementedError
```

**Usage in Main Execution:** `external/poe-world/run.py`

```python
elif config.post_synthesis_mode == 'agent':
    agent = Agent(config, learner)
    agent.plan_and_execute(get_goal_obj_type_by_game(config))  # Goal passed here
```

### Goal Detection and Processing

**Goal Object Detection:** `external/poe-world/agents/agent.py`

```python
def _get_goal_id(self, obj_list, goal_obj_type) -> int:
    """Get the ID of the goal object type"""
    goal_obj = obj_list.get_objs_by_obj_type(goal_obj_type)
    if goal_obj:
        return goal_obj[0].id  # Return first instance of goal object
    else:
        return -1  # No goal object found

def _get_goal_ids(self, obj_list, goal_obj_type) -> List[int]:
    """Get the IDs of all goal objects of a given type"""
    goal_objs = obj_list.get_objs_by_obj_type(goal_obj_type)
    if goal_objs:
        return [obj.id for obj in goal_objs]
    else:
        return []
```

**Goal Achievement Checking:** `external/poe-world/agents/mcts.py`

```python
def is_goal(self):
    return (self.get_abstract_state() == self.target_abstract_state
            ) and self.is_stable_state() and (not self.died)
```

### Goal-Driven Planning Loop

**Main Execution Loop:** `external/poe-world/agents/agent.py`

```python
def plan_and_execute(self, goal_obj_type: str = 'key') -> None:
    """Main execution loop that runs the full planning and execution pipeline."""
    n_goals_achieved = 0
    
    while True:
        # Try to find and execute a plan to reach the goal
        symbolic_plan = self._get_symbolic_plan(cur_obj_list, goal_obj_type, ...)
        
        if success:
            log.info(f"GOT GOAL! ({n_goals_achieved+1}/{self.config.agent.n_goals_to_achieve})")
            n_goals_achieved += 1
            
            if n_goals_achieved >= self.config.agent.n_goals_to_achieve:
                break  # Exit after achieving specified number of goals
```

**Multiple Goal Handling:** The agent pursues the **same goal type** multiple times, not different goals. For example:
- **Montezuma's Revenge**: Collect the key 1 time (default config)
- **Pong**: Hit the ball 100 times (Pong config)

**Goal Repetition Logic:** `external/poe-world/agents/agent.py`

```python
def _try_reach_multiple_goals(self, goal_ids: List[int], cur_obj_list, cur_game_state, n_budget_iterations) -> bool:
    """Try to reach multiple goals from last plan state"""
    # Find the best plan among available goal instances
    best_plan = None
    best_goal_id = None
    for goal_id in goal_ids:
        new_plan = self.mcts.search(cur_obj_list, str([goal_id]), world_model, ...)
        if new_plan is not None:
            if best_plan is None or len(new_plan) < len(best_plan):
                best_plan = new_plan
                best_goal_id = goal_id
    
    # Execute the best plan found
    for action in best_plan:
        cur_obj_list, cur_game_state = self.atari_env.step(action)
        # Check if goal achieved
        if self._abstract_state(cur_obj_list, target_id=best_goal_id) == str([best_goal_id]):
            success = True
            break
```

### Goal Configuration

**Goal Parameters:** Defined in configuration files and agent settings

- `goal_obj_type`: The type of object to target (e.g., 'key', 'ball')
- `n_goals_to_achieve`: Number of goals to achieve before stopping
- `target_abstract_state`: Abstract state representation of the goal

**Configuration Examples:**

**Default Config:** `external/poe-world/conf/config.yaml`
```yaml
agent:
  n_goals_to_achieve: 1  # Achieve goal once (Montezuma's Revenge)
```

**Pong Config:** `external/poe-world/conf/pong_agent.yaml`
```yaml
agent:
  n_goals_to_achieve: 100  # Achieve goal 100 times (Pong)
```

### Goal Limitations

1. **Fixed Goals**: Goals are hardcoded per game and cannot be dynamically generated
2. **Single Objective**: The agent pursues one goal type at a time (but may achieve it multiple times)
3. **No Goal Learning**: The system does not learn what goals are valuable
4. **No Goal Synthesis**: No mechanism for creating new goals based on learned capabilities
5. **External Control**: Goals are externally specified rather than emerging from the learning process
6. **Goal Repetition**: The system repeats the same goal type multiple times rather than pursuing different goals

### Goal Complexity and Scope

**PoE World Goals = Simple Object Reachability**

PoE World defines goals as **simple object contact** - literally just "touch this object":

```python
def _get_goal_id(self, obj_list, goal_obj_type) -> int:
    """Get the ID of the goal object type"""
    goal_obj = obj_list.get_objs_by_obj_type(goal_obj_type)
    if goal_obj:
        return obj[0].id  # Just find the object
    else:
        return -1
```

**Goal Achievement Check:**
```python
def is_goal(self):
    return (self.get_abstract_state() == self.target_abstract_state
            ) and self.is_stable_state() and (not self.died)
```

**Abstract State Definition:** `external/poe-world/agents/mcts.py`
```python
def get_abstract_state(self):
    if self.target_id is not None:
        try:
            player_obj = self.cur_obj_list.get_objs_by_obj_type('player')[0]
            target_obj = self.cur_obj_list.get_obj_by_id(self.target_id)
        except:
            return '[-1]'
        return str([self.target_id]) if player_obj.overlaps(target_obj) else '[-1]'
```

**Object Overlap Definition:** `external/poe-world/classes/helper.py`
```python
def overlaps(self, other):
    return pygame.Rect(self.x - 1, self.y, self.w + 2, self.h).colliderect(
        pygame.Rect(other.x, other.y, other.w, other.h))
```

**Reward Function Evidence:** `external/poe-world/run_rl.py`
```python
def reward_fn(obj_list) -> float:
    try:
        player_obj = obj_list.get_objs_by_obj_type('player')[0]
        key_obj = obj_list.get_objs_by_obj_type('key')[0]
        return 100 if player_obj.overlaps(key_obj) else 0  # Reward = 100 if touching key
    except:
        return 0
```

This translates to: **"Is the player touching the target object?"** - literally checking if the player's bounding box overlaps with the target object's bounding box.

**Examples from PoE World:**
- **Montezuma's Revenge**: "Touch the key" (not "collect the key and escape")
- **Pong**: "Touch the ball" (not "win the game by scoring points")

**What PoE World Goals Are:**
- Single spatial objectives (reach location, touch object)
- Immediate success conditions (object contact)
- Simple state changes (player position)
- Short-horizon planning (few actions to achieve)

**What PoE World Goals Are NOT:**
- Multi-step objectives (craft item then use it)
- Complex success conditions (inventory changes, crafting completion)
- Long-term planning (sequence of dependent actions)
- Resource management (collect materials, manage inventory)
- Hierarchical objectives (subgoals and goal decomposition)

**Comparison Example:**

**PoE World Goal (Simple):**
```
Goal: "Touch the key"
- Single action: Move to key location
- Success condition: Player object overlaps with key object
- No multi-step planning required
```

**Complex Goal (Not in PoE World):**
```
Goal: "Craft a powerful pickaxe and mine diamond"
- Step 1: Collect wood (multiple actions)
- Step 2: Craft wooden pickaxe (crafting action)
- Step 3: Mine stone (mining actions)
- Step 4: Craft stone pickaxe (crafting action)
- Step 5: Find diamond ore (exploration)
- Step 6: Mine diamond (mining action)
- Success condition: Diamond in inventory
```

**Why PoE World Uses Simple Goals:**

1. **Atari Game Constraints**: Atari games have simple success conditions with no inventory systems or complex state changes
2. **Planning Algorithm Limitations**: MCTS planner designed for short-horizon planning, no long-term goal decomposition
3. **World Model Scope**: Experts model immediate state transitions, not complex interactions or high-level actions

## Limitations: Externally Specified Goals

Goals are hardcoded based on the game (e.g., "collect key" for Montezuma's Revenge, "hit ball" for Pong). The system does not discover or generate its own goals.