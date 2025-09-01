# Issue #003: Creation vs Non-Creation Expert Separation in PoE-World

## Overview

PoE-World separates experts into two categories: **creation experts** and **non-creation experts**. This separation is not just data filtering - it requires fundamentally different synthesis strategies and prompts.

## What PoE-World Does

### 1. Object-Level Separation

PoE-World separates **objects** (not synthesizers) into two categories:

- **Non-creation objects**: Existing objects that persist across state transitions
- **Creation objects**: New objects that appear in the output state but weren't in the input state

**Source**: `external/poe-world/learners/obj_model_learner.py:430-450`

```python
def _separate_creation_in_observation(self, c: List[StateTransitionTriplet]):
    non_creation_c, creation_c = [], []
    for x in c:
        _, _, leftover_list2 = match_two_obj_lists(x.input_state, x.output_state)
        
        # Non-creation: objects that exist in both states
        non_creation_x = StateTransitionTriplet(
            x.input_state.deepcopy(), x.event,
            ObjList([x.output_state[idx] for idx in range(len(x.output_state))
                     if idx not in leftover_list2]))
        
        # Creation: objects that only exist in output state (leftover_list2)
        creation_x = StateTransitionTriplet(
            x.input_state.deepcopy(), x.event,
            ObjList([x.output_state[idx] for idx in leftover_list2]))
```

### 2. Object Matching Logic

PoE-World uses `match_two_obj_lists()` to identify which objects are "new":

**Source**: `external/poe-world/classes/helper.py:1338-1380`

```python
def match_two_obj_lists(obj_list1: "ObjList", obj_list2: "ObjList"):
    """
    Matches objects in two ObjLists by ID or obj_type, returning good pairs and leftover indices.
    """
    # Assumption: output from world model is obj_list1 and newly created objects all have id = -1
    for i, obj1 in enumerate(obj_list1):
        if obj1.id == -1:
            continue
        for j, obj2 in enumerate(obj_list2):
            if obj1.id == obj2.id:
                good_pairs.append((i, j))
                list1_hsh[i] = True
                list2_hsh[j] = True
                break
```

### 3. Different Prompt Sets

Creation and non-creation synthesizers use **completely different prompts**:

**Non-creation interpretation prompt** (`interpret_velocity_4_prompt`):
```
Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +2
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects that touch a ladder object with zero x-axis velocity set their x-axis velocity to +2
2. The player objects that touch a ladder object with positive y-axis velocity set their y-axis velocity to -4
```

**Creation interpretation prompt** (`interpret_velocity_creation_prompt`):
```
Example output list of object changes:
- The player object (id = 0) is deleted
- A new player object is created at (x=76,y=73)

Example reasons:
1. If player objects touch an unknown object, a new player object is created at (x=76, y+73)
2. If player objects touch a ladder object, a new player object is created at (x=76, y+73)
3. The player objects that touch an unknown object are deleted
4. The player objects that touch a ladder object are deleted
```

**Source**: `external/poe-world/prompts/synthesizer.py:1183-1220`

### 4. Creation Synthesizers Handle Both Creation and Deletion

Creation synthesizers are actually **"object lifecycle synthesizers"** that handle:

- **Object creation** - spawning new objects
- **Object deletion** - removing existing objects  
- **Object replacement** - deleting old objects and creating new ones

**Source**: `external/poe-world/prompts/synthesizer.py:1420-1480`

Example from `interpret_5_prompt`:
```
Example output list of object changes:
- The sword_hud object (id = 5) is deleted
- A new sword_hud object is created at (x=96, y=37)
- A new sword_hud object is created at (x=32, y=34)
- A new sword_hud object is created at (x=64, y=64)
```

### 5. Different Function Capabilities

The final functions generated are fundamentally different:

- **Non-creation functions**: Modify existing object attributes (velocity, position, health, etc.)
- **Creation functions**: Call `create_object()` to spawn new objects AND set `deleted` attribute to remove objects

**Source**: `external/poe-world/prompts/synthesizer.py:1534-1562`

```python
creation_starter_program = """\
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    obj_list = obj_list.deepcopy() # make a new copy of obj_list
    obj_list = obj_list.create_object(np.asarray([[0, 0]]), '{obj_type}', Position(x=RandomValues(obj_list.grid_size), y=RandomValues(obj_list.grid_size)))
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        {obj_type}_obj.deleted = RandomValues([1])
    return obj_list"""
```

## Key Insight

**Creation vs non-creation is a property of the data AND requires different synthesis strategies.** The separation is not just data filtering - it's about fundamentally different types of object changes that require different prompts, examples, and function generation approaches.

## What This Means for Our Implementation

We cannot use the same synthesizer for both creation and non-creation experts. We need:

1. **Separate prompt sets** for creation vs non-creation
2. **Object lifecycle detection** in our observable extractor (creation + deletion)
3. **Different synthesis strategies** for each expert type

## Design Decisions for Entity Lifecycle Handling

### **The Core Problem We Solved**

We needed to understand how to handle entity creation and deletion in our observable extractor. The key insight came from understanding how PoE-World actually works, not from trying to replicate its complexity.

### **How PoE-World Actually Works (The Key Insight)**

#### **1. Experts Predict by Modifying State, Not by Predicting Counts**
PoE-World experts don't predict "create 3 cows" - they predict by actually creating objects in the state:

```python
# Expert function creates entities by calling create_object()
def alter_cow_objects(obj_list: ObjList, action: str) -> ObjList:
    # Create a new cow at specific position
    obj_list = obj_list.create_object('cow', 50, 30)
    return obj_list

# Expert function deletes entities by setting deleted attribute
def alter_zombie_objects(obj_list: ObjList, action: str) -> ObjList:
    for zombie in obj_list.get_objs_by_obj_type('zombie'):
        zombie.deleted = 1  # Mark for deletion
    return obj_list
```

#### **2. The Creation Model Ensures At Least One Object Exists**
After all experts run, PoE-World's creation model checks:
```python
if self.name == 'creation':
    for objs_dist in objs_dists:
        if len(objs_dist) == 0:  # If no objects of this type exist
            new_objs_dist = new_objs_dist.create_object(self.obj_type, 0, 0)  # Create one at (0,0)
            new_objs_dist[0].deleted = RandomValues([1])  # Initially invisible
```

**This only happens if NO experts created any objects of that type.**

#### **3. Attribute Filling Happens After Expert Execution**
After experts run, `fill_unset_values_with_uniform` is called:
```python
def fill_unset_values_with_uniform(obj_list_dist: "ObjList", size_change_flag: bool = False):
    for obj_dist in obj_list_dist:
        if not isinstance(obj_dist.velocity_x, RandomValues):
            obj_dist.velocity_x = RandomValues(all_possible_velocities)  # Fill with uniform
        if not isinstance(obj_dist.velocity_y, RandomValues):
            obj_dist.velocity_y = RandomValues(all_possible_velocities)  # Fill with uniform
        if not isinstance(obj_dist.deleted, RandomValues):
            obj_dist.deleted = RandomValues(np.arange(2))  # Fill with uniform
```

**This only fills attributes that experts didn't set.**

#### **4. Object Matching and Loss Computation**
PoE-World uses `match_two_obj_lists()` to identify which objects are "new" and computes loss:
```python
def evaluate_logprobs_of_obj_list(obj_list_dist: "ObjList", obj_list: "ObjList", by_pos: bool = False):
    good_pairs, leftover_list1, leftover_list2 = match_two_obj_lists(obj_list_dist, obj_list)
    
    # Compute loss for matched objects
    for idx1, idx2 in good_pairs:
        obj_dist = obj_list_dist[idx1]
        obj = obj_list[idx2]
        # ... compute loss for position, velocity, deleted status
    
    # Handle leftover predicted objects (they should be deleted)
    for idx in leftover_list1:
        obj_dist = obj_list_dist[idx]
        if isinstance(obj_dist.deleted, RandomValues):
            logprobs += 100 * obj_dist.deleted.evaluate_logprobs(1)
    
    # Critical: If there are leftover observed objects, return LOG_IMPOSSIBLE_VALUE
    if len(leftover_list2) > 0:
        return LOG_IMPOSSIBLE_VALUE
```

**This means:**
- **If experts predict a cow at (50, 30)** and **a cow appears at (50, 30)** → Good match, compute loss
- **If experts predict a cow at (50, 30)** but **no cow appears** → Bad prediction, high loss  
- **If experts predict no cow** but **a cow appears** → LOG_IMPOSSIBLE_VALUE (very bad)

### **What This Means for Our Implementation**

#### **The Key Insight: We DO Need to Track Entity Lifecycle**
**PoE-World DOES check for newness** through object matching. The system needs to know: "Did the expert correctly predict that a cow would appear at position (50, 30)?"

#### **Why We Need Entity Lifecycle Observables**
1. **Entity existence tracking**: Track whether each specific entity exists (0/1) to confirm deletion
2. **Entity count tracking**: Track total count per type to confirm creation
3. **Entity attribute tracking**: Track position, health, etc. to confirm positioning

#### **The Architecture We Need**
**Entity lifecycle is handled through observable comparison, not through state manipulation:**

1. **Experts modify state** (create/delete entities directly)
2. **Observable extractor extracts** entity lifecycle observables
3. **Loss function compares** predicted vs observed observables
4. **Experts get penalized** for bad entity lifecycle predictions

### **Our Observable Extractor Implementation**

#### **1. `extract_attribute_predictions` Method**
```python
def extract_attribute_predictions(self, state: WorldState) -> dict[ObservableId, DiscreteDistribution]:
    predictions = {}
    
    # Entity existence tracking (per entity ID)
    # Track whether each specific entity exists (0 = deleted, 1 = exists)
    for entity in state.objects:
        entity_id = entity.entity_id
        predictions[ObservableId(f"entity_exists_{entity_id}")] = DiscreteDistribution.from_uniform([0, 1])
    
    # Entity count tracking (per entity type)
    # Track total count of each entity type (0, 1, 2, 3, 4, 5)
    for entity_type in ["cow", "zombie", "skeleton", "plant", "arrow", "fence"]:
        predictions[ObservableId(f"entity_count_{entity_type}")] = DiscreteDistribution.from_uniform([0, 1, 2, 3, 4, 5])
    
    # Entity attribute tracking (per entity ID)
    # Track position, health, and other attributes for each entity
    for entity in state.objects:
        entity_id = entity.entity_id
        predictions[ObservableId(f"entity_{entity_id}_position_x")] = DiscreteDistribution.from_uniform([0, 1, 2, ..., 100])
        predictions[ObservableId(f"entity_{entity_id}_position_y")] = DiscreteDistribution.from_uniform([0, 1, 2, ..., 100])
        predictions[ObservableId(f"entity_{entity_id}_health")] = DiscreteDistribution.from_uniform([0, 1, 2, ..., 100])
    
    return predictions
```

#### **2. `get_observed_outcomes` Method**
```python
def get_observed_outcomes(self, state: WorldState) -> dict[ObservableId, int]:
    observed = {}
    
    # Count entities by type
    entity_counts = {}
    for entity in state.objects:
        entity_type = entity.name  # "cow", "zombie", etc.
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Record entity counts
    for entity_type, count in entity_counts.items():
        observed[ObservableId(f"entity_count_{entity_type}")] = count
    
    # Record entity existence (all entities in state exist)
    for entity in state.objects:
        entity_id = entity.entity_id
        observed[ObservableId(f"entity_exists_{entity_id}")] = 1
    
    # Record entity attributes
    for entity in state.objects:
        entity_id = entity.entity_id
        observed[ObservableId(f"entity_{entity_id}_position_x")] = entity.position.x
        observed[ObservableId(f"entity_{entity_id}_position_y")] = entity.position.y
        observed[ObservableId(f"entity_{entity_id}_health")] = entity.health
    
    return observed
```

#### **3. `apply_expert_predictions` Method**
```python
def apply_expert_predictions(self, new_state: WorldState, expert_predictions: dict[ObservableId, list[DiscreteDistribution]], weights: torch.Tensor) -> WorldState:
    # We DON'T need to create/delete entities here
    # Experts have already modified the state by calling create_object(), etc.
    # We just return the state as-is
    
    # The key insight: experts predict by modifying state, not by predicting observables
    # Our observable extractor just observes what experts did to the state
    
    return new_state
```

### **Why This Design Works**

#### **1. Architecture Compatibility**
- **Weight fitter**: Automatically handles new observables without changes
- **World model**: Automatically handles new observables without changes  
- **Core protocols**: No changes needed
- **ObservableId**: No changes needed (it's just a string wrapper)

#### **2. Separation of Concerns**
- **Experts predict by modifying state**: They create/delete entities directly
- **Observable extractor extracts observables**: Tracks entity lifecycle through state comparison
- **Weight fitter computes loss**: Observable agnostic, just computes log probs
- **World model samples states**: Observable agnostic, just combines predictions

#### **3. Entity Lifecycle Evaluation**
- **Entity creation**: Evaluated through entity count observables
- **Entity deletion**: Evaluated through entity existence observables  
- **Entity positioning**: Evaluated through position observables
- **Entity attributes**: Evaluated through attribute observables

### **Handling Missing Predictions**

When experts don't predict certain observables, the system gracefully handles it:

```python
# If expert doesn't predict entity_exists_{id}, what happens?
# Observable extractor fills missing predictions with uniform distributions
predictions[ObservableId(f"entity_exists_{entity_id}")] = DiscreteDistribution.from_uniform([0, 1])

# This means:
# - Low confidence in the prediction
# - Medium penalty in loss computation (not catastrophic)
# - Experts can specialize in what they're good at predicting
```

**Uniform distributions = "I Don't Know"** - experts can specialize in what they want to predict.

### **Complete Example: Expert Predicts Cow Creation**

```python
# Input state: entity_1 (cow at (10, 20), health 9)
# Expert predicts: Create a new cow at (50, 30) with health 5
# Expert modifies state: obj_list.create_object('cow', 50, 30)

# Observable extractor extracts:
# - entity_count_cow: predicted=2, observed=2 → ✓ (cow was created)
# - entity_exists_1: predicted=1, observed=1 → ✓ (original cow survived)
# - entity_exists_2: predicted=1, observed=1 → ✓ (new cow was created)
# - entity_2_position_x: predicted=50, observed=50 → ✓ (new cow at correct X)
# - entity_2_position_y: predicted=30, observed=30 → ✓ (new cow at correct Y)
# - entity_2_health: predicted=5, observed=5 → ✓ (new cow has correct health)

# Loss computation rewards the expert for:
# 1. Correctly predicting cow creation
# 2. Correctly predicting cow position
# 3. Correctly predicting cow health
```

**The expert gets penalized for any wrong prediction** - wrong count, wrong position, wrong health, etc. This gives us the same precision as PoE-World's object matching system.

## References

- Object separation logic: `external/poe-world/learners/obj_model_learner.py:430-450`
- Object matching: `external/poe-world/classes/helper.py:1338-1380`
- Creation prompts: `external/poe-world/prompts/synthesizer.py:1183-1220`
- Creation examples: `external/poe-world/prompts/synthesizer.py:1420-1480`
- Function templates: `external/poe-world/prompts/synthesizer.py:1534-1562`
