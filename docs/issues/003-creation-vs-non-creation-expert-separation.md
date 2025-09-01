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
1. If player objects touch an unknown object, a new player object is created at (x=76, y=73)
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

## References

- Object separation logic: `external/poe-world/learners/obj_model_learner.py:430-450`
- Object matching: `external/poe-world/classes/helper.py:1338-1380`
- Creation prompts: `external/poe-world/prompts/synthesizer.py:1183-1220`
- Creation examples: `external/poe-world/prompts/synthesizer.py:1420-1480`
- Function templates: `external/poe-world/prompts/synthesizer.py:1534-1562`
