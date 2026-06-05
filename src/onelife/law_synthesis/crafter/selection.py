from crafter_oo.state_reconstruction import CowState, ZombieState

from .transition import Transition
from .change_detection import find_transitions_with_attribute_changes


def filter_zombie_damage_transitions(transitions: list[Transition]) -> list[Transition]:
    """
    Filter transitions to only include those where zombie health decreases.

    Args:
        transitions: List of all transitions

    Returns:
        List of transitions where at least one zombie took damage
    """
    zombie_damage_transitions = []
    for transition in transitions:
        # Check if any zombies have health changes
        state_zombies = [
            obj for obj in transition.state.objects if isinstance(obj, ZombieState)
        ]
        next_state_zombies = [
            obj for obj in transition.next_state.objects if isinstance(obj, ZombieState)
        ]

        # Create lookup by entity_id
        state_zombie_by_id = {z.entity_id: z for z in state_zombies}
        next_state_zombie_by_id = {z.entity_id: z for z in next_state_zombies}

        has_zombie_damage = False
        for entity_id, state_zombie in state_zombie_by_id.items():
            if entity_id in next_state_zombie_by_id:
                next_zombie = next_state_zombie_by_id[entity_id]
                if hasattr(state_zombie, "health") and hasattr(next_zombie, "health"):
                    if state_zombie.health > next_zombie.health:
                        has_zombie_damage = True
                        break

        if has_zombie_damage:
            zombie_damage_transitions.append(transition)

    return zombie_damage_transitions


def filter_cow_damage_transitions(transitions: list[Transition]) -> list[Transition]:
    """
    Filter transitions to only include those where cow health decreases.

    Args:
        transitions: List of all transitions

    Returns:
        List of transitions where at least one cow took damage
    """
    cow_damage_transitions = []
    for transition in transitions:
        # Check if any cows have health changes
        state_cows = [
            obj for obj in transition.state.objects if isinstance(obj, CowState)
        ]
        next_state_cows = [
            obj for obj in transition.next_state.objects if isinstance(obj, CowState)
        ]

        # Create lookup by entity_id
        state_cow_by_id = {c.entity_id: c for c in state_cows}
        next_state_cow_by_id = {c.entity_id: c for c in next_state_cows}

        has_cow_damage = False
        for entity_id, state_cow in state_cow_by_id.items():
            if entity_id in next_state_cow_by_id:
                next_cow = next_state_cow_by_id[entity_id]
                if hasattr(state_cow, "health") and hasattr(next_cow, "health"):
                    if state_cow.health > next_cow.health:
                        has_cow_damage = True
                        break

        if has_cow_damage:
            cow_damage_transitions.append(transition)

    return cow_damage_transitions


def filter_player_move_transitions(transitions: list[Transition]) -> list[Transition]:
    """
    Filter transitions to only include one transition for each move action type where player position changed.

    Args:
        transitions: List of all transitions

    Returns:
        List of transitions with one transition per move action type where player position changed
    """
    move_actions = {"Move North", "Move East", "Move South", "Move West"}
    player_move_transitions = []
    seen_actions = set()

    for transition in transitions:
        # Check if action is a move action and we haven't seen it yet
        if transition.action not in move_actions or transition.action in seen_actions:
            continue

        # Check if player position changed
        if transition.state.player and transition.next_state.player:
            old_pos = transition.state.player.position
            new_pos = transition.next_state.player.position
            if old_pos != new_pos:
                player_move_transitions.append(transition)
                seen_actions.add(transition.action)

    return player_move_transitions


def filter_player_health_damage_transitions(
    transitions: list[Transition],
) -> list[Transition]:
    """
    Filter transitions to only include those where player health decreases.

    Args:
        transitions: List of all transitions

    Returns:
        List of transitions where player health decreased
    """
    player_damage_transitions = []
    for transition in transitions:
        # Check if player health decreased
        if transition.state.player and transition.next_state.player:
            if transition.state.player.health > transition.next_state.player.health:
                player_damage_transitions.append(transition)

    return player_damage_transitions


def select_additional_interesting_transitions(
    all_transitions: list[Transition],
    current_interesting_indices: set[int],
    min_examples_per_attribute: int = 1,
) -> list[int]:
    """
    Select additional transitions to ensure we have examples of all entity attribute changes.

    Args:
        all_transitions: All available transitions
        current_interesting_indices: Indices of transitions already marked as interesting
        min_examples_per_attribute: Minimum number of examples to find per attribute

    Returns:
        List of additional transition indices to include
    """
    # Find all transitions showing attribute changes
    attribute_changes = find_transitions_with_attribute_changes(all_transitions)

    additional_indices = set()

    # For each entity type and attribute, ensure we have enough examples
    for entity_type, attributes in attribute_changes.items():
        for attr, transition_indices in attributes.items():
            # How many examples do we already have in current interesting transitions?
            existing_examples = len(
                [
                    idx
                    for idx in transition_indices
                    if idx in current_interesting_indices
                ]
            )

            # How many more do we need?
            needed = max(0, min_examples_per_attribute - existing_examples)

            # Add additional transitions that show this attribute change
            for idx in transition_indices:
                if idx not in current_interesting_indices and needed > 0:
                    additional_indices.add(idx)
                    needed -= 1

    return sorted(list(additional_indices))
