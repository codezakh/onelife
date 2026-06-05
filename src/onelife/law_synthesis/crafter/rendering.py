from .transition import Transition


from crafter_oo.state_reconstruction import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    PlayerState,
    SkeletonState,
    WorldState,
    ZombieState,
)


import difflib
import json
from pathlib import Path

import jinja2

INDUCTION_TEMPLATE_PATH = Path(__file__).parent / "law_induction_prompt_atomic.md"


def render_aspect_changes(transition: Transition, aspect: str) -> str:
    """
    Render a focused view of changes for a specific aspect of the state.

    Args:
        transition: The state transition
        aspect: The aspect to focus on (e.g., "zombie_health", "player_health", "player_inventory")

    Returns:
        Formatted string showing the relevant objects and their changes
    """
    lines = []

    if aspect == "player_health":
        # Show player health changes
        if transition.state.player and transition.next_state.player:
            state_health = transition.state.player.health
            next_health = transition.next_state.player.health

            lines.append("**Player Health:**")
            lines.append(f"  Current: {state_health}")
            lines.append(f"  Next: {next_health}")

            if state_health != next_health:
                health_change = next_health - state_health
                lines.append("  **Changes:**")
                lines.append(
                    f"    - health: {state_health} → {next_health} ({'+' if health_change > 0 else ''}{health_change})"
                )
            else:
                lines.append("  **Changes:** None")

    elif aspect == "zombie_health":
        # Show all zombies and their health changes
        state_zombies = {
            obj.entity_id: obj
            for obj in transition.state.objects
            if isinstance(obj, ZombieState)
        }
        next_zombies = {
            obj.entity_id: obj
            for obj in transition.next_state.objects
            if isinstance(obj, ZombieState)
        }

        # Show all zombies that exist in either state
        all_zombie_ids = set(state_zombies.keys()) | set(next_zombies.keys())

        for zombie_id in sorted(all_zombie_ids):
            state_zombie = state_zombies.get(zombie_id)
            next_zombie = next_zombies.get(zombie_id)

            lines.append(f"**Zombie {zombie_id}:**")

            # Show current state
            if state_zombie:
                lines.append(
                    f"  Current: health={state_zombie.health}, position=({state_zombie.position.x}, {state_zombie.position.y})"
                )
            else:
                lines.append("  Current: (zombie did not exist)")

            # Show next state
            if next_zombie:
                lines.append(
                    f"  Next: health={next_zombie.health}, position=({next_zombie.position.x}, {next_zombie.position.y})"
                )
            else:
                lines.append("  Next: (zombie no longer exists)")

            # Show changes
            changes = []
            if state_zombie and next_zombie:
                if state_zombie.health != next_zombie.health:
                    health_change = next_zombie.health - state_zombie.health
                    changes.append(
                        f"health {state_zombie.health} → {next_zombie.health} ({'+' if health_change > 0 else ''}{health_change})"
                    )
                if state_zombie.position != next_zombie.position:
                    changes.append(
                        f"position ({state_zombie.position.x}, {state_zombie.position.y}) → ({next_zombie.position.x}, {next_zombie.position.y})"
                    )
                if state_zombie.cooldown != next_zombie.cooldown:
                    cooldown_change = next_zombie.cooldown - state_zombie.cooldown
                    changes.append(
                        f"cooldown {state_zombie.cooldown} → {next_zombie.cooldown} ({'+' if cooldown_change > 0 else ''}{cooldown_change})"
                    )
            elif state_zombie and not next_zombie:
                changes.append("zombie was removed from the world")
            elif not state_zombie and next_zombie:
                changes.append("zombie was added to the world")

            if changes:
                lines.append("  **Changes:**")
                for change in changes:
                    lines.append(f"    - {change}")
            else:
                lines.append("  **Changes:** None")

            lines.append("")  # Empty line between zombies

    elif aspect == "player_inventory":
        # Show player inventory changes
        if transition.state.player and transition.next_state.player:
            state_inv = transition.state.player.inventory
            next_inv = transition.next_state.player.inventory

            lines.append("**Player Inventory:**")

            # List all inventory fields
            inventory_fields = [
                "health",
                "food",
                "drink",
                "energy",
                "sapling",
                "wood",
                "stone",
                "coal",
                "iron",
                "diamond",
                "wood_pickaxe",
                "stone_pickaxe",
                "iron_pickaxe",
                "wood_sword",
                "stone_sword",
                "iron_sword",
            ]

            changes = []
            for field in inventory_fields:
                state_val = getattr(state_inv, field, 0)
                next_val = getattr(next_inv, field, 0)
                if state_val != next_val:
                    change = next_val - state_val
                    changes.append(
                        f"{field}: {state_val} → {next_val} ({'+' if change > 0 else ''}{change})"
                    )

            if changes:
                lines.append("  **Changes:**")
                for change in changes:
                    lines.append(f"    - {change}")
            else:
                lines.append("  **Changes:** None")

    elif aspect == "player_position":
        # Show player position changes
        if transition.state.player and transition.next_state.player:
            state_pos = transition.state.player.position
            next_pos = transition.next_state.player.position

            lines.append("**Player Position:**")
            lines.append(f"  Current: ({state_pos.x}, {state_pos.y})")
            lines.append(f"  Next: ({next_pos.x}, {next_pos.y})")

            if state_pos != next_pos:
                lines.append("  **Changes:**")
                lines.append(
                    f"    - position: ({state_pos.x}, {state_pos.y}) → ({next_pos.x}, {next_pos.y})"
                )
            else:
                lines.append("  **Changes:** None")

    elif aspect in ["zombies", "cows", "skeletons", "arrows", "plants", "fences"]:
        # Generic entity change renderer
        entity_type_map = {
            "zombies": ZombieState,
            "cows": CowState,
            "skeletons": SkeletonState,
            "arrows": ArrowState,
            "plants": PlantState,
            "fences": FenceState,
        }

        entity_class = entity_type_map[aspect]
        state_entities = {
            obj.entity_id: obj
            for obj in transition.state.objects
            if isinstance(obj, entity_class)
        }
        next_entities = {
            obj.entity_id: obj
            for obj in transition.next_state.objects
            if isinstance(obj, entity_class)
        }

        # Show all entities that exist in either state
        all_entity_ids = set(state_entities.keys()) | set(next_entities.keys())

        for entity_id in sorted(all_entity_ids):
            state_entity = state_entities.get(entity_id)
            next_entity = next_entities.get(entity_id)

            lines.append(
                f"**{aspect[:-1].title()} {entity_id}:**"
            )  # Remove 's' and title case

            # Show current state
            if state_entity:
                lines.append(
                    f"  Current: health={state_entity.health}, position=({state_entity.position.x}, {state_entity.position.y})"
                )
                # Add type-specific attributes
                if isinstance(state_entity, ZombieState):
                    lines.append(f"    cooldown={state_entity.cooldown}")
                elif isinstance(state_entity, SkeletonState):
                    lines.append(f"    reload={state_entity.reload}")
                elif isinstance(state_entity, (ArrowState, PlayerState)):
                    lines.append(
                        f"    facing=({state_entity.facing.x}, {state_entity.facing.y})"
                    )
                elif isinstance(state_entity, PlantState):
                    lines.append(
                        f"    grown={state_entity.grown}, ripe={state_entity.ripe}"
                    )
            else:
                lines.append("  Current: (entity did not exist)")

            # Show next state
            if next_entity:
                lines.append(
                    f"  Next: health={next_entity.health}, position=({next_entity.position.x}, {next_entity.position.y})"
                )
                # Add type-specific attributes
                if isinstance(next_entity, ZombieState):
                    lines.append(f"    cooldown={next_entity.cooldown}")
                elif isinstance(next_entity, SkeletonState):
                    lines.append(f"    reload={next_entity.reload}")
                elif isinstance(next_entity, (ArrowState, PlayerState)):
                    lines.append(
                        f"    facing=({next_entity.facing.x}, {next_entity.facing.y})"
                    )
                elif isinstance(next_entity, PlantState):
                    lines.append(
                        f"    grown={next_entity.grown}, ripe={next_entity.ripe}"
                    )
            else:
                lines.append("  Next: (entity no longer exists)")

            # Show changes
            changes = []
            if state_entity and next_entity:
                if state_entity.health != next_entity.health:
                    health_change = next_entity.health - state_entity.health
                    changes.append(
                        f"health {state_entity.health} → {next_entity.health} ({'+' if health_change > 0 else ''}{health_change})"
                    )
                if state_entity.position != next_entity.position:
                    changes.append(
                        f"position ({state_entity.position.x}, {state_entity.position.y}) → ({next_entity.position.x}, {next_entity.position.y})"
                    )

                # Type-specific attribute changes
                if (
                    isinstance(state_entity, ZombieState)
                    and isinstance(next_entity, ZombieState)
                    and state_entity.cooldown != next_entity.cooldown
                ):
                    cooldown_change = next_entity.cooldown - state_entity.cooldown
                    changes.append(
                        f"cooldown {state_entity.cooldown} → {next_entity.cooldown} ({'+' if cooldown_change > 0 else ''}{cooldown_change})"
                    )
                elif (
                    isinstance(state_entity, SkeletonState)
                    and isinstance(next_entity, SkeletonState)
                    and state_entity.reload != next_entity.reload
                ):
                    reload_change = next_entity.reload - state_entity.reload
                    changes.append(
                        f"reload {state_entity.reload} → {next_entity.reload} ({'+' if reload_change > 0 else ''}{reload_change})"
                    )
                elif (
                    isinstance(state_entity, (ArrowState, PlayerState))
                    and isinstance(next_entity, (ArrowState, PlayerState))
                    and state_entity.facing != next_entity.facing
                ):
                    changes.append(
                        f"facing ({state_entity.facing.x}, {state_entity.facing.y}) → ({next_entity.facing.x}, {next_entity.facing.y})"
                    )
                elif isinstance(state_entity, PlantState) and isinstance(
                    next_entity, PlantState
                ):
                    if state_entity.grown != next_entity.grown:
                        grown_change = next_entity.grown - state_entity.grown
                        changes.append(
                            f"grown {state_entity.grown} → {next_entity.grown} ({'+' if grown_change > 0 else ''}{grown_change})"
                        )
                    if state_entity.ripe != next_entity.ripe:
                        changes.append(f"ripe {state_entity.ripe} → {next_entity.ripe}")

            elif state_entity and not next_entity:
                changes.append("entity was removed from the world")
            elif not state_entity and next_entity:
                changes.append("entity was added to the world")

            if changes:
                lines.append("  **Changes:**")
                for change in changes:
                    lines.append(f"    - {change}")
            else:
                lines.append("  **Changes:** None")

            lines.append("")  # Empty line between entities

    elif aspect == "map_tiles":
        # Show material changes in local view
        if transition.state.player:
            player_x, player_y = (
                transition.state.player.position.x,
                transition.state.player.position.y,
            )
            view_x, view_y = transition.state.view

            start_x = max(0, player_x - view_x)
            end_x = min(len(transition.state.materials), player_x + view_x + 1)
            start_y = max(0, player_y - view_y)
            end_y = min(
                len(transition.state.materials[0]) if transition.state.materials else 0,
                player_y + view_y + 1,
            )

            changes = []
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if (
                        x < len(transition.state.materials)
                        and y < len(transition.state.materials[x])
                        and x < len(transition.next_state.materials)
                        and y < len(transition.next_state.materials[x])
                    ):
                        state_mat = transition.state.materials[x][y]
                        next_mat = transition.next_state.materials[x][y]
                        if state_mat != next_mat:
                            changes.append(f"({x}, {y}): {state_mat} → {next_mat}")

            if changes:
                lines.append("**Map Tile Changes (Local View):**")
                for change in changes[:20]:  # Limit to first 20 changes
                    lines.append(f"  - {change}")
                if len(changes) > 20:
                    lines.append(f"  ... and {len(changes) - 20} more changes")
            else:
                lines.append("**Map Tile Changes:** None")

    if not lines:
        return f"No specific rendering implemented for aspect: {aspect}"

    return "\n".join(lines)


def create_ascii_map_with_legend(state: WorldState) -> tuple[str, str]:
    """
    Create a 2D ASCII map of the materials grid and entities centered on the player location.

    Args:
        state: The world state containing materials grid, objects, and player position

    Returns:
        A tuple of (map_string, legend_string)
    """
    if state.player is None:
        return "No player found in state", ""

    # Material to emoji mapping - only materials from constants.py
    material_emojis = {
        "grass": "🌱",
        "stone": "🪨",
        "coal": "⚫",
        "iron": "🔩",
        "diamond": "💎",
        "tree": "🌳",
        "water": "💧",
        "lava": "🌋",
        "sand": "🏖️",
        "path": "◽️",
        "table": "🪑",
        "furnace": "🔩",
        None: "⬜",  # Empty space
    }

    # Entity to emoji mapping
    entity_emojis = {
        "player": "👤",
        "cow": "🐄",
        "zombie": "🧟",
        "skeleton": "💀",
        "arrow": "🏹",
        "plant": "🌾",
        "fence": "🚧",
    }

    # Get player position
    player_x, player_y = state.player.position.x, state.player.position.y

    # Get view range (how far to show around player)
    view_x, view_y = state.view

    # Calculate bounds for the map - ensure player is centered
    start_x = max(0, player_x - view_x)
    end_x = min(len(state.materials), player_x + view_x + 1)
    start_y = max(0, player_y - view_y)
    end_y = min(
        len(state.materials[0]) if state.materials else 0, player_y + view_y + 1
    )

    # Create a map of entities by position for quick lookup
    entities_by_pos = {}
    for obj in state.objects:
        if hasattr(obj, "position") and obj.position:
            pos_key = (obj.position.x, obj.position.y)
            entities_by_pos[pos_key] = obj

    # Create the map
    map_lines = []
    map_lines.append(f"World Map (centered on player at ({player_x}, {player_y}))")
    map_lines.append(f"View range: {view_x}x{view_y}")
    map_lines.append(f"Map bounds: x[{start_x}:{end_x}], y[{start_y}:{end_y}]")
    map_lines.append(
        f"World size: {len(state.materials)}x{len(state.materials[0]) if state.materials else 0}"
    )
    map_lines.append(
        f"Player should appear at relative position: ({player_x - start_x}, {player_y - start_y})"
    )
    map_lines.append("=" * 50)

    # Add coordinate labels for x-axis
    x_labels = "    "  # Padding for y-axis labels
    for x in range(start_x, end_x):
        x_labels += f"{x:2d} "
    map_lines.append(x_labels)
    map_lines.append("")

    # Create the map grid
    for y in range(start_y, end_y):
        line = f"{y:2d} "  # Y coordinate label
        for x in range(start_x, end_x):
            pos_key = (x, y)

            # Check if there's an entity at this position
            if pos_key in entities_by_pos:
                entity = entities_by_pos[pos_key]
                entity_name = getattr(entity, "name", "unknown")
                emoji = entity_emojis.get(entity_name, "❓")
            elif x < len(state.materials) and y < len(state.materials[x]):
                # No entity, show material
                material = state.materials[x][y]
                emoji = material_emojis.get(material, "❓")
            else:
                emoji = "⬜"  # Out of bounds

            # Always show player at their position, even if there's another entity
            if x == player_x and y == player_y:
                emoji = "👤"

            line += f"{emoji} "
        map_lines.append(line)

    # Create legend separately
    legend_lines = []
    legend_lines.append("Legend:")

    # Add entity legend first
    legend_lines.append("Entities:")
    for entity_name, emoji in entity_emojis.items():
        legend_lines.append(f"  {emoji} = {entity_name}")

    legend_lines.append("")
    legend_lines.append("Materials:")
    for material, emoji in material_emojis.items():
        if material is not None:
            legend_lines.append(f"  {emoji} = {material}")

    return "\n".join(map_lines), "\n".join(legend_lines)


def diff(self: Transition) -> str:
    excluded_fields = {"event_bus", "serialized_random_state"}

    assert self.state.player is not None
    assert self.next_state.player is not None

    state_serialized = self.state.model_dump(exclude=excluded_fields)
    next_state_serialized = self.next_state.model_dump(exclude=excluded_fields)

    def format_serialized_state(state: dict) -> dict:
        # Remove the player field from the .objects list, so it isn't duplicated
        # since it is already in the .player field.
        state["objects"] = [obj for obj in state["objects"] if obj["name"] != "player"]

        # Sort the objects by entity_id
        state["objects"] = sorted(state["objects"], key=lambda x: x["entity_id"])

        # Sort the chunks by chunk_key
        state["chunks"] = sorted(state["chunks"], key=lambda x: x["chunk_key"])

        # For each chunk, sort the objects within the chunk
        for chunk in state["chunks"]:
            chunk["objects"] = sorted(chunk["objects"])

        # Remove the materials field, we show a local view instead and so
        # including it in the diff is redundant.
        state.pop("materials")

        return state

    state_serialized = format_serialized_state(state_serialized)
    next_state_serialized = format_serialized_state(next_state_serialized)

    # Generate the diff header lines
    header = [
        "diff --git a/state b/next_state",
        "--- a/state",
        "+++ b/next_state",
    ]

    # Generate the actual diff
    diff_lines = list(
        difflib.unified_diff(
            json.dumps(state_serialized, indent=2).splitlines(),
            json.dumps(next_state_serialized, indent=2).splitlines(),
            fromfile="a/state",
            tofile="b/next_state",
            lineterm="",
            n=3,  # Number of context lines, matching Git's default
        )
    )

    # Remove the original headers (first 2 lines) from unified_diff output
    # since we're providing our own Git-style headers
    diff_lines = diff_lines[2:]

    # Also add a diff between the local view of the state and the next state
    local_view, _ = create_ascii_map_with_legend(self.state)
    next_local_view, _ = create_ascii_map_with_legend(self.next_state)
    local_view_diff = list(
        difflib.unified_diff(
            local_view.splitlines(),
            next_local_view.splitlines(),
            fromfile="a/local_view",
            tofile="b/next_local_view",
            lineterm="",
            n=3,  # Number of context lines, matching Git's default
        )
    )

    return "\n".join(header + diff_lines + local_view_diff)


class PromptRenderer:
    def __init__(self):
        self.template = jinja2.Template(
            INDUCTION_TEMPLATE_PATH.read_text(), undefined=jinja2.StrictUndefined
        )

    def render_prompt(self, transition: Transition, aspect_of_state: str) -> str:

        excluded_fields = {"event_bus", "serialized_random_state"}

        state_serialized = transition.state.model_dump(exclude=excluded_fields)
        state_serialized["materials"] = "# SNIPPED TO SAVE SPACE, SEE LOCAL VIEW #"

        local_view, legend = create_ascii_map_with_legend(transition.state)

        # Render focused aspect changes
        aspect_changes = render_aspect_changes(transition, aspect_of_state)

        state_diff = diff(transition)

        return self.template.render(
            state=json.dumps(state_serialized, indent=2),
            action=transition.action,
            local_view=local_view,
            view_legend=legend,
            aspect_of_state=aspect_of_state,
            aspect_changes=aspect_changes,
            state_diff=state_diff,
        )
