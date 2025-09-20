#!/usr/bin/env python3
"""
Zombie Movement Visualization Script

This script demonstrates how to visualize zombie movement predictions from our laws
vs actual environment behavior. It creates a scenario with a zombie near the player,
samples multiple transitions, and overlays probability distributions on the rendered game.

COORDINATE MAPPING SYSTEM:
==========================

The Crafter environment uses multiple coordinate systems that need to be mapped:

1. WORLD COORDINATES:
   - Absolute positions in the game world (e.g., player at (16, 16))
   - Used by entities, laws, and state representations
   - Can be any integer values

2. VIEW COORDINATES:
   - Local coordinates within the player's view window
   - View is typically 9x9 tiles centered on the player
   - Player is at view center (4, 4) in a 9x9 view
   - Range: [0, view_width) x [0, view_height)

3. PIXEL COORDINATES:
   - Actual pixel positions in the rendered image
   - Image size: typically 256x256 pixels
   - Each view tile maps to ~28x28 pixels (256/9 ≈ 28)

COORDINATE TRANSFORMATION PIPELINE:
===================================

World → View → Pixel

1. World to View (with off-by-one fix):
   - Calculate relative position: (world_x - player_x, world_y - player_y)
   - Add view center offset: (view_center_x + rel_x, view_center_y + rel_y)
   - View center is (4, 3) for 9x9 view (Y coordinate has -1 adjustment)
   - This accounts for the off-by-one error discovered during testing
   - Check bounds: position must be within [0, view_width) x [0, view_height)

2. View to Pixel:
   - Calculate unit size: pixel_size = render_size / view_size
   - Map to pixels: (view_x * unit_x, view_y * unit_y)
   - Draw square: [pixel_x : pixel_x + unit_x, pixel_y : pixel_y + unit_y]

EXAMPLE:
========
Player at world (16, 16), zombie at world (17, 16) in 9x9 view:
- Relative position: (17-16, 16-16) = (1, 0)
- View center: (4, 3) for 9x9 view (Y has -1 adjustment)
- View coordinates: (4+1, 3+0) = (5, 3)
- Pixel coordinates: (5*28, 3*28) = (140, 84) in 256x256 image
- Draw 28x28 pixel square at (140, 84)
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import base64

# Import crafter components
from crafter.functional_env import (
    initial_state,
    transition,
    observation,
    export_world_state,
)
from crafter.state_export import WorldState, ZombieState
from crafter.state_reconstruction import reconstruct_world_from_state
from crafter import objects, constants
from crafter.testing_helpers import world_utils, player_utils

# Import our method components
from distant_sunburn.our_method.crafter.handwritten_laws import (
    CorrectEntityAILaw,
    LawFunctionWrapper,
)
from distant_sunburn.poe_world.core import DiscreteDistribution


# ============================================================================
# RANDOM STATE MANIPULATION FUNCTIONS
# ============================================================================


def create_random_state_with_seed(seed: int) -> np.random.RandomState:
    """Create a new RandomState with the given seed."""
    return np.random.RandomState(seed)


def serialize_random_state(random_state: np.random.RandomState) -> str:
    """Serialize a RandomState to base64 string."""
    state_bytes = pickle.dumps(random_state)
    return base64.b64encode(state_bytes).decode("ascii")


def modify_world_state_random_seed(state: WorldState, seed: int) -> WorldState:
    """
    Create a copy of the WorldState with a new random seed.

    Args:
        state: The original WorldState
        seed: New random seed to use

    Returns:
        New WorldState with modified random state
    """
    # Create a deep copy of the state
    new_state = copy.deepcopy(state)

    # Create new random state with the given seed
    new_random_state = create_random_state_with_seed(seed)

    # Serialize and set the new random state
    new_state.serialized_random_state = serialize_random_state(new_random_state)

    return new_state


def advance_random_state(state: WorldState, steps: int) -> WorldState:
    """
    Create a copy of the WorldState with the random state advanced by N steps.

    This is useful for getting different random outcomes without changing the seed.

    Args:
        state: The original WorldState
        steps: Number of random numbers to generate to advance the state

    Returns:
        New WorldState with advanced random state
    """
    # Create a deep copy of the state
    new_state = copy.deepcopy(state)

    # Get the current random state and advance it
    current_random_state = new_state.random_state
    for _ in range(steps):
        current_random_state.uniform()

    # Serialize and set the advanced random state
    new_state.serialized_random_state = serialize_random_state(current_random_state)

    return new_state


# ============================================================================
# COORDINATE MAPPING HELPER FUNCTIONS
# ============================================================================


def world_to_view_coordinates(
    world_x: int,
    world_y: int,
    player_x: int,
    player_y: int,
    view_width: int,
    view_height: int,
) -> tuple[int, int] | None:
    """
    Convert world coordinates to view coordinates.

    This function accounts for the off-by-one error in the Y coordinate
    that was discovered during testing to align with the game's sprite rendering.

    Args:
        world_x, world_y: World position to convert
        player_x, player_y: Player's world position
        view_width, view_height: Dimensions of the view window

    Returns:
        (view_x, view_y) if position is within view bounds, None otherwise
    """
    # Calculate relative position from player
    rel_x = world_x - player_x
    rel_y = world_y - player_y

    # Convert to view coordinates (view is centered on player)
    # FIXED: View center Y needs -1 adjustment to align with sprite rendering
    view_center_x = view_width // 2
    view_center_y = view_height // 2 - 1
    view_x = view_center_x + rel_x
    view_y = view_center_y + rel_y

    # Check if position is within view bounds
    if 0 <= view_x < view_width and 0 <= view_y < view_height:
        return (view_x, view_y)
    else:
        return None


def view_to_pixel_coordinates(
    view_x: int,
    view_y: int,
    view_width: int,
    view_height: int,
    render_width: int,
    render_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert view coordinates to pixel coordinates and calculate square size.

    Args:
        view_x, view_y: View position to convert
        view_width, view_height: Dimensions of the view window
        render_width, render_height: Dimensions of the rendered image

    Returns:
        (pixel_x, pixel_y, square_width, square_height)
    """
    # Calculate unit size for each grid cell
    unit_x = render_width // view_width
    unit_y = render_height // view_height

    # Calculate pixel coordinates
    pixel_x = view_x * unit_x
    pixel_y = view_y * unit_y

    return (pixel_x, pixel_y, unit_x, unit_y)


def world_to_pixel_coordinates(
    world_x: int,
    world_y: int,
    player_x: int,
    player_y: int,
    view_width: int,
    view_height: int,
    render_width: int,
    render_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Convert world coordinates directly to pixel coordinates using LocalView coordinate system.

    Args:
        world_x, world_y: World position to convert
        player_x, player_y: Player's world position
        view_width, view_height: Dimensions of the view window
        render_width, render_height: Dimensions of the rendered image

    Returns:
        (pixel_x, pixel_y, square_width, square_height) if within view, None otherwise
    """
    # First convert to view coordinates
    view_coords = world_to_view_coordinates(
        world_x, world_y, player_x, player_y, view_width, view_height
    )

    if view_coords is None:
        return None

    view_x, view_y = view_coords

    # Then convert to pixel coordinates
    return view_to_pixel_coordinates(
        view_x, view_y, view_width, view_height, render_width, render_height
    )


def world_to_view_coordinates_unified(
    world_pos: np.ndarray,
    player_pos: np.ndarray,
    view_dims: tuple[int, int],
) -> np.ndarray | None:
    """
    Unified function to convert world coordinates to view coordinates.

    This function accounts for the off-by-one error in the Y coordinate
    that was discovered during testing to align with the game's sprite rendering.

    Args:
        world_pos: World position as numpy array [x, y]
        player_pos: Player's world position as numpy array [x, y]
        view_dims: View dimensions as (width, height)

    Returns:
        View position as numpy array [x, y] if within bounds, None otherwise
    """
    # FIXED: Apply the same off-by-one fix for Y coordinate
    offset = np.array([view_dims[0] // 2, view_dims[1] // 2 - 1])
    center = player_pos

    view_pos = world_pos - center + offset

    # Check if position is within view bounds
    if 0 <= view_pos[0] < view_dims[0] and 0 <= view_pos[1] < view_dims[1]:
        return view_pos
    else:
        return None


# ============================================================================
# CUSTOM RENDERING FUNCTIONS
# ============================================================================


def render_with_distribution_overlay(
    world_state: WorldState,
    distributions: dict[tuple[int, int], float],
    view_dims: tuple[int, int] = (9, 9),
    render_size: tuple[int, int] = (256, 256),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Render the game world with probability distribution overlay between ground and sprites.

    This function replicates the LocalView rendering pipeline but adds our distribution
    overlay between the ground tiles and the sprites, so the distributions appear
    underneath the sprites but above the ground.

    Args:
        world_state: The world state to render
        distributions: Dict mapping (x, y) world positions to probability values
        view_dims: View dimensions of the game
        render_size: Render size of the image
        alpha: Transparency of the distribution overlay

    Returns:
        Rendered image with distribution overlay
    """
    from crafter import engine, objects, constants
    from crafter.state_reconstruction import reconstruct_world_from_state

    # Reconstruct the world from state
    world = reconstruct_world_from_state(world_state)

    # Find the player
    player = None
    for obj in world.objects:
        if isinstance(obj, objects.Player):
            player = obj
            break

    if player is None:
        raise ValueError("No player found in world state")

    # Set up rendering parameters
    size = np.array(render_size)
    unit = size // np.array(view_dims)
    canvas = np.zeros(tuple(size) + (3,), np.uint8) + 127

    # Create textures
    textures = engine.Textures(constants.root / "assets")

    # STEP 1: Draw ground tiles/materials (same as LocalView)
    center = np.array(player.pos)

    for x in range(view_dims[0]):
        for y in range(view_dims[1]):
            # Convert view coordinates to world coordinates
            view_pos = np.array([x, y])
            offset = np.array(view_dims) // 2
            world_pos = center + view_pos - offset

            if not _inside_world_bounds(world_pos, world.area):
                continue
            material, _ = world[world_pos]
            texture = textures.get(material, unit)
            _draw_texture(canvas, view_pos * unit, texture)

    # STEP 2: Draw probability distribution overlay
    # Use unified coordinate mapping for consistency
    for (world_x, world_y), prob in distributions.items():
        # Convert world coordinates to view coordinates using unified function
        world_pos = np.array([world_x, world_y])
        view_pos = world_to_view_coordinates_unified(world_pos, center, view_dims)

        if view_pos is None:
            continue

        # Convert to pixel coordinates
        pixel_pos = view_pos * unit

        # Create colored square based on probability
        color = _prob_to_color(prob)

        # Draw distribution square
        for dx in range(unit[0]):
            for dy in range(unit[1]):
                px, py = int(pixel_pos[0] + dx), int(pixel_pos[1] + dy)
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    # Blend with existing pixel (ground)
                    canvas[py, px] = (
                        alpha * np.array(color) + (1 - alpha) * canvas[py, px]
                    )

    # STEP 3: Draw sprites/objects (same as LocalView)
    for obj in world.objects:
        # Use unified coordinate mapping for consistency
        view_pos = world_to_view_coordinates_unified(obj.pos, center, view_dims)
        if view_pos is None:
            continue
        texture = textures.get(obj.texture, unit)
        _draw_alpha_texture(canvas, view_pos * unit, texture)

    # Apply lighting and other effects (same as LocalView)
    canvas = _apply_lighting(canvas, world.daylight)
    if player.sleeping:
        canvas = _apply_sleep_effect(canvas)

    # Handle the transpose that the original rendering does
    return canvas.transpose((1, 0, 2))


def _inside_world_bounds(pos: np.ndarray, area: tuple[int, int]) -> bool:
    """Check if position is within world bounds."""
    return 0 <= pos[0] < area[0] and 0 <= pos[1] < area[1]


def _inside_view_bounds(pos: np.ndarray, view_dims: tuple[int, int]) -> bool:
    """Check if position is within view bounds."""
    return 0 <= pos[0] < view_dims[0] and 0 <= pos[1] < view_dims[1]


def _draw_texture(canvas: np.ndarray, pos: np.ndarray, texture: np.ndarray) -> None:
    """Draw a texture at the given position."""
    if texture is None:
        return

    pos = pos.astype(int)
    size = np.array(texture.shape[:2])

    # Calculate bounds
    x1, y1 = pos
    x2, y2 = pos + size

    # Clamp to canvas bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(canvas.shape[1], x2)
    y2 = min(canvas.shape[0], y2)

    if x1 < x2 and y1 < y2:
        # Calculate texture bounds
        tx1, ty1 = x1 - pos[0], y1 - pos[1]
        tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)

        canvas[y1:y2, x1:x2] = texture[ty1:ty2, tx1:tx2]


def _draw_alpha_texture(
    canvas: np.ndarray, pos: np.ndarray, texture: np.ndarray
) -> None:
    """Draw a texture with alpha blending at the given position."""
    if texture is None:
        return

    pos = pos.astype(int)
    size = np.array(texture.shape[:2])

    # Calculate bounds
    x1, y1 = pos
    x2, y2 = pos + size

    # Clamp to canvas bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(canvas.shape[1], x2)
    y2 = min(canvas.shape[0], y2)

    if x1 < x2 and y1 < y2:
        # Calculate texture bounds
        tx1, ty1 = x1 - pos[0], y1 - pos[1]
        tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)

        # Alpha blend
        texture_slice = texture[ty1:ty2, tx1:tx2]
        canvas_slice = canvas[y1:y2, x1:x2]

        if texture_slice.shape[2] == 4:  # Has alpha channel
            alpha = texture_slice[:, :, 3:4] / 255.0
            rgb = texture_slice[:, :, :3]
            canvas[y1:y2, x1:x2] = (1 - alpha) * canvas_slice + alpha * rgb
        else:  # No alpha channel, just copy
            canvas[y1:y2, x1:x2] = texture_slice


def _prob_to_color(prob: float) -> tuple[int, int, int]:
    """Convert probability to RGB color (red for high, blue for low)."""
    # Clamp probability to [0, 1]
    prob = max(0.0, min(1.0, prob))

    # Create color gradient from blue (low) to red (high)
    if prob < 0.5:
        # Blue to purple
        intensity = prob * 2
        return (int(127 * intensity), 0, int(255 * (1 - intensity)))
    else:
        # Purple to red
        intensity = (prob - 0.5) * 2
        return (int(127 + 128 * intensity), 0, int(127 * (1 - intensity)))


def _apply_lighting(canvas: np.ndarray, daylight: float) -> np.ndarray:
    """Apply lighting effects to the canvas."""
    if daylight >= 0.5:
        return canvas

    # Apply night effect (simplified version)
    night_factor = 1 - daylight
    canvas = canvas * (1 - night_factor * 0.6)
    return canvas.astype(np.uint8)


def _apply_sleep_effect(canvas: np.ndarray) -> np.ndarray:
    """Apply sleep effect to the canvas."""
    # Convert to grayscale and tint blue
    gray = np.mean(canvas, axis=2, keepdims=True)
    blue_tint = np.array([0, 0, 16])
    return (gray * 0.5 + blue_tint * 0.5).astype(np.uint8)


# ============================================================================
# MAIN CLASSES
# ============================================================================


class ZombiePlacementHelper:
    """Helper class to place zombies near the player for testing."""

    @staticmethod
    def place_zombie_near_player(state: WorldState, distance: int = 3) -> WorldState:
        """
        Place a zombie near the player in the given state.

        Args:
            state: The world state to modify
            distance: Distance from player to place zombie

        Returns:
            Modified state with zombie placed near player
        """
        # Reconstruct world from state
        world = reconstruct_world_from_state(state)

        # Get player
        player = None
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                player = obj
                break

        if player is None:
            raise ValueError("No player found in world")

        # Clear all entities from the world (except the player)
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                continue
            world.remove(obj)

        # Set all tiles to grass for walkability
        for x in range(state.size[0]):
            for y in range(state.size[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        # Place player at center
        player_utils.set_player_position(
            player, (state.size[0] // 2, state.size[1] // 2)
        )

        # Add a zombie near the player
        zombie_pos = (player.pos[0] + distance, player.pos[1])
        zombie = objects.Zombie(world, zombie_pos, player)
        world.add(zombie)

        print(f"Placed zombie at {zombie_pos} near player at {player.pos}")

        # Export the modified state
        return export_world_state(world, view=state.view, step_count=state.step_count)


class DistributionVisualizer:
    """Helper class to visualize probability distributions on the game render."""

    @staticmethod
    def create_distribution_overlay(
        base_image: np.ndarray,
        distributions: dict[tuple[int, int], float],
        view_dims: tuple[int, int],
        render_size: tuple[int, int],
        player_pos: tuple[int, int],
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Create an overlay showing probability distributions as colored squares.

        Args:
            base_image: The base rendered game image
            distributions: Dict mapping (x, y) world positions to probability values
            view_dims: View dimensions of the game
            render_size: Render size of the image
            player_pos: Player's world position (x, y)
            alpha: Transparency of the overlay

        Returns:
            Image with distribution overlay
        """
        # Create a copy of the base image
        overlay_image = base_image.copy().astype(np.float32)

        view_width, view_height = view_dims
        render_width, render_height = render_size
        player_x, player_y = player_pos

        # Create overlay for each distribution position
        for (world_x, world_y), prob in distributions.items():
            # Use helper function to convert world coordinates to pixel coordinates
            pixel_coords = world_to_pixel_coordinates(
                world_x,
                world_y,
                player_x,
                player_y,
                view_width,
                view_height,
                render_width,
                render_height,
            )

            if pixel_coords is not None:
                pixel_x, pixel_y, square_width, square_height = pixel_coords

                # Create colored square based on probability
                color = DistributionVisualizer._prob_to_color(prob)

                # Draw square
                for dx in range(square_width):
                    for dy in range(square_height):
                        px, py = pixel_x + dx, pixel_y + dy
                        if (
                            0 <= px < overlay_image.shape[1]
                            and 0 <= py < overlay_image.shape[0]
                        ):
                            # Blend with existing pixel
                            overlay_image[py, px] = (
                                alpha * np.array(color)
                                + (1 - alpha) * overlay_image[py, px]
                            )

        return overlay_image.astype(np.uint8)

    @staticmethod
    def _prob_to_color(prob: float) -> tuple[int, int, int]:
        """Convert probability to RGB color (red for high, blue for low)."""
        # Clamp probability to [0, 1]
        prob = max(0, min(1, prob))

        # Red for high probability, blue for low probability
        red = int(255 * prob)
        blue = int(255 * (1 - prob))
        green = 0

        return (red, green, blue)


class ZombieMovementAnalyzer:
    """Main class for analyzing zombie movement predictions vs reality."""

    def __init__(self):
        self.law = LawFunctionWrapper.from_non_runtime_created(CorrectEntityAILaw())

    def sample_environment_transitions(
        self, initial_state: WorldState, action: str, n_samples: int = 50
    ) -> list[tuple[int, int]]:
        """
        Sample multiple transitions from the environment to get true zombie positions.

        Args:
            initial_state: Starting state
            action: Action to take
            n_samples: Number of samples to take

        Returns:
            List of (x, y) positions where zombie ended up
        """
        # Convert action string to action index
        action_idx = None
        for i, action_name in enumerate(constants.actions):
            if action_name == action:
                action_idx = i
                break

        if action_idx is None:
            raise ValueError(f"Unknown action: {action}")
        positions = []

        for i in range(n_samples):
            # Try advancing the random state instead of using different seeds
            # This might be more effective for getting different outcomes
            state_with_advanced_rng = advance_random_state(initial_state, steps=i * 10)

            # Take the transition
            next_state, _ = transition(state_with_advanced_rng, action_idx)

            # Find zombie position in the next state
            for obj in next_state.objects:
                if isinstance(obj, ZombieState):
                    positions.append((obj.position.x, obj.position.y))
                    break

        return positions

    def get_law_predictions(
        self, initial_state: WorldState, action: str
    ) -> dict[tuple[int, int], float]:
        """
        Get zombie movement predictions from our law.

        Args:
            initial_state: Starting state
            action: Action to take

        Returns:
            Dict mapping (x, y) positions to predicted probabilities
        """
        # Create a copy of the state
        state_copy = copy.deepcopy(initial_state)

        # Apply the law
        if self.law.precondition(state_copy, action):
            self.law.effect(state_copy, action)

        # Find zombie and extract its position distribution
        for obj in state_copy.objects:
            if isinstance(obj, ZombieState):
                # Check if position has DiscreteDistribution
                if isinstance(obj.position.x, DiscreteDistribution) and isinstance(
                    obj.position.y, DiscreteDistribution
                ):
                    # Extract distribution
                    x_dist = obj.position.x
                    y_dist = obj.position.y

                    # Convert to probability dict
                    predictions = {}
                    for i, x_val in enumerate(x_dist.support):
                        for j, y_val in enumerate(y_dist.support):
                            # Calculate combined probability
                            x_prob = np.exp(x_dist.log_probs[i])
                            y_prob = np.exp(y_dist.log_probs[j])
                            combined_prob = x_prob * y_prob

                            predictions[(int(x_val), int(y_val))] = combined_prob

                    return predictions

        return {}

    def create_test_visualization(self, initial_state: WorldState) -> None:
        """
        Create a test visualization with hand-specified distributions to test the rendering code.

        Args:
            initial_state: Starting state with zombie placed near player
        """
        print("Creating test visualization with hand-specified distributions...")

        # Render the initial state
        base_image = observation(initial_state, render_size=(256, 256))
        print(f"Base image shape: {base_image.shape}")

        # Get player position for coordinate mapping
        player_pos = initial_state.player.position
        print(f"Player position: ({player_pos.x}, {player_pos.y})")

        # Create a simple test: draw a blue square at a fixed position
        test_overlay = base_image.copy().astype(np.float32)

        # Draw a bright blue square in the center of the image
        center_x, center_y = base_image.shape[1] // 2, base_image.shape[0] // 2
        square_size = 32

        for dx in range(square_size):
            for dy in range(square_size):
                x = center_x - square_size // 2 + dx
                y = center_y - square_size // 2 + dy
                if 0 <= x < base_image.shape[1] and 0 <= y < base_image.shape[0]:
                    test_overlay[y, x] = [0, 0, 255]  # Bright blue

        # Test the distribution overlay function with debug info
        # Use world coordinates relative to the player position
        test_distribution = {
            (player_pos.x, player_pos.y): 1.0,  # Player position
            (player_pos.x + 1, player_pos.y): 0.5,  # One tile right of player
            (player_pos.x, player_pos.y + 1): 0.3,  # One tile down from player
            (player_pos.x - 1, player_pos.y): 0.2,  # One tile left of player
        }

        print(f"Testing distribution overlay with {len(test_distribution)} positions")

        # Test the distribution overlay
        dist_overlay = DistributionVisualizer.create_distribution_overlay(
            base_image,
            test_distribution,
            (9, 9),
            (256, 256),
            (player_pos.x, player_pos.y),
        )

        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Base image
        axes[0].imshow(base_image)
        axes[0].set_title("Base Game State")
        axes[0].axis("off")

        # Test overlay with blue square
        axes[1].imshow(test_overlay.astype(np.uint8))
        axes[1].set_title("Test: Blue Square Overlay")
        axes[1].axis("off")

        # Test distribution overlay
        axes[2].imshow(dist_overlay)
        axes[2].set_title("Test: Distribution Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("zombie_movement_test.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\nTest: Drew blue square at center ({center_x}, {center_y})")
        print(
            f"Test: Distribution overlay created with {len(test_distribution)} positions"
        )
        print("Test visualization complete! Check 'zombie_movement_test.png'")

    def create_environment_sampling_visualization(
        self, initial_state: WorldState, action: str = "move_right", n_samples: int = 50
    ) -> None:
        """
        Create a visualization showing environment sampling results.

        Args:
            initial_state: Starting state with zombie placed near player
            action: Action to take
            n_samples: Number of samples to take from environment
        """
        print(
            f"Creating environment sampling visualization with {n_samples} samples..."
        )

        # Show initial setup
        for obj in initial_state.objects:
            if isinstance(obj, ZombieState):
                print(f"Initial zombie position: ({obj.position.x}, {obj.position.y})")
                break

        print(
            f"Player position: ({initial_state.player.position.x}, {initial_state.player.position.y})"
        )
        print(f"Action: {action}")

        # Sample from the environment
        print("Sampling from environment...")
        env_positions = self.sample_environment_transitions(
            initial_state, action, n_samples
        )

        # Count position frequencies
        position_counts = {}
        for pos in env_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Convert counts to probabilities
        env_distribution = {
            pos: count / n_samples for pos, count in position_counts.items()
        }

        print(f"Environment sampling found {len(env_distribution)} unique positions")
        print(f"Position distribution: {env_distribution}")

        # Show coordinate mapping for sampled positions
        print("Coordinate mapping for sampled positions:")
        for pos, prob in env_distribution.items():
            view_coords = world_to_view_coordinates(
                pos[0],
                pos[1],
                initial_state.player.position.x,
                initial_state.player.position.y,
                9,
                9,
            )
            if view_coords:
                pixel_coords = view_to_pixel_coordinates(
                    view_coords[0], view_coords[1], 9, 9, 256, 256
                )
                print(
                    f"  World {pos} -> View {view_coords} -> Pixel {pixel_coords} (prob: {prob:.2f})"
                )

        # Show sample of positions for debugging
        if len(env_positions) > 10:
            print(f"Sample positions: {env_positions[:10]}...")
        else:
            print(f"All positions: {env_positions}")

        # Render the initial state
        base_image = observation(initial_state, render_size=(256, 256))
        player_pos = initial_state.player.position

        # Create environment distribution overlay
        env_overlay = DistributionVisualizer.create_distribution_overlay(
            base_image,
            env_distribution,
            (9, 9),
            (256, 256),
            (player_pos.x, player_pos.y),
            alpha=0.7,
        )

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Base image
        axes[0].imshow(base_image)
        axes[0].set_title("Base Game State")
        axes[0].axis("off")

        # Environment sampling overlay
        axes[1].imshow(env_overlay)
        axes[1].set_title(f"Environment Sampling ({n_samples} samples)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(
            "environment_sampling_visualization.png", dpi=150, bbox_inches="tight"
        )
        plt.show()

        print(
            "Environment sampling visualization complete! Check 'environment_sampling_visualization.png'"
        )

    def create_custom_rendering_visualization(
        self, initial_state: WorldState, action: str = "move_right", n_samples: int = 50
    ) -> None:
        """
        Create a visualization using custom rendering with distributions under sprites.

        Args:
            initial_state: Starting state with zombie placed near player
            action: Action to take
            n_samples: Number of samples to take from environment
        """
        print(f"Creating custom rendering visualization with {n_samples} samples...")

        # Sample from the environment
        print("Sampling from environment...")
        env_positions = self.sample_environment_transitions(
            initial_state, action, n_samples
        )

        # Count position frequencies
        position_counts = {}
        for pos in env_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Convert counts to probabilities
        env_distribution = {
            pos: count / n_samples for pos, count in position_counts.items()
        }

        print(f"Environment sampling found {len(env_distribution)} unique positions")
        print(f"Position distribution: {env_distribution}")

        # Render using custom rendering function
        custom_rendered = render_with_distribution_overlay(
            initial_state,
            env_distribution,
            view_dims=(9, 9),
            render_size=(256, 256),
            alpha=0.7,
        )

        # Also render the base image for comparison
        base_image = observation(initial_state, render_size=(256, 256))

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Base image (original rendering)
        axes[0].imshow(base_image)
        axes[0].set_title("Original Rendering (Overlay on Top)")
        axes[0].axis("off")

        # Custom rendering (distributions under sprites)
        axes[1].imshow(custom_rendered)
        axes[1].set_title("Custom Rendering (Distributions Under Sprites)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig("custom_rendering_visualization.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(
            "Custom rendering visualization complete! Check 'custom_rendering_visualization.png'"
        )


def main():
    """Main function to run the zombie movement visualization."""
    print("Creating initial game state...")

    # Create initial state
    state = initial_state(area=(32, 32), view=(9, 9), seed=42)

    # Place a zombie near the player
    state_with_zombie = ZombiePlacementHelper.place_zombie_near_player(
        state, distance=3
    )

    # Create analyzer
    analyzer = ZombieMovementAnalyzer()

    # Create test visualization with hand-specified distributions
    analyzer.create_test_visualization(state_with_zombie)

    # Create environment sampling visualization
    analyzer.create_environment_sampling_visualization(
        state_with_zombie, "move_right", n_samples=30
    )

    # Create custom rendering visualization
    analyzer.create_custom_rendering_visualization(
        state_with_zombie, "move_right", n_samples=30
    )


if __name__ == "__main__":
    main()
