import random
import numpy as np
from typing import List

from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.mixed_fsm_env import (
    initial_state,
    transition_function,
    Action,
    State,
)
from distant_sunburn.poe_world.mixed_fsm_env.handwritten_experts import (
    CORRECT_EXPERTS,
    INCORRECT_EXPERTS,
    ALL_EXPERTS,
)
from distant_sunburn.poe_world.weight_fitter import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.mixed_fsm_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.poe_world.mixed_fsm_env.handwritten_experts import (
    correct_deterministic_switch_expert,
    incorrect_deterministic_switch_expert_assumes_static,
    incorrect_deterministic_switch_expert_assumes_stochastic,
)
from distant_sunburn.poe_world.core import ExpertFunctionWrapper, WeightedExpert
from distant_sunburn.poe_world.world_model import PoEWorldModel


def generate_random_data(
    n_transitions: int, seed: int = 42
) -> List[SymbolicTransition[State]]:
    """
    Generate random transitions using the 1D environment.

    Args:
        n_transitions: Number of transitions to generate
        seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    transitions: List[SymbolicTransition[State]] = []
    current_state = initial_state(seed)

    for _ in range(n_transitions):
        # Choose random action
        action = rng.choice(list(Action))

        # Apply transition function
        next_state = transition_function(current_state, action)

        # Create symbolic transition
        transition = SymbolicTransition(
            prev_metadata=current_state, action=action, next_metadata=next_state
        )
        transitions.append(transition)

        # Update current state for next iteration
        current_state = next_state

    return transitions


def test_weight_fitting_distinguishes_single_good_from_bad_expert():
    n_transitions = 50
    transitions = generate_random_data(n_transitions, seed=42)

    # Split into train/test (75/25)
    split_point = int(0.75 * len(transitions))
    train_transitions = transitions[:split_point]
    test_transitions = transitions[split_point:]

    print(f"Generated {len(transitions)} transitions")
    print(f"Training on {len(train_transitions)} transitions")
    print(f"Testing on {len(test_transitions)} transitions")

    fitter = MaxLikelihoodWeightFitter(
        observable_extractor=ObservableExtractor(),
        learning_rate=0.1,
        max_iterations=5,
        batch_size=200,
        l1_weight=0.001,
    )

    good_expert = ExpertFunctionWrapper[State].from_non_runtime_created(
        correct_deterministic_switch_expert
    )
    bad_expert = ExpertFunctionWrapper[State].from_non_runtime_created(
        incorrect_deterministic_switch_expert_assumes_static
    )

    # Fit weights
    weighted_experts = fitter.fit([good_expert, bad_expert], train_transitions)

    for weighted_expert in weighted_experts:
        print(
            f"{weighted_expert.expert_function.__name__} weight: {weighted_expert.weight:.4f}"
        )

    good_weight = weighted_experts[0].weight
    bad_weight = weighted_experts[1].weight

    assert good_weight > bad_weight, (
        f"Good expert should have higher weight than bad expert. "
        f"Got good={good_weight:.4f}, bad={bad_weight:.4f}"
    )


def test_weight_fitting_distinguishes_good_from_bad_experts():
    n_transitions = 100
    transitions = generate_random_data(n_transitions, seed=42)

    # Split into train/test (75/25)
    split_point = int(0.75 * len(transitions))
    train_transitions = transitions[:split_point]
    test_transitions = transitions[split_point:]

    print(f"Generated {len(transitions)} transitions")
    print(f"Training on {len(train_transitions)} transitions")
    print(f"Testing on {len(test_transitions)} transitions")

    fitter = MaxLikelihoodWeightFitter(
        observable_extractor=ObservableExtractor(),
        learning_rate=0.1,
        max_iterations=5,
        batch_size=200,
        l1_weight=0.001,
    )

    # Fit weights
    weighted_experts = fitter.fit(ALL_EXPERTS, train_transitions)

    # Extract weights for correct and incorrect experts
    correct_weights = []
    incorrect_weights = []

    for i, weighted_expert in enumerate(weighted_experts):
        weight = weighted_expert.weight
        expert_func = weighted_expert.expert_function

        print(f"Expert {i} -- {expert_func.__name__}: weight = {weight:.4f}")

        if expert_func in CORRECT_EXPERTS:
            correct_weights.append(weight)
            print("  -> CORRECT expert")
        elif expert_func in INCORRECT_EXPERTS:
            incorrect_weights.append(weight)
            print("  -> INCORRECT expert")

    # Validate that correct experts have higher average weight
    avg_correct_weight = np.mean(correct_weights)
    avg_incorrect_weight = np.mean(incorrect_weights)

    print(f"\nAverage correct expert weight: {avg_correct_weight:.4f}")
    print(f"Average incorrect expert weight: {avg_incorrect_weight:.4f}")

    # Main assertion: correct experts should have higher weights on average
    assert avg_correct_weight > avg_incorrect_weight, (
        f"Correct experts should have higher weights than incorrect ones. "
        f"Got correct={avg_correct_weight:.4f}, incorrect={avg_incorrect_weight:.4f}"
    )


def test_jumpy_posterior():
    """
    This test checks that even in the case where we only have correct experts,
    the PoE-World inference algorithm puts a lot of probability mass on values that
    are not permitted by the true transition function.
    """

    weighted_experts = [
        WeightedExpert(expert_function=fn, weight=1.0, is_fitted=True)
        for fn in CORRECT_EXPERTS
    ]
    world_model = PoEWorldModel(
        observable_extractor=ObservableExtractor(),
        weighted_experts=weighted_experts,
    )

    n_samples = 100

    seed = 42

    deterministic_switch_values: list[int] = []
    stochastic_switch_values: list[int] = []
    static_switch_values: list[int] = []

    for _ in range(n_samples):
        state = initial_state(seed)
        action = Action.TOGGLE_DETERMINISTIC_SWITCH
        next_state = world_model.sample_next_state(state, action)
        deterministic_switch_values.append(next_state.deterministic_switch)
        stochastic_switch_values.append(next_state.stochastic_switch)
        static_switch_values.append(next_state.static_switch)

    for _ in range(n_samples):
        state = initial_state(seed)
        action = Action.TOGGLE_STOCHASTIC_SWITCH
        next_state = world_model.sample_next_state(state, action)
        deterministic_switch_values.append(next_state.deterministic_switch)
        stochastic_switch_values.append(next_state.stochastic_switch)
        static_switch_values.append(next_state.static_switch)

    for _ in range(n_samples):
        state = initial_state(seed)
        action = Action.TOGGLE_STATIC_SWITCH
        next_state = world_model.sample_next_state(state, action)
        deterministic_switch_values.append(next_state.deterministic_switch)
        stochastic_switch_values.append(next_state.stochastic_switch)
        static_switch_values.append(next_state.static_switch)

    # Convert to numpy arrays for cleaner computation
    deterministic_switch_array = np.array(deterministic_switch_values)
    stochastic_switch_array = np.array(stochastic_switch_values)
    static_switch_array = np.array(static_switch_values)

    # Compute empirical posteriors using numpy
    deterministic_switch_unique, deterministic_switch_counts = np.unique(
        deterministic_switch_array, return_counts=True
    )
    stochastic_switch_unique, stochastic_switch_counts = np.unique(
        stochastic_switch_array, return_counts=True
    )
    static_switch_unique, static_switch_counts = np.unique(
        static_switch_array, return_counts=True
    )

    # Convert counts to probabilities
    deterministic_switch_posterior = deterministic_switch_counts / len(
        deterministic_switch_array
    )
    stochastic_switch_posterior = stochastic_switch_counts / len(
        stochastic_switch_array
    )
    static_switch_posterior = static_switch_counts / len(static_switch_array)

    # Assert that no single value has more than max concentration of probability mass
    max_deterministic_switch_prob = np.max(deterministic_switch_posterior)
    max_stochastic_switch_prob = np.max(stochastic_switch_posterior)
    max_static_switch_prob = np.max(static_switch_posterior)

    max_concentration = 0.80

    # Display switch distributions in a clean 3x2 table
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Switch Distributions", show_header=True)
    table.add_column("Switch", style="cyan")
    table.add_column("Initial", justify="center", style="white")
    table.add_column("Off (0)", justify="right", style="green")
    table.add_column("On (1)", justify="right", style="yellow")

    # Extract probabilities for each switch value (0 or 1)
    def extract_switch_probabilities(observed_values, probabilities):
        value_to_probability = {
            val: prob for val, prob in zip(observed_values, probabilities)
        }
        return value_to_probability.get(0, 0.0), value_to_probability.get(1, 0.0)

    deterministic_off_prob, deterministic_on_prob = extract_switch_probabilities(
        deterministic_switch_unique, deterministic_switch_posterior
    )
    stochastic_off_prob, stochastic_on_prob = extract_switch_probabilities(
        stochastic_switch_unique, stochastic_switch_posterior
    )
    static_off_prob, static_on_prob = extract_switch_probabilities(
        static_switch_unique, static_switch_posterior
    )

    # Get initial state values
    initial_state_obj = initial_state(seed)

    table.add_row(
        "Deterministic",
        str(initial_state_obj.deterministic_switch),
        f"{deterministic_off_prob:.3f}",
        f"{deterministic_on_prob:.3f}",
    )
    table.add_row(
        "Stochastic",
        str(initial_state_obj.stochastic_switch),
        f"{stochastic_off_prob:.3f}",
        f"{stochastic_on_prob:.3f}",
    )
    table.add_row(
        "Static",
        str(initial_state_obj.static_switch),
        f"{static_off_prob:.3f}",
        f"{static_on_prob:.3f}",
    )

    console.print(table)
    assert (
        max_deterministic_switch_prob <= max_concentration
    ), f"Deterministic switch has max probability {max_deterministic_switch_prob:.3f} > {max_concentration}. "

    assert (
        max_stochastic_switch_prob <= max_concentration
    ), f"Stochastic switch has max probability {max_stochastic_switch_prob:.3f} > {max_concentration}. "

    assert (
        max_static_switch_prob <= max_concentration
    ), f"Static switch has max probability {max_static_switch_prob:.3f} > {max_concentration}. "
