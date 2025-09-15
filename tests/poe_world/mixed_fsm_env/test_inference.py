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
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from collections import Counter


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


@dataclass
class PosteriorOverAction:
    deterministic_switch_posterior: np.ndarray
    stochastic_switch_posterior: np.ndarray
    static_switch_posterior: np.ndarray


def compute_posterior_over_action(
    action: Action,
    world_model: PoEWorldModel,
    n_samples: int,
    seed: int,
) -> PosteriorOverAction:
    deterministic_switch_values: list[int] = []
    stochastic_switch_values: list[int] = []
    static_switch_values: list[int] = []

    for _ in range(n_samples):
        state = initial_state(seed)
        next_state = world_model.sample_next_state(state, action)
        deterministic_switch_values.append(next_state.deterministic_switch)
        stochastic_switch_values.append(next_state.stochastic_switch)
        static_switch_values.append(next_state.static_switch)

    deterministic_switch_counts = Counter(deterministic_switch_values)
    stochastic_switch_counts = Counter(stochastic_switch_values)
    static_switch_counts = Counter(static_switch_values)

    deterministic_switch_posterior = np.zeros(2)
    stochastic_switch_posterior = np.zeros(2)
    static_switch_posterior = np.zeros(2)

    for k, v in deterministic_switch_counts.items():
        deterministic_switch_posterior[k] = v / n_samples
    for k, v in stochastic_switch_counts.items():
        stochastic_switch_posterior[k] = v / n_samples
    for k, v in static_switch_counts.items():
        static_switch_posterior[k] = v / n_samples

    # Display switch distributions in a clean 3x2 table
    console = Console()
    table = Table(title=f"Switch Distributions for {action}", show_header=True)
    table.add_column("Switch", style="cyan")
    table.add_column("Initial", justify="center", style="white")
    table.add_column("Off (0)", justify="right", style="green")
    table.add_column("On (1)", justify="right", style="yellow")

    deterministic_off_prob, deterministic_on_prob = deterministic_switch_posterior
    stochastic_off_prob, stochastic_on_prob = stochastic_switch_posterior
    static_off_prob, static_on_prob = static_switch_posterior

    # Get initial state values
    initial_state_obj = initial_state(0)

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

    return PosteriorOverAction(
        deterministic_switch_posterior,
        stochastic_switch_posterior,
        static_switch_posterior,
    )


def test_posterior_good_experts_only():
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

    toggle_deterministic_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_DETERMINISTIC_SWITCH, world_model, n_samples, seed
    )

    # Deterministic switch should toggle from 0 --> 1 with high probability
    assert (
        toggle_deterministic_switch_posterior.deterministic_switch_posterior[1] >= 0.9
    )

    # Stochastic switch is never modified, but the implicit uniform prior
    # should cause it to be mixed between 0 and 1
    assert toggle_deterministic_switch_posterior.stochastic_switch_posterior[0] >= 0.3
    assert toggle_deterministic_switch_posterior.stochastic_switch_posterior[1] >= 0.3

    # Static switch is never modified, but the implicit uniform prior
    # should cause it to be mixed between 0 and 1
    assert toggle_deterministic_switch_posterior.static_switch_posterior[0] >= 0.3
    assert toggle_deterministic_switch_posterior.static_switch_posterior[1] >= 0.3

    toggle_stochastic_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STOCHASTIC_SWITCH, world_model, n_samples, seed
    )

    # Implicit priors should cause all observables to be mixed between 0 and 1

    assert toggle_stochastic_switch_posterior.deterministic_switch_posterior[0] >= 0.3
    assert toggle_stochastic_switch_posterior.deterministic_switch_posterior[1] >= 0.3

    assert toggle_stochastic_switch_posterior.static_switch_posterior[0] >= 0.3
    assert toggle_stochastic_switch_posterior.static_switch_posterior[1] >= 0.3

    assert toggle_stochastic_switch_posterior.static_switch_posterior[0] >= 0.3
    assert toggle_stochastic_switch_posterior.static_switch_posterior[1] >= 0.3

    toggle_static_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STATIC_SWITCH, world_model, n_samples, seed
    )

    # Implicit priors should cause all observables to be mixed between 0 and 1

    # Deterministic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_static_switch_posterior.deterministic_switch_posterior[0] >= 0.3
    assert toggle_static_switch_posterior.deterministic_switch_posterior[1] >= 0.3

    # Stochastic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_static_switch_posterior.stochastic_switch_posterior[0] >= 0.3
    assert toggle_static_switch_posterior.stochastic_switch_posterior[1] >= 0.3

    # Static switch expert puts heavy probability mass on 0
    assert toggle_static_switch_posterior.static_switch_posterior[0] >= 0.9


def test_posterior_after_fitting_all_experts():
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

    world_model = PoEWorldModel(
        observable_extractor=ObservableExtractor(),
        weighted_experts=weighted_experts,
    )

    n_test_samples = 100
    seed = 42

    toggle_deterministic_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_DETERMINISTIC_SWITCH, world_model, n_test_samples, seed
    )

    assert (
        toggle_deterministic_switch_posterior.deterministic_switch_posterior[1] >= 0.9
    )

    toggle_stochastic_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STOCHASTIC_SWITCH, world_model, n_test_samples, seed
    )

    # Assert all observables are mixed between 0 and 1
    # We can do this more easily by checking that the max probability is less than
    # a concentration limit

    assert max(toggle_stochastic_switch_posterior.deterministic_switch_posterior) <= 0.7
    assert max(toggle_stochastic_switch_posterior.stochastic_switch_posterior) <= 0.7
    assert max(toggle_stochastic_switch_posterior.static_switch_posterior) <= 0.7

    toggle_static_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STATIC_SWITCH, world_model, n_test_samples, seed
    )

    assert max(toggle_static_switch_posterior.deterministic_switch_posterior) <= 0.7
    assert max(toggle_static_switch_posterior.stochastic_switch_posterior) <= 0.7
    # Only the static switch expert puts heavy probability mass on 0
    assert toggle_static_switch_posterior.static_switch_posterior[0] >= 0.9
