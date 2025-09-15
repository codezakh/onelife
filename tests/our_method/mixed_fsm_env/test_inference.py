import random
import numpy as np
from typing import List

from distant_sunburn.our_method.core import SymbolicTransition
from distant_sunburn.mixed_fsm_env import (
    initial_state,
    transition_function,
    Action,
    State,
)
from distant_sunburn.our_method.mixed_fsm_env.handwritten_laws import (
    CORRECT_LAWS,
    INCORRECT_LAWS,
    ALL_LAWS,
)
from distant_sunburn.our_method.optimization import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.our_method.mixed_fsm_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.our_method.mixed_fsm_env.handwritten_laws import (
    CorrectDeterministicSwitchLaw,
    IncorrectDeterministicSwitchLawAssumesStatic,
)
from distant_sunburn.our_method.core import LawFunctionWrapper, WeightedLaw
from distant_sunburn.our_method.world_modeling import LawMixture
from rich.console import Console
from rich.table import Table
from dataclasses import dataclass
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
            prev_state=current_state, action=action, next_state=next_state
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

    good_law = LawFunctionWrapper[State].from_non_runtime_created(
        CorrectDeterministicSwitchLaw()
    )
    bad_law = LawFunctionWrapper[State].from_non_runtime_created(
        IncorrectDeterministicSwitchLawAssumesStatic()
    )

    # Fit weights
    weighted_laws = fitter.fit([good_law, bad_law], train_transitions)

    for weighted_law in weighted_laws:
        print(f"{weighted_law.law.__name__} weight: {weighted_law.weight:.4f}")

    good_weight = weighted_laws[0].weight
    bad_weight = weighted_laws[1].weight

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
    weighted_laws = fitter.fit(ALL_LAWS, train_transitions)

    # Extract weights for correct and incorrect experts
    correct_weights = []
    incorrect_weights = []

    for i, weighted_law in enumerate(weighted_laws):
        weight = weighted_law.weight
        law = weighted_law.law

        print(f"Expert {i} -- {law.__name__}: weight = {weight:.4f}")

        if law in CORRECT_LAWS:
            correct_weights.append(weight)
            print("  -> CORRECT expert")
        elif law in INCORRECT_LAWS:
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
    world_model: LawMixture,
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


def test_jumpy_posterior():
    weighted_laws = [
        WeightedLaw(law=fn, weight=1.0, is_fitted=True) for fn in CORRECT_LAWS
    ]
    world_model = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=weighted_laws,
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

    # Stochastic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_deterministic_switch_posterior.stochastic_switch_posterior[0] >= 0.9

    # Static switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_deterministic_switch_posterior.static_switch_posterior[0] >= 0.9

    toggle_stochastic_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STOCHASTIC_SWITCH, world_model, n_samples, seed
    )

    # Deterministic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_stochastic_switch_posterior.deterministic_switch_posterior[0] >= 0.9

    # Stochastic switch should be split between 0 and 1, with about equal probability
    # for each
    assert (
        abs(
            toggle_stochastic_switch_posterior.stochastic_switch_posterior[0]
            - toggle_stochastic_switch_posterior.stochastic_switch_posterior[1]
        )
        <= 0.1
    )

    # Static switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_stochastic_switch_posterior.static_switch_posterior[0] >= 0.9

    toggle_static_switch_posterior = compute_posterior_over_action(
        Action.TOGGLE_STATIC_SWITCH, world_model, n_samples, seed
    )

    # Deterministic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_static_switch_posterior.deterministic_switch_posterior[0] >= 0.9

    # Stochastic switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_static_switch_posterior.stochastic_switch_posterior[0] >= 0.9

    # Static switch is never modified, so it should be 0 --> 0 with high probability
    assert toggle_static_switch_posterior.static_switch_posterior[0] >= 0.9
