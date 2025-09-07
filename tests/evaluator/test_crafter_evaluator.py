import rich
from crafter.functional_env import EnvConfig
from crafter.functional_env import transition as crafter_transition_fn
from crafter.state_export import WorldState

from distant_sunburn.evaluator.baselines import NullWorldModel, TrueTransitionWorldModel
from distant_sunburn.evaluator.core import EvaluationConfig, Evaluator
from distant_sunburn.evaluator.crafter.components import _gamestate_to_json
from distant_sunburn.evaluator.crafter.factory import CrafterEvaluationFactory
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX


def test_evaluating_true_vs_null_world_model():
    """
    Test that the evaluation framework can successfully evaluate a world model
    and produce meaningful results using the factory and actual evaluator.
    """
    env_config = EnvConfig(size=(9, 9), view=(9, 9))
    factory = CrafterEvaluationFactory(env_config, policy_seed=42)
    config = EvaluationConfig(num_distractors=2)

    context = factory.create_context(config, num_transitions_per_scenario=30)

    def equality_check(state1: WorldState, state2: WorldState) -> bool:
        return _gamestate_to_json(state1) == _gamestate_to_json(state2)

    def wrap_true_transition_fn(state: WorldState, action) -> WorldState:
        next_state, _ = crafter_transition_fn(state, MAP_ACTION_TO_INDEX[action])
        return next_state

    true_model = TrueTransitionWorldModel(
        wrap_true_transition_fn, equal_fn=equality_check
    )
    null_model = NullWorldModel(equality_check)

    evaluator = Evaluator(context)

    true_wm_perf = evaluator.evaluate(true_model)

    null_wm_perf = evaluator.evaluate(null_model)

    assert (
        true_wm_perf.discriminative_accuracy == 1.0
    ), "True transition model should have accuracy greater than 0.5"

    #
    assert (
        true_wm_perf.edit_distance.raw == 0.0
    ), "True transition model should have low generative error"

    # Null model should perform worse than true model
    assert (
        null_wm_perf.discriminative_accuracy < true_wm_perf.discriminative_accuracy
    ), "Null model should perform worse than true model"

    rich.print("== True World Model Performance ==")
    rich.print(true_wm_perf)

    rich.print("== Null World Model Performance ==")
    rich.print(null_wm_perf)
