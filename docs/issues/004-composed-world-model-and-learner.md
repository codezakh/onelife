# Issue #004: Composed World Model and PoEWorldLearner Integration

## Context
We reimplemented core PoE-World components in a functional, stateless setting. We now need a composed world model that aggregates experts learned by per–object-type `ObjectModelOrchestrator`s, and a small learner/orchestrator that builds and maintains this composed model.

Our current pieces:
- `PoEWorldModel` implements `WorldModelProtocol` over a flat list of `WeightedExpert` and an `ObservableExtractorProtocol`.
- `ObjectModelOrchestrator` manages two `ExpertManager`s (creation vs non-creation), accumulates transitions, synthesizes experts, fits weights, prunes, and checkpoints.
- `ExpertManager` wraps `MaxLikelihoodWeightFitter` and `PoEWorldModel` to manage experts/weights.
- `ObservableExtractor` (Crafter and 1D variants) returns predictions for all observable attributes (not scoped per object type).

## High-Level Idea
- Compose a global world model by collecting all experts from each `ObjectModelOrchestrator` and instantiating a single `PoEWorldModel(observable_extractor, aggregated_experts)`.
- Keep the composed model read-only (inference surface). Data flows only into the orchestrators via `add_datapoint`. The learner coordinates learning cycles and (re)builds the composed model.
- Use the global extractor deliberately: unpredicted attributes are ignored in the likelihood; bad cross-type mutations get penalized → experts get pruned. This preserves simplicity and matches PoE-World "in spirit".

## Responsibilities
- PoEWorldLearner (ours)
  - Own one `ObjectModelOrchestrator` per object type (e.g., "player", ...).
  - Data ingestion: route `SymbolicTransition` to all orchestrators via `add_datapoint`.
  - Learning lifecycle:
    - `synthesize_world_model(transitions)`: full loop per orchestrator (`infer_moe()`), collect `ObjectTypeModel`s.
    - `update_world_model(transitions, fast=True)`: quick loop per orchestrator (`fast_infer_moe()` or full), then rebuild composed model.
    - `save_snapshot()` / `load_snapshot()` fan out to orchestrators.
  - Composition:
    - Aggregate experts: concat all object types’ non-creation + creation experts.
    - Build `PoEWorldModel(observable_extractor, aggregated_experts)` and expose as `WorldModelProtocol`.

- Composed World Model
  - Implemented by reusing `PoEWorldModel` directly over the aggregated experts.
  - Methods: `sample_next_state`, `evaluate_log_probability`, `experts`, `with_new_experts` (already provided by `PoEWorldModel`).

- ObjectModelOrchestrator (existing)
  - Accumulate transitions, detect surprise via managers, synthesize experts, fit weights, prune, checkpoint.

## Key Methods & Protocols
- `WorldModelProtocol` (core): `sample_next_state`, `evaluate_log_probability`, `with_new_experts`, `experts`.
- `ObservableExtractorProtocol` (core): `extract_attribute_predictions`, `get_observed_outcomes`, `apply_expert_predictions`.
- `ObjectModelOrchestrator`: `add_datapoint`, `infer_moe`, `fast_infer_moe`, `get_model`, checkpoint helpers.

## Dataflow
1) Environment produces `SymbolicTransition`s → PoEWorldLearner routes to orchestrators (`add_datapoint`).
2) Learner triggers learning rounds (full or fast) on orchestrators.
3) Learner collects experts and rebuilds the composed `PoEWorldModel`.
4) Downstream consumers (evaluation, prospective agent) use the composed model for sampling and scoring.

## Testing Plan (sketch)
- Unit: Composition behavior (Simple 1D)
  - We already have hand-written experts in `simple_1d_env/handwritten_experts.py`.
  - Build a small set across two pseudo-types; compose and check:
    - `evaluate_log_probability(s,a,s')` is higher for true vs wrong next state.
    - Unpredicted attributes do not impact log-prob.
- Integration: Orchestrator → Composed model (Crafter)
  - Run `infer_moe()` on a single orchestrator with simple transitions to synthesize experts, compose, then verify finite/improved log-prob on held-out transitions.
- Pruning effect visible in composed model
  - Include a bad expert, show log-prob improves after pruning and recomposition.

## Risks / Decisions
- Non-scoped extractor: Chosen for simplicity; safe because only predicted attributes contribute to likelihood; illegal cross-type mutations are penalized and pruned.
- Constraints layer: Omitted initially; can add later if needed.

## Implementation Sketch
- Add `poe_world_learner.py` with class `PoEWorldLearner`:
  - Holds: `dict[str, ObjectModelOrchestrator]`, `ObservableExtractorProtocol`, current `WorldModelProtocol`.
  - Methods: `synthesize_world_model`, `update_world_model`, `save_snapshot`, `load_snapshot`, `get_model`.
  - Helper: `_compose_world_model()` that flattens experts from orchestrators and returns `PoEWorldModel`.
- Optional helper module `composition.py` with `build_composed_world_model(extractor, object_type_models)` if we want to separate concerns.
- Add tests for composition and offline synthesis.
