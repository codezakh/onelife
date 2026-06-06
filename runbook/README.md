# Runbook

The runbook runs the full OneLife pipeline. It has three steps, run in order. Each step reads the previous step's output, so running them in order with no arguments chains the whole pipeline.

Steps 1 and 2 call an LLM and need `GEMINI_API_KEY` in your `.env`. Step 3 does not. Steps 1 and 2 also take a `--debug` flag that runs a short version for a quick smoke test.

## 1. Generate an exploration trajectory

```bash
uv run --env-file .env python runbook/01_generate_exploration_trajectory.py
```

Runs one unguided exploration episode in Crafter and writes the trajectory under `runbook_output/01_exploration/`.

## 2. Synthesize laws

```bash
uv run --env-file .env python runbook/02_synthesize_laws.py
```

Reads the trajectory from step 1 and writes the synthesized laws under `runbook_output/02_synthesis/laws/`.

## 3. Optimize and evaluate

```bash
uv run --env-file .env python runbook/03_optimize_and_evaluate.py
```

Fits the world model from the laws and the trajectory, evaluates it, and writes `fitted_world_model.json` and `evaluation_results.json` under `runbook_output/03_optimize_and_evaluate/`.

## Running a step on its own

Each step takes the previous output as a path, so you can run a later step without running the earlier ones. The trajectory and laws from the paper are on HuggingFace at [OneLife-Crafter](https://huggingface.co/datasets/codezakh/OneLife-Crafter). Download them and pass them in. For example, to fit and evaluate from the published trajectory and laws:

```bash
hf download codezakh/OneLife-Crafter --repo-type dataset --local-dir OneLife-Crafter
uv run --env-file .env python runbook/03_optimize_and_evaluate.py \
    --trajectory OneLife-Crafter/trajectory/open_ended_run_00.jsonl \
    --laws-dir OneLife-Crafter
```
