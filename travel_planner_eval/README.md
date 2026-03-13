# TravelPlanner

[TravelPlanner](https://arxiv.org/abs/2402.01622) is a benchmark for evaluating language agents on real-world multi-day travel planning. Given a natural-language query and pre-collected reference information (the "sole-planning" mode), the model must produce a complete day-by-day itinerary covering transportation, meals, attractions, and accommodation.

## Usage

### Installation

Install the package and its dependencies from the `travel_planner_eval/` directory:

```bash
uv sync
```

### Running evaluations

```bash
uv run inspect eval src/ --model openai/gpt-4o
```

You can also import the task as a normal Python object:

```python
from inspect_ai import eval
from travel_planner_eval import travel_planner
eval(travel_planner())
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval src/ --limit 20
uv run inspect eval src/ --max-connections 10
uv run inspect eval src/ --temperature 0.5
```

See `uv run inspect eval --help` for all available options.

## Parameters

### `travel_planner`

- `strategy` (Literal["direct", "cot", "react", "reflexion"]): Planning strategy to use. `direct` and `cot` are single-turn solvers; `react` and `reflexion` use Inspect's native `react()` agent with a cost-enquiry tool. (default: `"direct"`)
- `split` (Literal["validation", "train"]): Dataset split to evaluate. The test split is not supported (no ground-truth labels). (default: `"validation"`)
- `parse_model` (str | None): Model used to parse plan output into structured JSON for scoring. If `None`, the task model is used. (default: `None`)

```bash
# Evaluate with the ReAct strategy on the training split using a dedicated parse model
uv run inspect eval src/ \
  --model openai/gpt-4o \
  -T strategy=react \
  -T split=train \
  -T parse_model=openai/gpt-4o-mini
```

## Dataset

The dataset is loaded from [osunlp/TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) on HuggingFace.

| Split      | Samples | Levels               |
|------------|---------|----------------------|
| train      | 45      | easy / medium / hard |
| validation | 180     | easy / medium / hard |

The **test split** exists on HuggingFace but is not supported here. It omits the columns required for scoring (`people_number`, `budget`, `visiting_city_number`, `local_constraint`, `annotated_plan`)

Each sample is a travel query (e.g. "Plan a 5-day trip for 4 people from Seattle to California with a $12,000 budget, preferring Italian cuisine") paired with pre-collected reference information about available flights, hotels, restaurants, and attractions.

## Scoring

Scoring replicates the original paper's two-step process:

1. **Parsing**: An LLM converts the natural-language plan output into structured JSON (replicating the `postprocess/parsing.py` step from the original codebase). By default, the task model is used for parsing; set `parse_model` to override.

2. **Constraint checking**: The parsed plan is evaluated against two categories of constraints:

### Commonsense Constraints

| Check | Description |
|---|---|
| `is_reasonable_visiting_city` | Trip forms a valid closed-loop route |
| `is_valid_restaurants` | No restaurant visited more than once |
| `is_valid_attractions` | No attraction visited more than once |
| `is_valid_transportation` | No conflicting transport modes (e.g. flight + self-driving) |
| `is_valid_information_in_current_city` | Meals/attractions/accommodation match the day's city |
| `is_not_absent` | All required fields present and ≥50% filled |

### Hard Constraints

| Check | Description |
|---|---|
| `valid_transportation` | Transport mode satisfies the user's constraint (e.g. "no flight") |

The primary metric is **Commonsense Macro Pass Rate**: a sample passes if all implemented commonsense constraint checks pass, and the macro pass rate is the fraction of samples that pass.

## Evaluation Report

**Evaluation Version:** 1-A

| Model | Provider | Strategy | Split | Commonsense Macro Pass Rate | Stderr | Samples |
| ----- | -------- | -------- | ----- | --------------------------- | ------ | ------- |
| gpt-3.5-turbo-1106 | OpenAI | direct | validation | 0.050 | 0.050 | 20 |

**Notes:**

- More results to be added.
- Human expert baseline from the paper is 100% commonsense pass rate and 67.7% final pass rate (all constraints).

**Evaluation Commands:**

```bash
uv run inspect eval src/ --model openai/gpt-3.5-turbo-1106 --limit 20 -T split=validation
```

## Differences from Original Implementation

1. **Planning strategies**: Adds `react` and `reflexion` strategies using Inspect AI's native `react()` agent, in addition to the original `direct` and `cot` strategies.
2. **ReAct tooling**: The original codebase uses a custom agent loop that templates `{scratchpad}` directly into the prompt and parses `CostEnquiry[...]` / `Finish[...]` patterns from raw text. This implementation uses `inspect_ai.agent.react()`, which manages conversation history as structured messages and exposes `cost_enquiry` and `submit` as native model tools. As a result, `REACT_AGENT_SYSTEM_PROMPT` is a static system prompt with no `{scratchpad}` placeholder, and the per-query context (`{text}`, `{query}`) is passed as the user message via `REACT_USER_TEMPLATE`.
3. **Parsing model**: The original uses GPT-4 specifically for the JSON parsing step; this implementation defaults to the task model, keeping the evaluation self-contained. GPT-4 or any other model can still be set via `parse_model`.
4. **Evaluation framework**: Uses Inspect AI for structured task definition, scoring, and metrics rather than the original standalone scripts.
5. **Scorer output**: Constraint results are stored as structured `ScoreExplanation` metadata on each sample, making per-constraint analysis directly accessible in the Inspect log viewer.

## References

- **Paper**: [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622)
- **Original Implementation**: [OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)
- **Leaderboard**: [TravelPlanner Leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard)
