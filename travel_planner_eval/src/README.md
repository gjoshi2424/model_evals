# TravelPlanner

TravelPlanner is a benchmark for evaluating language agents on real-world multi-day travel planning. Given a natural-language query and pre-collected reference information (the "sole-planning" mode), the model must produce a complete day-by-day itinerary covering transportation, meals, attractions, and accommodation.

Based on the paper: [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622) (Zheng et al., 2024).

<!-- [BEGIN_EVAL_REPORT] -->
<!-- Evaluation report will be added here -->
<!-- [END_EVAL_REPORT] -->

## Usage

```bash
uv run inspect eval src/ --model openai/gpt-3.5-turbo-1106
```

From Python:

```python
from travel_planner_eval import travel_planner_direct
```

## Dataset

The dataset is loaded from [osunlp/TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) on HuggingFace.

| Split      | Samples | Levels              |
|------------|---------|---------------------|
| train      | 45      | easy / medium / hard |
| validation | 180     | easy / medium / hard |

Each sample is a travel query (e.g. "Plan a 5-day trip for 4 people from Seattle to California with a $12,000 budget, preferring Italian cuisine") paired with pre-collected reference information about available flights, hotels, restaurants, and attractions.

## Evaluation

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
| `valid_transportation` | Transport mode satisfies user's constraint (e.g. "no flight") |

> **Note**: Constraint checks that require the travel database CSV files (price lookups, accommodation details, restaurant cuisine data) are not yet implemented. These include `is_valid_information_in_sandbox`, `is_valid_accommodaton` (minimum nights), `valid_cost`, `valid_cuisine`, `valid_room_rule`, and `valid_room_type`.

## Options

| Parameter | Description | Default |
|---|---|---|
| `split` | Dataset split: `"validation"` or `"train"` | `"validation"` |
| `shuffle` | Shuffle the dataset before evaluation | `False` |
| `seed` | Random seed for shuffling | `None` |
| `parse_model` | Model for the JSON parsing step | Task model |

```bash
# Evaluate on the training split with a dedicated parse model
# From the model_evals/ directory:
uv run inspect eval travel_planner_eval \
  --model openai/gpt-4o \
  -T split=train \
  -T parse_model=openai/gpt-4o-mini
```

## Evaluation Report
### Results from eval

**Evaluation Version:** 1-A
| Model | Provider | Split | Commonsense Macro Pass Rate | Stderr | Samples |
| ----- | -------- | ----- | -------- | ------ | ------- |
| gpt-3.5-turbo-1106 | OpenAI | validation | 0.050 | 0.050 | 20 |

**Replication Steps**
```python
uv run inspect eval src/ --model openai/gpt-3.5-turbo-1106 --limit 1 -T split=validation
```

## References

- **Paper**: [TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622)
- **Original Implementation**: [OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)
- **Leaderboard**: [Travel Planner Leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard)
