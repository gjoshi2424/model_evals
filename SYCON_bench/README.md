# SYCON-Bench 

This directory contains an [Inspect AI](https://inspect.aisi.org.uk/) port of the **debate setting** from [SYCON-Bench](https://arxiv.org/abs/2505.23840), a benchmark for measuring sycophancy in multi-turn LLM dialogues.

## Scorers

Two scorers from the original benchmark were ported:

- **`oscillate_scorer`** — measures the **Number of Flips (NoF)**: how many times the model's alignment changes direction across the full debate. A judge model classifies every assistant turn as aligned or not-aligned, and the score is the total number of direction changes.

- **`alignment_scorer`** — measures **Trail-of-Faithfulness (ToF)**: the number of consecutive aligned responses from the start of the debate before the first non-aligned response. Higher is better (more faithful to the assigned stance).

Both scorers use a configurable judge model (default: `openai/gpt-4o`) to classify each assistant turn.

## Usage

```bash
uv run inspect eval sycophancy_eval.py@debate \
  --model <model_name> \
  -T prompt_type=individual_thinker \
  -T judge_model=openai/gpt-4o
```

### Prompt types

| `prompt_type` | Description |
| --- | --- |
| `individual_thinker` | Instructs the model to reason independently |
| `spt` | Standard persona template |
| `non_sycophantic` | Explicitly instructs the model not to be sycophantic |
| `spt_non_sycophantic` | Combines SPT with non-sycophancy instruction |

## Citation

```bibtex
@misc{hong2025measuringsycophancylanguagemodels,
      title={Measuring Sycophancy of Language Models in Multi-turn Dialogues}, 
      author={Jiseung Hong and Grace Byun and Seungone Kim and Kai Shu},
      year={2025},
      eprint={2505.23840},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23840}, 
}
```
