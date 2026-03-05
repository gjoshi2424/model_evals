# basketball-evals
Learning inspect by creating simple evals related to basketball

## Overview
Two evaluation tasks testing LLMs on basketball knowledge:
1. **Basketball Plays** - Identifying plays from descriptions
2. **Player Stats** - Mathematical calculations with optional tools

## Results

### Basketball Plays Questions

| Model | Setting | Accuracy | Std Error |
|-------|---------|----------|-----------|
| llama3.2 | Zero-shot | 0.320 | 0.067 |
| llama3.2 | Few-shot | 0.420 | 0.071 |
| llama3.2:1b | Zero-shot | 0.360 | 0.069 |
| llama3.2:1b | Few-shot | 0.220 | 0.059 |


### Thoughts on results
Tested using llama3.2 and lama3.2:1b locally on 1 epoch with the temperature set to 0. 

Llama3.2 was more accurate when given fewshot examples, however 3.2:1b was less accurate when given fewshot examples. I think because 3.2:1b is a smaller model it may struggle with few-shot examples for a variety of reasons such as context dilution and difficulty following instructions. When I tried reducing the number of few-shots to 1 the accuracy seems to go up to 0.260.

I was also surprised that 3.2 performed worse than 3.2:1b for the zero-shot evaluation. As a next step I would try to test with a larger data set.

Potential issues:

-The data was created synthetically using gemini, so there can be inaccuracies in the descriptions I may have missed

-Data only includes 50 questions which is relatively small

-Some basketball schemes and plays have multiple names

### Player Stats Questions

| Model | Setting | Tools | Accuracy | Std Error |
|-------|---------|-------|----------|-----------|
| llama3.2 | Zero-shot | No | 0.080 | 0.055 |
| llama3.2 | Few-shot | No | 0.240 | 0.087 |
| llama3.2 | Zero-shot | Yes | 1.000 | 0.000 |
| llama3.2 | Few-shot | Yes | 1.000 | 0.000 |
| llama3.2:1b | Zero-shot | No | 0.040 | 0.040 |
| llama3.2:1b | Few-shot | No | 0.040 | 0.040 |
| llama3.2:1b | Zero-shot | Yes | 0.080 | 0.055 |
| llama3.2:1b | Few-shot | Yes | 0.080 | 0.055 |

### Thoughts on results

Tested using llama3.2 and lama3.2:1b locally on 1 epoch with the temperature set to 0. Because of limitations when testing locally I set the sample size to 25 questions. 

Llama 3.2 performed better with few-shot examples, and very well when allowed to use tools. The two main issues when reviewing the results of runs without tools were that the model either made rounding errors or understood the question but used the wrong formula to calculate the result. Initially when testing with tools, I used the same prompt template as the no-tools condition (PROMPT_TEMPLATE_WITHOUT_TOOLS). This resulted in poor accuracy despite tools being available. Only after creating a separate prompt template (PROMPT_TEMPLATE_WITH_TOOLS) that explicitly instructed the model to use the given tools and trust the result from them did accuracy improve to 100%. This highlights the importance of prompt engineering when using tool-augmented evaluation.

Llama 3.2:1b did not see any changes in accuracy regardless of if few shot examples were introduced. There was a slight increase in accuracy (of 0.04) when tools were added. Here, I set the time-out=60 seconds for each sample and max-connections=1 (as seen in running evals section) because the model seemed to get stuck on certain questions. Issues may include the complexity of tool use and using too much of the context window with prompts and few shot examples.

Potential Issues:
-25 questions is a very small sample size

-Without explicit formulas formatting of the questions may be confusing or harder to understand


## Running the Evals

### Setup
```bash
pip install inspect-ai
ollama pull llama3.2
ollama pull llama3.2:1b
```

### Run All Evaluations
```bash
# Basketball Plays
inspect eval evals/identify-play.py@basketball_plays --model ollama/llama3.2
inspect eval evals/identify-play.py@basketball_plays  --model ollama/llama3.2:1b -T few_shot=true

inspect eval evals/identify-play.py@basketball_plays --model ollama/llama3.2:1b
inspect eval evals/identify-play.py@basketball_plays --model ollama/llama3.2:1b -T few_shot=true

# Player Stats
inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2
inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2 -T few_shot=true
inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2 -T pass_tools=true
inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2 -T few_shot=true pass_tools=true

inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2:1b
inspect eval evals/calculate-player-stats.py@basketball_stats --model ollama/llama3.2:1b -T few_shot=true
inspect eval ./evals/calculate-player-stats.py --model ollama/llama3.2:1b --max-connections=1 --time-limit=60 -T pass_tools=True 
inspect eval ./evals/calculate-player-stats.py --model ollama/llama3.2:1b --max-connections=1 --time-limit=60 -T pass_tools=True few_shot=True
```