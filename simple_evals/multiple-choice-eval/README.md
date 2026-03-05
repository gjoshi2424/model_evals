## Sciq Eval
This is a simple exercise to evaluate [sciq data set](https://huggingface.co/datasets/allenai/sciq/viewer/default/test?views%5B%5D=test) using gpt2. The accuracy for each question is measured by taking the logliklihood of the logits from the model response that correspond to the continuation tokens. 

## Learning Objective

This exercise is based on the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) in which I tried to break down their implementation of using logliklihood to evaluate a task using hf transformers to a very simple implementation (without batching, caching, etc.).
