## Perplexity Eval
This is a simple exercise to evaluate [wikitext-2 dataset](https://huggingface.co/datasets/EleutherAI/wikitext_document_level/viewer/wikitext-2-v1/test?views%5B%5D=wikitext_2_v1_test) using gpt2. The accuracy for each question is measured by taking the logliklihood of the logits from the model response that correspond to the continuation tokens. 

## Learning Objective

This exercise is based on the [lm evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) in which I tried to break down their implementation of using perplexity to evaluate wikitext using hf transformers (without batching, caching, etc.).
