# bragi

## Generating metered verse with LLaMA

This repo provides methods for using Meta's 6 billion parameter LLaMA model to generate song lyrics with a specific metric structure. If you've been yearning to rewrite the happy birthday song so that it's just about dogs, bragi can help :). 

The core functionality of `bragi` is provided via the `MetricGenerator` class. If you're wondering, Bragi is the [Norse god](https://en.wikipedia.org/wiki/Bragi) of poetry!

The library also provides wrappers around various methods for extracting metric information, such as syllable counts and rhyme schemes.

# Dev setup

1. You need to install `espeak`. 

On mac:
```
brew install espeak
```

On linux

```
apt-get update -y
apt-get install espeak -y
```

2. You also need to make sure torch is installed. I don't like installing torch with poetry, so it's not specified in the `pyproject.toml`. If your environment doesn't already have torch, run:

```
pip install torch
```


# Quick Start

See this [notebook](https://github.com/joehoover/bragi/blob/main/notebooks/llama-dev.ipynb). But, in general:

```python
from bragi.metric_generator import MetricGenerator
from transformers import LLaMAForCausalLM, LLaMATokenizer
import torch 

CACHE_DIR = 'weights'
SEP = "<sep>"
MODEL_PATH  = "/src/weights"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Load model and tokenizer
model = LLaMAForCausalLM.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, local_files_only=True).to(device)
tokenizer = LLaMATokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, local_files_only=True)

# Initialize `MetricGenerator`
generator = MetricGenerator(model=model, tokenizer=tokenizer, device=device)

# Generate
torch.manual_seed(2)
output = generator(
    prompt = prompt,
    text_init = text_init,
    free_tokens=['||', '?', '.', ','],
    # syllable_budget = torch.Tensor([6., 6.]),
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    remove_invalid_values=True,
    do_sample=True,
    top_k=25,
    temperature=.7,
    max_length = 100,
    new_line_token='||',
    bad_words_ids=[[8876]],
)

print('---text_init----')
print(text_init)
print('\n')

print('----output-----')
print(output)
print('\n')

print('----Syllables-----')
print(f"Syllables per line in output: {generator.calculate_syllable_budget(output)}")
print(f"Syllables per line in `text_init`: {generator.calculate_syllable_budget(text_init)}")

```