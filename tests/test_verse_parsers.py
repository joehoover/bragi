
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_token_syllable_scores_with_lexical_constraint():
    
    # This is dumb and bad, need to add tokenizer mock.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokens_to_include=['dog', 'cat', 'run']
    
    syllable_scores = token_syllable_scores(tokenizer, pt=True, tokens_to_include=['dog', 'cat', 'run'])
    
    token_id = tokenizer('cow')
    assert syllable_scores[token_id['input_ids'][0]].item() == float('Inf')
    token_id = tokenizer('dog')
    assert syllable_scores[token_id['input_ids'][0]].item() == 1.