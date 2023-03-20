import torch
import transformers 

from transformers.generation.logits_process import LogitsWarper

# TODO
# 1. Add support for beam search
class SyllableRestrictionWarper(LogitsWarper):
    def __init__(
        self, 
        prompt: str,
        tokenizer: transformers.PreTrainedTokenizerFast,
        syllable_budget: torch.Tensor,
        syllable_scorer: callable,
        filter_value: float = -float("Inf"), 
        min_tokens_to_keep: int = 1,
        free_tokens=['!', ',', ':', '?', ';', ' ',],
        new_line_token = '\n',
        num_beams: int = 10,
    ):
#         if not isinstance(syllable_budget, int) or syllable_budget < 0:
#             raise ValueError(f"`syllable_budget` has to be a strictly positive or zero integer, but is {syllable_budget}")

        self.syllable_budget = syllable_budget.repeat(num_beams)
        self.filter_value = filter_value
        self.syllable_scores = syllable_scorer(tokenizer, pt=True, free_tokens=free_tokens)#.repeat(num_beams, 1)
        self.tokenizer = tokenizer
        self.new_line_token = '\n'
        self.new_line_token_id = tokenizer('\n', return_tensors='pt')['input_ids'].item()
        self.prompt_offset = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
        self.line_number = 0
        self.line_budget = syllable_budget.shape[0] - 1
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]


        syllable_scores = self.syllable_scores
        
        if self.line_number > self.syllable_budget.shape[0] - 1:
            syllable_budget = torch.Tensor([0])
        else:
            syllable_budget = self.syllable_budget[self.line_number, None]

        # Update syllable budget 
        syllable_budget = self.update_syllable_budget(input_ids, syllable_budget, syllable_scores)
        
        
        # Remove all tokens with more syllables than `syllable_budget`
        syllable_scores = syllable_scores.repeat(batch_size, 1)
        indices_to_remove =  syllable_scores > syllable_budget[:,None]
        
        # Check if line has been completed
        line_completed = syllable_budget <= 0
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        if True in line_completed:
  
            
            # Force EOS if line budget is spent
            if self.line_budget < 0 or self.line_number > self.syllable_budget.shape[0]:

                indices_to_remove[line_completed,:] = torch.full_like(indices_to_remove[line_completed,:], True)
                scores[line_completed, self.tokenizer.eos_token_id] = -1
            # Otherwise, force new line and move to next line budget
            else:
                scores[line_completed,  self.new_line_token_id] = -1
                
            self.line_number += 1
            self.line_budget -= 1

        return scores

    
    def update_syllable_budget(self, input_ids, syllable_budget, syllable_scores):
        if input_ids.shape[1] > self.prompt_offset:
            syllable_cost = syllable_scores[input_ids[:,-1]]
            syllable_budget -= syllable_cost

        return syllable_budget 