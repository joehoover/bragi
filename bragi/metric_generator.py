
import torch
from typing import Optional, List

from transformers.generation.logits_process import LogitsProcessorList

from .verse_parsers import PoesyParsedVerseHandler
from .logit_warpers import SyllableRestrictionWarper

class MetricGenerator():
    def __init__(
        self, 
        model, 
        tokenizer, 
        syllable_scorer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.syllable_scorer = syllable_scorer
        self.verse_handler = verse_handler = PoesyParsedVerseHandler()
        
    
    def calculate_syllable_budget(
        self,
        text_init
    ):
        
        _, syllable_budget = self.verse_handler.example(text_init)
        syllable_budget = torch.Tensor(syllable_budget)
        return syllable_budget
    
    def generate(
        self, 
        prompt, 
        text_init: Optional[str] = None,
        syllable_budget: Optional[torch.Tensor] = None, 
        free_tokens: Optional[List] = ['\n', '!', ',', ':', '?', ';', ' '],
        num_beams: Optional[int] = 1, 
        return_full_text: bool = False,
        **kwargs
    ):
        
        if text_init and syllable_budget:
            raise Error("You cannot specify both `syllable_budget` and `text_init`. Choose one or the other.")
        
        if not text_init and not torch.is_tensor(syllable_budget):
            raise Error("You must provide either `syllable_budget` or `text_init`.")
        
        if text_init:
            syllable_budget = self.calculate_syllable_budget(text_init)
            
        processors = LogitsProcessorList()
        
        processors.append(
            SyllableRestrictionWarper(
                tokenizer=self.tokenizer,
                syllable_budget=syllable_budget,
                syllable_scorer=self.syllable_scorer,
                free_tokens=free_tokens,
                num_beams = num_beams,
                prompt = prompt,
            )

        )
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        outputs = self.model.generate(
            input_ids,
            num_beams=num_beams,
            logits_processor=processors,
            **kwargs
        )

        outputs = outputs[:, input_ids.shape[1]:]
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return output