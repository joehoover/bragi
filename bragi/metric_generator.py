
import torch
from typing import Optional, List

from transformers.generation.logits_process import LogitsProcessorList

from .verse_parsers import PoesyParsedVerseHandler, token_syllable_scores
from .logit_warpers import SyllableRestrictionWarper

class MetricGenerator():
    def __init__(
        self, 
        model, 
        tokenizer, 
        syllable_scorer = token_syllable_scores,
        device = 'cpu',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.syllable_scorer = syllable_scorer
        self.verse_handler = verse_handler = PoesyParsedVerseHandler()
        self.device = device


        try:
            self.tokenizer.vocab
        except AttributeError:
            self.tokenizer.vocab = self.tokenizer.get_vocab()
        
    
    def calculate_syllable_budget(
        self,
        text_init,
        pt: str = True,
    ):
        
        _, syllable_budget = self.verse_handler.example(text_init)
        
        if pt:
            syllable_budget = torch.Tensor(syllable_budget)
        
        return syllable_budget
    
    def generate(
        self, 
        prompt, 
        text_init: Optional[str] = None,
        syllable_budget: Optional[torch.Tensor] = None, 
        free_tokens: Optional[List] = ['\n', '!', ',', ':', '?', ';', ' ', '||'],
        num_beams: Optional[int] = 1, 
        return_full_text: bool = False,
        new_line_token: str = '||',
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
                new_line_token = new_line_token,
                device = self.device,
            )

        )
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)


        outputs = self.model.generate(
            input_ids,
            num_beams=num_beams,
            logits_processor=processors,
            **kwargs
        )

        outputs = outputs[:, input_ids.shape[1]:]
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        output = self.postprocess(output, new_line_token)
        return output

    def postprocess(self, output, new_line_token):
        return output.replace(new_line_token, '\n').replace('\n ', '\n').strip()
