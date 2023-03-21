
import torch
from typing import Optional, List

from transformers.generation.logits_process import LogitsProcessorList

from .verse_parsers import PoesyParsedVerseHandler, token_syllable_scores
from .logit_warpers import SyllableRestrictionWarper

class MetricGenerator():
    """
    This class can be used to generate strings that adhere to a specific metric structure.
    You need to provide an initialized model and tokenizer. 
    Currently, only `gpt2` and `LLaMA` are tested.
    """
    def __init__(
        self, 
        model, 
        tokenizer, 
        syllable_scorer: callable = token_syllable_scores,
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
        """
        This method injects the `SyllableRestrictionWarper` and returns token ids.
        You can pass any standard `Transformers.model.generate()` methods.
        However, you have to be careful with `new_line_token`. Under the current implementation, 
        it needs to resolve to a single token. For LLaMA, you should use the default '||'. 
        For GPT2, you can just set `new_line_token='\n'`.
        """
        
        if text_init and syllable_budget:
            raise Error("You cannot specify both `syllable_budget` and `text_init`. Choose one or the other.")
        
        if not text_init and not torch.is_tensor(syllable_budget):
            raise Error("You must provide either `syllable_budget` or `text_init`.")

        if isinstance(syllable_budget, list):
            syllable_budget = torch.Tensor(syllable_budget)
        
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

        return outputs


    def postprocess(self, output, new_line_token):
        return output.replace(new_line_token, '\n').replace('\n ', '\n').strip()
    
    def __call__(
        self, 
        prompt, 
        text_init: Optional[str] = None,
        syllable_budget: Optional[torch.Tensor] = None, 
        free_tokens: Optional[List] = ['\n', '!', ',', ':', '?', ';', ' ', '||'],
        num_beams: Optional[int] = 1, 
        return_full_text: bool = False,
        new_line_token: str = '||',
        **kwargs
    ) -> str:

        """
        This method injects the `SyllableRestrictionWarper` and returns decoded token ids.
        You can pass any standard `Transformers.model.generate()` methods.
        However, you have to be careful with `new_line_token`. Under the current implementation, 
        it needs to resolve to a single token. For LLaMA, you should use the default '||'. 
        For GPT2, you can just set `new_line_token='\n'`.

        Example:

        ```

        from bragi.metric_generator import MetricGenerator
        from transformers import LLaMAForCausalLM, LLaMATokenizer
        import torch 

        device = ...
        model = LLaMAForCausalLM.from_pretrained(...).to(device)
        tokenizer = LLaMATokenizer.from_pretrained(...)
        generator = MetricGenerator(model=model, tokenizer=tokenizer, device=device)

        text_init = "Happy birthday to you,\nHappy birthday to you,\nHappy birthday dear Marvin,\nHappy birthday to you"
        
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

        assert output == 'I love how the dogs can \nbe so good, so bad, and \nso strange. Sometimes I love them \nwhen they are the most good'
        ```
        """
        

        outputs = self.generate(
            prompt = prompt,
            text_init = text_init,
            syllable_budget = syllable_budget,
            free_tokens = free_tokens,
            num_beams = num_beams,
            return_full_text = return_full_text,
            new_line_token = new_line_token,
            **kwargs,
        )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        decoded_output = self.postprocess(decoded_output, new_line_token)

        return decoded_output