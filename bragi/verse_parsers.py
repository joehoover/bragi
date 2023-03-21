import poesy
import pronouncing
import torch

from typing import Optional


def cmu_syllable_counter(word):
    """
    Returns inf for OOV tokens.
    Note: This prohibits things like numbers and punctuation. Very naive and dumb.
    """
    pronunciation_list = pronouncing.phones_for_word(word)
    if len(pronunciation_list) > 0:
        syllable_count = pronouncing.syllable_count(pronunciation_list[0])
    else:
        return float("Inf")
    
    return syllable_count

def syllable_mapper(vocab):
    syllable_map = {}
    for token, idx in vocab.items():
        n_syllables = cmu_syllable_counter(token)
        try:
            syllable_map[n_syllables].append(idx)
        except KeyError:
            syllable_map[n_syllables] = [idx]
    return syllable_map

def token_syllable_scores(tokenizer, pt=True, free_tokens=['\n', '!', ',', ':', '?', ';', ' ', '||']):
    """
    Returns list or torch tensor of size==tokenizer.vocab_size where element i is the count of syllables
    for token i.
    """
    sorted_vocab = {k: v for k, v in sorted(tokenizer.vocab.items(), key=lambda item: item[1])}
    syllable_scores = []
    for token, idx in sorted_vocab.items():

        # Have to decode the vocab item to deal with special characters, e.g. '\n' is represented as 'Ċ'
        
   
        decoded_token = tokenizer.decode(sorted_vocab[token])
        if decoded_token != ' ':
            decoded_token = decoded_token.strip()
        if decoded_token not in free_tokens:
            n_syllables = cmu_syllable_counter(decoded_token)
        else:
            n_syllables = 0

        syllable_scores.append(n_syllables)
    
    # Scoring is wrong for '\n' patching
    for token in free_tokens:
        syllable_scores[tokenizer(token, add_special_tokens=False)['input_ids'][0]] = 0
    
    if pt:
        return torch.Tensor(syllable_scores)
    return syllable_scores


class BaseParsedVerseHandler():
    """
    Base abstraction for classes of parsed text.
    """

    def __init__(
            self, 
            parser,
            new_line_chars: str = "\n",
            control_code_start_tag: str = "<PREF>", 
            control_code_end_tag: str = "</PREF>",
            control_code_length_tag: str = "<SYLLABLES: {length}>",
            control_code_end_syllable_tag: str = "<END: {syllable}>", 
            control_code_rhyme_tag: str = "<RHYME: {rhyme_id}>"
    ):
        
        self.parser = parser
        self.new_line_chars = new_line_chars
        self.control_code_start_tag = control_code_start_tag 
        self.control_code_end_tag = control_code_end_tag
        self.control_code_length_tag = control_code_length_tag 
        self.control_code_end_syllable_tag = control_code_end_syllable_tag
        self.control_code_rhyme_tag = control_code_rhyme_tag
        
        self.vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}

    def control_code(self, parsed_text):
        """
        Return the control code for verse encoded in `self.parsed_text`.
        """
        raise NotImplementedError()
    
    def example(self):
        """
        Generate example
        """


    def syllabic_text(self, line_break_chars = "\n"):
        """
        Returns formatted parse of text
        """
        raise NotImplementedError()
    
class PoesyParsedVerseHandler(BaseParsedVerseHandler):
    """
    This class provides wrappers around various poesy methods. It can be used 
    to calculate the syllabic structure of an input text as well as
    format training data for fine-tuning with control codes. 

    Example:

        ```
        from bragi.verse_parsers import PoesyParsedVerseHandler
        verse_handler = PoesyParsedVerseHandler()

        text = "An elderly man called Keith,\
        Mislaid his set of false teeth.\
        They'd been laid on a chair,\
        He'd forgot they were there,\
        Sat down, and was bitten beneath."

        print(text)

        example, syllable_budget = verse_handler.example(text)
        print(f"\nThe lines of the limerick above have the following syllable counts: {syllable_budget}")
        ```
    """

    def __init__(self,):

        super().__init__(parser=poesy.Poem)

    def _merge_end_syllables(self, split_syllables):
        """
        Sometimes an end syllable won't have vowels, 
        so we need to merge with the previous syllable to ensure
        a rhyme can be formed. E.g., "Mary" -> "Ma|ry" -> "Mary".
        """
        end_syllable = split_syllables[-1].strip()

        if self.vowels.isdisjoint(end_syllable):
            for i in reversed(range(len(split_syllables) - 1)):
                end_syllable = split_syllables[i].strip() + end_syllable
                if not self.vowels.isdisjoint(end_syllable):
                    break

        return(end_syllable)
    
    def parse_text(self, text, **kwargs):
        return self.parser(text, **kwargs)
    
    def _format_example(self, text, control_code):
        
        return control_code + '\n'*2 + text

    def example(self, text, **kwargs):
        parsed_text = self.parse_text(text, **kwargs)
        control_code, syllable_counts = self.control_code(parsed_text)
        return self._format_example(text, control_code), syllable_counts


    def control_code(
            self,
            parsed_text: poesy.Poem,
            syllabic_text = False, 
        ):
        
        lines = ""
        # syllable_counts 

        control_code = self.control_code_start_tag + self.new_line_chars
        syllable_counts = []
        rhyme_id_counter = {}

        for line in parsed_text.lineld:

            # Note: it would be nice to retain some capitals (e.g. proper nouns,
            # capitals in original text) but, I'll leave that for another time.
            syllables = line.get("parse").replace('*', '').lower().split('.')
            split_syllables = [i.split('|') for i in syllables]
            split_syllables = [item for sublist in split_syllables for item in sublist]
 
            end_syllable = self._merge_end_syllables(split_syllables)
            
            
            # end_syllable = syllables[-1].split('|')[-1].strip()
            syllable_counts.append(line.get("num_sylls"))
            control_code += self.control_code_length_tag.format(length=line.get("num_sylls"))
            control_code += self.control_code_rhyme_tag.format(rhyme_id=line.get("rhyme").upper())
            # Control code shouldn't constrain rhyme, it should be 
            # freely generated. 
            # control_code += self.control_code_end_syllable_tag.format(syllable=end_syllable)
            control_code += self.new_line_chars
            
            if syllabic_text:
                # Note: it would be nice to retain some capitals (e.g. proper nouns,
                # capitals in original text) but, I'll leave that for another time.
                lines += ' '.join(syllables) + self.new_line_chars
            
        control_code += self.control_code_end_tag
        return control_code, syllable_counts