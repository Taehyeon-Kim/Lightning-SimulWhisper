import mlx.core as mx
import sys
class TokenBuffer:

    def __init__(self, text="", tokenizer=None, prefix_token_ids=[]):
        self.text = text
        self.prefix_token_ids = prefix_token_ids
        self.tokenizer = tokenizer

    def as_token_ids(self, tokenizer=None):

        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not set.") 
        return self.prefix_token_ids + self.tokenizer.encode(self.text)

    def as_tensor(self):
        tok_ids = self.as_token_ids()
        return mx.array(tok_ids, dtype=mx.int64)[None, :]

    def as_tensor_beam(self, beam):
        t = self.as_tensor()
        return mx.repeat(t, repeats=beam, axis=0)


    def as_text(self):
        return self.text

    @staticmethod
    def empty(*a, **kw):
        return TokenBuffer(*a,**kw)

    @staticmethod
    def from_text(text, *a, **kw):
        return TokenBuffer(*a, text=text, **kw)
    
    def is_empty(self):
        return self.text is None or self.text == ""

    def trim_words(self, num=1, after=0):
        '''
        num: how many words to trim from the beginning
        after: how many characters to skip (length of the static prompt)
        '''
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."

        ids = tokenizer.encode(self.text[after:])
        words, wids = self.tokenizer.split_to_word_tokens(ids)
#        print(words, file=sys.stderr)
#        print(wids, file=sys.stderr)
        if not words:
            return 0
        self.text = self.text[:after] + "".join(words[num:])
        return sum(len(wi) for wi in wids[:num])

    def append_token_ids(self, token_ids):
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."
        self.text += self.tokenizer.decode(token_ids)

    def as_split_word_tokens(self):
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."
        ids = tokenizer.encode(self.text)
        return tokenizer.split_to_word_tokens(ids)
