# special tokens remove
# pretokenize
# merge using merges array

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        "initalize vocab, merges from BPE training"
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        raise

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as f:
            # do regex? and convert to list

            return
        with open(merges_filepath, 'r') as f:
            return
        self.special_tokens = special_tokens

            

    def encode(self, text: str) -> list[int]:
        raise

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise

    def decode(self, ids: list[int]) -> str:
        raise