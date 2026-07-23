import os
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from typing import BinaryIO
from functools import partial
import regex as re
from collections import defaultdict
from itertools import pairwise

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''
        Trains the byte-pair tokenizer.
        '''
        with open(input_path, "rb") as f:
                num_processes = os.cpu_count()
                boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
                pretokens = defaultdict(int)
                with Pool() as pool:
                        chunk_pretokens = pool.map(partial(chunk_pretokenize, 
                                                           input_path=input_path, 
                                                           special_tokens=special_tokens, 
                                                           vocab_size=vocab_size), zip(boundaries[:-1], boundaries[1:]))
                        for chunk_pretoken in chunk_pretokens:
                                for key, value in chunk_pretoken.items():
                                        pretokens[key]+=value
        # return pretokens

        pretoken_list = list(pretokens.keys())
        pretoken_freqs = list(pretokens.values())

        pair_count = defaultdict(int)
        pair_loc = defaultdict(set)
        for i, pretoken in enumerate(pretokens):
                for pair in zip(pretoken, pretoken[1:]):
                        pair_count[pair] += pretokens[pretoken]
                        pair_loc[pair].add(i)

        
                
        current_size = 0
        # initialize vocab array
        vocab = defaultdict(bytes)
        for i in range(0, 256):
                vocab[i] = bytes([i])
        for i, spl in enumerate(special_tokens):
                vocab[i+255] = spl
        current_size = 256 + len(special_tokens)
        # initialize merges array
        merges = []
        print(pair_count)

        while current_size < vocab_size:
                # order all pairs
                greatest = max(pair_count, key=lambda k: (pair_count[k], k)) # lexicographical
                vocab[current_size]=greatest[0]+greatest[1]
                merges.append(greatest)
                current_size+=1

                for loc in pair_loc[greatest]:
                        for pair in pairwise(pretoken_list[loc]):
                                if pair==greatest:
                                        # merge
                                        continue

                        
                        # rerun to add new pairs
                        for pair in loc:
                             continue

                # remove most freq from pair_count, pair_loc also
                pair_count.pop(greatest, None)
                pair_loc.pop(greatest, None)
                continue

        return vocab, merges


def chunk_pretokenize(
                boundaries: tuple[int, int],
                input_path: str | os.PathLike,
                vocab_size: int,
                special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
        start, end = boundaries
        with open(input_path, "rb") as file:
                file.seek(start)
                chunk = file.read(end - start).decode("utf-8", errors="ignore")

                escaped_tokens = [re.escape(token) for token in special_tokens]
                split_chunks = re.split("|".join(escaped_tokens), chunk)

                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

                d = defaultdict(int)
                for split in split_chunks:
                        for match in re.finditer(PAT, split):
                                word = match.group()
                                key = tuple(bytes([b]) for b in word.encode("utf-8"))
                                d[key]+=1
                return d



if __name__ == "__main__":
        print(train_bpe("/Users/dodo/code/llm/assignment1-basics/data/a.txt", 255, ["<|endoftext|>"]))