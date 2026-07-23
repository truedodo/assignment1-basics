import os
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from typing import BinaryIO
from functools import partial
import regex as re
from collections import defaultdict

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
        pair_count = defaultdict(int)
        pair_loc = defaultdict(set)
        for i, pretoken in enumerate(pretokens):
                for pair in zip(pretoken, pretoken[1:]):
                        pair_count[pair] += pretokens[pretoken]
                        pair_loc[pair].add(i)
        # after this loop, pretokens variable is unnecessary now
        
        current_size = 0
        # initialize vocab array
        vocab = defaultdict(bytes)
        for i in range(0, 127):
                vocab[i] = chr(i).encode("utf-8")
        vocab[128] = b"<|endoftext|>"
        current_size=129
        # initialize merges array
        merges = []
        print(pair_count)

        while current_size <= vocab_size:
                # order all pairs
                max_freq = max(pair_count.values())
                most_freq = [k for k,v in pair_count.items() if v == max_freq]
                greatest = sorted(most_freq)[-1]
                greatest = max(pair_count, key=lambda k: (pair_count[k], k)) # lexicographically greatest
                # now for most freq,
                # add to vocab array
                vocab[current_size]=greatest[0]+greatest[1]
                current_size+=1
                # add to merges array
                merges.append(greatest)

                for loc in pair_loc[greatest]:
                        for pair in loc:
                                if pair==greatest:
                                        # merge
                                        
                                        continue
                                if (greatest[0] or greatest[1]) in pair:
                                        pair_count[pair]-=1
                                        
                                        # delete count
                                        continue
                        
                        # rerun to add new pairs
                        for pair in loc:
                             continue

                # remove most freq from pair_count, pair_loc also
                pair_count.pop(greatest, None)
                pair_loc.pop(greatest, None)
                continue


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