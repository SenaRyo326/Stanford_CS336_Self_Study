from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
import regex

GPT2_SPLIT_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_PATTERN = regex.compile(GPT2_SPLIT_REGEX)


def _count_chunk(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        pattern = "|".join(regex.escape(t) for t in special_tokens)
        docs = regex.split(pattern, chunk)
    else:
        docs = [chunk]

    counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for doc in docs:
        for match in _GPT2_PATTERN.finditer(doc):
            word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            counts[word] += 1
    return counts


def _find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size == 0:
            return [0, 0]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        mini_chunk_size = 4096
        for bi in range(1, len(boundaries) - 1):
            initial_pos = boundaries[bi]
            f.seek(initial_pos)
            while True:
                mini_chunk = f.read(mini_chunk_size)
                if mini_chunk == b"":
                    boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    boundaries[bi] = initial_pos + found_at
                    break
                initial_pos += mini_chunk_size

    return sorted(set(boundaries))


def bpe_train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    input_path_str = os.fspath(input_path)
    file_size = os.path.getsize(input_path_str)
    word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    num_workers = os.cpu_count() or 1
    if file_size > 1_000_000 and num_workers > 1 and special_tokens:
        split_tok = special_tokens[0].encode("utf-8")
        boundaries = _find_chunk_boundaries(input_path_str, num_workers, split_tok)
        tasks = [
            (input_path_str, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
            if boundaries[i] < boundaries[i + 1]
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for chunk_counts in executor.map(_count_chunk, tasks):
                for word, freq in chunk_counts.items():
                    word_counts[word] += freq
    else:
        word_counts = _count_chunk((input_path_str, 0, file_size, special_tokens))

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    while len(vocab) < vocab_size:
        active_pairs = {p: c for p, c in pair_counts.items() if c > 0}
        if not active_pairs:
            break

        best_pair = max(active_pairs, key=lambda p: (active_pairs[p], p))
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        affected_words = list(pair_to_words[best_pair])
        for word in affected_words:
            freq = word_counts.pop(word, 0)
            if freq == 0:
                continue

            for i in range(len(word) - 1):
                old_pair = (word[i], word[i + 1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(old_pair, None)
                pair_to_words[old_pair].discard(word)

            new_word_list: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word_list.append(merged_token)
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)

            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(new_word)

        pair_to_words.pop(best_pair, None)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self._cache: dict[bytes, list[int]] = {}

        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = regex.compile(
                "(" + "|".join(regex.escape(t) for t in sorted_special) + ")"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): v.encode("latin1") if isinstance(v, str) else bytes(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, encoding="utf-8") as f:
            merges = [
                (a.encode("latin1"), b.encode("latin1"))
                for a, b in (line.rstrip("\n").split(" ") for line in f if line.strip())
            ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        if token_bytes in self._cache:
            return self._cache[token_bytes]

        parts = [bytes([b]) for b in token_bytes]
        while len(parts) >= 2:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.bytes_to_id[p] for p in parts]
        self._cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        chunks = self._special_pattern.split(text) if self._special_pattern else [text]
        special_set = set(self.special_tokens)
        result: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                result.append(self.bytes_to_id[chunk.encode("utf-8")])
            else:
                for match in _GPT2_PATTERN.finditer(chunk):
                    result.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")