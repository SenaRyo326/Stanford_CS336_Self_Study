import regex
from collections import defaultdict
def bpe_train(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
):

    vocab = {}
    for i in range(256):
        vocab[i]=bytes([i])
    for token in special_tokens: vocab[len(vocab)]=token.encode("utf-8")
    merges = []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        pattern ='|'.join(regex.escape(token)for token in special_tokens)
        corpus = regex.split(pattern,text)
    else: corpus=[text]

    GPT2_SPLIT_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    word_counts = defaultdict(int)

    for chuck in corpus:
        for match in regex.finditer(GPT2_SPLIT_REGEX, chuck):
            word=tuple(bytes([b]) for b in match.group().encode("utf-8"))
            word_counts[word]+=1

    while len(vocab)<vocab_size:
        pair_counts=defaultdict(int)
        for word, freqs in word_counts.items():
            if len(word)<2:continue

            for i in range(len(word)-1):
                pair=(word[i],word[i+1])
                pair_counts[pair]+=freqs
        if not pair_counts: break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        vocab[len(vocab)]=best_pair[0]+best_pair[1]
        merges.append(best_pair)

        new_word_counts = {}
        for word, freqs in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = freqs

        word_counts = new_word_counts



    return vocab, merges