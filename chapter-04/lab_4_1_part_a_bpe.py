from collections import Counter

def get_pair_counts(ids):
    """Count how often each adjacent pair of token IDs occurs."""
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe(text, vocab_size):
    """Learn byte-level BPE merges until the vocabulary reaches vocab_size."""
    ids = list(text.encode("utf-8"))            # start from raw bytes: 256 base tokens
    merges = {}                                  # (id1, id2) -> new_id
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)       # most frequent adjacent pair
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        # apply the merge that was learned earliest (lowest new_id) first
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

if __name__ == "__main__":   # needs input.txt from chapter-03/download_data.py
    corpus = open("input.txt", encoding="utf-8").read()[:200_000]   # Tiny Shakespeare
    merges, vocab = train_bpe(corpus, vocab_size=512)
    print("first five merges:", [vocab[i].decode("utf-8", "replace") for i in range(256, 261)])
    print("last five merges: ", [vocab[i].decode("utf-8", "replace") for i in range(507, 512)])

    sample = "What light through yonder window breaks?"
    ids = encode(sample, merges)
    print(len(sample.encode("utf-8")), "bytes ->", len(ids), "tokens")
    print([vocab[i].decode("utf-8", "replace") for i in ids])
    assert decode(ids, vocab) == sample                              # lossless round trip

    modern = "The transformer architecture uses self-attention."
    print(len(modern.encode("utf-8")), "bytes ->", len(encode(modern, merges)), "tokens (unfamiliar modern text)")
