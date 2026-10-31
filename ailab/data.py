"""Identical splits for every model; no network downloads or test-set tuning."""
import hashlib
import random
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

VOCAB = ['<bos>', '<eos>', 'the', 'red', 'blue', 'green', 'cat', 'dog', 'bird',
         'sees', 'likes', 'a', '.', 'runs', 'sleeps']


def recall(n, length, seed):
    g = torch.Generator().manual_seed(seed)
    # First symbol is the target, middle symbols are independent distractors.
    symbols = torch.randint(0, 8, (n, length), generator=g)
    y = symbols[:, 0].clone()
    symbols[:, -1] = 8  # Query: "what was the first symbol?"
    return torch.nn.functional.one_hot(symbols, 10).float(), y


def language(n, length, seed):
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        tokens = [0]
        while len(tokens) < length + 1:
            color, animal = rng.randrange(3, 6), rng.randrange(6, 9)
            if rng.random() < .5:
                tokens += [2, color, animal, rng.randrange(13, 15), 12, 1, 0]
            else:
                tokens += [2, color, animal, rng.randrange(9, 11), 11,
                           rng.randrange(3, 6), rng.randrange(6, 9), 12, 1, 0]
        docs.append(tokens[:length + 1])
    z = torch.tensor(docs)
    return z[:, :-1], z[:, 1:]


def datasets(task, length=32, quick=False):
    if task == 'digits':
        d = load_digits()
        indices = list(range(len(d.target)))
        train, rest = train_test_split(indices, test_size=.3, random_state=2026, stratify=d.target)
        val, test = train_test_split(rest, test_size=.5, random_state=2026, stratify=d.target[rest])
        x = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16
        y = torch.tensor(d.target, dtype=torch.long)
        result = [(x[i], y[i]) for i in (train, val, test)]
    else:
        make = recall if task == 'recall' else language
        result = [make(n, length, seed) for n, seed in ((4096, 101), (1024, 202), (1024, 303))]
    if quick:
        result = [(x[:min(len(x), 128)], y[:min(len(y), 128)]) for x, y in result]
    return result


def fingerprint(splits):
    h = hashlib.sha256()
    for x, y in splits:
        h.update(x.contiguous().numpy().tobytes())
        h.update(y.contiguous().numpy().tobytes())
    return h.hexdigest()
