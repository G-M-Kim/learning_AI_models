"""Small, readable models. Inputs: images [B,1,8,8], recall [B,T,10], tokens [B,T]."""
import math
import torch
from torch import nn


class Spike(torch.autograd.Function):
    """Hard threshold forward; fast-sigmoid surrogate derivative backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + 5 * x.abs()).square()


def lif(current, membrane):
    # Leaky accumulation, threshold=1, detached subtractive reset.
    voltage = 0.9 * membrane + current
    spike = Spike.apply(voltage - 1)
    return spike, voltage - spike.detach()


class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.net(x)


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                                 nn.ReLU(), nn.Flatten(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


class ImageSNN(nn.Module):
    def __init__(self, convolution=False, steps=8):
        super().__init__()
        self.steps = steps
        self.encoder = (nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.AvgPool2d(2), nn.Flatten())
                        if convolution else nn.Sequential(nn.Flatten(), nn.Linear(64, 64)))
        self.head = nn.Linear(128 if convolution else 64, 10)

    def forward(self, x):
        current = self.encoder(x)  # Direct current encoding, NOT Poisson spikes.
        membrane = torch.zeros_like(current)
        rate = torch.zeros_like(current)
        for _ in range(self.steps):
            spike, membrane = lif(current, membrane)
            rate = rate + spike
        return self.head(rate / self.steps)


class RecallMLP(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(length * 10, 64), nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net(x)


class RecallRNN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.rnn = getattr(nn, kind)(10, 48, batch_first=True)
        self.head = nn.Linear(48, 8)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.head(output[:, -1])


class SRNN(nn.Module):
    """Recurrent LIF with learned spike feedback; SRNN has no single canonical cell."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(10, 48)
        self.recurrent = nn.Linear(48, 48, bias=False)
        self.head = nn.Linear(48, 8)

    def forward(self, x):
        membrane = x.new_zeros(x.size(0), 48)
        spike = torch.zeros_like(membrane)
        for sample in x.unbind(1):
            spike, membrane = lif(self.input(sample) + self.recurrent(spike), membrane)
        # Analog membrane readout retains subthreshold information.
        return self.head(membrane)


class RecallAttention(nn.Module):
    def __init__(self, length, spiking=False):
        super().__init__()
        self.spiking = spiking
        self.embed = nn.Linear(10, 32)
        self.position = nn.Parameter(torch.randn(1, length, 32) * .02)
        if spiking:
            self.qkv = nn.Linear(32, 96)
            self.norm = nn.LayerNorm(32)
        else:
            layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(32, 8)

    def forward(self, x):
        h = self.embed(x) + self.position
        if self.spiking:
            # Single-step spike Q/K/V, softmax-free normalized coincidence attention.
            # Teaching hybrid: analog embedding/residual/readout. NOT Spikformer reproduction.
            q, k, v = (Spike.apply(t) for t in self.qkv(h).chunk(3, -1))
            scores = q @ k.transpose(-1, -2) / math.sqrt(32)
            weights = scores / scores.sum(-1, keepdim=True).clamp_min(1)
            h = self.norm(h + weights @ v)
        else:
            h = self.encoder(h)
        return self.head(h[:, -1])


class LanguageModel(nn.Module):
    def __init__(self, kind, vocab, length):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab, 48)
        self.position = nn.Embedding(length, 48)
        if kind == 'gpt':
            layer = nn.TransformerEncoderLayer(48, 4, 96, dropout=0, batch_first=True)
            self.backbone = nn.TransformerEncoder(layer, 2)
        else:
            self.backbone = nn.LSTM(48, 48, batch_first=True)
        self.head = nn.Linear(48, vocab)

    def forward(self, tokens):
        h = self.embed(tokens)
        if self.kind == 'gpt':
            h = h + self.position(torch.arange(tokens.size(1), device=tokens.device))
            # True blocks future tokens: no target leakage.
            mask = torch.triu(torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool), 1)
            h = self.backbone(h, mask=mask)
        else:
            h, _ = self.backbone(h)
        return self.head(h)


MODELS = {'digits': ['dnn', 'cnn', 'snn', 'scnn'],
          'recall': ['dnn', 'rnn', 'lstm', 'gru', 'srnn', 'transformer', 'spike_attention'],
          'language': ['lstm', 'gpt']}


def build(task, model, length=32, vocab=0):
    if model not in MODELS[task]:
        raise ValueError(f'{task} supports {MODELS[task]}')
    if task == 'digits':
        return {'dnn': ImageMLP, 'cnn': ImageCNN, 'snn': ImageSNN,
                'scnn': lambda: ImageSNN(convolution=True)}[model]()
    if task == 'language':
        return LanguageModel(model, vocab, length)
    if model == 'dnn':
        return RecallMLP(length)
    if model in ('rnn', 'lstm', 'gru'):
        return RecallRNN({'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU'}[model])
    if model == 'srnn':
        return SRNN()
    return RecallAttention(length, spiking=model == 'spike_attention')
