"""Inspect held-out predictions, or generate text with a trained tiny LM."""
import argparse
from pathlib import Path
import torch
from .data import VOCAB, datasets
from .models import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Local runs/.../best.pt; only load trusted files')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    if args.count < 1:
        p.error('count must be positive')
    checkpoint = torch.load(Path(args.checkpoint), weights_only=True, map_location='cpu')
    c = checkpoint['config']
    model = build(c['task'], c['model'], c['length'], len(VOCAB)).to(args.device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    x, y = datasets(c['task'], c['length'], c['quick'])[2]
    with torch.no_grad():
        if c['task'] == 'language':
            tokens = torch.tensor([[0]], device=args.device)
            torch.manual_seed(42)
            for _ in range(60):
                logits = model(tokens[:, -c['length']:])[:, -1]
                next_token = torch.multinomial(logits.softmax(-1), 1)
                tokens = torch.cat([tokens, next_token], 1)
            print(' '.join(VOCAB[t] for t in tokens[0].tolist()))
        else:
            predictions = model(x[:args.count].to(args.device)).argmax(-1).cpu()
            for i, guess in enumerate(predictions.tolist()):
                print(f'test[{i}] expected={y[i].item()} predicted={guess}')
                if c['task'] == 'recall':
                    print('input:', x[i].argmax(-1).tolist(), '(8=query)')


if __name__ == '__main__':
    main()
