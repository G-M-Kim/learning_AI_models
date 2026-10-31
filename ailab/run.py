"""Train one model, select by validation loss, evaluate held-out test once."""
import argparse
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import VOCAB, datasets, fingerprint
from .models import MODELS, build


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=MODELS, default='digits')
    p.add_argument('--model', default='dnn')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--quick', action='store_true', help='128 examples per split; functionality check, not benchmark')
    p.add_argument('--output', default='runs')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Seeds reduce noise; floating-point results need not be bit-identical across platforms.


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y = y.to(device)
            flat = logits.reshape(-1, logits.size(-1))
            loss_sum += nn.functional.cross_entropy(flat, y.reshape(-1), reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
    return loss_sum / count, correct / count


def train(args):
    if min(args.epochs, args.batch_size) < 1 or args.length < 4 or args.lr <= 0:
        raise ValueError('epochs/batch-size > 0, length >= 4, lr > 0 required')
    if args.model not in MODELS[args.task]:
        raise ValueError(f'{args.task} models: {MODELS[args.task]}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run python -m ailab.doctor and see docs/SETUP_WINDOWS.md. CPU requires explicit --device cpu.')
    device = torch.device(args.device)
    torch.set_num_threads(2)  # Tiny workloads: avoid excessive CPU threading.
    seed_everything(args.seed)
    splits = datasets(args.task, args.length, args.quick)
    data_hash = fingerprint(splits)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(TensorDataset(x, y), batch_size=args.batch_size,
                          shuffle=(i == 0), generator=generator if i == 0 else None,
                          num_workers=0) for i, (x, y) in enumerate(splits)]
    model = build(args.task, args.model, args.length, len(VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    folder = Path(args.output) / f'{args.task}_{args.model}_seed{args.seed}_{stamp}'
    folder.mkdir(parents=True)
    source_hash = hashlib.sha256()
    for name in ('data.py', 'models.py', 'run.py'):
        source_hash.update((Path(__file__).parent / name).read_bytes())
    config = vars(args).copy()
    config.update(source_sha256=source_hash.hexdigest(), numpy=np.__version__, platform=platform.platform())
    config.update(data_sha256=data_hash, torch=str(torch.__version__), sklearn=sklearn.__version__,
                  python=platform.python_version(), device_name=(torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor() or 'CPU'))
    # Equal data, budget and software/hardware; model and seed intentionally excluded.
    protocol = {k: v for k, v in config.items() if k not in ('model', 'seed', 'output')}
    config['comparison_group'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best, history = float('inf'), []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: reduce learning rate; run not benchmarked.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        val_loss, val_acc = evaluate(model, loaders[1], device)
        history.append(dict(epoch=epoch + 1, train_loss=total/count, val_loss=val_loss, val_accuracy=val_acc))
        print(f'{args.model} epoch {epoch+1:02d}: train={total/count:.4f} val={val_loss:.4f} val_acc={val_acc:.3f}', flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': config, 'best_epoch': epoch + 1}, folder / 'best.pt')
    sync(device)
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state'])
    # Test labels are used only here, after checkpoint selection.
    sync(device)
    start = time.perf_counter()
    test_loss, test_acc = evaluate(model, loaders[2], device)
    sync(device)
    result = dict(task=args.task, model=args.model, seed=args.seed,
                  comparison_group=config['comparison_group'], data_sha256=data_hash,
                  quick=args.quick, parameters=sum(p.numel() for p in model.parameters()),
                  best_epoch=checkpoint['best_epoch'], test_loss=test_loss, test_accuracy=test_acc,
                  perplexity=math.exp(min(test_loss, 50)) if args.task == 'language' else None,
                  training_seconds=training_seconds, test_seconds=time.perf_counter()-start,
                  peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
                  device=config['device_name'], history=history)
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (folder / 'history.csv').write_text('epoch,train_loss,val_loss,val_accuracy\n' + '\n'.join(
        f"{r['epoch']},{r['train_loss']},{r['val_loss']},{r['val_accuracy']}" for r in history), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2))
    print(f'Saved: {folder}')
    return folder


if __name__ == '__main__':
    train(arguments())
