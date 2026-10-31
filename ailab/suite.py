"""Run comparable models, one at a time to keep GPU memory bounded."""
import argparse
import subprocess
import sys
from .models import MODELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=MODELS, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--models', nargs='+', help='Optional subset')
    p.add_argument('--output', default='runs')
    args, rest = p.parse_known_args()
    for model in args.models or MODELS[args.task]:
        if model not in MODELS[args.task]:
            p.error(f'Unsupported model: {model}')
        for seed in args.seeds:
            subprocess.run([sys.executable, '-m', 'ailab.run', '--task', args.task,
                            '--model', model, '--seed', str(seed), '--output', args.output, *rest], check=True)
    subprocess.run([sys.executable, '-m', 'ailab.report', '--root', args.output], check=True)


if __name__ == '__main__':
    main()
