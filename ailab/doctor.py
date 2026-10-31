"""python -m ailab.doctor --require-cuda"""
import argparse
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require-cuda', action='store_true')
    args = p.parse_args()
    print('Python:', sys.version.split()[0], '\nPyTorch:', torch.__version__)
    print('Wheel CUDA runtime:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print('GPU:', p.name, '\nVRAM GiB:', round(p.total_memory/1024**3, 2))
        x = torch.randn(128, 128, device='cuda', requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        print('CUDA forward/backward: PASS')
    elif args.require_cuda:
        print('FAIL: check NVIDIA driver and CUDA PyTorch wheel. See docs/SETUP_WINDOWS.md.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
