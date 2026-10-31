# AI model learning lab: contributor instructions

## Purpose and audience

This is a Korean educational toy project for learning how DNN/MLP, CNN, RNN,
LSTM, GRU, Transformer and spiking variants behave on comparable test data.
The learner is comfortable with RTL/hardware concepts but is new to Python ML.
Prefer short readable PyTorch code, explicit tensor shapes, and Korean teaching
material over training frameworks or unnecessary abstractions.

Read `README.md`, `docs/CODEX_HANDOFF.md`, and the relevant implementation before
editing. The repository and the user's current request are the source of truth;
do not assume access to an earlier ChatGPT conversation.

## Runtime and entry points

- Target PC: Windows 11, RTX 4070 Ti (12 GB VRAM), 32 GB RAM.
- Documented environment: Python 3.11, torch 2.7.1 CUDA 12.6, requirements.txt.
- `ailab/data.py`: deterministic splits and generated data.
- `ailab/models.py`: small model implementations and model registry.
- `ailab/run.py`: train/validation selection/test once, artifacts, protocol hash.
- `ailab/suite.py`: sequential model/seed runs. `ailab/report.py`: offline report.
- `scripts/build_study.py` generates `docs/study.html`; update the generator first.
- GPU is the normal default. CPU fallback must be explicit, not silent.

## Preserve scientific meaning

- Compare models within the same task, data split and experiment protocol.
- Never tune on test labels or choose the checkpoint by test performance.
- Keep causal masking and input/target shifting correct for language models.
- Do not manufacture monotonic accuracy improvements with model generation.
- Record measured results with seed/device/config. Do not label CPU data as GPU.
- Existing `examples/cpu_reference/` is historical evidence; do not overwrite it
  merely because code or results change. Add a clearly labeled new reference.
- SNNs use surrogate gradients; reset sample/batch state unless the task defines
  stateful streaming. The spike attention toy is not a Spikformer reproduction.
- No universal canonical spiking LSTM/GRU correspondence should be invented.
- `runs/`, `.venv/`, credentials, and trained checkpoints stay out of Git.

## Verification and handoff

Use the Python interpreter for the active project environment. On Windows this
can be `.\.venv\Scripts\python.exe`; on other systems inspect the environment.

```text
python -m unittest discover -s tests -v
python -m ailab.run --task digits --model dnn --quick --epochs 1 --device cpu
python -m ailab.report
```

Run a relevant small smoke experiment for changed model/task paths. For a
CUDA-specific change, use `python -m ailab.doctor --require-cuda` and a small GPU
run when GPU is available. If unavailable, say GPU validation remains pending.
Do not repeat the full 20 reference trainings for a documentation-only change.

Update the explanation and setup instructions when behavior changes. If the
HTML generator changes, regenerate the checked-in HTML. Report changed files,
verification, limitations, branch/commit, and exact next command to the learner.
Use a feature branch/PR for shared changes; leave merging to the user unless
explicitly asked to merge. Preserve unrelated local edits and existing runs.
