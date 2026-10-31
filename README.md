# AI 모델 발전 실습실 · DNN → Transformer + SNN

**코드를 읽고 → 작은 모델을 학습하고 → 같은 test set에서 결과를 비교하는 한국어 실습.** Windows 11 / RTX 4070 Ti / Python 3.11 / PyTorch 2.7.1 CUDA 12.6을 기준으로 합니다. API 키가 필요하지 않습니다.

처음 사용하는 PC라면 **[설치부터 실행까지](docs/SETUP_WINDOWS.md)** 를 먼저 따라 하세요. 시각화 교재 **[docs/study.html](docs/study.html)** 은 내려받은 뒤 브라우저로 열어야 렌더링됩니다. 외부 CDN 없이 오프라인에서 작동합니다.

## 권장 학습 순서

| 단계 | 입력/과제 | 기본 모델 | 함께 볼 SNN 계열 | 주요 관찰 |
|---|---|---|---|---|
| 1 | 8×8 손글씨 숫자 / 10분류 | DNN(MLP) | LIF SNN | 역전파, surrogate gradient |
| 2 | 같은 숫자, 같은 split | CNN | Spiking CNN | local receptive field, weight sharing |
| 3 | 첫 기호 기억 / 8분류 | RNN | SRNN | 시간 상태, BPTT, 기억 소실 |
| 4 | 같은 기억 과제 | LSTM, GRU | SRNN을 다시 비교 | gate와 leak의 차이 |
| 5 | 같은 기억 과제 | Transformer encoder | spike attention toy | Q/K/V, 위치 정보, 직접 참조 |
| 6 | 짧은 문법 문장 / 다음 token | LSTM LM, tiny GPT | attention 원리 실습으로 연결 | causal mask, teacher forcing, 생성 |

DNN은 큰 범주이며 이 실습의 `dnn`은 fully-connected MLP를 뜻합니다. CNN/RNN/Transformer는 단순한 일렬 대체 관계가 아닙니다. **과제 사이의 정확도는 직접 비교하지 않습니다. 새 모델의 우승을 보장하지 않습니다.** SNN도 전체 모델 계열이며 LSTM/GRU 각각에 표준 1:1 spiking 대응이 있는 것은 아닙니다. `spike_attention`은 혼합형 교육 모델로 Spikformer 재현이나 spiking LLM을 뜻하지 않습니다.

## 설치가 끝난 뒤

아래 `python`은 `.venv`의 Python입니다. [Windows 안내](docs/SETUP_WINDOWS.md)의 Set-Alias를 쓰거나 `.\.venv\Scripts\python.exe`로 바꾸세요.

```powershell
python -m ailab.doctor --require-cuda
python -m ailab.run --task digits --model dnn --quick --epochs 2
python -m ailab.suite --task digits --epochs 15 --seeds 42 43 44
python -m ailab.report
Start-Process .\runs\comparison.html
Start-Process .\docs\study.html
```

단일 모델로 공부할 때:

```powershell
python -m ailab.run --task digits --model cnn --epochs 15
python -m ailab.run --task digits --model scnn --epochs 15
python -m ailab.run --task recall --model rnn --length 32 --epochs 15
python -m ailab.run --task recall --model srnn --length 32 --epochs 15
python -m ailab.run --task recall --model lstm --length 32 --epochs 15
python -m ailab.run --task recall --model gru --length 32 --epochs 15
python -m ailab.run --task recall --model transformer --length 32 --epochs 15
python -m ailab.run --task recall --model spike_attention --length 32 --epochs 15
python -m ailab.run --task language --model lstm --epochs 15
python -m ailab.run --task language --model gpt --epochs 15
```

전체 비교 및 기억 거리 실험:

```powershell
python -m ailab.suite --task recall --length 8 --epochs 15 --seeds 42 43 44
python -m ailab.suite --task recall --length 32 --epochs 15 --seeds 42 43 44
python -m ailab.suite --task language --epochs 15 --seeds 42 43 44
```

길이 8과 32는 서로 다른 group입니다. 길이에 따라 각 모델이 얼마나 떨어지는지 보되 동일 group에서 먼저 비교하세요. CPU를 명시적으로 쓰려면 `--device cpu`. 기본은 CUDA이며 GPU 설정 실패를 조용히 CPU로 숨기지 않습니다.

## 결과와 비교의 규칙

- `runs/과제_모델_seed_시간/`: config.json, history.csv, result.json, best.pt.
- `runs/comparison.html`: train/validation loss 그래프, seed별 결과, seed 평균과 표본 SD, 파라미터 수·시간·GPU 할당 메모리.
- `runs/summary.csv`: Excel에서 열 수 있는 전체 기록. 언어 모델은 token accuracy와 perplexity도 기록.
- test split은 checkpoint 선택에 사용하지 않습니다. validation loss가 가장 낮은 checkpoint를 고른 뒤 test를 한 번 평가합니다.
- 데이터 fingerprint·학습 예산·길이·batch·lr·장치·주요 버전이 같을 때만 같은 group. seed는 initialization/minibatch 순서만 바꾸며 split은 고정합니다. 반복된 동일 seed는 평균에 최신 결과 하나만 사용합니다.
- 파라미터 수를 강제로 맞추지는 않습니다. 이는 **동일 데이터·epoch 예산 비교**이며, 동일 FLOPs/시간/파라미터 또는 각 모델 최적 튜닝 비교가 아닙니다.
- quick run은 기능 점검. seed 하나는 결론의 근거로 약합니다. SNN spike sparsity는 이 GPU 시뮬레이터에서 에너지 절감 측정값이 아닙니다.

## 다음 수정과 Codex 인수인계

[docs/CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md)에 현재 구현, 파일 지도, 실행 환경, 수정 요청 예시를 정리했습니다. 루트 [AGENTS.md](AGENTS.md)는 에이전트용 작업 지침입니다. 저장소를 VS Code로 열고 두 문서를 먼저 읽도록 요청한 뒤 이번 수정 사항을 전달하면 됩니다.

## 파일을 읽는 순서

1. `ailab/data.py`: 입력과 정답, 분할, 언어 shift.
2. `ailab/models.py`: 작은 모델 전체와 tensor shape 주석.
3. `ailab/run.py`: 학습·검증·test·checkpoint.
4. `ailab/predict.py`: 학습한 모델로 held-out sample 예측/문장 생성.
5. `docs/study.html`: 각 모델 설명·구조 그림·수식·실험·예상과 반례·LIF/causal mask 조작.
6. `docs/EXPERIMENTS.md`: 공정한 비교 방법, CPU에서 실제 얻은 예시 결과.

## 데이터와 범위

Digits는 scikit-learn 내장 UCI optical recognition digits, 1,797개(8×8 gray), 70/15/15 stratified split입니다. 라이브러리 설치 후 추가 다운로드가 없습니다. 나머지는 seed를 고정한 생성기와 vocabulary를 저장소에 포함했습니다. `examples/test_samples.json`에는 세 과제의 실제 입력·정답 예시가 있습니다. 학습/검증/test는 독립 seed로 생성합니다. 언어 데이터는 자연어 능력을 검증하는 corpus가 아니라 문법을 배우는 초소형 toy입니다. 일반적인 대규모 모델의 성능/학습시간으로 확대 해석하지 않습니다.

출처: [scikit-learn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [UCI 원자료](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). 설치 wheel·논문·SNN 자료는 교재 마지막에 연결했습니다.

## 검증

```powershell
python -m unittest discover -s tests -v
```

CI는 Linux/Windows에서 CPU로 모든 모델의 gradient, 언어 causal mask, split 재현성, SNN 상태 리셋, metric 집계를 검사합니다. NVIDIA GPU 검증은 사용자 PC의 doctor와 실제 run으로 확인합니다.
